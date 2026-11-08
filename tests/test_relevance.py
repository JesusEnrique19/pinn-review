import numpy as np
import pandas as pd

from pinnreview import relevance
from pinnreview.relevance import threshold_for_recall, wilson


def test_threshold_reaches_target_recall():
    rng = np.random.default_rng(0)
    y = np.array([1] * 100 + [0] * 100)
    scores = np.concatenate([rng.uniform(0.3, 1.0, 100), rng.uniform(0.0, 0.7, 100)])
    thr = threshold_for_recall(y, scores, target=0.95)
    recall = ((scores >= thr) & (y == 1)).sum() / 100
    assert recall >= 0.95
    # and it is the highest such threshold: the next positive score up would drop below target
    higher = np.sort(scores[y == 1])[np.sort(scores[y == 1]) > thr]
    if len(higher):
        assert ((scores >= higher[0]) & (y == 1)).sum() / 100 < 0.95


def test_wilson_interval_known_value():
    lo, hi = wilson(81, 100)
    assert lo < 0.81 < hi
    assert abs(lo - 0.722) < 0.002 and abs(hi - 0.874) < 0.002


def test_human_audit_overrides_llm_label(tmp_path, monkeypatch):
    monkeypatch.setattr(relevance, "G1", tmp_path)
    pd.DataFrame({"id": ["W1", "W2"], "title": ["a", "b"], "abstract": ["", ""], "in_scope": ["y", "n"]}).to_csv(
        tmp_path / "labels_x.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame({"id": ["W1", "W2"], "human_label": ["n", ""], "llm2_label": ["y", "y"]}).to_csv(
        tmp_path / "audit_v1.csv", index=False, encoding="utf-8-sig")
    df = relevance.load_labels("x", ["labels_{v}.csv"])
    # W1: human "n" beats both LLM passes; W2: second pass "y" replaces first pass "n"
    assert dict(zip(df["id"], df["y"])) == {"W1": 0, "W2": 1}
    assert dict(zip(df["id"], df["human"])) == {"W1": True, "W2": False}
    assert dict(zip(df["id"], df["second_pass"])) == {"W1": False, "W2": True}
