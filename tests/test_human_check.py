import json

import pandas as pd
import pytest

from pinnreview import human_check


def test_score_reports_agreement_and_kappa_per_task(tmp_path, monkeypatch):
    sample, key, snap = tmp_path / "s.csv", tmp_path / "k.csv", tmp_path / "h.json"
    monkeypatch.setattr(human_check, "SAMPLE", sample)
    monkeypatch.setattr(human_check, "KEY", key)
    monkeypatch.setattr(human_check, "SNAPSHOT", snap)
    tasks = ["relevance"] * 4 + ["screening"] * 4
    pd.DataFrame({"item": range(1, 9), "task": tasks, "human_label": list("yynn") + list("YyNn")}).to_csv(
        sample, index=False, encoding="utf-8-sig")
    pd.DataFrame({"item": range(1, 9), "task": tasks, "id": range(8), "llm_label": list("yyny") + list("yynn")}).to_csv(
        key, index=False)
    human_check.score()
    out = json.loads(snap.read_text(encoding="utf-8"))
    assert out["tasks"]["screening"]["kappa"] == 1.0
    assert out["tasks"]["relevance"]["agreement"] == 0.75
    assert out["tasks"]["relevance"]["disagreements"] == [4]
    assert out["tasks"]["all"]["n"] == 8


def test_score_refuses_incomplete_labels(tmp_path, monkeypatch):
    sample, key = tmp_path / "s.csv", tmp_path / "k.csv"
    monkeypatch.setattr(human_check, "SAMPLE", sample)
    monkeypatch.setattr(human_check, "KEY", key)
    pd.DataFrame({"item": [1, 2], "task": ["relevance"] * 2, "human_label": ["y", ""]}).to_csv(sample, index=False)
    pd.DataFrame({"item": [1, 2], "task": ["relevance"] * 2, "id": [0, 1], "llm_label": ["y", "n"]}).to_csv(key, index=False)
    with pytest.raises(SystemExit):
        human_check.score()
