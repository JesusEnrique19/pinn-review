"""Human check of LLM labels (protocol Section 11; threats to validity).

A human author labels a random sample blind to the LLM labels; agreement (Cohen's kappa) is reported in the paper.

    python -m pinnreview.human_check make    # -> data/processed/human_check/sample_v1.csv (blind) + key_v1.csv
    python -m pinnreview.human_check score   # after filling human_label -> snapshots/human_check_v1.json

Sample: 30 records of the independent relevance test set (IC1 in scope, y/n; these labels give the classifier's
precision and recall) and 20 core-set screening decisions (IC4 met, y/n), drawn at random with the project seed.
The sample file shows only title, abstract and the question; the LLM labels are in the key file, which the human
must not open before scoring. Only the author fills in `human_label` (y or n) and, optionally, `human_notes`.
"""

import argparse
import json
import math

import numpy as np
import pandas as pd

from pinnreview.codebook import cohen_kappa
from pinnreview.paths import PROCESSED
from pinnreview.validation import SEED

OUT = PROCESSED / "human_check"
SAMPLE, KEY = OUT / "sample_v1.csv", OUT / "key_v1.csv"
SNAPSHOT = PROCESSED / "snapshots" / "human_check_v1.json"
QUESTIONS = {
    "relevance": "IC1: does the work use neural networks to solve, learn or identify differential equations or "
                 "their solution operators (forward, inverse or parametric), with the differential equation in the "
                 "loss, the architecture or the learned map? (y/n)",
    "screening": "IC4: does the work propose or empirically evaluate a PINN/neural-operator method, or analyse their "
                 "training or approximation theoretically? Application papers that only use an off-the-shelf "
                 "PINN/operator are n. (y/n)",
}


def make(n_relevance=30, n_screening=20):
    if SAMPLE.exists():
        raise SystemExit(f"{SAMPLE} exists; it may already hold human labels. Delete it explicitly to redraw.")
    rng = np.random.default_rng(SEED)
    rel = pd.read_csv(PROCESSED / "g1" / "relevance_test_v3.csv")
    scr = pd.read_csv(PROCESSED / "g3" / "screening_v3.csv").dropna(subset=["decision"])
    parts = [
        rel.iloc[rng.choice(len(rel), n_relevance, replace=False)].assign(task="relevance", llm_label=lambda d: d["in_scope"]),
        scr.iloc[rng.choice(len(scr), n_screening, replace=False)].assign(task="screening", llm_label=lambda d: d["decision"]),
    ]
    s = pd.concat(parts, ignore_index=True)
    s = s.iloc[rng.permutation(len(s))].reset_index(drop=True)  # mix tasks so the order does not reveal anything
    s["item"] = np.arange(1, len(s) + 1)
    s["question"] = s["task"].map(QUESTIONS)
    s["abstract"] = s["abstract"].fillna("(no abstract available)")
    OUT.mkdir(parents=True, exist_ok=True)
    blind = s[["item", "task", "question", "title", "abstract"]].assign(human_label="", human_notes="")
    blind.to_csv(SAMPLE, index=False, encoding="utf-8-sig")  # BOM so that Excel shows accents correctly
    s[["item", "task", "id", "llm_label"]].to_csv(KEY, index=False)
    print(f"{SAMPLE.relative_to(PROCESSED.parent.parent)}: {len(s)} items ({n_relevance} relevance, {n_screening} screening)")


def kappa_ci(a, b, n_boot=2000):
    rng = np.random.default_rng(SEED)
    a, b = np.asarray(a), np.asarray(b)
    ks = []
    for _ in range(n_boot):
        i = rng.integers(0, len(a), len(a))
        k = cohen_kappa(list(a[i]), list(b[i]))
        if not math.isnan(k):
            ks.append(k)
    return [float(np.percentile(ks, 2.5)), float(np.percentile(ks, 97.5))]


def score():
    s = pd.read_csv(SAMPLE, encoding="utf-8-sig", dtype={"human_label": str})
    key = pd.read_csv(KEY)
    s["human_label"] = s["human_label"].fillna("").str.strip().str.lower()
    missing = s.loc[~s["human_label"].isin(["y", "n"]), "item"].tolist()
    if missing:
        raise SystemExit(f"human_label must be y or n for every item; missing or invalid: {missing}")
    d = s.merge(key, on=["item", "task"])
    out = {"labeller": "human author", "n": len(d), "tasks": {}}
    for task, g in [("all", d)] + list(d.groupby("task")):
        a, b = g["human_label"].tolist(), g["llm_label"].tolist()
        out["tasks"][task] = {"n": len(g), "agreement": float(np.mean(np.array(a) == np.array(b))),
                              "kappa": cohen_kappa(a, b), "kappa_ci95": kappa_ci(a, b),
                              "disagreements": g.loc[g["human_label"] != g["llm_label"], "item"].tolist()}
    SNAPSHOT.write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(json.dumps(out, indent=1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["make", "score"])
    args = ap.parse_args()
    make() if args.cmd == "make" else score()


if __name__ == "__main__":
    main()
