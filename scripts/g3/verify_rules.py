"""Verification of the rule-based quality items against the text snippets (deviations.md, Section 7).

A random 20 % of the random-stratum papers with full text is drawn (seed 20261030). For Q1, Q3 and Q4 the
rule value is shown with the snippets that triggered it; the LLM verifier records its own reading of the
snippets in VERDICTS (paper -> 3 characters for Q1, Q3, Q4: y / p / n). Accuracy = agreement with the rule.

    python scripts/g3/verify_rules.py show
    python scripts/g3/verify_rules.py score
"""
import json
import sys

import pandas as pd

from pinnreview.coding import load_coded
from pinnreview.core import G3
from pinnreview.paths import INTERIM, PROCESSED
from pinnreview.validation import SEED

ITEMS = {"q1_strong_baseline": ["solver_comparison", "classical_solver"],
         "q3_seeds_variance": ["seeds", "dispersion"],
         "q4_code_available": ["code_link", "code_statement", "on_request"]}
# short id -> verifier reading of the snippets for Q1, Q3, Q4 (LLM verifier, 2026-10-30)
VERDICTS = {
    "W3021668893": "nyy", "W4403346014": "pny", "W4385627137": "nny", "W4386719917": "pyn", "W3175293984": "nnp",
    "W4387355756": "nyn", "W4413251953": "pnp", "W4402703748": "nnp", "W3162074777": "nnn", "W3176958767": "pyy",
    "W3110941460": "nnn", "W4320086389": "pyn", "W4386557166": "nyy", "W4414689385": "pnn", "W4307312380": "pyn",
    "W2971386244": "ynn", "W4414942905": "nnp", "W4386779736": "nny", "W4406331680": "pnn", "W4361865663": "nnn",
    "W4414976420": "nnn",
}


def sample():
    coded = load_coded("v3")
    core = pd.read_csv(G3 / "core_v3.csv")
    rnd = set(core.loc[core["stratum"] == "random", "id"])
    ev = pd.read_csv(G3 / "evidence_v3.csv")
    pool = ev[ev["full_text"] & ev["id"].isin(rnd)]
    picked = pool.sample(frac=0.2, random_state=SEED)["id"].tolist()
    return picked, coded


def show():
    picked, coded = sample()
    dossiers = {d["id"]: d for d in map(json.loads, open(INTERIM / "dossiers_v3.jsonl", encoding="utf-8"))}
    for pid in picked:
        q = coded[pid]["quality"]
        print(f"\n## {pid.rsplit('/', 1)[-1]} rule Q1={q['q1_strong_baseline']} Q3={q['q3_seeds_variance']} Q4={q['q4_code_available']}")
        for item, keys in ITEMS.items():
            for k in keys:
                for s in dossiers[pid]["evidence"].get(k, [])[:1]:
                    print(f"  {item[:2]}/{k}: {s[:230]}")


def score():
    picked, coded = sample()
    m = {"y": "yes", "p": "partial", "n": "no"}
    out = {}
    for i, item in enumerate(ITEMS):
        pairs = [(coded[p]["quality"][item], m[VERDICTS[p.rsplit('/', 1)[-1]][i]]) for p in picked if p.rsplit('/', 1)[-1] in VERDICTS]
        out[item] = {"n": len(pairs), "agreement": sum(a == b for a, b in pairs) / len(pairs) if pairs else None,
                     "rule_yes_confirmed": sum(a == b == "yes" for a, b in pairs), "rule_yes": sum(a == "yes" for a, _ in pairs),
                     "verifier_yes": sum(b == "yes" for _, b in pairs)}
    (PROCESSED / "snapshots" / "rule_verification_v3.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    show() if sys.argv[1] == "show" else score()
