"""Phase 3 — codebook coding of the core set and RQ3/RQ4 synthesis.

    python -m pinnreview.coding next v3 [--n 15]   # print the next uncoded dossiers (compact)
    python -m pinnreview.coding validate v3        # validate every coded record against the codebook
    python -m pinnreview.coding summary v3         # gap map (RQ3) and evidence-quality prevalence (RQ4)

Coded records live in data/processed/g3/coding_{v}.jsonl, one JSON object per paper, written by
scripts/g3/code_records.py. Coder: LLM (Claude); see protocol/deviations.md. Papers without an
open-access full text are coded from the abstract; quality items needing the full text are "na" with
``"evidence_basis": "abstract"`` in notes. RQ4 prevalences use the random stratum only.
"""

import argparse
import json
from collections import Counter

import pandas as pd

from pinnreview.codebook import load_codebook, validate_record
from pinnreview.core import G3
from pinnreview.paths import INTERIM, PROCESSED, TABLES
from pinnreview.relevance import wilson


def coding_file(version):
    return G3 / f"coding_{version}.jsonl"


def load_coded(version):
    path = coding_file(version)
    if not path.exists():
        return {}
    return {r["openalex_id"]: r for r in (json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip())}


def dossiers(version):
    with open(INTERIM / f"dossiers_{version}.jsonl", encoding="utf-8") as fh:
        return [json.loads(l) for l in fh]


def next_batch(version, n=15, abstract_chars=450):
    """Compact view for taxonomy coding: title, abstract head and first contribution sentence."""
    coded = load_coded(version)
    manifest = pd.read_csv(G3 / f"fulltext_{version}.csv").set_index("id")["status"]
    # only papers whose full-text status is final: text extracted, or no retrievable open-access text
    # (no OA version, or retrieval still failing after retries: HTTP error, timeout, not a PDF)
    ready = [d for d in dossiers(version) if d["full_text"] or manifest.get(d["id"]) != "ok"]
    todo = [d for d in ready if d["id"] not in coded][:n]
    for d in todo:
        contrib = d["contribution"][0][:200] if d["contribution"] else ""
        print(f"{d['id'].rsplit('/', 1)[-1]}|ft={int(d['full_text'])}| {d['title']} || {(d['abstract'] or '')[:abstract_chars]} || C: {contrib}")
    print(f"\n{len(coded)} coded, {len(dossiers(version)) - len(coded)} remaining")


def validate(version):
    cb = load_codebook()
    errors = {pid: validate_record({k: v for k, v in r.items() if k != "stratum"}, cb) for pid, r in load_coded(version).items()}
    return {pid: e for pid, e in errors.items() if e}


def summary(version):
    cb = load_codebook()
    coded = pd.DataFrame(load_coded(version).values())
    core = pd.read_csv(G3 / f"core_{version}.csv")
    coded = coded.merge(core[["id", "stratum", "year"]], left_on="openalex_id", right_on="id", how="left")

    # RQ3 gap map: method family x failure mode (all coded papers)
    gap = Counter((m, f) for ms, fs in zip(coded["method_family"], coded["failure_mode_targeted"]) for m in ms for f in fs)
    methods = list(cb["fields"]["method_family"]["values"])
    failures = list(cb["fields"]["failure_mode_targeted"]["values"])
    gap_table = pd.DataFrame([[gap.get((m, f), 0) for f in failures] for m in methods], index=methods, columns=failures)
    TABLES.mkdir(parents=True, exist_ok=True)
    gap_table.to_csv(TABLES / "rq3_gap_map.csv")

    # RQ4: prevalence of each quality item (random stratum, full-text papers, items not "na")
    rnd = coded[(coded["stratum"] == "random")]
    quality = pd.DataFrame(list(rnd["quality"]))
    prev = {}
    for item in cb["quality"]["items"]:
        vals = quality[item]
        applicable = vals[vals != "na"]
        k_yes, k_partial, n = int((applicable == "yes").sum()), int((applicable == "partial").sum()), int(len(applicable))
        prev[item] = {"n": n, "yes": k_yes, "partial": k_partial, "share_yes": k_yes / n if n else None,
                      "ci95_yes": wilson(k_yes, n)}
    out = {"coded": int(len(coded)), "random_stratum": int(len(rnd)),
           "method_family_counts": Counter(m for ms in coded["method_family"] for m in ms),
           "failure_mode_counts": Counter(f for fs in coded["failure_mode_targeted"] for f in fs),
           "empty_gap_cells": int((gap_table == 0).to_numpy().sum()), "quality_prevalence": prev}
    (PROCESSED / "snapshots" / f"rq34_{version}.json").write_text(json.dumps(out, indent=2, default=dict), encoding="utf-8")
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("action", choices=["next", "validate", "summary"])
    parser.add_argument("version")
    parser.add_argument("--n", type=int, default=15)
    args = parser.parse_args()
    if args.action == "next":
        next_batch(args.version, args.n)
    elif args.action == "validate":
        errors = validate(args.version)
        print("all records valid" if not errors else json.dumps(errors, indent=2))
    else:
        print(json.dumps(summary(args.version), indent=2, default=dict))


if __name__ == "__main__":
    main()
