"""Semi-automated codebook coding of the core set (coder: LLM (Claude) + fixed rules; see deviations.md).

Taxonomy fields and Q7 are coded by the LLM from title, abstract and contribution sentences.
Quality items Q1–Q6 and Q8 are derived by the fixed rules in ``quality_from_evidence`` from the regex
indicators of ``pinnreview.evidence`` (full-text papers only; "na" for abstract-only papers). The rules
are verified against the snippets on a random 20 % sample (scripts/g3/verify_rules.py).

Entry: short id -> (contribution, method_family, failure_mode, pde_class, problem_type, application_domain,
abstract_metric, q7, notes). Multi-valued fields are joined with "+". q7: y / p / n / a.

    python scripts/g3/code_records.py
"""
import json

import pandas as pd

from pinnreview.codebook import load_codebook, validate_record
from pinnreview.coding import coding_file
from pinnreview.core import G3
from q1_direct import Q1_DIRECT  # noqa: E402  (same folder)

Q = {"y": "yes", "p": "partial", "n": "no", "a": "na"}

RECORDS = {}  # filled batch by batch (see code_batches.py)


def quality_from_evidence(e):
    """Fixed rules from indicator counts (row of g3/evidence_v3.csv) to Q1-Q6, Q8."""
    if not e["full_text"]:
        return dict.fromkeys(["q1_strong_baseline", "q2_cost_parity", "q3_seeds_variance", "q4_code_available",
                              "q5_tuning_parity", "q6_metric_defined", "q8_limitations"], "na")
    return {
        "q1_strong_baseline": "yes" if e["solver_comparison"] else ("partial" if e["classical_solver"] else "no"),
        "q2_cost_parity": "yes" if e["cost"] and e["solver_comparison"] else ("partial" if e["cost"] else "no"),
        "q3_seeds_variance": "yes" if e["seeds"] and e["dispersion"] else ("partial" if e["seeds"] or e["dispersion"] else "no"),
        "q4_code_available": "yes" if e["code_link"] else ("partial" if e["code_statement"] or e["on_request"] else "no"),
        "q5_tuning_parity": "partial" if e["tuning"] else "no",   # parity itself cannot be detected automatically
        "q6_metric_defined": "yes" if e["metric_rel_l2"] else ("partial" if e["metric_other"] else "no"),
        "q8_limitations": "yes" if e["limitations"] >= 2 else ("partial" if e["limitations"] == 1 else "no"),
    }


def metric_from_evidence(e, abstract_metric):
    if not e["full_text"]:
        return abstract_metric
    return "rel_l2" if e["metric_rel_l2"] else ("abs_l2_or_mse" if e["metric_other"] else "none")


def expand(short_id, entry, evidence):
    contrib, method, failure, pde, problem, domain, metric, q7, notes = entry
    pid = f"https://openalex.org/{short_id}"
    e = evidence.loc[pid]
    quality = quality_from_evidence(e)
    quality["q7_generalisation_test"] = Q[q7]
    if short_id in Q1_DIRECT:  # direct coding replaces the rejected Q1 rule (deviations.md)
        quality["q1_strong_baseline"] = Q[Q1_DIRECT[short_id]]
    return {
        "openalex_id": pid, "coder": "LLM+rules",
        "contribution_type": contrib.split("+"), "method_family": method.split("+"),
        "failure_mode_targeted": failure.split("+"), "pde_class": pde.split("+"), "problem_type": problem.split("+"),
        "application_domain": domain, "error_metric": metric_from_evidence(e, metric),
        "quality": quality, "notes": notes + ("" if e["full_text"] else " [abstract only]"),
    }


def main():
    from code_batches import BATCHES  # noqa: E402  (same folder)

    RECORDS.update({k: v for batch in BATCHES for k, v in batch.items()})
    cb = load_codebook()
    evidence = pd.read_csv(G3 / "evidence_v3.csv").set_index("id")
    records = [expand(k, v, evidence) for k, v in RECORDS.items()]
    bad = {r["openalex_id"]: validate_record(r, cb) for r in records}
    bad = {k: v for k, v in bad.items() if v}
    if bad:
        raise SystemExit(json.dumps(bad, indent=2))
    coding_file("v3").write_text("".join(json.dumps(r) + "\n" for r in records), encoding="utf-8")
    print(f"{len(records)} records valid and written")


if __name__ == "__main__":
    main()
