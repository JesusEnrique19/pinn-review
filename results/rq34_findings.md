# RQ3–RQ4 findings — draft (numbers from `data/processed/snapshots/rq34_v3.json`)

Core set: 250 papers (54 seeds + 196 year-stratified random). Coding: LLM taxonomy + rule-based quality items,
Q1 coded directly by the LLM (rule rejected in verification). Interpretation is for the authors to confirm.

## RQ3 — method × failure-mode map (`fig_rq3_gap_map`, `results/tables/rq3_gap_map.csv`)

* Most frequent families: operator learning (58), vanilla PINN applied to new settings (52), architecture
  changes (36), hybrid numerical–neural methods (19), training strategies (17).
* 27 % of papers state no specific failure mode they address (68/250).
* The two pathologies that dominate the methodological literature's narrative — gradient imbalance and
  ill-conditioning — are each targeted by only 6 papers. Gradient imbalance is addressed almost only by loss
  balancing (6 of 7 links); ill-conditioning is spread thinly across six families (optimisers 2, one each for
  training strategy, hard constraints, vanilla PINN, architecture and weak forms). Spectral bias is addressed
  mainly by architecture changes (4); causality violation by training strategies (3).
* Gaps: no coded paper combines loss balancing with complex geometry or parametric generalisation; adaptive
  sampling, hard constraints and optimisers are never used against spectral bias or causality violation; operator
  learning never targets sharp gradients or ill-conditioning. In the plotted map (15 families × 11 failure modes,
  excluding "other" and "none stated") 87 of 165 cells are empty.

## RQ4 — evidence quality (random stratum, papers with full text, n = 105)

| Item | Yes | 95 % CI | Basis |
|---|---|---|---|
| Q1 Comparison against a classical solver run by the authors | 17 % (18) | 11–26 % | direct LLM coding |
| Q2 Cost reported for proposed method and baselines | 15 % (16) | 10–23 % | rule (not verified) |
| Q3 ≥ 3 runs with dispersion | 15 % (16) | 10–23 % | rule, "yes" verified 5/6 |
| Q4 Public code link | 22 % (23) | 15–31 % | rule, agreement 0.86 |
| Q6 Error metric named and defined | 46 % (48) | 37–55 % | rule (not verified) |
| Q7 Generalisation tested (when applicable, n = 81) | 68 % (55) | 57–77 % | LLM coding |
| Q8 Limitations reported | 65 % (68) | 55–73 % | rule (not verified) |

Q5 (tuning parity) cannot be established automatically and is not reported as evidence.

**Reading:** fewer than one in five methodological papers compares against a conventional solver that the
authors run themselves, and about one in six reports run-to-run variability — consistent with the weak-baseline
and reporting-bias concerns of McGreivy & Hakim (2024), here measured on PINN and operator literature.
These figures motivate RQ5 (controlled re-evaluation against a classical solver).
