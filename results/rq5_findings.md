# RQ5 findings (draft for the authors, 2026-11-02)

Source of every number: `data/processed/snapshots/rq5_v3.json` and `results/tables/rq5_*.csv`
(`python -m pinnreview.bench.analysis`, 115/115 runs). The manuscript reads the same files; this note only
summarises them for the authors' discussion (protocol Section 11). Reading by the LLM, to be checked by the authors.

## Results

* **H5a (each remedy beats M0 on its target problem): supported for 1 of 5.**
  * M2 (NTK weights) on P4: median error 22× lower (95 % CI 7.5–60), better in 5/5 seeds.
  * M4 (causal) on P2: no change (factor 1.0, CI 0.98–1.1). M3 (RAD) on P3: inconclusive (CI 0.22–6.5).
  * M5 (exact conditions + periodic embedding) on P1: 11× **worse** (0.72 vs 0.063).
  * M1 (learning-rate annealing) on P4: 6.6× **worse**; trivial solution u ≈ 0 in all seeds (error 1.00), and the
    same on P1 (1.00). Diagnostic (`scripts/g4/m1_probe.py`, `results/bench/diagnostics/`): the boundary weight exceeds
    10^3 within 100 steps and the residual loss never leaves its initial value (~7 × 10^3), with either moving-average
    convention. Reported as a replication failure of the pre-registered implementation (deviation 2026-11-02).
* **Off target:** M5 is the best PINN on P2, P3, P4; M2 beats M0 on all four problems.
* **H5b (classical solver better and cheaper): supported on 4/4.** Classical solver reaches the best PINN error in
  ≤ 0.49 s versus 20–63 min of PINN training (5.7 × 10^3 to 1.5 × 10^5 times faster at matched accuracy); its finest
  resolution is more accurate than every PINN on every problem.
* **Friedman/Nemenyi (P1–P3, 15 blocks):** χ² = 33.7, p < 0.001, CD = 1.95. Mean ranks M2 1.8, M5 2.6, M4 3.0,
  M3 4.1, M0 4.3, M1 5.2. Significant: M2 < M3, M0, M1; M5 < M1; M4 < M1. M2 and M5 cannot be separated.
* **Replication anchors: 2 of 5 reproduced** (vanilla PINN on P4, both depths). Vanilla PINN on P1 did *not* fail
  (0.063 vs 0.897 reported by Krishnapriyan et al.); M1 on P4 did not reach 0.0112 / 0.00743.

## Points for the authors' discussion (not written in the paper)

* Gains are problem-specific: no remedy dominates, and two remedies hurt their own target problem.
* The cost gap to classical solvers on these forward problems is 4–5 orders of magnitude, consistent with RQ4
  (few papers compare against a classical solver run by the authors).
* Limits: default hyperparameters, one network size, CPU budget, four problems; tuned versions may do better.
