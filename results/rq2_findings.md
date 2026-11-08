# RQ2 findings — draft (numbers from `data/processed/snapshots/rq2_*_v3.json`)

Cluster names below are **candidate labels proposed by the LLM**; naming and interpretation are decided by
the authors at Gate G2.

## Co-citation (1 000 most-cited references, ≥ 32 citations each; 11 724 citing papers)

Leiden at resolution 1.0: 7 communities. Bootstrap stability ARI 0.72 (95 % CI 0.61–0.83). Same partition on
the classifier-intersection subset (ARI 0.89 on 970 shared nodes). The Block A subset (PINN/operator names only)
partitions differently (ARI 0.42), as expected: it drops the generic neural-PDE-solver literature.

| # | Size | Median year | Candidate label | Anchors |
|---|---|---|---|---|
| 0 | 371 | 2021 | PINN core and applications | Raissi 2019; Karniadakis 2021; DeepXDE |
| 1 | 208 | 2017 | Deep-learning infrastructure and early PINN preprints | Adam; Raissi 2017; PyTorch |
| 2 | 198 | 2018 | Neural PDE solvers outside PINN branding | DGM; Lagaris 1998; Deep Ritz |
| 3 | 172 | 2021 | Operator learning | DeepONet; FNO; ResNet |
| 4 | 25 | 2021 | PINNs for integrable systems and solitons | two-stage PINN; NLS localized waves |
| 5 | 13 | 2021 | Seismic and wave-equation PINNs | PINN FWI; PINNeik; VTI wave equation |
| 6 | 13 | 2020 | "Intelligent computing" with Levenberg–Marquardt ANNs (nanofluids, epidemics) | Cheema 2020; Shoaib 2021; Choi 1995 |

Full node list: `results/tables/rq2_cocitation_nodes.csv`.

## Research fronts (bibliographic coupling, 11 724 papers) and topics (SPECTER2, 13 961 papers)

Fronts: 25 clusters ≥ 50 papers, seed-stability ARI 0.73. Topics: 16 clusters, seed-stability ARI 0.59.
Agreement between the citation view and the text view: NMI 0.26 (fronts are organised by method lineage,
topics by application domain). Tables: `results/tables/rq2_fronts.csv`, `results/tables/rq2_topics.csv`.

Candidate front labels (for G2): PINN core (2 064); operator learning (1 072); theory, error and convergence
(811); reduced-order models and dynamical systems (759); structural/solid mechanics (751); LM-ANN nanofluids
(543); high-dimensional stochastic PDEs and pricing (542); Bayesian inverse problems and UQ (483); generative
super-resolution and imaging (473); graph networks and turbulence (462); reservoirs and porous media (454);
control (407); seismic (380); electromagnetics and inverse design (366); weather and climate (321); solitons
(311); interfaces and domain decomposition (263); traffic and PDE backstepping (230); power systems and wakes
(209); quantum/molecular (186); batteries (171); blood flow (159); **Kolmogorov–Arnold networks (75, median
year 2025 — the youngest front)**; plasma and fusion (74); additive manufacturing (57).

Topic dynamics 2021–2025: the generic "numerical PDE solving" topic loses share (−2.8 points/year) and the
operator topic −1.2 points/year, while application topics (heat/flow +0.8, structures +0.7, reservoirs +0.5)
gain: the field is shifting from method development to domain application.

## Bursts and main path (`results/tables/rq2_bursts.csv`, `results/tables/rq2_main_path.csv`)

Kleinberg (quarterly, 435 keywords in ≥ 30 papers, s = 2, γ = 1): 114 bursts; the top 20 are identical at γ = 0.5
and γ = 2. Two regimes: 2017–2022 bursts on foundational vocabulary (deep/feedforward networks, curse of
dimensionality, HJB equations, PDE-constrained optimisation, model reduction, elliptic equations); **no new
top-20 burst in 2023–2024**; then 2025 bursts on **Kolmogorov–Arnold networks**, digital twins, SHAP
(explainability) and thermal radiation (nanofluid applications).

Main path (SPC, 103 821 internal citation edges, 35 papers): Raissi et al. 2017 → hidden-physics and
uncertainty-aware PINNs (2018–2019) → multi-fidelity and parareal PINNs (2019–2020) → approximation theory for
high-dimensional PDEs (2020–2021) → neural operators and their comparison (2021) → operator learning in
mechanics (2022) → surveys (2022) → operator transformers and FNO applications (2023–2024) → PIKANs (2025) →
hybrid quantum PINNs (2025).

Figures: `fig_rq2_cocitation_map`, `fig_rq2_topics_heatmap`, `fig_rq2_bursts` (PDF + PNG in `results/figures`).
