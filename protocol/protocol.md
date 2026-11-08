# Review Protocol (pre-registration draft)

**Working title.** From Promise to Evidence: A Bibliometric Mapping, Evidence Quality Audit and
Controlled Re-evaluation of Physics-Informed Neural Networks and Neural Operators

| Field | Value |
|---|---|
| Protocol version | 1.0 (approved at Gate G0 without changes, 2026-10-30) |
| Date | 2026-10-30 |
| Registry | OSF Registries (to be filed by the authors before any screening or analysis starts) |
| Target journal | International Journal of Advanced Computer Science and Applications (IJACSA, SAI) |
| Review type | Two-tier: bibliometric mapping study + systematic review with evidence audit + controlled computational re-evaluation |
| Reporting guidelines | PRISMA 2020, PRISMA-S, Kitchenham and Charters (2007) SLR guidelines, Petersen et al. (2015) mapping-study guidelines |
| Authors | *To be completed by the authors* |

---

## 1. Rationale and positioning

Physics-informed neural networks (PINNs) and neural operators have become the two dominant
families of scientific machine learning (SciML) methods for differential equations. Several
syntheses already exist, and this review must add something they do not provide:

| Prior work | Type | Corpus | What it does not cover |
|---|---|---|---|
| Lawal et al., *Big Data Cogn. Comput.* 2022 | PRISMA SLR + bibliometrics | 120 articles (Scopus, WoS) | Neural operators; post-2022 growth; evidence quality |
| Emmert-Streib et al., *IEEE Access* 2024 | Scientometrics, topic modeling | Scopus | Method-to-failure-mode mapping; empirical verification |
| Ghalambaz et al., *Int. J. Numer. Methods Heat Fluid Flow* 2024 | Bibliometrics | 996 WoS records (2019–2022) | Operators; quality of evidence |
| Cuomo et al. 2022; Karniadakis et al. 2021; Hao et al. 2023 | Narrative surveys | Not systematic | Reproducible selection; quantitative mapping |

McGreivy and Hakim (*Nat. Mach. Intell.* 2024) showed that machine learning studies of
fluid-related PDEs frequently compare against weak baselines and are affected by reporting
biases. No review has applied this lens systematically to PINNs and neural operators, nor tested
whether the remedies most cited in the literature hold up under equal-budget conditions.

**Contribution.** Three linked layers of evidence:

1. *What is published* — reproducible bibliometric mapping of the field (RQ1–RQ2).
2. *What is claimed* — full-text coding of methodological studies into a method × failure mode ×
   PDE class map (RQ3) and an audit of evidence quality (RQ4).
3. *What holds* — a controlled, equal-budget re-evaluation of representative remedies against a
   classical numerical baseline (RQ5).

## 2. Research questions

| RQ | Question | Operationalisation | Nature |
|---|---|---|---|
| RQ1 | How has the PINN and neural operator literature grown and concentrated between 2017 and 2025? | Annual output; exponential vs logistic vs Gompertz growth (AIC/BIC); Bradford zones; Lotka exponent (MLE, KS test); country collaboration network | Descriptive, model selection |
| RQ2 | Which knowledge communities structure the field, and when did operator learning and PDE foundation models emerge as distinct communities? | Co-citation and bibliographic-coupling networks with Leiden clustering; dynamic topic model; Kleinberg bursts; main path analysis | Descriptive, temporal |
| RQ3 | Which remedies target which failure modes, on which PDE classes, and which combinations remain unstudied? | Codebook fields `method_family` × `failure_mode` × `pde_class`; gap map (evidence gap matrix) | Systematic mapping |
| RQ4 | How strong is the published evidence? | Quality rubric Q1–Q8 (baseline strength, cost reporting, seeds, code, tuning parity, metric definition, generalisation test, limitations); temporal trend | Audit |
| RQ5 | Do the gains reported for representative remedies hold under a controlled, equal-budget re-evaluation against a classical solver? | Benchmark in Section 9; relative L2 error, wall-clock, time-to-accuracy; Friedman/Nemenyi | Confirmatory (pre-specified hypotheses) |

Applications (CFD, digital twins, inverse problems, cyber-physical systems) are not a separate
RQ; they are coded as the `application_domain` dimension of RQ3.

## 3. Design overview

```
OpenAlex snapshot + arXiv ──► Tier 1: bibliometric corpus (automated, ~10^4 records)
                                   │  title/abstract screening (ASReview + human)
                                   ▼
                              Tier 2: core methodological set (~150–250 full texts)
                                   │  full-text coding (codebook v1) + quality rubric
                                   ▼
                              Gap map + evidence audit ──► selection of remedies for RQ5
                                                                │
                                                                ▼
                                               Controlled re-evaluation (benchmark)
```

## 4. Information sources and search strategy

### 4.1 Sources

| Source | Role | Access |
|---|---|---|
| OpenAlex | Primary bibliographic source (metadata, references, concepts/topics) | REST API, CC0 data; snapshot date recorded |
| arXiv | Preprints (much of SciML appears there first) | arXiv API |
| Semantic Scholar | Reference lists and citation data for preprints and proceedings papers whose OpenAlex record lacks references | Graph API |
| Crossref | DOI validation of every cited reference in the manuscript | REST API |
| Scopus / WoS | Coverage check only (seed set lookup), not a retrieval source | Institutional access, if available |

### 4.1.1 Coverage check performed at Phase 0

All 62 seed-set entries (54 in-scope seeds, 8 foundational references) were matched to OpenAlex
records (`data/processed/seed_set_resolved.csv`). Two coverage problems were found that shape the
design:

* **Duplicate records:** 15 of 62 papers (24 %) exist as two or three separate OpenAlex works
  (preprint, proceedings, journal). Without the deduplication rules of Section 4.5, citation counts
  would be split and the same paper would appear as several nodes in the networks.
* **Missing reference lists:** for 14 of 62 canonical records (23 %), OpenAlex stores no references.
  They are mostly arXiv, NeurIPS, ICML and ICLR papers, which are central to the field. Their
  reference lists are therefore taken from Semantic Scholar for the co-citation and
  bibliographic-coupling analyses, and the share of records supplemented this way is reported.
* Isolated metadata errors were also observed (e.g. a NeurIPS paper attributed to an unrelated
  venue); venue fields are checked for the Bradford analysis.

### 4.2 Time window

Publication years 2017–2025 (complete years). The 2017 lower bound corresponds to the first PINN
preprints (Raissi et al.). Records from 2026 are retrieved and reported descriptively but are
excluded from growth-model fitting because the year is incomplete and indexing lags.

### 4.3 Search string, version 0

Searched in title and abstract. Version 0 is a starting point; it will be iterated in Phase 1 and
every version logged in `protocol/search_log.md` (PRISMA-S item 8).

```
Block A (method families)
"physics-informed neural network" OR "physics informed neural network" OR "PINNs"
OR "physics-informed deep learning" OR "physics-informed machine learning"
OR "neural operator" OR "DeepONet" OR "Fourier neural operator"
OR "operator learning" OR "scientific machine learning"
OR "physics-constrained neural network" OR "physics-guided neural network"
```

Changes with respect to the original proposal and their reasons:

* The bare term `PINN` and the acronyms `FNO` and `SciML` are dropped from the primary string
  because they are ambiguous (author surnames, unrelated acronyms). Their marginal yield is
  measured in Phase 1 and they are re-added only if precision stays above target.
* The mandatory second AND block (`differential equation`, `loss balancing`, ...) is removed: it
  silently excludes relevant work whose abstract does not use those words. Topical relevance is
  instead enforced at screening.
* Wildcards (`*`) are not used because OpenAlex full-text search applies stemming and does not
  guarantee wildcard semantics; plural forms are covered by stemming and verified in Phase 1.

### 4.4 Search validation (stopping criteria for query iteration)

* **Recall:** the query must retrieve ≥ 95 % of the in-scope seed set
  (`protocol/seed_set.csv`, role = `seed`), which is fixed at G0 before any retrieval.
* **Precision:** on a random sample of 200 retrieved records, ≥ 80 % must be on-topic
  (judged by a human). If precision is lower, exclusion terms are added and their recall cost
  re-measured.

### 4.5 Deduplication

1. Exact DOI match.
2. arXiv identifier match.
3. Fuzzy title match (token-set ratio ≥ 95 with the same first-author surname and |Δyear| ≤ 1).
   A preprint and its published version are merged into one record that keeps the published
   metadata and the earliest date.

## 5. Eligibility criteria

### 5.1 Tier 1 (bibliometric corpus)

| | Criterion |
|---|---|
| IC1 | The work uses neural networks to solve, learn or identify differential equations or the operators that map their inputs to solutions (forward, inverse, or parametric problems). |
| IC2 | Journal article, conference paper (including NeurIPS, ICML and ICLR proceedings without DOI), review, or arXiv preprint. |
| IC3 | English language; publication year 2017–2025. |
| EC1 | Physics enters only as input features or post-hoc checks, with no differential-equation structure in the loss, the architecture, or the learned map. |
| EC2 | Editorials, errata, retraction notices, abstracts-only, theses, book front matter. |
| EC3 | Retracted items (OpenAlex `is_retracted`). |

### 5.2 Tier 2 (core methodological set)

Records included in Tier 1 that additionally satisfy:

| | Criterion |
|---|---|
| IC4 | The work proposes or empirically evaluates a method belonging to at least one family of the taxonomy (codebook field `method_family`), or provides a theoretical analysis of PINN/operator training or approximation. |
| IC5 | The full text is accessible (open access, arXiv, or institutional access). |
| EC4 | Application papers that use an off-the-shelf PINN/operator without methodological contribution or comparative evaluation (these remain in Tier 1 and feed the application dimension). |

If more than 300 records satisfy IC4–IC5, a stratified random sample of 250 is drawn
(strata: publication year × method family; seed `20261030`). Seed-set papers that meet the
criteria are always included.

## 6. Selection process

1. **Title/abstract screening (Tier 2)** with ASReview (active learning), following the SAFE
   procedure: random screening of 10 % to calibrate, active learning, a second model to find
   remaining relevant records, and a quality check.
   **Stopping rule:** stop when 200 consecutive records are judged irrelevant *and* all
   eligible seed-set papers have been found.
2. **LLM assistance** may be used only as an additional screener and as a pre-filler of extraction
   forms. Every inclusion decision and every extracted value is confirmed by a human author.
   The LLM's agreement with the human decisions (recall, precision) is reported.
3. **Full-text eligibility** is decided by a human author; reasons for exclusion are recorded.
4. **Reliability:** a second human coder independently screens and codes a random 20 % of the
   Tier 2 set. Cohen's κ is reported per field; fields with κ < 0.70 have their definitions revised
   and are re-coded for the whole set.

## 7. Data extraction

Extraction follows `protocol/codebook.yaml` (version 1), validated by
`src/pinnreview/codebook.py`. Full texts are parsed with GROBID. Main dimensions:
bibliographic identifiers, `method_family`, `failure_mode`, `pde_class`, `problem_type`,
`application_domain`, benchmark PDEs, reported error metric and value, baselines, compute reporting,
seeds, code availability, and the quality rubric Q1–Q8.

## 8. Analysis plan for RQ1–RQ4

| RQ | Analysis | Statistical treatment |
|---|---|---|
| RQ1 | Annual counts 2017–2025; exponential, logistic and Gompertz fits | Nonlinear least squares; AIC/BIC model selection; doubling time with bootstrap 95 % CI |
| RQ1 | Bradford zones (equal-thirds of articles); Leimkuhler fit | Goodness of fit (R²); sensitivity to zone definition |
| RQ1 | Lotka's law on author productivity | Discrete power-law MLE with estimated x_min; KS bootstrap test (Clauset, Shalizi and Newman 2009); comparison with log-normal by likelihood ratio |
| RQ1 | Country collaboration | Co-authorship network at country level; share of internationally co-authored papers |
| RQ2 | Co-citation (references cited ≥ threshold) and bibliographic coupling | Leiden clustering; resolution sweep; stability by adjusted Rand index over 100 bootstrap resamples |
| RQ2 | Dynamic topic model on titles + abstracts | BERTopic with SPECTER2 embeddings; topics per year; robustness to `min_topic_size` |
| RQ2 | Kleinberg two-state burst detection on author keywords and OpenAlex topics | Fixed parameters (s = 2, γ = 1), reported with sensitivity |
| RQ2 | Main path analysis on the citation network of Tier 1 | Search path count weights; global and key-route main paths |
| RQ3 | Gap map: `method_family` × `failure_mode`, faceted by `pde_class` | Counts; empty and sparse cells listed explicitly |
| RQ4 | Prevalence of each quality indicator Q1–Q8 | Proportions with Wilson 95 % CI; trend over years (Cochran–Armitage test); comparison by venue type |

Analyses in RQ1–RQ4 are exploratory; no confirmatory hypotheses are claimed for them.

## 9. Controlled re-evaluation (RQ5)

### 9.1 Problems

| ID | PDE | Domain and data | Known failure mode | Reference solution |
|---|---|---|---|---|
| P1 | Convection u_t + β u_x = 0, β = 30 (sensitivity: β = 50) | x ∈ [0, 2π], t ∈ [0, 1], u(x,0) = sin x, periodic | Ill-conditioned loss landscape (Krishnapriyan et al. 2021) | Analytic: sin(x − βt) |
| P2 | Allen–Cahn u_t − 10⁻⁴ u_xx + 5u³ − 5u = 0 | x ∈ [−1, 1], t ∈ [0, 1], u(x,0) = x² cos(πx), periodic | Causality violation, sharp interfaces | Fourier pseudo-spectral ETDRK4 (N = 512) |
| P3 | Burgers u_t + u u_x − (0.01/π) u_xx = 0 | x ∈ [−1, 1], t ∈ [0, 1], u(x,0) = −sin(πx), u(±1,t) = 0 | Shock-like gradient | Cole–Hopf solution by Gauss–Hermite quadrature |
| P4 | Helmholtz u_xx + u_yy + k²u = q, k = 1 | [−1, 1]², exact u = sin(πx) sin(4πy), Dirichlet | Gradient imbalance, multi-frequency (Wang, Teng and Perdikaris 2021) | Analytic |

### 9.2 Methods

| ID | Method | Source | Applicable to |
|---|---|---|---|
| M0 | Vanilla PINN, fixed unit loss weights | Raissi et al. 2019 | P1–P4 |
| M1 | Learning-rate annealing loss weights | Wang, Teng and Perdikaris 2021 | P1–P4 |
| M2 | NTK-based loss weights | Wang, Yu and Perdikaris 2022 | P1–P4 |
| M3 | Residual-based adaptive distribution (RAD) | Wu et al. 2023 | P1–P4 |
| M4 | Causal training | Wang, Sankaran and Perdikaris 2024 | P1–P3 (time-dependent) |
| M5 | Exact boundary/initial conditions + periodic Fourier embedding | Sukumar and Srivastava 2022; Dong and Ni 2021 | P1–P4 |
| C | Classical solver (spectral or finite differences), resolution sweep | — | P1–P4 |

The final list of remedies may be revised at G3 if the gap map shows that a different method family
dominates the literature. Any change is logged as a deviation before the benchmark is run.

### 9.3 Fairness rules (equal budget)

* Same architecture for all PINN methods: MLP, 4 hidden layers × 64 units, tanh, Glorot init.
* Same optimiser and budget: Adam, learning rate 10⁻³ with exponential decay 0.9 every 2 000 steps,
  20 000 steps; 4 096 residual points, 256 boundary and 256 initial points per step.
* Method-specific hyperparameters are taken from the original papers' defaults; no per-method tuning
  beyond those defaults (stated as a limitation).
* 5 seeds per (problem, method) pair. All runs on the same CPU, single thread count fixed;
  wall-clock time recorded.
* The classical solver is run on the same machine across a resolution sweep to produce an
  error-versus-time curve.
* Before any run, the error values reported by the original papers for the same problems are
  extracted and recorded in `protocol/replication_anchors.csv`.

### 9.4 Metrics and statistics

* Primary: relative L2 error on the reference grid. Secondary: wall-clock time, time to reach
  relative L2 error 10⁻², final loss components.
* Per problem: median and IQR across seeds; bootstrap 95 % CI of the log-error ratio versus M0.
* Across problems: Friedman test on P1–P3 with blocks = (problem, seed) for M0–M5, followed by
  Nemenyi post-hoc test and a critical difference diagram (Demšar 2006). P4 is reported separately
  because M4 does not apply.
* Pareto plot of error versus wall-clock including the classical solver.

### 9.5 Pre-specified hypotheses

* **H5a.** Each remedy M1–M5 achieves a lower median relative L2 error than M0 on the problem
  whose failure mode it was designed to address.
* **H5b.** At equal wall-clock time on CPU, the classical solver attains lower error than every PINN
  variant on P1–P4.
* **Replication criterion.** A published result is considered reproduced when the direction of the
  improvement over M0 matches and our median error lies within one order of magnitude of the
  reported value.

## 10. Reproducibility and data availability

* The OpenAlex snapshot date and query version are stored in a manifest; raw dumps, processed data,
  code and figures are deposited on Zenodo with a DOI at submission.
* Environment pinned with `uv.lock`; continuous integration re-runs tests and, at Phase 6, the full
  pipeline from the frozen snapshot.
* Every number in the manuscript is generated from code (LaTeX macros written from JSON outputs).
* Every reference in the manuscript is validated against Crossref or arXiv metadata.

## 11. Roles and use of generative AI

IJACSA requires disclosure of generative AI use and requires that ideas, results and conclusions
originate from the human authors. Roles are therefore fixed in advance:

| Task | Generative AI (Claude) | Human authors |
|---|---|---|
| Code for retrieval, analysis, benchmark, figures | Writes and tests | Review |
| Query iteration and validation metrics | Proposes and computes | Approve (G1) |
| Screening | Optional second screener | Decide every inclusion |
| Full-text extraction | Pre-fills forms | Verify every value; double-code 20 % |
| Cluster and topic labelling, interpretation | Suggests candidate labels | Decide (G2) |
| Benchmark conclusions | Computes statistics | Decide which claims are supported (G4) |
| Manuscript | Drafts methods and results text; language editing | Write discussion and conclusions; approve all text (G5) |

A declaration on generative AI will be included in the manuscript describing this division.

## 12. Anticipated threats to validity

* **Construct:** heterogeneous terminology (PINN vs physics-constrained vs physics-guided); mitigated by
  seed-set recall validation.
* **Coverage:** OpenAlex metadata gaps (missing references, affiliations); measured against the seed set
  and, where possible, Scopus.
* **Selection:** single-database dependence and preprint/published duplication; mitigated by
  deduplication rules and arXiv retrieval.
* **Coding:** subjective classification; mitigated by κ thresholds and a public codebook.
* **Benchmark:** small number of problems, default hyperparameters, CPU-only budget; results are
  framed as a controlled check, not a leaderboard.
* **Temporal:** incomplete 2026 data and citation-age bias toward older papers.

## 13. Timeline and gates

| Week | Phase | Gate |
|---|---|---|
| 1 | 0 Protocol and environment | G0 approve protocol, file on OSF |
| 2 | 1 Retrieval | G1 approve query and corpus |
| 3–4 | 2 Bibliometrics | G2 interpret clusters |
| 3–5 | 3 Screening and coding | G3 adjudication, double coding |
| 4–6 | 4 Controlled re-evaluation | G4 supported claims |
| 6–8 | 5 Writing | G5 discussion and conclusions |
| 8–9 | 6 QA, preprint, submission | Authors submit |

## 14. Amendments

Any change after registration is recorded in `protocol/deviations.md` with date, description,
reason and expected impact, and reported in the manuscript.
