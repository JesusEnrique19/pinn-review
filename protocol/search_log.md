# Search log (PRISMA-S)

One entry per query version run against a source. Seed recall counts a seed as found if the query
matches any of its OpenAlex versions. Precision is judged by a human on a random sample of 200
records; hold-out recall on in-scope records from independent reviews, labelled blind to retrieval.

| Version | Date | Source | Query / filters | Records | Seed recall | Hold-out recall | Precision (n=200) | Decision |
|---|---|---|---|---|---|---|---|---|
| v0 | 2026-10-30 | OpenAlex API | `queries.yaml` v0, title+abstract, 2017–2025 | 15 724 | 0.833 (45/54) | — | — | Rejected: misses neural PDE solvers outside PINN/operator vocabulary |
| v1 | 2026-10-30 | OpenAlex API | `queries.yaml` v1 (v0 + Blocks B–D), title+abstract, 2017–2025 | 20 646 | 1.000 (54/54)* | 0.926 (75/81)† | 0.655 (131/200)† | Rejected: both targets missed; v2 in preparation |

| v2 | 2026-10-30 | OpenAlex API | `queries.yaml` v2 (type + language filters, qualified generic terms, Block E) | 18 807 | 0.981 (53/54) | 0.988 (80/81)† | 0.738 on retained v1-sample records (n=172)† | Superseded: qualifier dropped three eligible records |
| v3 | 2026-10-30 | OpenAlex API | `queries.yaml` v3 (v2 + dynamical-system/survey context, generative models in Block E) | 19 158 | 1.000 (54/54) | 0.988 (80/81)† | ≈ 0.69 estimated‡ | **Candidate for G1**; harvested |

\* Optimistic: the seed set informed Blocks B–D. The hold-out recall is the unbiased estimate.
† LLM labels pending human audit (see `deviations.md`).
‡ Stratified estimate: 17 377 records shared with v1 at 0.737 (n=175) and 1 781 v3-only records at
0.25 (15/60, `precision_v3_new.csv`): (17 377 × 0.737 + 1 781 × 0.25) / 19 158 ≈ 0.69.

## Recall–precision trade-off (G1 decision)

| Option | Records | Seed recall | Hold-out recall | Precision |
|---|---|---|---|---|
| v3 (broad) | 19 158 | 1.000 | 0.988 | ≈ 0.69 |
| v3 Block A only (method-family names) | 11 497 | 0.778 | 0.901 | 0.835 |

No Boolean query tested reaches both targets. The broad query meets recall. The remaining noise is
semantic ("physics-informed" used without any differential equation, supervised regression on
simulation outputs, PDE-inspired architectures for images or graphs), and keywords cannot separate it.

## Tier 1 relevance classifier (2026-10-30)

Training: 660 labelled records (LLM first pass, audited samples replaced by LLM second pass). Model
selection by 5×3 repeated cross-validation on training only; threshold fixed for recall ≥ 0.95 by inner
cross-validation. Independent test: 200 random corpus records, prevalence 0.655, never used before this step.

| Model | CV AUC | CV AP | Test precision (95 % CI) | Test recall (95 % CI) | Tier 1 papers |
|---|---|---|---|---|---|
| TF-IDF + LR | 0.901 | 0.918 | 0.801 (0.73–0.86) | 0.954 (0.90–0.98) | 14 425 |
| **SPECTER2 + LR (selected by CV)** | 0.916 | 0.933 | **0.810 (0.74–0.87)** | **0.947 (0.89–0.97)** | **13 961** |
| No classifier (v3 as harvested) | — | — | 0.655 | 1.000 | 18 111 |

The two filters agree on 12 878 papers (Jaccard 0.83; Spearman 0.78 between scores). Their intersection
is the planned robustness subset for Phase 2, together with the Block A subset.

## Harvest and deduplication of v3 (2026-10-30)

* Manifest: `data/processed/snapshots/openalex_v3.json` (sha256 `3c6e5879…`). 19 186 lines harvested
  for 19 158 reported; 43 ids were repeated by cursor paging, and 15 reported records were not
  returned (index changes during paging; < 0.1 %).
* Deduplication (`data/processed/snapshots/dedup_v3.json`): 19 143 unique records → **18 111 papers**;
  827 papers had 2+ versions (merges: 838 by title, 165 by arXiv id, 29 by DOI). A spot check of 12
  random title merges found 12 true duplicates.
* 4 678 papers (25.8 %) have no reference list in OpenAlex; Semantic Scholar supplementation is
  needed before co-citation analysis (protocol 4.1.1).

## v1 diagnosis (input for v2)

* **Hold-out misses (6/81):** five use "physics-informed" with a network type not in the query
  ("physics informed deep neural network", "physics-informed generative adversarial networks",
  "physics-informed Fourier networks", "physics-informed multi-LSTM", "physics-informed deep 1D CNN");
  one is a model-discovery review. Candidate fix: a block pairing "physics-informed" with any
  network / deep-learning term.
* **Out-of-scope records in the precision sample (69/200), by cause:**
  ineligible document types or language (dataset, dissertation, erratum, peer-review letter, front
  matter, poster, non-English; about 12); non-ML uses of "operator learning" (clinical learning curve,
  scheduling); PDE-inspired architectures for images, graphs or text matched by Block B;
  "physics-informed" used loosely in fault diagnosis or empirical regression with no differential
  equation; supervised regression on numerical results (nanofluid studies).
  Candidate fixes: OpenAlex type and language filters; removing or qualifying "operator learning";
  measuring per-term precision before deciding on Block B.
* The per-term precision analysis stopped when the shared, keyless OpenAlex daily budget ran out
  (HTTP 429); it resumes once an API key is available.

## Notes

* **Per-term yields, v0 (2017–2025):** "physics-informed neural network" 9 059; PINNs 7 587;
  "neural operator" 2 276; "physics-informed machine learning" 1 777; "operator learning" 1 044;
  "Fourier neural operator" 982; "scientific machine learning" 832; "physics-informed deep learning" 757;
  DeepONet 709; "physics-guided neural network" 266; "physics-constrained neural network" 101.
  The hyphenated and unhyphenated forms of "physics-informed neural network" return identical counts
  (tokenisation treats the hyphen as a space).
* **Block yields, v1:** B (neural networks AND PDEs) 8 113; C (named solvers) 375; D (pre-trained models) 297.
* **Abstract coverage:** 72 % of the v1 precision sample has an abstract in OpenAlex. For the remaining
  28 % the title is the only searchable field, which is why hold-out recall is required, and why
  abstracts for screening are supplemented from Semantic Scholar / Crossref.
* **Hold-out reviews** (rule: 3 most-cited 2024–2025 reviews with Block A terms in the title, ≥150
  references, outside the seed set): W4414050850, W4406417557, W4410412417. 509 of their 619
  references fall in 2017–2025; 150 sampled (seed 20261030). The first rule tried (Block A in title *or*
  abstract) selected application reviews whose references were mostly out of scope; it was replaced
  before any labelling.
* **API access:** since 2026 OpenAlex requires a free API key for cursor paging. Counts, recall checks
  and the samples above use single-page requests; the full harvest needs `OPENALEX_API_KEY`.
