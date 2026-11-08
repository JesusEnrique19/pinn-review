# PINN / Neural Operator Review — reproducible pipeline

Working title: *From Promise to Evidence: A Bibliometric Mapping, Evidence Quality Audit and
Controlled Re-evaluation of Physics-Informed Neural Networks and Neural Operators*
(target: IJACSA, SAI paper format).

Every number, table and figure in the manuscript is produced by code in this repository from a
frozen data snapshot. Nothing is copied by hand into the paper.

## Layout

| Path | Content |
|---|---|
| `protocol/` | Protocol (frozen in this repository), seed set, codebook, quality rubric, deviations log |
| `src/pinnreview/` | Pipeline code (retrieval, screening support, bibliometrics, benchmark) |
| `tests/` | Integrity tests for protocol artefacts and pipeline code |
| `data/raw`, `data/cache` | API dumps (not versioned; regenerated from snapshot manifest) |
| `data/processed` | Versioned derived datasets |
| `results/figures`, `results/tables` | Generated outputs used by the manuscript |
| `paper/` | Manuscript sources (Phase 5) |

## Phases and gates

| Phase | Output | Human gate |
|---|---|---|
| 0 Protocol and environment | `protocol/protocol.md`, verified seed set, codebook v1 | **G0** approve and freeze protocol |
| 1 Retrieval | Frozen corpus, PRISMA-S search report | **G1** validate query and corpus |
| 2 Bibliometrics | Growth, Bradford, Lotka, networks, topics, bursts | **G2** interpret and name clusters |
| 3 Screening and coding | Coded core set, kappa, gap map, quality audit | **G3** adjudicate, double-code 20 % |
| 4 Controlled re-evaluation | Benchmark tables, Pareto and CD diagrams | **G4** decide supported claims |
| 5 Writing | Manuscript in SAI template | **G5** authors write discussion/conclusions |
| 6 QA and submission | CI reproduction, checklists, preprint | Authors submit |

## Running

```bash
uv sync                      # core environment (Python 3.12)
uv run pytest                # protocol and pipeline integrity tests
uv run python -m pinnreview.seedset --resolve   # resolve seed set against OpenAlex
```

Optional environment variables for OpenAlex: `OPENALEX_API_KEY`, `OPENALEX_MAILTO`.
