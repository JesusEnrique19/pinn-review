# G1 query-validation scripts

Run from the repository root with `OPENALEX_API_KEY` set. They read the labelled samples in
`data/processed/g1/` and only query OpenAlex for counts and id matches.

| Script | Purpose |
|---|---|
| `diagnose_v1.py` | Precision of each v1 term, by number of matching terms and by document type |
| `measure_version.py <version>` | Seed recall, hold-out recall and precision on retained v1-sample records |
| `strata.py` | Size of v3, of its v3-only stratum, and of its Block A subset |
| `new_stratum.py` | Random sample of the v3-only stratum (already run; do not re-run) |
| `label_v3_new.py` | LLM labels for that sample |
| `core_option.py` | Precision and recall of the Block A subset (narrow Tier 1 option) |
