"""Precision/recall of the specific-name subset (Block A of v3) — option 'narrow Tier 1'."""
import pandas as pd
from pinnreview import retrieval

v3 = retrieval.load_query("v3")
core = {**v3, "terms": v3["terms"][:8]}
G1 = "data/processed/g1/"
prec = pd.read_csv(G1 + "precision_v1.csv", encoding="utf-8-sig")
kept = retrieval.matching_ids(core, prec["id"].tolist())
sub = prec[prec["id"].isin(kept)]
print(f"Block A precision on v1 sample: {(sub['in_scope'] == 'y').mean():.3f} (n={len(sub)})")
hold = pd.read_csv(G1 + "holdout_v1.csv", encoding="utf-8-sig")
rel = hold[hold["in_scope"] == "y"]
found = retrieval.matching_ids(core, rel["id"].tolist())
print(f"Block A hold-out recall: {len(found) / len(rel):.3f} ({len(found)}/{len(rel)})")
r, seeds = retrieval.seed_recall("v3")
seeds_core = pd.read_csv("data/processed/seed_set_resolved.csv")
seeds_core = seeds_core[seeds_core.role == "seed"]
ids = sorted({i for v in seeds_core.version_ids.fillna(seeds_core.openalex_id) for i in v.split(";")})
f = retrieval.matching_ids(core, ids)
hit = seeds_core.version_ids.fillna(seeds_core.openalex_id).map(lambda v: any(i in f for i in v.split(";")))
print(f"Block A seed recall: {hit.mean():.3f} ({hit.sum()}/{len(hit)})")
