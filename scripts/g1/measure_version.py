"""Measure a query version against the labelled v1 samples.

Precision on v1-sample records that the new version also retrieves is unbiased for the
v_new ∩ v1 stratum; records only in v_new need a fresh sample (reported as share of total).
"""
import sys

import pandas as pd

from pinnreview import openalex, retrieval

version = sys.argv[1]
G1 = "data/processed/g1/"
q = retrieval.load_query(version)
total = openalex.get("works", {"filter": retrieval.search_filter(q), "per_page": 1})["meta"]["count"]

prec = pd.read_csv(G1 + "precision_v1.csv", encoding="utf-8-sig")
kept = retrieval.matching_ids(q, prec["id"].tolist())
sub = prec[prec["id"].isin(kept)]
dropped = prec[~prec["id"].isin(kept)]

hold = pd.read_csv(G1 + "holdout_v1.csv", encoding="utf-8-sig")
rel = hold[hold["in_scope"] == "y"]
found = retrieval.matching_ids(q, rel["id"].tolist())

seed_recall, seeds = retrieval.seed_recall(version)

print(f"{version}: total records {total}")
print(f"seed recall {seed_recall:.3f} ({seeds['found'].sum()}/{len(seeds)}); missed: {list(seeds.loc[~seeds['found'], 'key'])}")
print(f"hold-out recall {len(found) / len(rel):.3f} ({len(found)}/{len(rel)})")
for t in rel.loc[~rel["id"].isin(found), "title"]:
    print("   missed:", t)
print(f"precision on v1-sample records kept: {(sub['in_scope'] == 'y').mean():.3f} (n={len(sub)})")
print(f"v1-sample records dropped: {len(dropped)} (of which in scope: {(dropped['in_scope'] == 'y').sum()})")
for t in dropped.loc[dropped["in_scope"] == "y", "title"]:
    print("   in-scope dropped:", t[:110])
