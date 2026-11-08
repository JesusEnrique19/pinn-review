"""Per-term precision on the labelled v1 sample, and why hold-out misses were missed."""
import pandas as pd
from pinnreview import retrieval, openalex

G1 = "data/processed/g1/"
q = retrieval.load_query("v1")
prec = pd.read_csv(G1 + "precision_v1.csv", encoding="utf-8-sig")
ids = prec["id"].tolist()
rows = []
for term in q["terms"]:
    found = retrieval.matching_ids({**q, "terms": [term]}, ids)
    sub = prec[prec["id"].isin(found)]
    only = prec[prec["id"].isin(found)]
    rows.append({"term": term[:70], "n": len(sub), "prec": round((sub["in_scope"] == "y").mean(), 2) if len(sub) else None})
    prec[term[:30]] = prec["id"].isin(found)
print(pd.DataFrame(rows).to_string(index=False))

term_cols = [t[:30] for t in q["terms"]]
prec["n_terms"] = prec[term_cols].sum(axis=1)
print("\nprecision by number of matching terms:")
print(prec.groupby("n_terms")["in_scope"].apply(lambda s: f"{(s == 'y').mean():.2f} (n={len(s)})"))
print("\nprecision by type:")
print(prec.groupby("type")["in_scope"].apply(lambda s: f"{(s == 'y').mean():.2f} (n={len(s)})"))
only_b = prec[(prec["n_terms"] == 1) & prec[term_cols[12]]]
print(f"\nrecords matched ONLY by Block B: n={len(only_b)}, precision={(only_b['in_scope'] == 'y').mean():.2f}")
