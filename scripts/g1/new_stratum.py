"""Size of the v3-only stratum (records in v3 not retrieved by v1) and a random sample of it.

Already run; re-running overwrites precision_v3_new.csv and its labels (see label_v3_new.py).
Note: with ``sample`` set, OpenAlex reports the sample size as meta.count; strata.py gives the true size.
"""
import pandas as pd

from pinnreview import openalex, retrieval
from pinnreview.validation import RECORD_FIELDS, SEED, to_rows, EXCEL_ENCODING

v1, v3 = retrieval.load_query("v1"), retrieval.load_query("v3")
expr = f"({' OR '.join(v3['terms'])}) AND NOT ({' OR '.join(v1['terms'])})"
flt = retrieval.search_filter({**v3, "terms": [expr]})
body = openalex.get("works", {"filter": flt, "sample": 60, "seed": SEED, "per_page": 60, "select": ",".join(RECORD_FIELDS)})
print("v3-only records:", body["meta"]["count"])
rows = to_rows(body["results"])
rows.to_csv("data/processed/g1/precision_v3_new.csv", index=False, encoding=EXCEL_ENCODING)
for i, r in rows.iterrows():
    print(f"{i}|{r.year}|{r.type}|{str(r.venue)[:30]}|{r.title}|{str(r.abstract)[:160]}")
