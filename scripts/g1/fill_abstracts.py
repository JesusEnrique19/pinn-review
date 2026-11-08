"""Fill missing abstracts in the second-pass review files from Semantic Scholar (by DOI, else by title).

Writes ``abstract_s2`` and ``abstract_source`` columns; titles are matched with a ratio >= 90 check.
"""
import gzip
import json

import pandas as pd
from rapidfuzz import fuzz

from pinnreview import s2
from pinnreview.paths import INTERIM, PROCESSED
from pinnreview.seedset import normalize
from pinnreview.validation import EXCEL_ENCODING, G1

FIELDS = "title,abstract,tldr"
dois = {}
for name in ["precision_v1.csv", "holdout_v1.csv", "precision_v3_new.csv"]:
    df = pd.read_csv(G1 / name, encoding=EXCEL_ENCODING)
    dois.update(dict(zip(df["id"], df["doi"])))
corpus = pd.read_csv(PROCESSED / "corpus_v3.csv.gz")
dois.update(dict(zip(corpus["id"], corpus["doi"])))

for name in ["audit_v1.csv", "audit_relevance_v3.csv"]:
    path = G1 / name
    df = pd.read_csv(path, encoding=EXCEL_ENCODING)
    filled, sources = [], []
    for row in df.itertuples():
        if isinstance(row.abstract, str) and row.abstract.strip():
            filled.append(""), sources.append("openalex")
            continue
        paper, doi = None, dois.get(row.id)
        if isinstance(doi, str) and doi:
            paper = s2.paper_by_doi(doi, FIELDS)
        if not paper:
            paper = s2.paper_by_title(row.title, FIELDS)
            if paper and fuzz.ratio(normalize(paper.get("title") or ""), normalize(row.title)) < 90:
                paper = None
        text = (paper or {}).get("abstract") or ((paper or {}).get("tldr") or {}).get("text") or ""
        kind = "s2_abstract" if (paper or {}).get("abstract") else ("s2_tldr" if text else "none")
        filled.append(text), sources.append(kind)
    df["abstract_s2"], df["abstract_source"] = filled, sources
    df.to_csv(path, index=False, encoding=EXCEL_ENCODING)
    print(name, pd.Series(sources).value_counts().to_dict())
