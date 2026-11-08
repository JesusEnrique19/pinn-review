"""Tier 1 bibliometric dataset (Phase 2 input).

    python -m pinnreview.biblio v3

One record per Tier 1 paper, merged over its OpenAlex versions: earliest year, source, authors,
countries, references (version ids of corpus papers rewritten to their canonical id), keywords and
topics, plus membership of the two robustness subsets:
* ``in_tfidf``: also accepted by the TF-IDF classifier;
* ``block_a``: title or abstract contains a Block A method-family name. This is a local regular-expression
  approximation of the OpenAlex search, used only for the sensitivity analysis.
"""

import argparse
import gzip
import json
import re

import pandas as pd

from pinnreview import retrieval
from pinnreview.paths import INTERIM, PROCESSED
from pinnreview.validation import abstract_text

BLOCK_A = re.compile(
    r"physics[\s-]+informed\s+neural\s+networks?|\bPINNs?\b|neural\s+operators?|deeponets?|"
    r"fourier\s+neural\s+operators?|physics[\s-]+constrained\s+neural\s+networks?|"
    r"physics[\s-]+guided\s+neural\s+networks?",
    re.I,
)


def dataset_file(version):
    return INTERIM / f"tier1_{version}.jsonl.gz"


def build(version):
    raw = {w["id"]: w for w in retrieval.iter_works(version)}
    corpus = pd.read_csv(PROCESSED / f"corpus_{version}.csv.gz")
    tier1 = pd.read_csv(PROCESSED / f"tier1_{version}.csv.gz")
    tfidf = pd.read_csv(PROCESSED / f"tier1_{version}_tfidf.csv.gz")
    keep = set(tier1.loc[tier1["tier1"], "id"])
    in_tfidf = set(tfidf.loc[tfidf["tier1"], "id"])
    canonical = {v: row.id for row in corpus.itertuples() for v in row.version_ids.split(";")}

    n = 0
    with gzip.open(dataset_file(version), "wt", encoding="utf-8") as out:
        for row in corpus[corpus["id"].isin(keep)].itertuples():
            versions = [raw[v] for v in row.version_ids.split(";") if v in raw]
            canon = raw.get(row.id, versions[0])
            refs = {canonical.get(r, r) for v in versions for r in (v.get("referenced_works") or [])}
            refs.discard(row.id)
            abstract = next((t for t in (abstract_text(v.get("abstract_inverted_index")) for v in versions) if t), "")
            authors, countries, institutions = [], set(), set()
            for a in canon.get("authorships") or []:
                authors.append({"id": a["author"].get("id"), "name": a["author"].get("display_name")})
                countries.update(a.get("countries") or [])
                institutions.update(i.get("id") for i in a.get("institutions") or [] if i.get("id"))
            record = {
                "id": row.id,
                "year": int(row.year),
                "title": row.title,
                "type": row.type,
                "source_id": row.venue_id if isinstance(row.venue_id, str) else None,
                "source": row.venue if isinstance(row.venue, str) else None,
                "authors": authors,
                "countries": sorted(countries),
                "n_institutions": len(institutions),
                "references": sorted(refs),
                "refs_source": "openalex" if refs else None,
                "keywords": sorted({k["display_name"].lower() for v in versions for k in (v.get("keywords") or [])}),
                "primary_topic": ((canon.get("primary_topic") or {}).get("display_name")),
                "cited_by_count": int(row.cited_by_count),
                "doi": row.doi if isinstance(row.doi, str) else None,
                "in_tfidf": row.id in in_tfidf,
                "block_a": bool(BLOCK_A.search(f"{row.title or ''} {abstract}")),
            }
            out.write(json.dumps(record) + "\n")
            n += 1
    return n


def load(version):
    with gzip.open(dataset_file(version), "rt", encoding="utf-8") as fh:
        return pd.DataFrame([json.loads(line) for line in fh])


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("version")
    args = parser.parse_args()
    n = build(args.version)
    df = load(args.version)
    print(json.dumps({
        "papers": n,
        "years": [int(df["year"].min()), int(df["year"].max())],
        "with_references": int((df["references"].str.len() > 0).sum()),
        "with_countries": int((df["countries"].str.len() > 0).sum()),
        "with_source": int(df["source_id"].notna().sum()),
        "subset_intersection": int(df["in_tfidf"].sum()),
        "subset_block_a": int(df["block_a"].sum()),
    }, indent=2))


if __name__ == "__main__":
    main()
