"""Query validation sets for Gate G1 (protocol Section 4.4).

    python -m pinnreview.validation precision v1   # random sample of retrieved records
    python -m pinnreview.validation holdout v1     # independent recall set from other reviews
    python -m pinnreview.validation score v1       # compute precision and hold-out recall from labels

The seed set was used to build query v1, so its recall is optimistic. The hold-out set is drawn from
the reference lists of independent reviews chosen by a fixed rule, labelled blind to retrieval
status, and gives an unbiased recall estimate.
"""

import argparse
import json
import random

import pandas as pd

from pinnreview import openalex, retrieval
from pinnreview.paths import PROCESSED
from pinnreview.seedset import RESOLVED_FILE

SEED = 20261030
EXCEL_ENCODING = "utf-8-sig"  # BOM so Excel opens label files with correct accents
G1 = PROCESSED / "g1"
LABEL_COLUMNS = ["in_scope", "labeller", "notes"]
RECORD_FIELDS = ["id", "doi", "display_name", "publication_year", "type", "primary_location", "abstract_inverted_index"]


def abstract_text(inverted):
    if not inverted:
        return ""
    positions = [(pos, word) for word, poss in inverted.items() for pos in poss]
    return " ".join(word for _, word in sorted(positions))


def to_rows(works):
    rows = []
    for w in works:
        source = ((w.get("primary_location") or {}).get("source") or {})
        rows.append({
            "id": w["id"],
            "doi": w.get("doi"),
            "year": w.get("publication_year"),
            "type": w.get("type"),
            "venue": source.get("display_name"),
            "title": w.get("display_name"),
            "abstract": abstract_text(w.get("abstract_inverted_index")),
            **{c: "" for c in LABEL_COLUMNS},
        })
    return pd.DataFrame(rows)


def precision_sample(version, n=200):
    query = retrieval.load_query(version)
    params = {"filter": retrieval.search_filter(query), "sample": n, "seed": SEED, "per_page": n, "select": ",".join(RECORD_FIELDS)}
    return to_rows(openalex.get("works", params)["results"])


def select_holdout_reviews(version, k=3, min_refs=150):
    """Fixed rule: the k most-cited 2024-2025 reviews whose *title* matches Block A, outside the seed set, with >= min_refs references."""
    seeds = pd.read_csv(RESOLVED_FILE)
    seed_ids = {i for v in seeds["version_ids"].fillna(seeds["openalex_id"]) for i in v.split(";")}
    block_a = {**retrieval.load_query("v0"), "years": [2024, 2025], "field": "title.search"}
    params = {
        "filter": f"{retrieval.search_filter(block_a)},type:review,referenced_works_count:>{min_refs - 1}",
        "sort": "cited_by_count:desc",
        "per_page": 25,
        "select": "id,display_name,publication_year,cited_by_count,referenced_works,referenced_works_count,primary_location",
    }
    reviews = [w for w in openalex.get("works", params)["results"] if w["id"] not in seed_ids]
    return reviews[:k]


def fetch_works(ids, batch=50):
    short = [i.rsplit("/", 1)[-1] for i in ids]
    works = []
    for start in range(0, len(short), batch):
        params = {"filter": f"openalex:{'|'.join(short[start:start + batch])}", "per_page": 200, "select": ",".join(RECORD_FIELDS)}
        works += openalex.get("works", params)["results"]
    return works


def holdout(version, n=150):
    reviews = select_holdout_reviews(version)
    refs = sorted({r for w in reviews for r in w["referenced_works"]})
    works = [w for w in fetch_works(refs) if 2017 <= (w.get("publication_year") or 0) <= 2025]
    sample = random.Random(SEED).sample(works, min(n, len(works)))
    retrieved = retrieval.matching_ids(retrieval.load_query(version), [w["id"] for w in sample])
    labels = to_rows(sample)
    key = pd.DataFrame({"id": labels["id"], "retrieved": labels["id"].isin(retrieved)})
    meta = {
        "rule": "k=3 most-cited 2024-2025 reviews with Block A terms in the title, type=review, >=150 references, not in seed set",
        "reviews": [{"id": w["id"], "title": w["display_name"], "year": w["publication_year"], "cited_by": w["cited_by_count"]} for w in reviews],
        "references_total": len(refs),
        "references_2017_2025": len(works),
        "sampled": len(sample),
        "seed": SEED,
    }
    return labels, key, meta


def audit_sample(version, high_fraction=0.15):
    """Records for human audit of LLM labels: every low-confidence label plus a random share of high-confidence ones."""
    parts = []
    for name in ("precision", "holdout"):
        df = pd.read_csv(G1 / f"{name}_{version}.csv", encoding=EXCEL_ENCODING)
        low = df[df["confidence"] == "low"]
        high = df[df["confidence"] == "high"].sample(frac=high_fraction, random_state=SEED)
        part = pd.concat([low, high])[["id", "year", "type", "venue", "title", "abstract", "in_scope", "confidence", "notes"]]
        parts.append(part.assign(sample=name).rename(columns={"in_scope": "llm_label", "notes": "llm_notes"}))
    out = pd.concat(parts, ignore_index=True)
    out["human_label"] = ""
    out["human_notes"] = ""
    return out


def score(version):
    out = {}
    prec = pd.read_csv(G1 / f"precision_{version}.csv", encoding=EXCEL_ENCODING)
    labelled = prec[prec["in_scope"].isin(["y", "n"])]
    if len(labelled):
        out["precision"] = (labelled["in_scope"] == "y").mean()
        out["precision_n"] = len(labelled)
    hold = pd.read_csv(G1 / f"holdout_{version}.csv", encoding=EXCEL_ENCODING).merge(pd.read_csv(G1 / f"holdout_{version}_key.csv", encoding=EXCEL_ENCODING), on="id")
    relevant = hold[hold["in_scope"] == "y"]
    if len(relevant):
        out["holdout_recall"] = relevant["retrieved"].mean()
        out["holdout_relevant_n"] = len(relevant)
        out["holdout_missed"] = relevant.loc[~relevant["retrieved"], "title"].tolist()
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("action", choices=["precision", "holdout", "audit", "score"])
    parser.add_argument("version")
    args = parser.parse_args()
    G1.mkdir(parents=True, exist_ok=True)
    if args.action == "precision":
        rows = precision_sample(args.version)
        rows.to_csv(G1 / f"precision_{args.version}.csv", index=False, encoding=EXCEL_ENCODING)
        print(f"{len(rows)} records; with abstract: {(rows['abstract'] != '').mean():.0%}")
    elif args.action == "holdout":
        labels, key, meta = holdout(args.version)
        labels.to_csv(G1 / f"holdout_{args.version}.csv", index=False, encoding=EXCEL_ENCODING)
        key.to_csv(G1 / f"holdout_{args.version}_key.csv", index=False, encoding=EXCEL_ENCODING)
        (G1 / f"holdout_{args.version}_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
        print(json.dumps(meta, indent=2))
    elif args.action == "audit":
        out = audit_sample(args.version)
        out.to_csv(G1 / f"audit_{args.version}.csv", index=False, encoding=EXCEL_ENCODING)
        print(out.groupby(["sample", "confidence"]).size().to_string())
    else:
        print(json.dumps(score(args.version), indent=2))


if __name__ == "__main__":
    main()
