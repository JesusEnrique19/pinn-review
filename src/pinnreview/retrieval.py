"""Corpus retrieval from OpenAlex (protocol Section 4, Phase 1).

    python -m pinnreview.retrieval count v0     # per-term and total counts
    python -m pinnreview.retrieval harvest v0   # download corpus + write snapshot manifest
    python -m pinnreview.retrieval recall v0    # seed-set recall of a harvested version
"""

import argparse
import datetime as dt
import gzip
import hashlib
import json
import os

import pandas as pd
import yaml

from pinnreview import openalex
from pinnreview.paths import PROCESSED, PROTOCOL, RAW
from pinnreview.seedset import RESOLVED_FILE

QUERIES_FILE = PROTOCOL / "queries.yaml"
SNAPSHOTS = PROCESSED / "snapshots"
HARVEST_FIELDS = [
    "id", "doi", "display_name", "publication_year", "publication_date", "type", "language",
    "authorships", "primary_location", "locations", "best_oa_location", "open_access", "ids",
    "cited_by_count", "counts_by_year", "referenced_works", "referenced_works_count",
    "abstract_inverted_index", "keywords", "topics", "primary_topic", "concepts", "is_retracted",
    "fwci",
]


def load_query(version):
    with open(QUERIES_FILE, encoding="utf-8") as fh:
        return yaml.safe_load(fh)[version]


def search_filter(query, terms=None):
    terms = terms or query["terms"]
    first, last = query["years"]
    extra = f",{query['filters']}" if query.get("filters") else ""
    return f"{query['field']}:{' OR '.join(terms)},publication_year:{first}-{last}{extra}"


def count(version):
    query = load_query(version)
    rows = [
        {"term": term, "count": openalex.get("works", {"filter": search_filter(query, [term]), "per_page": 1})["meta"]["count"]}
        for term in query["terms"]
    ]
    total = openalex.get("works", {"filter": search_filter(query), "per_page": 1})["meta"]["count"]
    return pd.DataFrame(rows), total


def raw_file(version):
    return RAW / f"openalex_{version}.jsonl.gz"


def manifest_file(version):
    return SNAPSHOTS / f"openalex_{version}.json"


def harvest(version):
    """Download every matching work with cursor paging; responses are not cached (too large)."""
    if not os.environ.get("OPENALEX_API_KEY"):
        raise SystemExit("OpenAlex requires a free API key for cursor paging: set OPENALEX_API_KEY "
                         "(get one at https://openalex.org/settings/api).")
    query = load_query(version)
    params = {"filter": search_filter(query), "per_page": 200, "select": ",".join(HARVEST_FIELDS), "cursor": "*"}
    RAW.mkdir(parents=True, exist_ok=True)
    path = raw_file(version)
    n = 0
    expected = None
    with gzip.open(path, "wt", encoding="utf-8") as out:
        while params["cursor"]:
            body = openalex.get("works", params, use_cache=False)
            expected = body["meta"]["count"]
            for work in body["results"]:
                out.write(json.dumps(work) + "\n")
            n += len(body["results"])
            params["cursor"] = body["meta"].get("next_cursor")
            if not body["results"]:
                break
            print(f"\r{n}/{expected}", end="", flush=True)
    print()
    manifest = {
        "version": version,
        "source": "OpenAlex API",
        "filter": search_filter(query),
        "retrieved_at": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "records": n,
        "api_reported_count": expected,
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "file": path.name,
    }
    SNAPSHOTS.mkdir(parents=True, exist_ok=True)
    manifest_file(version).write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def iter_works(version):
    with gzip.open(raw_file(version), "rt", encoding="utf-8") as fh:
        for line in fh:
            yield json.loads(line)


def matching_ids(query, ids, batch=50):
    """Subset of OpenAlex ``ids`` matched by ``query``, asked directly of the API (no download needed)."""
    short = [i.rsplit("/", 1)[-1] for i in ids]
    found = set()
    for start in range(0, len(short), batch):
        chunk = short[start:start + batch]
        params = {"filter": f"{search_filter(query)},openalex:{'|'.join(chunk)}", "per_page": 200, "select": "id"}
        found |= {w["id"] for w in openalex.get("works", params)["results"]}
    return found


def seed_recall(version):
    """Recall of in-scope seeds: a seed counts as found if the query matches any of its OpenAlex versions."""
    seeds = pd.read_csv(RESOLVED_FILE)
    seeds = seeds[seeds["role"] == "seed"].copy()
    versions = seeds["version_ids"].fillna(seeds["openalex_id"]).str.split(";")
    found = matching_ids(load_query(version), sorted({i for v in versions for i in v}))
    seeds["found"] = versions.map(lambda v: any(i in found for i in v))
    return seeds["found"].mean(), seeds


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("action", choices=["count", "harvest", "recall"])
    parser.add_argument("version")
    args = parser.parse_args()
    if args.action == "count":
        terms, total = count(args.version)
        print(terms.to_string(index=False))
        print(f"TOTAL (OR of all terms): {total}")
    elif args.action == "harvest":
        print(json.dumps(harvest(args.version), indent=2))
    else:
        recall, seeds = seed_recall(args.version)
        print(f"seed recall: {recall:.3f} ({seeds['found'].sum()}/{len(seeds)})")
        missed = seeds[~seeds["found"]]
        if len(missed):
            print("missed:")
            print(missed[["key", "family", "oa_title", "oa_year", "oa_source"]].to_string(index=False))


if __name__ == "__main__":
    main()
