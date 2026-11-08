"""Seed set: known papers that the retrieval query must recover (protocol Section 4.4).

``python -m pinnreview.seedset --resolve`` matches every entry of ``protocol/seed_set.csv`` to an
OpenAlex work and writes ``data/processed/seed_set_resolved.csv`` for human verification at G0.
"""

import argparse
import re
import unicodedata

import pandas as pd
from rapidfuzz import fuzz

from pinnreview import openalex
from pinnreview.paths import PROCESSED, PROTOCOL

SEED_FILE = PROTOCOL / "seed_set.csv"
RESOLVED_FILE = PROCESSED / "seed_set_resolved.csv"
REQUIRED_COLUMNS = ["key", "role", "family", "first_author", "year", "title", "venue"]
ROLES = {"seed", "foundation"}
TITLE_MATCH_MIN = 90
SELECT = [
    "id", "doi", "display_name", "publication_year", "type", "authorships",
    "cited_by_count", "referenced_works_count", "primary_location", "is_retracted",
]


def load_seed_set(path=SEED_FILE):
    return pd.read_csv(path, dtype={"year": int})


def normalize(text):
    text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode()
    return re.sub(r"[^a-z0-9 ]+", " ", text.lower()).strip()


def first_author_matches(work, surname):
    """True if ``surname`` is a token of the first author's name (handles "Zhao, Zhiyuan" order)."""
    if not work or not work.get("authorships"):
        return False
    tokens = normalize(work["authorships"][0]["author"].get("display_name") or "").split()
    return normalize(surname).replace(" ", "") in tokens or normalize(surname).split()[-1] in tokens


def _titles(entry):
    alt = entry.get("alt_title")
    return [normalize(entry["title"])] + ([normalize(alt)] if isinstance(alt, str) and alt.strip() else [])


def title_score(entry, work):
    return max(fuzz.ratio(t, normalize(work.get("display_name") or "")) for t in _titles(entry))


def match_entry(entry, candidates):
    """Return (best_work, title_score, versions) for a seed entry among OpenAlex candidates.

    OpenAlex often holds several records for one paper (preprint, proceedings, journal). All
    records whose full title matches are kept as ``versions``; retrieving any of them counts
    towards seed-set recall. The canonical record is the most cited version by the same first
    author.
    """
    scored = [(title_score(entry, w), w) for w in candidates]
    versions = [w for score, w in scored if score >= TITLE_MATCH_MIN]
    if versions:
        best = max(versions, key=lambda w: (first_author_matches(w, entry["first_author"]), w.get("cited_by_count") or 0))
        return best, title_score(entry, best), versions
    if not scored:
        return None, 0, []
    score, best = max(scored, key=lambda s: s[0])
    return best, score, []


def resolve(seed):
    rows = []
    for entry in seed.to_dict("records"):
        candidates = []
        for title in _titles(entry):
            candidates += openalex.search_works(title, per_page=10, select=SELECT)
        candidates = list({w["id"]: w for w in candidates}.values())
        work, score, versions = match_entry(entry, candidates)
        source = ((work or {}).get("primary_location") or {}).get("source") or {}
        rows.append({
            "key": entry["key"],
            "role": entry["role"],
            "family": entry["family"],
            "openalex_id": (work or {}).get("id"),
            "doi": (work or {}).get("doi"),
            "oa_title": (work or {}).get("display_name"),
            "oa_year": (work or {}).get("publication_year"),
            "oa_type": (work or {}).get("type"),
            "oa_source": source.get("display_name"),
            "title_score": score,
            "first_author_match": first_author_matches(work, entry["first_author"]),
            "year_delta": (work or {}).get("publication_year", 0) - entry["year"] if work else None,
            "n_versions": len(versions),
            "version_ids": ";".join(v["id"] for v in versions),
            "cited_by_count": (work or {}).get("cited_by_count"),
            "referenced_works_count": (work or {}).get("referenced_works_count"),
            "is_retracted": (work or {}).get("is_retracted"),
            "resolved": score >= TITLE_MATCH_MIN,
        })
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--resolve", action="store_true", help="match the seed set against OpenAlex")
    args = parser.parse_args()
    seed = load_seed_set()
    print(f"{len(seed)} entries: {seed['role'].value_counts().to_dict()}")
    if args.resolve:
        resolved = resolve(seed)
        PROCESSED.mkdir(parents=True, exist_ok=True)
        resolved.to_csv(RESOLVED_FILE, index=False)
        ok = resolved["resolved"].sum()
        print(f"resolved {ok}/{len(resolved)} -> {RESOLVED_FILE}")
        print(f"entries with duplicate OpenAlex records: {(resolved['n_versions'] > 1).sum()}")
        no_refs = resolved[resolved["referenced_works_count"] == 0]
        print(f"entries whose canonical record has no reference list: {len(no_refs)} ({', '.join(no_refs['key'])})")
        flagged = resolved[~resolved["resolved"] | ~resolved["first_author_match"]]
        if len(flagged):
            print("\nentries needing human check:")
            print(flagged[["key", "title_score", "first_author_match", "year_delta", "n_versions", "oa_title"]].to_string(index=False))


if __name__ == "__main__":
    main()
