"""Deduplication of a harvested corpus (protocol Section 4.5).

    python -m pinnreview.dedup v3

Rules, applied in order and merged with union-find:
1. exact DOI match; 2. arXiv identifier match; 3. same first-author surname, |Δyear| <= 1 and
title similarity >= 95 (Levenshtein ratio on normalised titles; token-set ratio is not used because it
scores a title that is a subset of another as 100).
The canonical record of each group is a published version (article, review, conference paper) when
one exists, then the most cited; the group keeps the earliest year and the union of references.
"""

import argparse
import gzip
import json
import re
from collections import defaultdict

import pandas as pd
from rapidfuzz import fuzz

from pinnreview import retrieval
from pinnreview.paths import PROCESSED
from pinnreview.seedset import normalize

TITLE_MIN = 95
PUBLISHED = {"article", "review", "conference-paper"}
ARXIV_RE = re.compile(r"arxiv\.org/(?:abs|pdf)/([0-9]{4}\.[0-9]{4,5}|[a-z\-]+/[0-9]{7})", re.I)


class UnionFind:
    def __init__(self, items):
        self.parent = {i: i for i in items}

    def find(self, x):
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[rb] = ra


def arxiv_id(work):
    for loc in work.get("locations") or []:
        for url in (loc.get("landing_page_url"), loc.get("pdf_url")):
            if url and (m := ARXIV_RE.search(url)):
                return m.group(1).lower()
    doi = (work.get("doi") or "").lower()
    if "10.48550/arxiv." in doi:
        return doi.split("10.48550/arxiv.")[-1]
    return None


def first_surname(work):
    authors = work.get("authorships") or []
    if not authors:
        return ""
    parts = normalize(authors[0]["author"].get("display_name") or "").split()
    return parts[-1] if parts else ""


def find_groups(works):
    """Return {work_id: group_root_id} and the rule that first linked each merged pair."""
    uf = UnionFind([w["id"] for w in works])
    reasons = []

    def link(key_fn, rule):
        index = defaultdict(list)
        for w in works:
            if key := key_fn(w):
                index[key].append(w["id"])
        for ids in index.values():
            for other in ids[1:]:
                if uf.find(ids[0]) != uf.find(other):
                    reasons.append((ids[0], other, rule))
                uf.union(ids[0], other)

    link(lambda w: (w.get("doi") or "").lower() or None, "doi")
    link(arxiv_id, "arxiv")

    by_author = defaultdict(list)
    for w in works:
        by_author[first_surname(w)].append(w)
    for surname, group in by_author.items():
        if not surname or len(group) < 2:
            continue
        titles = [normalize(w.get("display_name") or "") for w in group]
        for i in range(len(group)):
            for j in range(i + 1, len(group)):
                yi, yj = group[i].get("publication_year") or 0, group[j].get("publication_year") or 0
                if abs(yi - yj) > 1 or not titles[i] or uf.find(group[i]["id"]) == uf.find(group[j]["id"]):
                    continue
                if fuzz.ratio(titles[i], titles[j]) >= TITLE_MIN:
                    reasons.append((group[i]["id"], group[j]["id"], "title"))
                    uf.union(group[i]["id"], group[j]["id"])
    return {w["id"]: uf.find(w["id"]) for w in works}, reasons


def canonical(members):
    return max(members, key=lambda w: (w.get("type") in PUBLISHED, w.get("cited_by_count") or 0, w["id"]))


def deduplicate(works):
    """Return (records DataFrame with one row per paper, merge log DataFrame)."""
    unique = list({w["id"]: w for w in works}.values())
    roots, reasons = find_groups(unique)
    groups = defaultdict(list)
    for w in unique:
        groups[roots[w["id"]]].append(w)
    rows = []
    for members in groups.values():
        canon = canonical(members)
        source = ((canon.get("primary_location") or {}).get("source") or {})
        rows.append({
            "id": canon["id"],
            "doi": canon.get("doi"),
            "title": canon.get("display_name"),
            "year": min(m.get("publication_year") or 9999 for m in members),
            "canonical_year": canon.get("publication_year"),
            "type": canon.get("type"),
            "venue": source.get("display_name"),
            "venue_id": source.get("id"),
            "cited_by_count": sum(m.get("cited_by_count") or 0 for m in members),
            "n_versions": len(members),
            "version_ids": ";".join(sorted(m["id"] for m in members)),
            "n_references": len({r for m in members for r in (m.get("referenced_works") or [])}),
        })
    log = pd.DataFrame(reasons, columns=["kept_side", "merged_side", "rule"])
    return pd.DataFrame(rows), log, len(works) - len(unique)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("version")
    args = parser.parse_args()
    works = list(retrieval.iter_works(args.version))
    records, log, repeated = deduplicate(works)
    out = PROCESSED / f"corpus_{args.version}.csv.gz"
    records.to_csv(out, index=False, compression="gzip")
    log.to_csv(PROCESSED / f"dedup_log_{args.version}.csv", index=False)
    summary = {
        "harvested_lines": len(works),
        "repeated_ids_from_paging": repeated,
        "unique_records": len(records) + int((records["n_versions"] - 1).sum()),
        "papers_after_dedup": len(records),
        "merged_by_rule": log["rule"].value_counts().to_dict(),
        "papers_with_multiple_versions": int((records["n_versions"] > 1).sum()),
        "records_without_references": int((records["n_references"] == 0).sum()),
    }
    (PROCESSED / "snapshots" / f"dedup_{args.version}.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
