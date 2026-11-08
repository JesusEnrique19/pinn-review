"""Reference lists from Semantic Scholar for Tier 1 papers without references in OpenAlex (protocol 4.1.1).

    python -m pinnreview.references v3

Steps: (1) identify each paper by DOI or arXiv id; (2) fetch its references from the S2 batch endpoint;
(3) map each reference to a corpus paper (DOI or arXiv id) or, for references cited at least twice
across the supplemented papers, to an OpenAlex work by DOI; (4) rewrite the Tier 1 dataset with
``refs_source = "s2"`` for supplemented papers. References that cannot be mapped are counted, not kept.
"""

import argparse
import gzip
import json
from collections import Counter

import pandas as pd

from pinnreview import biblio, openalex, retrieval, s2
from pinnreview.dedup import arxiv_id
from pinnreview.paths import PROCESSED

MIN_EXTERNAL_CITES = 2


def _norm_doi(doi):
    return doi.lower().removeprefix("https://doi.org/") if isinstance(doi, str) and doi else None


def supplement(version):
    df = biblio.load(version)
    corpus = pd.read_csv(PROCESSED / f"corpus_{version}.csv.gz")
    raw = {w["id"]: w for w in retrieval.iter_works(version)}
    canonical = {v: row.id for row in corpus.itertuples() for v in row.version_ids.split(";")}
    doi_to_id, arxiv_to_id = {}, {}
    for wid, w in raw.items():
        cid = canonical.get(wid, wid)
        if d := _norm_doi(w.get("doi")):
            doi_to_id[d] = cid
        if a := arxiv_id(w):
            arxiv_to_id[a] = cid

    # 1. identifiers for papers without references
    missing = df[df["references"].str.len() == 0]
    ids = {}
    for row in missing.itertuples():
        versions = [raw[v] for v in corpus.loc[corpus["id"] == row.id, "version_ids"].iloc[0].split(";") if v in raw]
        doi = next((d for d in (_norm_doi(v.get("doi")) for v in versions) if d and "arxiv" not in d), None)
        arx = next((a for a in (arxiv_id(v) for v in versions) if a), None)
        if doi:
            ids[row.id] = f"DOI:{doi}"
        elif arx:
            ids[row.id] = f"ARXIV:{arx}"

    # 2. references from S2
    keys = list(ids)
    s2_refs = {}
    for k in range(0, len(keys), 500):
        chunk = keys[k:k + 500]
        body = s2.batch([ids[c] for c in chunk], "references.externalIds")
        for cid, paper in zip(chunk, body):
            if paper and paper.get("references") is not None:
                s2_refs[cid] = [r.get("externalIds") or {} for r in paper["references"]]

    # 3. map references
    def internal(ext):
        if (d := _norm_doi(ext.get("DOI"))) and d in doi_to_id:
            return doi_to_id[d]
        if (a := (ext.get("ArXiv") or "").lower()) and a in arxiv_to_id:
            return arxiv_to_id[a]
        return None

    external = Counter(_norm_doi(e.get("DOI")) for refs in s2_refs.values() for e in refs
                       if internal(e) is None and _norm_doi(e.get("DOI")))
    wanted = [d for d, n in external.items() if n >= MIN_EXTERNAL_CITES]
    resolved = {}
    for k in range(0, len(wanted), 50):
        chunk = wanted[k:k + 50]
        body = openalex.get("works", {"filter": "doi:" + "|".join(chunk), "per_page": 50, "select": "id,doi"})
        resolved.update({_norm_doi(w["doi"]): w["id"] for w in body["results"] if w.get("doi")})

    stats = Counter()
    mapped = {}
    for cid, refs in s2_refs.items():
        out = set()
        for e in refs:
            stats["references"] += 1
            if target := internal(e):
                out.add(target)
                stats["mapped_internal"] += 1
            elif (d := _norm_doi(e.get("DOI"))) and d in resolved:
                out.add(resolved[d])
                stats["mapped_external"] += 1
            else:
                stats["unmapped"] += 1
        out.discard(cid)
        if out:
            mapped[cid] = sorted(out)

    # 4. rewrite the dataset
    path = biblio.dataset_file(version)
    rows = [json.loads(line) for line in gzip.open(path, "rt", encoding="utf-8")]
    with gzip.open(path, "wt", encoding="utf-8") as out:
        for r in rows:
            if r["id"] in mapped:
                r["references"], r["refs_source"] = mapped[r["id"]], "s2"
            out.write(json.dumps(r) + "\n")
    summary = {
        "papers_without_openalex_refs": int(len(missing)),
        "with_doi_or_arxiv": len(ids), "found_in_s2_with_refs": len(s2_refs), "supplemented": len(mapped),
        "reference_mentions": stats["references"], "mapped_internal": stats["mapped_internal"],
        "mapped_external": stats["mapped_external"], "unmapped": stats["unmapped"],
        "external_dois_resolved": len(resolved), "external_dois_requested": len(wanted),
        "papers_with_references_after": int(sum(1 for r in rows if r["references"] or r["id"] in mapped)),
    }
    (PROCESSED / "snapshots" / f"references_{version}.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("version")
    args = parser.parse_args()
    print(json.dumps(supplement(args.version), indent=2))


if __name__ == "__main__":
    main()
