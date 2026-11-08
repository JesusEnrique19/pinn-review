"""RQ2 — intellectual base: co-citation network of the most-cited references (protocol Section 8).

    python -m pinnreview.rq2_network v3

* Nodes: the TOP_N references most cited by Tier 1 papers. Edge weight: number of Tier 1 papers citing
  both, normalised by association strength c_ij / (c_i c_j) (van Eck and Waltman 2009).
* Communities: Leiden (RB-configuration, weighted) at resolution 1.0, with a resolution sweep.
* Stability: 100 bootstrap resamples of the citing papers; adjusted Rand index (ARI) against the main
  partition on shared nodes.
* Reference metadata (title, year, first author) is fetched from OpenAlex for labelling.
Outputs: results/tables/rq2_cocitation_nodes.csv, data/processed/snapshots/rq2_cocitation_{v}.json.
"""

import argparse
import json
from collections import Counter
from itertools import combinations

import igraph as ig
import leidenalg
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.metrics import adjusted_rand_score

from pinnreview import biblio, openalex
from pinnreview.paths import PROCESSED, TABLES
from pinnreview.validation import SEED

TOP_N = 1000
RESOLUTIONS = [0.5, 0.75, 1.0, 1.25, 1.5, 2.0]
MAIN_RESOLUTION = 1.0


def cocitation(ref_lists, nodes):
    """Association-strength co-citation graph over ``nodes`` from a list of reference lists."""
    index = {n: i for i, n in enumerate(nodes)}
    rows, cols = [], []
    for p, refs in enumerate(ref_lists):
        for r in refs:
            if r in index:
                rows.append(p)
                cols.append(index[r])
    A = sparse.csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(len(ref_lists), len(nodes)))
    C = (A.T @ A).tocoo()
    cites = np.asarray(A.sum(axis=0)).ravel()
    mask = C.row < C.col
    i, j, c = C.row[mask], C.col[mask], C.data[mask]
    w = c / (cites[i] * cites[j])
    g = ig.Graph(n=len(nodes), edges=list(zip(i.tolist(), j.tolist())), directed=False)
    g.es["weight"] = (w / w.max()).tolist()
    g.es["count"] = c.tolist()
    return g, cites


def leiden(g, resolution, seed=SEED):
    part = leidenalg.find_partition(g, leidenalg.RBConfigurationVertexPartition, weights="weight",
                                    resolution_parameter=resolution, seed=seed, n_iterations=-1)
    return np.array(part.membership), part.modularity


def fetch_metadata(ids):
    meta = {}
    short = [i.rsplit("/", 1)[-1] for i in ids]
    for k in range(0, len(short), 50):
        body = openalex.get("works", {"filter": "openalex:" + "|".join(short[k:k + 50]), "per_page": 50,
                                      "select": "id,display_name,publication_year,authorships,primary_location"})
        for w in body["results"]:
            au = w.get("authorships") or []
            src = ((w.get("primary_location") or {}).get("source") or {})
            meta[w["id"]] = {"title": w.get("display_name"), "year": w.get("publication_year"),
                             "first_author": au[0]["author"]["display_name"] if au else None,
                             "source": src.get("display_name")}
    return meta


def analyse(df, n_boot=100):
    ref_lists = [r for r in df["references"] if r]
    counts = Counter(r for refs in ref_lists for r in refs)
    nodes = [n for n, _ in counts.most_common(TOP_N)]
    g, cites = cocitation(ref_lists, nodes)
    sweep = {}
    for res in RESOLUTIONS:
        memb, q = leiden(g, res)
        sizes = np.bincount(memb)
        sweep[res] = {"clusters": int(len(sizes)), "clusters_ge10": int((sizes >= 10).sum()), "quality": float(q)}
    main_memb, _ = leiden(g, MAIN_RESOLUTION)

    rng = np.random.default_rng(SEED)
    aris = []
    for b in range(n_boot):
        sample = [ref_lists[i] for i in rng.integers(0, len(ref_lists), len(ref_lists))]
        gb, _ = cocitation(sample, nodes)
        memb_b, _ = leiden(gb, MAIN_RESOLUTION, seed=SEED + b + 1)
        aris.append(adjusted_rand_score(main_memb, memb_b))
    return g, nodes, cites, main_memb, {
        "citing_papers": len(ref_lists), "distinct_references": len(counts), "nodes": len(nodes),
        "min_citations_in_network": int(cites.min()), "edges": g.ecount(),
        "resolution_sweep": sweep, "main_resolution": MAIN_RESOLUTION,
        "clusters": int(main_memb.max() + 1),
        "bootstrap_ari_mean": float(np.mean(aris)), "bootstrap_ari_ci95": [float(np.percentile(aris, 2.5)), float(np.percentile(aris, 97.5))],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("version")
    args = parser.parse_args()
    df = biblio.load(args.version)
    g, nodes, cites, memb, summary = analyse(df)
    meta = fetch_metadata(nodes)
    table = pd.DataFrame({"id": nodes, "citations": cites.astype(int), "cluster": memb})
    table = table.join(pd.DataFrame([meta.get(n, {}) for n in nodes]))
    table["in_tier1"] = table["id"].isin(set(df["id"]))
    table["degree_strength"] = g.strength(weights="count")
    TABLES.mkdir(parents=True, exist_ok=True)
    table.sort_values(["cluster", "citations"], ascending=[True, False]).to_csv(TABLES / "rq2_cocitation_nodes.csv", index=False)
    clusters = []
    for c, grp in table.groupby("cluster"):
        top = grp.nlargest(5, "citations")
        clusters.append({"cluster": int(c), "size": int(len(grp)), "citations": int(grp["citations"].sum()),
                         "median_year": float(grp["year"].median()) if grp["year"].notna().any() else None,
                         "top": [f"{r.first_author} ({r.year}) {r.title}" for r in top.itertuples()]})
    summary["cluster_summaries"] = sorted(clusters, key=lambda c: -c["size"])
    robust = {}
    for name, sub in [("robust_intersection", df[df["in_tfidf"]]), ("robust_block_a", df[df["block_a"]])]:
        _, nodes_s, _, memb_s, s = analyse(sub, n_boot=20)
        common = {n: i for i, n in enumerate(nodes_s)}
        shared = [k for k, n in enumerate(nodes) if n in common]
        s["ari_vs_main_on_shared_nodes"] = float(adjusted_rand_score(memb[shared], memb_s[[common[nodes[k]] for k in shared]]))
        s["shared_nodes"] = len(shared)
        robust[name] = {k: s[k] for k in ("nodes", "clusters", "bootstrap_ari_mean", "ari_vs_main_on_shared_nodes", "shared_nodes")}
    summary["robustness"] = robust
    (PROCESSED / "snapshots" / f"rq2_cocitation_{args.version}.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in summary.items() if k != "cluster_summaries"}, indent=2))
    for c in summary["cluster_summaries"]:
        print(f"\n[{c['cluster']}] n={c['size']} median_year={c['median_year']}")
        for t in c["top"][:3]:
            print("   ", t[:120])


if __name__ == "__main__":
    main()
