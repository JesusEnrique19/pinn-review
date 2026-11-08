"""RQ2 — research fronts (bibliographic coupling) and topics (SPECTER2 embeddings).

    python -m pinnreview.rq2_fronts v3

* Coupling: papers × references matrix, IDF-weighted and L2-normalised; cosine k-nearest-neighbour graph
  (k = 15) clustered with Leiden.
* Topics: cosine kNN graph (k = 15) on SPECTER2 embeddings clustered with Leiden; topics labelled with
  class-based TF-IDF (c-TF-IDF, as in BERTopic) on titles and abstracts. This replaces BERTopic's
  UMAP + HDBSCAN step (see protocol/deviations.md).
* Stability: ARI across 5 Leiden seeds; convergent validity: NMI between fronts and topics.
* Topic dynamics: yearly share of each topic and its 2021–2025 trend (OLS slope of the share).
Outputs: results/tables/rq2_{fronts,topics}.csv, data/processed/snapshots/rq2_fronts_{v}.json.
"""

import argparse
import json

import igraph as ig
import leidenalg
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

from pinnreview import biblio, relevance
from pinnreview.paths import PROCESSED, TABLES
from pinnreview.validation import SEED

K = 15
MIN_CLUSTER = 50
# Words shared by almost every Tier 1 paper; removed only from c-TF-IDF labels so labels discriminate
DOMAIN_STOP = [
    "neural", "network", "networks", "informed", "physics", "pinn", "pinns", "deep", "learning", "machine",
    "data", "model", "models", "method", "methods", "based", "using", "approach", "proposed", "paper", "study",
    "results", "equation", "equations", "differential", "pde", "pdes", "solving", "solution", "solutions",
    "problem", "problems", "framework", "novel", "accuracy", "accurate", "training", "prediction", "performance",
]


def knn_graph(X, k=K, chunk=2000):
    """Symmetric weighted kNN graph from row-normalised X (dense ndarray or sparse CSR) by cosine similarity."""
    n = X.shape[0]
    src, dst, w = [], [], []
    for s in range(0, n, chunk):
        sims = X[s:s + chunk] @ X.T
        sims = sims.toarray() if sparse.issparse(sims) else sims
        for r in range(sims.shape[0]):
            sims[r, s + r] = -1
        idx = np.argpartition(-sims, k, axis=1)[:, :k]
        for r in range(idx.shape[0]):
            for j in idx[r]:
                if sims[r, j] > 0:
                    src.append(s + r)
                    dst.append(int(j))
                    w.append(float(sims[r, j]))
    g = ig.Graph(n=n, edges=list(zip(src, dst)), directed=False)
    g.es["weight"] = w
    return g.simplify(combine_edges="max")


def leiden(g, resolution=1.0, seed=SEED):
    part = leidenalg.find_partition(g, leidenalg.RBConfigurationVertexPartition, weights="weight",
                                    resolution_parameter=resolution, seed=seed, n_iterations=-1)
    return np.array(part.membership)


def merge_small(memb, min_size=MIN_CLUSTER):
    """Relabel clusters smaller than ``min_size`` as -1 (reported as 'other')."""
    sizes = np.bincount(memb)
    out = np.where(sizes[memb] >= min_size, memb, -1)
    keep = sorted(set(out) - {-1}, key=lambda c: -sizes[c])
    remap = {c: i for i, c in enumerate(keep)}
    return np.array([remap.get(c, -1) for c in out])


def ctfidf_labels(docs, memb, top=8):
    from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

    stop = sorted(ENGLISH_STOP_WORDS | set(DOMAIN_STOP))
    vec = CountVectorizer(stop_words=stop, ngram_range=(1, 2), min_df=5, max_df=0.5)
    X = vec.fit_transform(docs)
    terms = np.array(vec.get_feature_names_out())
    classes = sorted(set(memb) - {-1})
    tf = np.vstack([np.asarray(X[memb == c].sum(axis=0)).ravel() for c in classes])
    avg_words = tf.sum() / len(classes)
    idf = np.log(1 + avg_words / tf.sum(axis=0).clip(min=1))
    score = (tf / tf.sum(axis=1, keepdims=True)) * idf
    return {c: ", ".join(terms[np.argsort(-score[i])[:top]]) for i, c in enumerate(classes)}


def coupling_matrix(refs):
    vocab = {}
    rows, cols = [], []
    for p, rs in enumerate(refs):
        for r in rs:
            cols.append(vocab.setdefault(r, len(vocab)))
            rows.append(p)
    A = sparse.csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(len(refs), len(vocab)))
    df_ = np.asarray((A > 0).sum(axis=0)).ravel()
    A = A @ sparse.diags(np.log(len(refs) / df_))
    norms = np.sqrt(np.asarray(A.multiply(A).sum(axis=1)).ravel())
    return sparse.diags(1 / np.where(norms > 0, norms, 1)) @ A


def seed_stability(g, resolution, seeds=5):
    parts = [leiden(g, resolution, seed=SEED + s) for s in range(seeds)]
    return float(np.mean([adjusted_rand_score(parts[0], p) for p in parts[1:]]))


def trend(shares, years):
    sel = (years >= 2021) & (years <= 2025)
    return float(np.polyfit(years[sel], shares[sel], 1)[0])


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("version")
    args = parser.parse_args()
    df = biblio.load(args.version)
    texts = relevance.load_texts(args.version).set_index("id")
    docs = [f"{texts.at[i, 'title']} {texts.at[i, 'abstract']}" for i in df["id"]]

    # topics on embeddings (all Tier 1 papers)
    emb = np.load(relevance.embeddings_file(args.version))
    index = {i: k for k, i in enumerate(emb["ids"])}
    E = emb["vectors"][[index[i] for i in df["id"]]]
    g_topic = knn_graph(E)
    topic_raw = leiden(g_topic, resolution=1.0)
    topics = merge_small(topic_raw)
    topic_labels = ctfidf_labels(docs, topics)

    # fronts on coupling (papers with references)
    has_refs = df["references"].str.len() > 0
    sub = df[has_refs].reset_index(drop=True)
    g_front = knn_graph(coupling_matrix(sub["references"]))
    fronts = merge_small(leiden(g_front, resolution=1.0))
    front_labels = ctfidf_labels([docs[i] for i in np.where(has_refs)[0]], fronts)

    years = np.arange(2017, 2026)
    rows = []
    for c in sorted(set(topics) - {-1}):
        per_year = pd.Series(df["year"][topics == c]).value_counts().reindex(years, fill_value=0)
        total = df["year"].value_counts().reindex(years, fill_value=0)
        share = (per_year / total.clip(lower=1)).to_numpy()
        rows.append({"topic": c, "size": int((topics == c).sum()), "label": topic_labels[c],
                     "median_year": float(df["year"][topics == c].median()), "share_2025": float(share[-1]),
                     "trend_2021_2025": trend(share, years),
                     "top_cited": "; ".join(df[topics == c].nlargest(3, "cited_by_count")["title"].str[:80])})
    topic_table = pd.DataFrame(rows)
    front_rows = [{"front": c, "size": int((fronts == c).sum()), "label": front_labels[c],
                   "median_year": float(sub["year"][fronts == c].median()),
                   "top_cited": "; ".join(sub[fronts == c].nlargest(3, "cited_by_count")["title"].str[:80])}
                  for c in sorted(set(fronts) - {-1})]
    front_table = pd.DataFrame(front_rows)
    TABLES.mkdir(parents=True, exist_ok=True)
    topic_table.to_csv(TABLES / "rq2_topics.csv", index=False)
    front_table.to_csv(TABLES / "rq2_fronts.csv", index=False)
    pd.DataFrame({"id": df["id"], "topic": topics, "front": pd.Series(fronts, index=np.where(has_refs)[0]).reindex(range(len(df))).fillna(-2).astype(int)}).to_csv(
        PROCESSED / f"rq2_assignments_{args.version}.csv.gz", index=False, compression="gzip")

    both = (topics[has_refs.to_numpy()] >= 0) & (fronts >= 0)
    summary = {
        "topics": {"papers": len(df), "clusters_ge50": int(len(topic_table)), "other_share": float((topics == -1).mean()),
                   "seed_stability_ari": seed_stability(g_topic, 1.0)},
        "fronts": {"papers": len(sub), "clusters_ge50": int(len(front_table)), "other_share": float((fronts == -1).mean()),
                   "seed_stability_ari": seed_stability(g_front, 1.0)},
        "convergent_validity_nmi": float(normalized_mutual_info_score(topics[has_refs.to_numpy()][both], fronts[both])),
    }
    (PROCESSED / "snapshots" / f"rq2_fronts_{args.version}.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print("\nTOPICS (size, median year, trend 2021-25):")
    for r in topic_table.sort_values("size", ascending=False).itertuples():
        print(f"  [{r.topic}] {r.size:5d} {r.median_year:.0f} {r.trend_2021_2025:+.4f} | {r.label}")
    print("\nFRONTS:")
    for r in front_table.itertuples():
        print(f"  [{r.front}] {r.size:5d} {r.median_year:.0f} | {r.label}")


if __name__ == "__main__":
    main()
