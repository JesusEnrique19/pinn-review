"""RQ2 — temporal dynamics: Kleinberg bursts, main path, and RQ2 figures.

    python -m pinnreview.rq2_dynamics v3

* Bursts: two-state batched Kleinberg (2002) automaton on quarterly document streams (2017Q1–2025Q4) for
  every OpenAlex keyword in >= 30 Tier 1 papers; s = 2, gamma = 1 (protocol), sensitivity at gamma = 0.5 / 2.
* Main path: search path count (SPC) weights on the citation DAG among Tier 1 papers (edges cited -> citing,
  only when the cited paper is strictly older); global main path = maximum-weight source-to-sink path.
* Figures: co-citation map, topic-share heatmap, burst timeline.
"""

import argparse
import json
import math
from collections import defaultdict

import numpy as np
import pandas as pd
from scipy.special import gammaln

from pinnreview import biblio, figstyle, retrieval
from pinnreview.paths import PROCESSED, TABLES

MIN_DOCS = 30


# ---------- Kleinberg ----------

def _cost(r, d, p):
    p = min(max(p, 1e-12), 1 - 1e-12)
    return -(gammaln(d + 1) - gammaln(r + 1) - gammaln(d - r + 1) + r * math.log(p) + (d - r) * math.log(1 - p))


def kleinberg(r, d, s=2.0, gamma=1.0):
    """Viterbi state sequence (0 = base, 1 = burst) and per-step weight for counts r out of totals d."""
    n = len(r)
    p0 = r.sum() / d.sum()
    p1 = min(s * p0, 0.9999)
    trans = gamma * math.log(n)
    cost = np.array([[_cost(r[t], d[t], p) for p in (p0, p1)] for t in range(n)])
    best, back = cost[0].copy(), np.zeros((n, 2), int)
    best[1] += trans
    for t in range(1, n):
        new = np.empty(2)
        for j in (0, 1):
            cands = [best[0] + (trans if j == 1 else 0), best[1]]
            back[t, j] = int(np.argmin(cands))
            new[j] = min(cands) + cost[t, j]
        best = new
    states = np.zeros(n, int)
    states[-1] = int(np.argmin(best))
    for t in range(n - 1, 0, -1):
        states[t - 1] = back[t, states[t]]
    return states, cost[:, 0] - cost[:, 1]


def bursts(df, quarters, gamma=1.0):
    totals = df.groupby("quarter").size().reindex(quarters, fill_value=0).to_numpy()
    exploded = df[["quarter", "keywords"]].explode("keywords").dropna()
    counts = exploded.groupby(["keywords", "quarter"]).size().unstack(fill_value=0).reindex(columns=quarters, fill_value=0)
    counts = counts[counts.sum(axis=1) >= MIN_DOCS]
    rows = []
    for kw, r in counts.iterrows():
        states, w = kleinberg(r.to_numpy(), totals, gamma=gamma)
        t = 0
        while t < len(states):
            if states[t] == 1:
                start = t
                while t < len(states) and states[t] == 1:
                    t += 1
                rows.append({"keyword": kw, "start": quarters[start], "end": quarters[t - 1],
                             "weight": float(w[start:t].sum()), "papers": int(r.sum())})
            t += 1
    return pd.DataFrame(rows).sort_values("weight", ascending=False)


# ---------- main path ----------

def main_path(df, dates):
    ids = list(df["id"])
    pos = {i: k for k, i in enumerate(ids)}
    d = np.array([dates[i] for i in ids], dtype="datetime64[D]")
    edges = [(pos[r], pos[p]) for p, refs in zip(df["id"], df["references"]) for r in refs
             if r in pos and d[pos[r]] < d[pos[p]]]
    n = len(ids)
    out_adj, in_adj = defaultdict(list), defaultdict(list)
    for u, v in edges:
        out_adj[u].append(v)
        in_adj[v].append(u)
    order = np.argsort(d, kind="stable")  # edges go strictly forward in time, so date order is topological
    n_minus = np.zeros(n)
    for u in order:
        n_minus[u] = 1.0 if not in_adj[u] else sum(n_minus[w] for w in in_adj[u])
    n_plus = np.zeros(n)
    for u in order[::-1]:
        n_plus[u] = 1.0 if not out_adj[u] else sum(n_plus[w] for w in out_adj[u])
    spc = {(u, v): n_minus[u] * n_plus[v] for u, v in edges}
    total = max(spc.values())
    best = np.full(n, -np.inf)
    prev = np.full(n, -1)
    for u in order:
        if not in_adj[u]:
            best[u] = 0.0
        for v in out_adj[u]:
            cand = best[u] + spc[(u, v)] / total
            if cand > best[v]:
                best[v], prev[v] = cand, u
    sinks = [u for u in range(n) if not out_adj[u] and in_adj[u]]
    end = max(sinks, key=lambda u: best[u])
    path = []
    while end != -1:
        path.append(end)
        end = prev[end]
    path = path[::-1]
    return [{"id": ids[u], "date": str(d[u]), "title": df["title"].iloc[u], "citations": int(df["cited_by_count"].iloc[u])} for u in path], len(edges)


# ---------- figures ----------

def fig_topics_heatmap(df, assign, labels):
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap

    years = np.arange(2018, 2026)
    tab = pd.crosstab(assign["topic"], df["year"]).reindex(columns=years, fill_value=0)
    share = tab / tab.sum(axis=0)
    share = share.loc[share.mean(axis=1).sort_values(ascending=False).index]
    fig, ax = plt.subplots(figsize=(figstyle.FULL_W, 3.0))
    im = ax.imshow(share.to_numpy() * 100, aspect="auto", cmap=ListedColormap(figstyle.SEQUENTIAL))
    ax.set_xticks(range(len(years)), years)
    ax.set_yticks(range(len(share)), [f"T{t}: {labels[t]}" for t in share.index])
    ax.tick_params(axis="y", labelsize=6)
    ax.grid(False)
    for i in range(share.shape[0]):
        for j in range(share.shape[1]):
            v = share.iloc[i, j] * 100
            ax.text(j, i, f"{v:.0f}", ha="center", va="center", fontsize=5.5,
                    color="#ffffff" if v > share.to_numpy().max() * 60 else figstyle.INK)
    cb = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.01)
    cb.set_label("Share of the year's papers (%)")
    cb.outline.set_visible(False)
    ax.set_xlabel("Publication year")
    figstyle.save(fig, "fig_rq2_topics_heatmap")


def fig_bursts(b, top=20):
    import matplotlib.pyplot as plt

    b = b.head(top).sort_values(["start", "weight"], ascending=[False, True])  # earliest burst at the top
    q = lambda s: int(s[:4]) + (int(s[-1]) - 1) / 4
    fig, ax = plt.subplots(figsize=(figstyle.COLUMN_W, 3.2))
    for i, r in enumerate(b.itertuples()):
        ax.barh(i, q(r.end) + 0.25 - q(r.start), left=q(r.start), height=0.6, color=figstyle.SERIES[0])
    ax.set_yticks(range(len(b)), b["keyword"])
    ax.tick_params(axis="y", labelsize=6)
    ax.set_xlabel("Burst period (quarters)")
    ax.grid(axis="x")
    ax.grid(axis="y", visible=False)
    ax.set_xlim(2017, 2026)
    figstyle.save(fig, "fig_rq2_bursts")


def fig_cocitation(version):
    import igraph as ig
    import matplotlib.pyplot as plt

    from pinnreview.rq2_network import cocitation

    df = biblio.load(version)
    nodes = pd.read_csv(TABLES / "rq2_cocitation_nodes.csv")
    ref_lists = [r for r in df["references"] if r]
    g, cites = cocitation(ref_lists, list(nodes["id"]))
    w = np.array(g.es["weight"])
    g = g.subgraph_edges(np.where(w >= np.quantile(w, 0.9))[0].tolist(), delete_vertices=False)
    import random
    random.seed(20261030)  # igraph layouts use Python's RNG
    layout = np.array(g.layout_fruchterman_reingold(weights="weight", niter=500).coords)
    from matplotlib.lines import Line2D

    fig, ax = plt.subplots(figsize=(figstyle.FULL_W, 4.0))
    clusters = nodes["cluster"].to_numpy()
    connected = np.array(g.degree()) > 0  # nodes isolated by the edge filter are not drawn
    for c in sorted(set(clusters)):
        m = (clusters == c) & connected
        ax.scatter(layout[m, 0], layout[m, 1], s=4 + 40 * cites[m] / cites.max(), color=figstyle.SERIES[c % 8],
                   edgecolors="#ffffff", linewidths=0.3, zorder=2)
        cx, cy = np.median(layout[m], axis=0)
        ax.annotate(f"C{c}", (cx, cy), fontsize=8, color=figstyle.INK, weight="bold", zorder=3)
    ax.set_axis_off()
    handles = [Line2D([], [], marker="o", linestyle="", markersize=5, color=figstyle.SERIES[c % 8], label=f"C{c}")
               for c in sorted(set(clusters))]
    ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 0.0), ncol=7)
    figstyle.save(fig, "fig_rq2_cocitation_map")


# ---------- driver ----------

def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("version")
    args = parser.parse_args()
    df = biblio.load(args.version)
    corpus = pd.read_csv(PROCESSED / f"corpus_{args.version}.csv.gz").set_index("id")
    raw_dates = {w["id"]: w.get("publication_date") for w in retrieval.iter_works(args.version)}
    year = dict(zip(df["id"], df["year"]))
    dates = {}
    for i in df["id"]:
        ds = [raw_dates.get(v) for v in corpus.at[i, "version_ids"].split(";") if raw_dates.get(v)]
        dates[i] = min(ds) if ds else f"{year[i]}-07-01"
    df["date"] = df["id"].map(dates)
    df["quarter"] = pd.to_datetime(df["date"]).dt.to_period("Q").astype(str)
    quarters = [str(q) for q in pd.period_range("2017Q1", "2025Q4", freq="Q")]
    df = df[df["quarter"].isin(quarters)]

    b = bursts(df, quarters)
    sens = {g: set(bursts(df, quarters, gamma=g)["keyword"]) for g in (0.5, 2.0)}
    TABLES.mkdir(parents=True, exist_ok=True)
    b.to_csv(TABLES / "rq2_bursts.csv", index=False)
    path, n_edges = main_path(df.reset_index(drop=True), dates)
    pd.DataFrame(path).to_csv(TABLES / "rq2_main_path.csv", index=False)

    assign = pd.read_csv(PROCESSED / f"rq2_assignments_{args.version}.csv.gz").set_index("id").loc[df["id"]].reset_index()
    topics = pd.read_csv(TABLES / "rq2_topics.csv").set_index("topic")["label"]
    labels = {t: ", ".join(l.split(", ")[:3]) for t, l in topics.items()}
    figstyle.apply()
    fig_topics_heatmap(df.reset_index(drop=True), assign[assign["topic"] >= 0], labels)
    fig_bursts(b)
    fig_cocitation(args.version)

    top = set(b["keyword"].head(20))
    summary = {"keywords_tested": int(df[["keywords"]].explode("keywords").value_counts().ge(MIN_DOCS).sum()),
               "bursts": int(len(b)), "top20_overlap_gamma_0.5": len(top & sens[0.5]) / 20,
               "top20_overlap_gamma_2.0": len(top & sens[2.0]) / 20,
               "citation_edges_internal": n_edges, "main_path_length": len(path)}
    (PROCESSED / "snapshots" / f"rq2_dynamics_{args.version}.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(b.head(20).to_string(index=False))
    print("\nMAIN PATH:")
    for p in path:
        print(f"  {p['date']} [{p['citations']}] {p['title'][:100]}")


if __name__ == "__main__":
    main()
