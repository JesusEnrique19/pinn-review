"""RQ5 analysis of the controlled re-evaluation (protocol Sections 9.4-9.5).

    python -m pinnreview.bench.analysis

* Per (problem, method): median and IQR of relative L2 error and of active training time over seeds.
* Paired log10 error ratio vs M0 (same seed) with a bootstrap 95 % CI of the median ratio.
* H5a: each remedy beats M0 on the problem whose failure mode it targets (TARGETS below, fixed before results).
* H5b: the classical solver reaches a lower error than every PINN variant at a fraction of the PINN time;
  time-to-accuracy of the classical solver at the best PINN median error (log-log interpolation).
* Friedman test on P1-P3 (blocks = problem x seed, methods M0-M5) with Nemenyi critical difference.
* Replication: anchors in protocol/replication_anchors.csv vs our medians (direction + within one order).
Outputs: results/tables/rq5_*.csv, data/processed/snapshots/rq5_v3.json, fig_rq5_pareto, fig_rq5_cd.
Works on partial results: cells or tests without enough runs are reported as incomplete.
"""

import json
import math

import numpy as np
import pandas as pd
from scipy import stats

from pinnreview import figstyle
from pinnreview.bench.timing import active_seconds
from pinnreview.paths import PROCESSED, PROTOCOL, RESULTS, TABLES
from pinnreview.validation import SEED

BENCH = RESULTS / "bench"
METHODS = ["M0", "M1", "M2", "M3", "M4", "M5"]
# Failure mode each remedy was designed for (protocol 9.2 / 9.5), fixed before the analysis:
TARGETS = {"M1": "P4", "M2": "P4", "M3": "P3", "M4": "P2", "M5": "P1"}
Q_NEMENYI_05 = {2: 1.960, 3: 2.343, 4: 2.569, 5: 2.728, 6: 2.850}  # Demsar (2006), alpha = 0.05


def load_runs():
    rows = []
    for f in sorted(BENCH.glob("P*_M*_s*.json")):
        if f.stem.endswith("_smoke"):
            continue
        d = json.loads(f.read_text(encoding="utf-8"))
        active, gaps = active_seconds(d["history"])
        rows.append({"problem": d["problem"], "method": d["method"], "seed": d["seed"], "rel_l2": d["rel_l2"],
                     "active_seconds": active, "suspension_gaps": gaps})
    return pd.DataFrame(rows)


def bootstrap_median(x, n=5000):
    rng = np.random.default_rng(SEED)
    x = np.asarray(x)
    meds = np.median(rng.choice(x, size=(n, len(x)), replace=True), axis=1)
    return [float(np.percentile(meds, 2.5)), float(np.percentile(meds, 97.5))]


def cell_table(runs):
    g = runs.groupby(["problem", "method"])
    out = g.agg(n=("seed", "size"), median_rel_l2=("rel_l2", "median"),
                q1=("rel_l2", lambda s: s.quantile(0.25)), q3=("rel_l2", lambda s: s.quantile(0.75)),
                median_active_min=("active_seconds", lambda s: s.median() / 60)).reset_index()
    return out


def ratios_vs_m0(runs):
    rows = []
    base = runs[runs["method"] == "M0"].set_index(["problem", "seed"])["rel_l2"]
    for (p, m), grp in runs[runs["method"] != "M0"].groupby(["problem", "method"]):
        pairs = [(np.log10(r.rel_l2 / base[(p, r.seed)])) for r in grp.itertuples() if (p, r.seed) in base.index]
        if not pairs:
            continue
        rows.append({"problem": p, "method": m, "n_pairs": len(pairs), "median_log10_ratio": float(np.median(pairs)),
                     "ci95": bootstrap_median(pairs) if len(pairs) >= 3 else None,
                     "wins": int(sum(v < 0 for v in pairs))})
    return pd.DataFrame(rows)


def h5a(ratios):
    out = {}
    for m, p in TARGETS.items():
        r = ratios[(ratios["method"] == m) & (ratios["problem"] == p)]
        if r.empty:
            out[m] = {"problem": p, "status": "no runs"}
            continue
        r = r.iloc[0]
        ci = r["ci95"]
        out[m] = {"problem": p, "n_pairs": int(r["n_pairs"]), "median_log10_ratio": r["median_log10_ratio"],
                  "ci95": ci, "wins": int(r["wins"]),
                  "supported": bool(ci is not None and ci[1] < 0), "complete": int(r["n_pairs"]) >= 5}
    return out


def time_to_accuracy(curve, target):
    """Classical wall time to reach `target` error by log-log interpolation along the resolution sweep."""
    c = curve.sort_values("wall_seconds")
    e, t = np.log10(c["rel_l2"].to_numpy()), np.log10(c["wall_seconds"].to_numpy())
    if target < 10 ** e.min():
        return None
    for i in range(len(e)):
        if e[i] <= np.log10(target):
            if i == 0:
                return float(10 ** t[0])
            frac = (np.log10(target) - e[i - 1]) / (e[i] - e[i - 1])
            return float(10 ** (t[i - 1] + frac * (t[i] - t[i - 1])))
    return None


def h5b(cells, classical):
    out = {}
    for p, grp in cells.groupby("problem"):
        best = grp.loc[grp["median_rel_l2"].idxmin()]
        curve = classical[classical["problem"] == p]
        tta = time_to_accuracy(curve, best["median_rel_l2"])
        out[p] = {"best_pinn_method": best["method"], "best_pinn_median_rel_l2": float(best["median_rel_l2"]),
                  "best_pinn_active_min": float(best["median_active_min"]),
                  "classical_best_rel_l2": float(curve["rel_l2"].min()),
                  "classical_max_seconds": float(curve["wall_seconds"].max()),
                  "classical_seconds_to_match_best_pinn": tta,
                  "speed_ratio": (best["median_active_min"] * 60 / tta) if tta else None,
                  "supported": bool(curve["rel_l2"].min() < best["median_rel_l2"]
                                    and curve["wall_seconds"].max() < best["median_active_min"] * 60)}
    return out


def friedman_nemenyi(runs):
    sub = runs[runs["problem"].isin(["P1", "P2", "P3"])]
    table = sub.pivot_table(index=["problem", "seed"], columns="method", values="rel_l2").reindex(columns=METHODS)
    table = table.dropna()
    if len(table) < 3:
        return {"status": f"incomplete: {len(table)} complete blocks"}
    stat, p = stats.friedmanchisquare(*[table[m] for m in METHODS])
    ranks = table.rank(axis=1).mean()
    k, n = len(METHODS), len(table)
    cd = Q_NEMENYI_05[k] * math.sqrt(k * (k + 1) / (6 * n))
    return {"blocks": n, "chi2": float(stat), "p_value": float(p), "mean_ranks": ranks.round(3).to_dict(), "critical_difference": cd}


def replication(cells):
    anchors = pd.read_csv(PROTOCOL / "replication_anchors.csv")
    anchors = anchors[pd.to_numeric(anchors["reported_value"], errors="coerce").notna()]
    rows = []
    for a in anchors.itertuples():
        c = cells[(cells["problem"] == a.problem) & (cells["method"] == a.method)]
        if c.empty:
            continue
        ours = float(c["median_rel_l2"].iloc[0])
        rows.append({"problem": a.problem, "method": a.method, "source": a.source_key, "setting": a.source_setting,
                     "reported": float(a.reported_value), "ours_median": ours, "n": int(c["n"].iloc[0]),
                     "within_one_order": bool(abs(math.log10(ours / float(a.reported_value))) <= 1)})
    return pd.DataFrame(rows)


def fig_pareto(cells, classical):
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(figstyle.FULL_W, 4.4))
    for ax, p in zip(axes.ravel(), ["P1", "P2", "P3", "P4"]):
        curve = classical[classical["problem"] == p].sort_values("wall_seconds")
        ax.plot(curve["wall_seconds"], curve["rel_l2"], "-o", color=figstyle.INK2, markersize=3, label="Classical solver")
        for i, m in enumerate(METHODS):
            c = cells[(cells["problem"] == p) & (cells["method"] == m)]
            if c.empty:
                continue
            r = c.iloc[0]
            ax.errorbar(r["median_active_min"] * 60, r["median_rel_l2"],
                        yerr=[[r["median_rel_l2"] - r["q1"]], [r["q3"] - r["median_rel_l2"]]],
                        fmt="o", color=figstyle.SERIES[i], markersize=4, elinewidth=0.8, label=m)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_title(p, loc="left")
        ax.grid(axis="both", which="major")
        ax.set_xlabel("Time (s, CPU)")
        ax.set_ylabel("Relative L2 error")
    # one legend entry per series over all panels (a method may be missing from one panel, e.g. M4 on P4)
    entries = {}
    for ax in axes.ravel():
        for h, lab in zip(*ax.get_legend_handles_labels()):
            entries.setdefault(lab, h)
    order = ["Classical solver"] + [m for m in METHODS if m in entries]
    fig.legend([entries[k] for k in order], order, loc="lower center", ncol=7, bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    figstyle.save(fig, "fig_rq5_pareto")


def cd_groups(ranks, cd):
    """Maximal runs of methods (sorted by mean rank) whose rank span is below the critical difference."""
    names = sorted(ranks, key=ranks.get)
    groups = []
    for i in range(len(names)):
        j = i
        while j + 1 < len(names) and ranks[names[j + 1]] - ranks[names[i]] < cd:
            j += 1
        if j > i and not any(g[0] <= i and j <= g[1] for g in groups):
            groups.append((i, j))
    return [[names[k] for k in range(a, b + 1)] for a, b in groups]


def fig_cd(fn):
    """Demsar (2006) critical-difference diagram from the Friedman/Nemenyi result."""
    import matplotlib.pyplot as plt

    ranks, cd = fn["mean_ranks"], fn["critical_difference"]
    names = sorted(ranks, key=ranks.get)
    k = len(names)
    half = (k + 1) // 2
    fig, ax = plt.subplots(figsize=(figstyle.COLUMN_W, 1.3))
    ax.set_xlim(0.8, k + 0.2)
    ax.set_ylim(-0.5 - 0.22 * half, 1.0)
    ax.axis("off")
    ax.hlines(0, 1, k, color=figstyle.INK2, linewidth=0.8)
    for r in range(1, k + 1):
        ax.vlines(r, 0, 0.08, color=figstyle.INK2, linewidth=0.8)
        ax.text(r, 0.15, str(r), ha="center", va="bottom", fontsize=7)
    for i, m in enumerate(names):
        left = i < half
        y = -0.35 - 0.22 * (i if left else k - 1 - i)
        x_text = 0.9 if left else k + 0.1
        ax.plot([ranks[m], ranks[m], x_text], [0, y, y], color=figstyle.INK2, linewidth=0.6)
        ax.text(x_text, y, f"{m} ({ranks[m]:.2f})", ha="right" if left else "left", va="center", fontsize=7)
    for g_i, g in enumerate(cd_groups(ranks, cd)):
        y = -0.12 - 0.08 * g_i
        ax.hlines(y, ranks[g[0]] - 0.03, ranks[g[-1]] + 0.03, color=figstyle.SERIES[0], linewidth=2.0)
    ax.hlines(0.75, 1, 1 + cd, color=figstyle.INK2, linewidth=1.0)
    ax.text(1 + cd / 2, 0.8, f"CD = {cd:.2f}", ha="center", va="bottom", fontsize=7)
    figstyle.save(fig, "fig_rq5_cd")


def main():
    runs = load_runs()
    classical = pd.DataFrame(json.loads((BENCH / "classical.json").read_text(encoding="utf-8")))
    cells = cell_table(runs)
    ratios = ratios_vs_m0(runs)
    TABLES.mkdir(parents=True, exist_ok=True)
    cells.to_csv(TABLES / "rq5_cells.csv", index=False)
    ratios.to_csv(TABLES / "rq5_ratios_vs_m0.csv", index=False)
    rep = replication(cells)
    rep.to_csv(TABLES / "rq5_replication.csv", index=False)
    summary = {"runs": int(len(runs)), "expected_runs": 115, "runs_with_suspension_gap": int((runs["suspension_gaps"] > 0).sum()),
               "h5a": h5a(ratios), "h5b": h5b(cells, classical), "friedman_nemenyi": friedman_nemenyi(runs),
               "replication": rep.to_dict("records")}
    (PROCESSED / "snapshots" / "rq5_v3.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    figstyle.apply()
    fig_pareto(cells, classical)
    if "mean_ranks" in summary["friedman_nemenyi"]:
        fig_cd(summary["friedman_nemenyi"])
    print(json.dumps(summary, indent=2, default=str))


if __name__ == "__main__":
    main()
