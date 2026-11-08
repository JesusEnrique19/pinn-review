"""RQ3/RQ4 figures from the coded core set.

    python -m pinnreview.rq34 v3

* fig_rq3_gap_map: method family × failure mode counts (all 250 coded papers); empty cells are the gaps.
* fig_rq4_quality: share of random-stratum full-text papers meeting each quality item ("yes", with Wilson
  95 % CI) and "partial" stacked. Q3 "partial" and Q5 are not shown as evidence (rule limits, deviations.md).
"""

import argparse
import json

import numpy as np
import pandas as pd

from pinnreview import coding, figstyle
from pinnreview.codebook import load_codebook
from pinnreview.paths import PROCESSED, TABLES

LABELS = {
    "q1_strong_baseline": "Q1 Classical-solver comparison", "q2_cost_parity": "Q2 Cost reported vs baselines",
    "q3_seeds_variance": "Q3 Seeds and dispersion", "q4_code_available": "Q4 Public code",
    "q6_metric_defined": "Q6 Metric defined", "q7_generalisation_test": "Q7 Generalisation test",
    "q8_limitations": "Q8 Limitations reported",
}


def fig_gap_map():
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap

    gap = pd.read_csv(TABLES / "rq3_gap_map.csv", index_col=0)
    gap = gap.drop(index=[r for r in ("other",) if r in gap.index], columns=[c for c in ("none_stated",) if c in gap.columns])
    gap = gap.loc[gap.sum(axis=1).sort_values(ascending=False).index, gap.sum(axis=0).sort_values(ascending=False).index]
    fig, ax = plt.subplots(figsize=(figstyle.FULL_W, 3.6))
    ax.imshow(gap.to_numpy(), aspect="auto", cmap=ListedColormap(["#f4f4f2"] + figstyle.SEQUENTIAL[1:]))
    ax.set_xticks(range(gap.shape[1]), [c.replace("_", " ") for c in gap.columns], rotation=35, ha="right")
    ax.set_yticks(range(gap.shape[0]), [r.replace("_", " ") for r in gap.index])
    ax.tick_params(labelsize=6.5)
    ax.grid(False)
    vmax = gap.to_numpy().max()
    for i in range(gap.shape[0]):
        for j in range(gap.shape[1]):
            v = gap.iat[i, j]
            ax.text(j, i, "·" if v == 0 else str(v), ha="center", va="center", fontsize=6,
                    color=figstyle.MUTED if v == 0 else ("#ffffff" if v > vmax * 0.55 else figstyle.INK))
    ax.set_xlabel("Failure mode targeted")
    ax.set_ylabel("Method family")
    figstyle.save(fig, "fig_rq3_gap_map")


def fig_quality(prev):
    import matplotlib.pyplot as plt

    items = [k for k in LABELS if k in prev]
    yes = np.array([prev[k]["yes"] / prev[k]["n"] for k in items]) * 100
    part = np.array([0 if k == "q3_seeds_variance" else prev[k]["partial"] / prev[k]["n"] for k in items]) * 100
    lo = np.array([prev[k]["ci95_yes"][0] for k in items]) * 100
    hi = np.array([prev[k]["ci95_yes"][1] for k in items]) * 100
    fig, ax = plt.subplots(figsize=(figstyle.COLUMN_W, 2.6))
    y = np.arange(len(items))[::-1]
    ax.barh(y, yes, height=0.6, color=figstyle.SERIES[0], label="Yes")
    ax.barh(y, part, left=yes, height=0.6, color=figstyle.SEQUENTIAL[1], label="Partial")
    ax.errorbar(yes, y, xerr=[yes - lo, hi - yes], fmt="none", ecolor=figstyle.INK, elinewidth=0.6, capsize=1.5)
    for yi, v, k in zip(y, yes, items):
        ax.text(v + (hi - yes)[items.index(k)] + 2, yi, f"{v:.0f}% (n={prev[k]['n']})", va="center", fontsize=6, color=figstyle.INK2)
    ax.set_yticks(y, [LABELS[k] for k in items])
    ax.tick_params(axis="y", labelsize=6.5)
    ax.set_xlim(0, 100)
    ax.set_xlabel("Share of papers (%)")
    ax.grid(axis="x")
    ax.grid(axis="y", visible=False)
    ax.legend(loc="upper right")
    figstyle.save(fig, "fig_rq4_quality")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("version")
    args = parser.parse_args()
    summary = coding.summary(args.version)
    figstyle.apply()
    fig_gap_map()
    fig_quality(summary["quality_prevalence"])
    print(json.dumps({k: (v["yes"], v["n"]) for k, v in summary["quality_prevalence"].items()}))


if __name__ == "__main__":
    main()
