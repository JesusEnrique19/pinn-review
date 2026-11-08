"""PRISMA 2020 flow diagram (fig_prisma) drawn from the snapshot values used in the manuscript.

    python -m pinnreview.prisma
"""

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

from pinnreview import figstyle
from pinnreview.manuscript import values

BOX_W, EX_W = 0.47, 0.47
MAIN_X, EX_X = 0.02, 0.53
LINE_H = 0.027


def _box(ax, x, y, w, h, text, fill):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=0.008",
                                linewidth=0.6, edgecolor=figstyle.INK2, facecolor=fill))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=6, linespacing=1.2)


def _arrow(ax, x0, y0, x1, y1):
    ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                arrowprops={"arrowstyle": "-|>", "color": figstyle.INK2, "linewidth": 0.6, "mutation_scale": 6})


def fig_prisma():
    v = values()
    fig, ax = plt.subplots(figsize=(figstyle.COLUMN_W, 4.6))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    main, side = "#eef3fb", "#f6f6f4"
    rows = [  # (y, height, main text, exclusion text)
        (0.865, 0.11, f"Records identified in OpenAlex\n(query v3, title + abstract,\n2017–2025, {v['query_date']})\nn = {v['n_harvested']}",
         f"Removed before screening:\npaging repeats n = {v['n_paging_repeats']}\nversions merged n = {v['n_merged']}\n(title {v['n_merged_title']}, arXiv {v['n_merged_arxiv']},\nDOI {v['n_merged_doi']})"),
        (0.70, 0.085, f"Papers screened by the\nrelevance classifier\nn = {v['n_papers']}",
         f"Excluded by classifier\n(test P = {v['clf_precision']}, R = {v['clf_recall']})\nn = {v['n_clf_excluded']}"),
        (0.545, 0.085, f"Tier 1: bibliometric corpus\n(RQ1–RQ2)\nn = {v['n_tier1']}", None),
        (0.375, 0.10, f"Year-stratified random sample\nscreened on title/abstract (IC4)\nn = {v['n_screened']}",
         f"Excluded n = {v['n_excluded_ta']}\napplication only (EC4) {v['ex_ec4_application']}\nno differential equation {v['ex_ec1_no_de']}\noff-topic {v['ex_off_topic']}; notices {v['ex_ec2_notice']}"),
        (0.21, 0.10, f"Core set (RQ3) n = {v['n_core']}\n{v['n_included_random']} random + {v['n_seeds_core']} seed papers",
         f"Full text not available\n(abstract-only coding)\nn = {v['n_abstract_only']}\n(no open access {v['n_no_oa']}, failed {v['n_ft_failed']})"),
        (0.05, 0.10, f"Coded from full text n = {v['n_fulltext']}\nRQ4 prevalence: random\nstratum with full text n = {v['rs_fulltext_n']}", None),
    ]
    for i, (y, h, text, ex) in enumerate(rows):
        _box(ax, MAIN_X, y, BOX_W, h, text, main)
        if ex:
            eh = max(h, LINE_H * (ex.count(chr(10)) + 1) + 0.02)
            _box(ax, EX_X, y + h / 2 - eh / 2, EX_W, eh, ex, side)
            _arrow(ax, MAIN_X + BOX_W, y + h / 2, EX_X, y + h / 2)
        if i + 1 < len(rows):
            ny, nh = rows[i + 1][0], rows[i + 1][1]
            _arrow(ax, MAIN_X + BOX_W / 2, y, MAIN_X + BOX_W / 2, ny + nh)
    for y, label in ((0.92, "Identification"), (0.62, "Screening"), (0.15, "Included")):
        ax.text(-0.005, y, label, rotation=90, ha="right", va="center", fontsize=6.5, color=figstyle.INK2)
    figstyle.save(fig, "fig_prisma")


def main():
    figstyle.apply()
    fig_prisma()


if __name__ == "__main__":
    main()
