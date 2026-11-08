"""Shared figure style for the manuscript.

Palette and mark rules follow the dataviz reference palette (fixed categorical order, thin marks,
recessive grid, single y-axis). The typeface is Times New Roman at 8 pt because the IJACSA/SAI template
prescribes it for figure labels; the template overrides the palette's sans-serif default.
"""

import matplotlib as mpl
import matplotlib.pyplot as plt

from pinnreview.paths import FIGURES

SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
SEQUENTIAL = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
INK, INK2, MUTED = "#0b0b0b", "#52514e", "#898781"
GRID, AXIS, SURFACE = "#e1e0d9", "#c3c2b7", "#ffffff"

COLUMN_W = 3.5   # inches, one column of the two-column SAI template
FULL_W = 7.16    # inches, figure spanning both columns


def apply():
    mpl.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        "font.size": 8, "axes.titlesize": 8, "axes.labelsize": 8,
        "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7,
        "text.color": INK, "axes.labelcolor": INK2, "xtick.color": MUTED, "ytick.color": MUTED,
        "axes.edgecolor": AXIS, "axes.linewidth": 0.6,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "axes.grid.axis": "y", "grid.color": GRID, "grid.linewidth": 0.5,
        "axes.axisbelow": True, "axes.prop_cycle": mpl.cycler(color=SERIES),
        "lines.linewidth": 1.2, "lines.markersize": 3.5,
        "legend.frameon": False,
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "figure.dpi": 150, "savefig.dpi": 600, "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
        "pdf.fonttype": 42, "ps.fonttype": 42,
    })


def save(fig, name):
    """Write vector PDF (manuscript) and PNG (preview) under results/figures."""
    FIGURES.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES / f"{name}.pdf")
    fig.savefig(FIGURES / f"{name}.png", dpi=300)
    plt.close(fig)
