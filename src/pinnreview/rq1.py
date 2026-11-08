"""RQ1 — growth and concentration of the field (protocol Section 8).

    python -m pinnreview.rq1 v3

Outputs: results/tables/rq1_*.csv, results/figures/fig_rq1_*.{pdf,png}, and
data/processed/snapshots/rq1_{version}.json with every number the manuscript quotes. Each analysis is
repeated on the robustness subsets (classifier intersection, Block A); figures use the main Tier 1 set.
"""

import argparse
import json
import math
from collections import Counter
from itertools import combinations

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit

from pinnreview import biblio, figstyle
from pinnreview.paths import PROCESSED, TABLES
from pinnreview.validation import SEED

YEARS = np.arange(2017, 2026)


# ---------- growth ----------

def exponential(t, a, b):
    return a * np.exp(b * t)


def logistic(t, k, r, t0):
    return k / (1 + np.exp(-r * (t - t0)))


def gompertz(t, k, b, c):
    return k * np.exp(-b * np.exp(-c * t))


GROWTH_MODELS = {
    "exponential": (exponential, lambda y: [max(y[0], 1), 0.5]),
    "logistic": (logistic, lambda y: [y.max() * 1.5, 0.8, 7]),
    "gompertz": (gompertz, lambda y: [y.max() * 1.5, 10, 0.4]),
}


def _ic(rss, n, k):
    return n * math.log(rss / n) + 2 * k, n * math.log(rss / n) + k * math.log(n)


def growth(counts, n_boot=2000):
    """Fit each model to annual counts (t = year - 2017); AIC/BIC from least squares."""
    t = (counts.index.to_numpy() - YEARS[0]).astype(float)
    y = counts.to_numpy().astype(float)
    fits = {}
    for name, (f, p0) in GROWTH_MODELS.items():
        try:
            params, _ = curve_fit(f, t, y, p0=p0(y), maxfev=20000)
        except RuntimeError:
            continue
        rss = float(((y - f(t, *params)) ** 2).sum())
        aic, bic = _ic(rss, len(y), len(params))
        fits[name] = {"params": [float(p) for p in params], "rss": rss, "aic": aic, "bic": bic}
    best = min(fits, key=lambda m: fits[m]["aic"])
    for m in fits:
        fits[m]["delta_aic"] = fits[m]["aic"] - fits[best]["aic"]

    # doubling time of the exponential fit with a residual bootstrap CI
    a, b = fits["exponential"]["params"]
    resid = y - exponential(t, a, b)
    rng = np.random.default_rng(SEED)
    boots = []
    for _ in range(n_boot):
        yb = exponential(t, a, b) + rng.choice(resid, size=len(resid), replace=True)
        try:
            (_, bb), _ = curve_fit(exponential, t, yb, p0=[a, b], maxfev=20000)
            boots.append(math.log(2) / bb)
        except RuntimeError:
            continue
    yoy = (y[1:] / y[:-1] - 1)
    return {
        "counts": {int(k): int(v) for k, v in counts.items()},
        "models": fits, "best_by_aic": best,
        "doubling_time_years": math.log(2) / b,
        "doubling_time_ci95": [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))],
        "yoy_growth_last3": [float(g) for g in yoy[-3:]],
    }


# ---------- sources (Bradford) ----------

def bradford(df):
    per_source = df.dropna(subset=["source_id"]).groupby("source_id").agg(papers=("id", "size"), name=("source", "first"))
    per_source = per_source.sort_values("papers", ascending=False).reset_index()
    total = per_source["papers"].sum()
    cum = per_source["papers"].cumsum()
    zone = np.where(cum <= total / 3, 1, np.where(cum <= 2 * total / 3, 2, 3))
    zone[0] = 1
    per_source["zone"] = zone
    sizes = per_source.groupby("zone").size().to_dict()
    multipliers = [sizes.get(2, 0) / max(sizes.get(1, 1), 1), sizes.get(3, 0) / max(sizes.get(2, 1), 1)]

    # Leimkuhler: cumulative share of papers R(r) = a * ln(1 + b r) over source rank r
    r = np.arange(1, len(per_source) + 1, dtype=float)
    share = cum.to_numpy() / total
    (la, lb), _ = curve_fit(lambda r, a, b: a * np.log1p(b * r), r, share, p0=[0.1, 1.0], maxfev=20000)
    pred = la * np.log1p(lb * r)
    r2 = 1 - ((share - pred) ** 2).sum() / ((share - share.mean()) ** 2).sum()
    return per_source, {
        "sources": int(len(per_source)), "papers_with_source": int(total),
        "zone_sizes": {int(k): int(v) for k, v in sizes.items()},
        "bradford_multipliers": multipliers,
        "leimkuhler": {"a": float(la), "b": float(lb), "r2": float(r2)},
        "top10": per_source.head(10)[["name", "papers"]].to_dict("records"),
    }


# ---------- authors (Lotka) ----------

def lotka(df, n_boot=100):
    import powerlaw

    counts = Counter(a["id"] for authors in df["authors"] for a in authors if a.get("id"))
    x = np.array(list(counts.values()))
    fit = powerlaw.Fit(x, discrete=True, verbose=False)
    R_ln, p_ln = fit.distribution_compare("power_law", "lognormal", normalized_ratio=True)
    R_exp, p_exp = fit.distribution_compare("power_law", "exponential", normalized_ratio=True)

    # semi-parametric bootstrap KS test with x_min held fixed (approximation of Clauset et al. 2009).
    # Synthetic tails are drawn by exact inverse transform from the discrete power-law pmf on
    # [x_min, 10^5]; powerlaw's own generator is ~6 s per sample at this size.
    rng = np.random.default_rng(SEED)
    tail = x[x >= fit.xmin]
    body = x[x < fit.xmin]
    p_tail = len(tail) / len(x)
    D_obs = fit.power_law.D
    support = np.arange(int(fit.xmin), 100_000, dtype=float)
    pmf = support ** -fit.power_law.alpha
    pmf /= pmf.sum()
    exceed = 0
    for _ in range(n_boot):
        n_tail = rng.binomial(len(x), p_tail)
        synth = rng.choice(support, size=n_tail, p=pmf)
        if len(body):
            synth = np.concatenate([synth, rng.choice(body, len(x) - n_tail)])
        f = powerlaw.Fit(synth, discrete=True, xmin=fit.xmin, verbose=False)
        exceed += f.power_law.D >= D_obs
    lotka_classic = sum(1 for v in x if v == 1) / len(x)
    return {
        "authors": int(len(x)), "single_paper_share": float(lotka_classic),
        "max_papers": int(x.max()), "alpha": float(fit.power_law.alpha), "xmin": int(fit.xmin),
        "tail_authors": int(len(tail)), "ks_D": float(D_obs), "ks_p_boot": exceed / n_boot, "n_boot": n_boot,
        "vs_lognormal": {"R": float(R_ln), "p": float(p_ln)},
        "vs_exponential": {"R": float(R_exp), "p": float(p_exp)},
    }, x, fit


# ---------- countries ----------

def countries(df):
    with_c = df[df["countries"].str.len() > 0]
    full = Counter(c for cs in with_c["countries"] for c in cs)
    frac = Counter()
    for cs in with_c["countries"]:
        for c in cs:
            frac[c] += 1 / len(cs)
    intl = (with_c["countries"].str.len() > 1).mean()
    edges = Counter(tuple(sorted(p)) for cs in with_c["countries"] for p in combinations(cs, 2))
    table = pd.DataFrame({"papers_full": pd.Series(full), "papers_fractional": pd.Series(frac)}).fillna(0)
    table = table.sort_values("papers_full", ascending=False)
    table["share_full"] = table["papers_full"] / len(with_c)
    by_year = with_c.explode("countries").groupby(["year", "countries"]).size().unstack(fill_value=0)
    return table, by_year, {
        "papers_with_country": int(len(with_c)), "countries": int(len(full)),
        "international_share": float(intl),
        "top10": [{"country": c, "papers": int(n), "share": n / len(with_c)} for c, n in full.most_common(10)],
        "top_pairs": [{"pair": list(p), "papers": int(n)} for p, n in edges.most_common(10)],
    }


# ---------- figures ----------

def fig_growth(g):
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(figstyle.COLUMN_W, 2.2))
    years = np.array(list(g["counts"]))
    y = np.array(list(g["counts"].values()))
    ax.bar(years, y, width=0.7, color=figstyle.SERIES[0], label="Papers per year")
    ax.set_yscale("log")
    ax.grid(axis="y", which="major")
    t = np.linspace(0, years[-1] - YEARS[0], 200)
    styles = {"exponential": "--", "logistic": "-", "gompertz": ":"}
    for i, (name, fit) in enumerate(g["models"].items()):
        f = GROWTH_MODELS[name][0]
        ax.plot(t + YEARS[0], f(t, *fit["params"]), styles[name], color=figstyle.INK2 if name == g["best_by_aic"] else figstyle.MUTED,
                linewidth=1.0, label=f"{name.capitalize()} (ΔAIC {fit['delta_aic']:.1f})")
    ax.set_xlabel("Publication year")
    ax.set_ylabel("Papers (log scale)")
    ax.set_xticks(years)
    ax.legend(loc="upper left")
    figstyle.save(fig, "fig_rq1_growth")


def fig_bradford(per_source, b):
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(figstyle.COLUMN_W, 2.2))
    r = np.arange(1, len(per_source) + 1)
    share = per_source["papers"].cumsum() / per_source["papers"].sum()
    ax.plot(r, share, color=figstyle.SERIES[0], label="Observed")
    ax.plot(r, b["leimkuhler"]["a"] * np.log1p(b["leimkuhler"]["b"] * r), "--", color=figstyle.INK2, linewidth=1.0,
            label=f"Leimkuhler fit (R² {b['leimkuhler']['r2']:.3f})")
    edges = per_source.groupby("zone").size().cumsum()
    for z, edge in edges.items():
        if z < 3:
            ax.axvline(edge, color=figstyle.AXIS, linewidth=0.6)
            ax.text(edge * 1.1, 0.05, f"Zone {z}|{z + 1}", color=figstyle.MUTED, fontsize=6)
    ax.set_xscale("log")
    ax.set_xlabel("Source rank (log scale)")
    ax.set_ylabel("Cumulative share of papers")
    ax.legend(loc="upper left")
    figstyle.save(fig, "fig_rq1_bradford")


def fig_lotka(x, l, fit):
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(figstyle.COLUMN_W, 2.2))
    values, freq = np.unique(x, return_counts=True)
    ccdf = 1 - np.cumsum(freq) / freq.sum() + freq / freq.sum()
    ax.loglog(values, ccdf, "o", color=figstyle.SERIES[0], markersize=2.5, label="Authors")
    xs = np.arange(l["xmin"], values.max() + 1)
    tail_share = (x >= l["xmin"]).mean()
    # exact discrete CCDF of the fitted power law, P(X >= x), on the truncated support used for sampling
    support = np.arange(l["xmin"], 100_000, dtype=float)
    pmf = support ** -l["alpha"]
    pmf /= pmf.sum()
    pl_ccdf = pmf[::-1].cumsum()[::-1][: len(xs)]
    ax.loglog(xs, tail_share * pl_ccdf, "--", color=figstyle.INK2, linewidth=1.0,
              label=f"Power law (Lotka), α = {l['alpha']:.2f}")
    ax.loglog(xs, tail_share * fit.lognormal.ccdf(xs), ":", color=figstyle.SERIES[1], linewidth=1.2,
              label="Lognormal")
    ax.set_xlabel("Papers per author")
    ax.set_ylabel("Share of authors with ≥ x papers")
    ax.grid(axis="both", which="major")
    ax.legend(loc="lower left")
    figstyle.save(fig, "fig_rq1_lotka")


def fig_countries(by_year, top):
    import matplotlib.pyplot as plt

    names = [c["country"] for c in top[:6]]
    fig, ax = plt.subplots(figsize=(figstyle.COLUMN_W, 2.2))
    data = by_year.reindex(columns=names, fill_value=0).loc[YEARS[0]:YEARS[-1]]
    for i, c in enumerate(names):
        ax.plot(data.index, data[c], color=figstyle.SERIES[i], marker="o", label=c)
        if i < 2:  # direct labels only for the two leading series; the rest are identified by the legend
            ax.annotate(c, (data.index[-1], data[c].iloc[-1]), xytext=(3, 0), textcoords="offset points",
                        fontsize=6, color=figstyle.INK2, va="center")
    ax.set_xlabel("Publication year")
    ax.set_ylabel("Papers (full counting)")
    ax.set_xticks(data.index)
    ax.legend(loc="upper left", ncol=2)
    figstyle.save(fig, "fig_rq1_countries")


# ---------- driver ----------

def analyse(df, figures=False):
    counts = df[df["year"].between(YEARS[0], YEARS[-1])].groupby("year").size().reindex(YEARS, fill_value=0)
    g = growth(counts)
    per_source, b = bradford(df)
    l, x, fit = lotka(df)
    ctable, by_year, c = countries(df)
    if figures:
        figstyle.apply()
        fig_growth(g)
        fig_bradford(per_source, b)
        fig_lotka(x, l, fit)
        fig_countries(by_year, c["top10"])
        TABLES.mkdir(parents=True, exist_ok=True)
        per_source.to_csv(TABLES / "rq1_sources_bradford.csv", index=False)
        ctable.to_csv(TABLES / "rq1_countries.csv")
    # preprint servers (arXiv, SSRN, ...) inflate the Bradford core; journals-only as sensitivity
    _, b_journals = bradford(df[df["type"] != "preprint"])
    return {"papers": int(len(df)), "growth": g, "bradford": b, "bradford_without_preprints": b_journals,
            "preprint_share": float((df["type"] == "preprint").mean()), "lotka": l, "countries": c}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("version")
    args = parser.parse_args()
    df = biblio.load(args.version)
    out = {
        "tier1": analyse(df, figures=True),
        "robust_intersection": analyse(df[df["in_tfidf"]]),
        "robust_block_a": analyse(df[df["block_a"]]),
    }
    path = PROCESSED / "snapshots" / f"rq1_{args.version}.json"
    path.write_text(json.dumps(out, indent=2, default=float), encoding="utf-8")
    for name, r in out.items():
        g, b, l, c = r["growth"], r["bradford"], r["lotka"], r["countries"]
        print(f"{name}: n={r['papers']} best={g['best_by_aic']} doubling={g['doubling_time_years']:.2f}y "
              f"bradford_zones={b['zone_sizes']} lotka_alpha={l['alpha']:.2f} (xmin {l['xmin']}, p_boot {l['ks_p_boot']:.2f}) "
              f"intl={c['international_share']:.3f} top={[t['country'] for t in c['top10'][:3]]}")


if __name__ == "__main__":
    main()
