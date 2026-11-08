"""Build the manuscript in the journal's Word template from paper/manuscript.md (protocol Section 10).

    python -m pinnreview.manuscript            # -> paper/build/manuscript.docx
    python -m pinnreview.manuscript --pdf      # also export PDF with Microsoft Word (Windows) and page previews

Every number in the text is a Jinja2 variable computed here from the analysis snapshots (StrictUndefined: a
missing value fails the build), so the manuscript cannot drift from the results.

Markup accepted in paper/manuscript.md (a small Markdown subset mapped to the template's styles):
  front matter between '---' lines (YAML: title, authors, affiliations, abstract, keywords)
  '# Heading {#sec:label}'  numbered Heading 1 (Roman, by the template); '##' Heading 2; '###' Heading 3
  '#! Heading'              unnumbered heading (Acknowledgment, References)
  '- item'                  bullet list
  '![Caption](fig_name){#fig:label wide}'  figure from results/figures (wide = spans both columns)
  'Table: Caption {#tbl:label wide}' followed by a pipe table
  inline: **bold**, *italic*, ^sup^, ~sub~, [@key; @key2] citations, @fig:x / @tbl:x / @sec:x references,
          [[TODO ...]] highlighted note for the authors
"""

import argparse
import copy
import gzip
import json
import re
import subprocess
from pathlib import Path

import pandas as pd
import yaml
from docx import Document
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_COLOR_INDEX
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Inches, Pt
from jinja2 import Environment, StrictUndefined

from pinnreview.bibliography import Bibliography
from pinnreview.paths import FIGURES, PROCESSED, RAW, ROOT, TABLES

PAPER = ROOT / "paper"
BUILD = PAPER / "build"
TEMPLATE = ROOT / "SAI_PAPER_FORMAT.docx"  # journal template (thesai.org), kept local
SNAP = PROCESSED / "snapshots"
COLUMN_IN, FULL_IN = 3.5, 7.1
ROMAN = ["I", "II", "III", "IV", "V", "VI", "VII", "VIII", "IX", "X", "XI", "XII"]


# ---------------------------------------------------------------- numbers

def _snap(name):
    return json.loads((SNAP / f"{name}.json").read_text(encoding="utf-8"))


def n(x):
    return f"{int(round(x)):,}"


def pct(x, digits=0):
    return f"{100 * x:.{digits}f}%"


def ci_pct(ci, digits=0):
    return f"{100 * ci[0]:.{digits}f}–{100 * ci[1]:.{digits}f}%"


def sci(x, digits=2):
    """6.3 × 10^−2^ in the manuscript markup; plain decimals for values between 0.1 and 10."""
    if 0.1 <= abs(x) < 10:
        return f"{x:.{digits}f}"
    mant, exp = f"{x:.{digits - 1}e}".split("e")
    return f"{mant} × 10^{int(exp)}^".replace("-", "−")


def rq5_values(rq5, cells):
    """Text variables and the results table of RQ5 from the rq5 snapshot and results/tables/rq5_cells.csv."""
    v = {}
    names = {"M1": "learning-rate annealing (M1)", "M2": "NTK weights (M2)", "M3": "adaptive sampling (M3)",
             "M4": "causal training (M4)", "M5": "exact conditions (M5)"}
    h5a = rq5["h5a"]
    for m, d in h5a.items():
        # factor > 1: the remedy lowered the median error by that factor; < 1: it raised it
        v[f"h5a_{m}_factor"] = f"{10 ** -d['median_log10_ratio']:.2g}"
        v[f"h5a_{m}_inv"] = f"{10 ** d['median_log10_ratio']:.2g}"
        v[f"h5a_{m}_ci"] = f"{10 ** -d['ci95'][1]:.2g}–{10 ** -d['ci95'][0]:.2g}"
        v[f"h5a_{m}_wins"] = d["wins"]
    supported = [m for m, d in h5a.items() if d["supported"]]
    worse = [m for m, d in h5a.items() if d["ci95"][0] > 0]
    words = ["no", "one", "two", "three", "four", "five"]
    v["h5a_n_supported"], v["h5a_n_supported_w"], v["h5a_n_worse_w"] = len(supported), words[len(supported)], words[len(worse)]
    med = cells.pivot(index="method", columns="problem", values="median_rel_l2")
    k = int((med.loc["M2"] < med.loc["M0"]).sum())
    v["m2_better"] = "all four" if k == 4 else f"{words[k]} of four"
    v["h5a_supported"] = " and ".join(names[m] for m in supported) or "none"
    v["h5a_n_worse"] = len(worse)
    v["h5a_worse"] = " and ".join(f"{names[m]} on {h5a[m]['problem']}" for m in worse) or "none"

    h5b = rq5["h5b"]
    v["h5b_n"], v["h5b_supported"] = len(h5b), sum(d["supported"] for d in h5b.values())
    speed = [d["speed_ratio"] for d in h5b.values()]
    v["h5b_speed_lo"], v["h5b_speed_hi"] = sci(min(speed)), sci(max(speed))
    match = [d["classical_seconds_to_match_best_pinn"] for d in h5b.values()]
    v["h5b_match_hi"] = f"{max(match):.2g}"
    pinn_min = [d["best_pinn_active_min"] for d in h5b.values()]
    v["h5b_pinn_min_lo"], v["h5b_pinn_min_hi"] = f"{min(pinn_min):.0f}", f"{max(pinn_min):.0f}"
    best = [d["best_pinn_method"] for d in h5b.values()]
    v["h5b_best_m5_w"] = words[sum(b == "M5" for b in best)]

    fn = rq5["friedman_nemenyi"]
    v["fr_blocks"], v["fr_chi2"], v["fr_cd"] = fn["blocks"], f"{fn['chi2']:.1f}", f"{fn['critical_difference']:.2f}"
    v["fr_p"] = "< 0.001" if fn["p_value"] < 0.001 else f"= {fn['p_value']:.3f}"
    ranks = fn["mean_ranks"]
    v["fr_ranks"] = ", ".join(f"{m} {r:.1f}" for m, r in sorted(ranks.items(), key=lambda kv: kv[1]))
    pairs = [(a, b) for a in ranks for b in ranks if ranks[b] - ranks[a] > fn["critical_difference"]]
    v["fr_sig_pairs"] = "; ".join(f"{a} < {b}" for a, b in sorted(pairs, key=lambda p: (ranks[p[0]], ranks[p[1]])))
    top = sorted(ranks, key=ranks.get)[:2]
    v["fr_top"] = " and ".join(top)
    v["fr_top_sig"] = abs(ranks[top[0]] - ranks[top[1]]) > fn["critical_difference"]

    rep = rq5["replication"]
    v["rep_n_w"], v["rep_ok_w"] = words[len(rep)], words[sum(r["within_one_order"] for r in rep)].capitalize()
    p1 =next(r for r in rep if r["problem"] == "P1")
    v["rep_p1_reported"], v["rep_p1_ours"] = f"{p1['reported']:.2f}", f"{p1['ours_median']:.3f}"
    m1 = [r for r in rep if r["method"] == "M1"]
    v["rep_m1_reported"] = " and ".join(f"{r['reported']:.3g}" for r in m1)
    v["rep_m1_ours"] = f"{m1[0]['ours_median']:.2f}"
    v["m1_p1"] = f"{med.loc['M1', 'P1']:.2f}"
    v["rq5_gap_runs"] = rq5["runs_with_suspension_gap"]

    # results table: median relative L2 error over seeds; best PINN per problem in bold
    probs = ["P1", "P2", "P3", "P4"]
    rows = []
    for m in med.index:
        row = [m]
        for p in probs:
            x = med.loc[m, p] if p in med.columns else float("nan")
            if pd.isna(x):
                row.append("–")
            else:
                row.append(f"**{sci(x)}**" if x == med[p].min() else sci(x))
        rows.append(row)
    rows.append(["C"] + [sci(h5b[p]["classical_best_rel_l2"]) for p in probs])
    t = cells.groupby("problem")["median_active_min"]
    rows.append(["PINN time (min)"] + [f"{t.min()[p]:.0f}–{t.max()[p]:.0f}" for p in probs])
    rows.append(["C time (s)"] + [f"{h5b[p]['classical_max_seconds']:.2g}" for p in probs])
    v["rq5_table"] = "\n".join("| " + " | ".join(r) + " |" for r in rows)
    return v


def search_recall():
    """Seed-set and hold-out recall of the v3 harvest, recomputed offline from the raw dump."""
    ids = set()
    with gzip.open(RAW / "openalex_v3.jsonl.gz", "rt", encoding="utf-8") as f:
        for line in f:
            ids.add(json.loads(line)["id"])
    seeds = pd.read_csv(PROCESSED / "seed_set_resolved.csv")
    seeds = seeds[seeds["role"] == "seed"]
    found = seeds["version_ids"].map(lambda v: bool(set(v.split(";")) & ids))
    hold = pd.read_csv(PROCESSED / "g1" / "holdout_v1.csv", encoding="utf-8-sig")
    rel = hold[hold["in_scope"] == "y"]
    return {"seed_found": int(found.sum()), "seed_n": len(seeds),
            "holdout_found": int(rel["id"].isin(ids).sum()), "holdout_n": len(rel)}


def values():
    oa, dd, rel, core = _snap("openalex_v3"), _snap("dedup_v3"), _snap("relevance_final_v3"), _snap("core_v3")
    rq1, cc, fr, dyn = _snap("rq1_v3"), _snap("rq2_cocitation_v3"), _snap("rq2_fronts_v3"), _snap("rq2_dynamics_v3")
    q, refs = _snap("rq34_v3"), _snap("references_v3")
    t1, robust = rq1["tier1"], rq1["robust_intersection"]
    v = {}

    # retrieval and selection (PRISMA)
    v["query_date"] = oa["retrieved_at"][:10]
    v["n_harvested"], v["n_unique"], v["n_papers"] = n(dd["harvested_lines"]), n(dd["unique_records"]), n(dd["papers_after_dedup"])
    v["n_paging_repeats"] = n(dd["repeated_ids_from_paging"])
    v["n_merged"] = n(dd["unique_records"] - dd["papers_after_dedup"])
    v["n_multi_version"] = n(dd["papers_with_multiple_versions"])
    v["n_merged_title"], v["n_merged_arxiv"], v["n_merged_doi"] = (n(dd["merged_by_rule"][k]) for k in ("title", "arxiv", "doi"))
    v["n_tier1"] = n(rel["tier1_n"])
    v["n_clf_excluded"] = n(rel["corpus_n"] - rel["tier1_n"])
    v["clf_train_n"], v["clf_test_n"] = n(rel["train_n"]), n(rel["test_n"])
    v["clf_precision"], v["clf_recall"] = f"{rel['test_precision']:.3f}", f"{rel['test_recall']:.3f}"
    v["clf_precision_ci"] = f"{rel['test_precision_ci95'][0]:.2f}–{rel['test_precision_ci95'][1]:.2f}"
    v["clf_recall_ci"] = f"{rel['test_recall_ci95'][0]:.2f}–{rel['test_recall_ci95'][1]:.2f}"
    rec = search_recall()
    v["seed_recall"] = f"{rec['seed_found']}/{rec['seed_n']}"
    v["holdout_recall"] = f"{rec['holdout_found'] / rec['holdout_n']:.3f} ({rec['holdout_found']}/{rec['holdout_n']})"

    # references
    v["refs_before"] = pct((rel["tier1_n"] - refs["papers_without_openalex_refs"]) / rel["tier1_n"])
    v["refs_after"] = pct(refs["papers_with_references_after"] / rel["tier1_n"])
    v["n_s2_supplemented"] = n(refs["supplemented"])

    # RQ1
    g = t1["growth"]
    years = sorted(g["counts"])
    v["first_year"], v["last_year"] = years[0], years[-1]
    v["count_first"], v["count_last"] = n(g["counts"][years[0]]), n(g["counts"][years[-1]])
    v["doubling"] = f"{g['doubling_time_years']:.2f}"
    v["doubling_ci"] = f"{g['doubling_time_ci95'][0]:.2f}–{g['doubling_time_ci95'][1]:.2f}"
    v["doubling_robust"] = f"{robust['growth']['doubling_time_years']:.2f}"
    v["yoy_last3"] = ", ".join(pct(x) for x in g["yoy_growth_last3"])
    aic = {m: d["aic"] for m, d in g["models"].items()}
    best = min(aic, key=aic.get)
    v["growth_best"] = best.capitalize()
    v["daic_logistic"], v["daic_exponential"] = f"{aic['logistic'] - aic[best]:.1f}", f"{aic['exponential'] - aic[best]:.1f}"
    b, bj = t1["bradford"], t1["bradford_without_preprints"]
    v["n_sources"] = n(b["sources"])
    v["preprint_share"] = pct(rq1["tier1"]["preprint_share"])
    v["bradford_zones"] = " / ".join(n(b["zone_sizes"][k]) for k in ("1", "2", "3"))
    v["bradford_mult"] = " and ".join(f"{x:.1f}" for x in b["bradford_multipliers"])
    v["bradford_j_zones"] = " / ".join(n(bj["zone_sizes"][k]) for k in ("1", "2", "3"))
    v["bradford_j_mult"] = " and ".join(f"{x:.1f}" for x in bj["bradford_multipliers"])
    v["bradford_j_top"] = bj["top10"][0]["name"]
    v["bradford_j_top_papers"] = n(bj["top10"][0]["papers"])
    lk = t1["lotka"]
    v["n_authors"], v["single_share"] = n(lk["authors"]), pct(lk["single_paper_share"], 1)
    v["lotka_alpha"], v["lotka_ks_p"] = f"{lk['alpha']:.2f}", ("< 0.01" if lk["ks_p_boot"] < 0.01 else f"= {lk['ks_p_boot']:.2f}")
    v["lotka_R"] = f"{lk['vs_lognormal']['R']:.1f}".replace("-", "−")
    v["lotka_robust_alpha"], v["lotka_robust_xmin"] = f"{robust['lotka']['alpha']:.2f}", robust["lotka"]["xmin"]
    v["lotka_robust_p"] = f"{robust['lotka']['ks_p_boot']:.2f}"
    c = t1["countries"]
    v["country_coverage"] = pct(c["papers_with_country"] / t1["papers"], 1)
    v["n_countries"] = c["countries"]
    v["intl_share"] = pct(c["international_share"], 1)
    top = {r["country"]: r for r in c["top10"]}
    for code in ("CN", "US", "DE", "GB", "IN"):
        v[f"share_{code}"] = pct(top[code]["share"], 1)
    v["pair_cn_us"] = n(c["top_pairs"][0]["papers"])

    # RQ2
    v["cc_citing"], v["cc_refs"], v["cc_nodes"] = n(cc["citing_papers"]), n(cc["distinct_references"]), n(cc["nodes"])
    v["cc_min_cit"], v["cc_edges"], v["cc_clusters"] = cc["min_citations_in_network"], n(cc["edges"]), cc["clusters"]
    v["cc_ari"] = f"{cc['bootstrap_ari_mean']:.2f}"
    v["cc_ari_ci"] = f"{cc['bootstrap_ari_ci95'][0]:.2f}–{cc['bootstrap_ari_ci95'][1]:.2f}"
    ri = cc["robustness"]["robust_intersection"]
    v["cc_ari_intersection"], v["cc_shared_nodes"] = f"{ri['ari_vs_main_on_shared_nodes']:.2f}", n(ri["shared_nodes"])
    v["cc_ari_block_a"] = f"{cc['robustness']['robust_block_a']['ari_vs_main_on_shared_nodes']:.2f}"
    sizes = sorted((s["size"] for s in cc["cluster_summaries"]), reverse=True)
    v["cc_sizes"] = ", ".join(str(s) for s in sizes)
    v["n_fronts"], v["fronts_ari"] = fr["fronts"]["clusters_ge50"], f"{fr['fronts']['seed_stability_ari']:.2f}"
    v["n_topics"], v["topics_ari"] = fr["topics"]["clusters_ge50"], f"{fr['topics']['seed_stability_ari']:.2f}"
    v["nmi"] = f"{fr['convergent_validity_nmi']:.2f}"
    fronts = pd.read_csv(TABLES / "rq2_fronts.csv")
    kan = fronts[fronts["label"].str.contains("kolmogorov", case=False)].iloc[0]
    v["kan_front_size"], v["kan_front_year"] = int(kan["size"]), int(kan["median_year"])
    v["fronts_median_years"] = f"{int(fronts['median_year'].min())}–{int(fronts['median_year'].max())}"
    topics = pd.read_csv(TABLES / "rq2_topics.csv").set_index("topic")
    # topic ids as labelled at Gate G2: 1 = operator learning, 2 = generic numerical PDE solving
    v["trend_numerical"] = f"{-100 * topics.at[2, 'trend_2021_2025']:.1f}"  # declines, stated as 'fell by'
    v["trend_operator"] = f"{-100 * topics.at[1, 'trend_2021_2025']:.1f}"
    v["trend_heat"] = f"+{100 * topics.at[3, 'trend_2021_2025']:.1f}"
    v["trend_structures"] = f"+{100 * topics.at[4, 'trend_2021_2025']:.1f}"
    bursts = pd.read_csv(TABLES / "rq2_bursts.csv").sort_values("weight", ascending=False).head(20)
    start_year = bursts["start"].str[:4].astype(int)
    v["top20_bursts_2023_2024"] = int(start_year.between(2023, 2024).sum())
    v["top20_bursts_2025"] = int((start_year == 2025).sum())
    kan_burst = bursts[bursts["keyword"].str.contains("kolmogorov")].iloc[0]
    v["kan_burst_papers"] = int(kan_burst["papers"])
    v["n_keywords"], v["n_bursts"] = dyn["keywords_tested"], dyn["bursts"]
    v["n_cit_edges"], v["main_path_len"] = n(dyn["citation_edges_internal"]), dyn["main_path_length"]

    # core set
    v["n_screened"], v["n_included_random"] = core["screened_title_abstract"], core["included_random"]
    v["n_excluded_ta"] = core["screened_title_abstract"] - core["included_random"]
    for k, val in core["excluded_by_reason"].items():
        v[f"ex_{k.lower()}"] = val
    v["n_seeds_core"], v["n_core"] = core["seeds_added"], core["core_total"]
    v["ic4_prev"] = f"{core['ic4_prevalence_tier1_weighted']:.2f}"
    v["n_ic4_est"] = n(core["estimated_ic4_papers_in_tier1"])
    ft = pd.read_csv(PROCESSED / "g3" / "fulltext_v3.csv")
    v["n_fulltext"] = int((ft["status"] == "ok").sum())
    v["n_no_oa"] = int((ft["status"] == "no_open_access").sum())
    v["n_ft_failed"] = int((~ft["status"].isin(["ok", "no_open_access"])).sum())
    v["n_abstract_only"] = v["n_no_oa"] + v["n_ft_failed"]

    # RQ3
    fam, fail = q["method_family_counts"], q["failure_mode_counts"]
    for k, val in {**fam, **fail}.items():
        v[f"c_{k}"] = val
    v["none_stated_share"] = pct(fail["none_stated"] / q["coded"])
    v["max_named_pathology"] = max(fail[k] for k in ("gradient_imbalance", "spectral_bias", "ill_conditioning",
                                                     "causality_violation"))
    gap = pd.read_csv(TABLES / "rq3_gap_map.csv", index_col=0)
    gap = gap.drop(index=[r for r in ("other",) if r in gap.index], columns=[c for c in ("none_stated",) if c in gap.columns])
    v["gap_rows"], v["gap_cols"] = gap.shape
    v["gap_cells"], v["gap_empty"] = gap.size, int((gap == 0).to_numpy().sum())

    # RQ4
    for k, d in q["quality_prevalence"].items():
        tag = k.split("_")[0]
        v[f"{tag}_n"], v[f"{tag}_yes"], v[f"{tag}_share"] = d["n"], d["yes"], pct(d["share_yes"])
        v[f"{tag}_ci"] = ci_pct(d["ci95_yes"])
    v["rs_fulltext_n"] = q["quality_prevalence"]["q1_strong_baseline"]["n"]

    # RQ5 (may be partial while the grid runs)
    rq5 = _snap("rq5_v3")
    v["rq5_runs"], v["rq5_expected"] = rq5["runs"], rq5["expected_runs"]
    v["rq5_complete"] = rq5["runs"] >= rq5["expected_runs"]
    v.update(rq5_values(rq5, pd.read_csv(TABLES / "rq5_cells.csv")))

    # human check of LLM labels (pinnreview.human_check); absent until the author has labelled the sample
    hc = SNAP / "human_check_v1.json"
    v["hc_done"] = hc.exists()
    if v["hc_done"]:
        h = json.loads(hc.read_text(encoding="utf-8"))["tasks"]
        for task in ("all", "relevance", "screening"):
            v[f"hc_{task}_n"], v[f"hc_{task}_agree"] = h[task]["n"], pct(h[task]["agreement"])
            v[f"hc_{task}_kappa"] = f"{h[task]['kappa']:.2f}"
            v[f"hc_{task}_kappa_ci"] = f"{h[task]['kappa_ci95'][0]:.2f}–{h[task]['kappa_ci95'][1]:.2f}"
    return v


# ---------------------------------------------------------------- template surgery

def _sect(template_sect, cols):
    s = copy.deepcopy(template_sect)
    for t in s.findall(qn("w:type")):
        s.remove(t)
    t = OxmlElement("w:type")
    t.set(qn("w:val"), "continuous")
    s.insert(0, t)
    c = s.find(qn("w:cols"))
    if cols == 1:
        for a in list(c.attrib):
            if a != qn("w:space"):
                del c.attrib[a]
        for child in list(c):
            c.remove(child)
    else:
        c.set(qn("w:num"), str(cols))
    return s


class Writer:
    """Fills the SAI template: front matter in place, generated body before the two-column section break."""

    def __init__(self, front, bib):
        self.doc = Document(str(TEMPLATE))
        self.bib = bib
        self.n_wide = 0
        body = self.doc.element.body
        children = list(body.iterchildren())
        paras = {i: el for i, el in enumerate(children)}
        self.title_p, self.subtitle_p = children[0], children[1]
        self.author_p, self.aff_ps = children[3], [children[4], children[5]]
        abstract_i = next(i for i, el in paras.items() if el.tag == qn("w:p") and self._text(el).startswith("Abstract"))
        self.abstract_p, self.keywords_p = children[abstract_i], children[abstract_i + 1]
        sect_is = [i for i, el in paras.items() if el.find(".//" + qn("w:sectPr")) is not None and el.tag == qn("w:p")]
        self.anchor = children[sect_is[-1]]  # paragraph that ends the two-column section
        self.two_col_sect = self.anchor.find(".//" + qn("w:sectPr"))
        frames = [el for i, el in paras.items() if abstract_i + 1 < i < sect_is[-1] and el.find(".//" + qn("w:framePr")) is not None]
        for el in children[abstract_i + 2:sect_is[-1]]:
            if el not in frames:
                body.remove(el)
        self.frames = frames
        self._front(front)

    @staticmethod
    def _text(el):
        return "".join(t.text or "" for t in el.iter(qn("w:t")))

    def _para_obj(self, el):
        from docx.text.paragraph import Paragraph
        return Paragraph(el, self.doc._body)

    def _set_text(self, el, text):
        p = self._para_obj(el)
        for r in list(p.runs):
            r._r.getparent().remove(r._r)
        self.inline(p, text)
        return p

    def _front(self, front):
        self._set_text(self.title_p, front["title"])
        self.subtitle_p.getparent().remove(self.subtitle_p)
        self._set_text(self.author_p, front["authors"])
        for el, aff in zip(self.aff_ps, front["affiliations"] + [None] * 2):
            if aff is None:
                el.getparent().remove(el)
            else:
                self._set_text(el, aff)
        for el, label, text in ((self.abstract_p, "Abstract", front["abstract"]), (self.keywords_p, "Keywords", front["keywords"])):
            p = self._para_obj(el)
            for r in list(p.runs):
                r._r.getparent().remove(r._r)
            lead = p.add_run(label)
            lead.italic = True
            p.add_run("—")
            self.inline(p, text.strip())
        sponsor, corresponding = (self.frames + [None, None])[:2]
        if sponsor is not None:
            self._set_text(sponsor, front.get("funding", "[[TODO funding statement]]"))
        if corresponding is not None:
            self._set_text(corresponding, front.get("corresponding", "[[TODO * Corresponding author (email)]]"))

    # ---- body
    def _new(self, style):
        p = self.doc.add_paragraph(style=style)
        self.anchor.addprevious(p._p)
        return p

    def section_break(self, cols):
        p = self._new("Body Text")
        p.paragraph_format.space_after = Pt(0)
        pPr = p._p.get_or_add_pPr()
        pPr.append(_sect(self.two_col_sect, cols))
        return p

    def heading(self, level, text):
        style = {1: "Heading 1", 2: "Heading 2", 3: "Heading 3", 5: "Heading 5"}[level]
        self.inline(self._new(style), text)

    def paragraph(self, text, style="Body Text"):
        self.inline(self._new(style), text)

    def _top_frame(self, paragraph):
        """Float a paragraph at the top of its page across both columns (Word frame, as in the template).

        Word merges adjacent paragraphs with identical frame properties into one frame, so each figure's frame
        is a few twips narrower than the previous one (word_export.ps1 identifies wide frames to within 2 pt)."""
        frame = OxmlElement("w:framePr")
        for k, v in (("w", str(int(FULL_IN * 1440) - 2 * self.n_wide)), ("hSpace", "0"), ("vSpace", "180"), ("wrap", "notBeside"),
                     ("vAnchor", "margin"), ("hAnchor", "margin"), ("xAlign", "center"), ("yAlign", "top")):
            frame.set(qn(f"w:{k}"), v)
        pPr = paragraph._p.get_or_add_pPr()
        pPr.insert(1 if pPr.find(qn("w:pStyle")) is not None else 0, frame)

    def figure(self, name, caption, wide):
        p = self._new("Body Text")
        p.paragraph_format.first_line_indent = Pt(0)
        p.paragraph_format.space_before = Pt(6)
        p.alignment = 1
        p.paragraph_format.keep_with_next = True
        # Native size keeps the 8 pt figure text at 8 pt; larger figures are scaled down to the available width
        from PIL import Image

        path = FIGURES / f"{name}.png"
        native_in = Image.open(path).size[0] / 300
        p.add_run().add_picture(str(path), width=Inches(min(native_in, FULL_IN if wide else COLUMN_IN)))
        cap = self._new("16_Figure Caption")
        self.inline(cap, caption)
        if wide:  # figure and caption share one frame at the top of the page, like a LaTeX figure*
            self._top_frame(p)
            self._top_frame(cap)
            self.n_wide += 1

    @staticmethod
    def column_widths(rows, total, char_in=0.068, pad_in=0.2):
        """Widths (inches) proportional to the capped longest cell text, but never narrower than the longest
        unbreakable word (spaces break, no-break spaces do not), so short columns stay narrow and do not wrap."""
        plain = [[re.sub(r"\[@[^\]]+\]|[*~^]", "", c).strip() for c in r] for r in rows]
        cols = range(len(rows[0]))
        weights = [max(min(len(r[j]), 32) for r in plain) + 7 for j in cols]
        floor = [max(len(w) for r in plain for w in r[j].split(" ")) * char_in + pad_in for j in cols]
        pinned = set()
        while True:  # pin columns below their floor, share the rest of the width by weight
            free = [j for j in cols if j not in pinned]
            left = total - sum(floor[j] for j in pinned)
            widths = [floor[j] if j in pinned else left * weights[j] / sum(weights[k] for k in free) for j in cols]
            short = [j for j in free if widths[j] < floor[j]]
            if not short or len(short) == len(free):
                return widths
            pinned.update(short)

    def table(self, caption, rows, wide):
        if wide:
            self.section_break(2)
        cap = self._new("14_Table Head")
        cap.paragraph_format.keep_with_next = True
        self.inline(cap, caption)
        header, body = rows[0], rows[1:]
        t = self.doc.add_table(rows=len(rows), cols=len(header))
        self.anchor.addprevious(t._tbl)
        t.alignment = WD_TABLE_ALIGNMENT.CENTER
        t.autofit = False
        widths = [Inches(w) for w in self.column_widths(rows, FULL_IN if wide else COLUMN_IN)]
        for i, row in enumerate(rows):
            for j, cell_text in enumerate(row):
                cell = t.cell(i, j)
                cell.width = widths[j]
                para = cell.paragraphs[0]
                para.style = self.doc.styles["table col head" if i == 0 else "table copy"]
                para.paragraph_format.first_line_indent = Pt(0)
                para.paragraph_format.keep_with_next = i < len(rows) - 1  # keep the table in one column
                para.alignment = 0
                self.inline(para, cell_text.strip())
            trPr = t.rows[i]._tr.get_or_add_trPr()
            trPr.append(OxmlElement("w:cantSplit"))
        self._rules(t, len(rows))
        spacer = self._new("Body Text")
        spacer.paragraph_format.space_after = Pt(2)
        if wide:
            self.section_break(1)

    @staticmethod
    def _rules(t, nrows):
        """Booktabs-style rules: top, below header, bottom."""
        def border(cell, side, sz):
            tcPr = cell._tc.get_or_add_tcPr()
            b = tcPr.find(qn("w:tcBorders"))
            if b is None:
                b = OxmlElement("w:tcBorders")
                tcPr.append(b)
            e = OxmlElement(f"w:{side}")
            e.set(qn("w:val"), "single")
            e.set(qn("w:sz"), str(sz))
            e.set(qn("w:color"), "000000")
            b.append(e)
        for j in range(len(t.columns)):
            border(t.cell(0, j), "top", 8)
            border(t.cell(0, j), "bottom", 4)
            border(t.cell(nrows - 1, j), "bottom", 8)

    def references(self):
        for key, text, manual in self.bib.rendered():
            self.inline(self._new("17_References"), text)

    # ---- inline markup
    TOKEN = re.compile(r"(\[\[.+?\]\]|\[@[^\]]+\]|\*\*.+?\*\*|\*[^*\s][^*]*?\*|\^[^^]+\^|~[^~]+~)")

    @staticmethod
    def smart_quotes(text):
        text = re.sub(r'"([^"]*)"', "“\\1”", text)
        return text.replace("'", "’")

    def inline(self, p, text):
        for part in self.TOKEN.split(self.smart_quotes(text)):
            if not part:
                continue
            if part.startswith("[["):
                r = p.add_run(part[2:-2].strip())
                r.font.highlight_color = WD_COLOR_INDEX.YELLOW
            elif part.startswith("[@"):
                keys = [k.strip().lstrip("@") for k in part[1:-1].split(";")]
                p.add_run(self.cite(keys))
            elif part.startswith("**"):  # may contain ^sup^ / ~sub~ (e.g. a bold table value 1.7 × 10^−2^)
                n_runs = len(p.runs)
                self.inline(p, part[2:-2])
                for r in p.runs[n_runs:]:
                    r.bold = True
            elif part.startswith("*"):
                p.add_run(part[1:-1]).italic = True
            elif part.startswith("^"):
                p.add_run(part[1:-1]).font.superscript = True
            elif part.startswith("~"):
                p.add_run(part[1:-1]).font.subscript = True
            else:
                p.add_run(part)

    def cite(self, keys):
        nums = sorted({self.bib.cite(k) for k in keys})
        groups, start = [], nums[0]
        for a, b in zip(nums, nums[1:] + [None]):
            if b != a + 1:
                groups.append(f"[{start}]" if start == a else (f"[{start}], [{a}]" if a == start + 1 else f"[{start}]–[{a}]"))
                start = b
        return ", ".join(groups)


# ---------------------------------------------------------------- markdown

LABEL = re.compile(r"\s*\{([^}]*)\}\s*$")


def _attrs(line):
    m = LABEL.search(line)
    if not m:
        return line.strip(), None, False
    parts = m.group(1).split()
    label = next((p[1:] for p in parts if p.startswith("#")), None)
    return line[: m.start()].strip(), label, "wide" in parts


def parse(text):
    """Blocks: ('h', level, text, label) | ('p', text) | ('li', text) | ('fig', name, caption, label, wide)
    | ('tbl', caption, rows, label, wide) | ('refs',)."""
    blocks, buf, lines, i = [], [], text.split("\n"), 0

    def flush():
        if buf:
            blocks.append(("p", " ".join(s.strip() for s in buf)))
            buf.clear()

    while i < len(lines):
        line = lines[i]
        s = line.strip()
        if not s or s.startswith("<!--"):
            flush()
            if s.startswith("<!--"):
                while "-->" not in lines[i]:
                    i += 1
        elif s == "[[REFERENCES]]":
            flush()
            blocks.append(("refs",))
        elif s.startswith("#!"):
            flush()
            blocks.append(("h", 5, s[2:].strip(), None))
        elif s.startswith("#"):
            flush()
            level = len(s) - len(s.lstrip("#"))
            title, label, _ = _attrs(s.lstrip("#"))
            blocks.append(("h", level, title, label))
        elif s.startswith("- "):
            flush()
            item = [s[2:]]
            while i + 1 < len(lines) and lines[i + 1].startswith("  ") and lines[i + 1].strip():
                i += 1
                item.append(lines[i].strip())
            blocks.append(("li", " ".join(item)))
        elif s.startswith("!["):
            flush()
            m = re.match(r"!\[(.*)\]\(([^)]+)\)(.*)$", s)
            _, label, wide = _attrs("x" + m.group(3))
            blocks.append(("fig", m.group(2), m.group(1), label, wide))
        elif s.startswith("Table:"):
            flush()
            caption, label, wide = _attrs(s[len("Table:"):])
            rows = []
            while i + 1 < len(lines) and lines[i + 1].strip().startswith("|"):
                i += 1
                cells = lines[i].strip().strip("|").split("|")
                if not all(re.fullmatch(r"\s*:?-+:?\s*", c) for c in cells):
                    rows.append(cells)
            blocks.append(("tbl", caption, rows, label, wide))
        else:
            buf.append(s)
        i += 1
    flush()
    return blocks


def resolve_refs(blocks):
    """Number sections (Roman / letters), figures and tables in order; @sec:x -> 'Section II', @fig:x -> 'Fig. 3',
    @tbl:x -> 'Table IV' (IEEE style)."""
    names, h1, h2, nfig, ntab = {}, 0, 0, 0, 0
    for b in blocks:
        if b[0] == "h" and b[1] == 1:
            h1, h2 = h1 + 1, 0
            if b[3]:
                names[b[3]] = f"Section {ROMAN[h1 - 1]}"
        elif b[0] == "h" and b[1] == 2:
            h2 += 1
            if b[3]:
                names[b[3]] = f"Section {ROMAN[h1 - 1]}-{chr(64 + h2)}"
        elif b[0] == "fig":
            nfig += 1
            if b[3]:
                names[b[3]] = f"Fig. {nfig}"
        elif b[0] == "tbl":
            ntab += 1
            if b[3]:
                names[b[3]] = f"Table {ROMAN[ntab - 1]}"

    def sub(m):
        key = m.group(1)
        if key not in names:
            raise KeyError(f"unknown cross-reference @{key}")
        return names[key]

    pat = re.compile(r"@((?:sec|fig|tbl):[\w-]+)")
    out = []
    for b in blocks:
        out.append(tuple(pat.sub(sub, x) if isinstance(x, str) else
                         ([[pat.sub(sub, c) for c in row] for row in x] if isinstance(x, list) else x) for x in b))
    return out


def build(pdf=False):
    src = (PAPER / "manuscript.md").read_text(encoding="utf-8")
    # '{#label}' marks figures and sections, so Jinja comments use '{##' instead of '{#'
    env = Environment(undefined=StrictUndefined, keep_trailing_newline=True,
                      comment_start_string="{##", comment_end_string="##}")
    text = env.from_string(src).render(**values())
    _, front_raw, body = text.split("---\n", 2)
    front = yaml.safe_load(front_raw)
    bib = Bibliography()
    w = Writer(front, bib)
    for b in resolve_refs(parse(body)):
        kind = b[0]
        if kind == "h":
            w.heading(b[1], b[2])
        elif kind == "p":
            w.paragraph(b[1])
        elif kind == "li":
            w.paragraph(b[1], "11_Bullet List")
        elif kind == "fig":
            w.figure(b[1], b[2], b[4])
        elif kind == "tbl":
            w.table(b[1], b[2], b[4])
        elif kind == "refs":
            w.references()
    BUILD.mkdir(parents=True, exist_ok=True)
    out = BUILD / "manuscript.docx"
    w.doc.save(str(out))
    manual = [k for k, _, m in bib.rendered() if m]
    print(f"{out.relative_to(ROOT)}: {len(bib.order)} references ({len(manual)} typed by hand: {', '.join(manual)})")
    if pdf:
        export_pdf(out)
    return out


def export_pdf(docx_path):
    """Word (COM) renders the PDF exactly as the journal will see it; page images go to paper/build/pages."""
    pdf = docx_path.with_suffix(".pdf")
    try:
        pdf.unlink(missing_ok=True)
    except PermissionError:  # open in a PDF viewer: do not touch the viewer, write next to it
        pdf = docx_path.with_suffix(".new.pdf")
        print(f"warning: {docx_path.with_suffix('.pdf').name} is open in another program; writing {pdf.name}")
        pdf.unlink(missing_ok=True)
    script = Path(__file__).with_name("word_export.ps1")
    subprocess.run(["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", str(script),
                    "-Docx", str(docx_path), "-Pdf", str(pdf), "-WidePoints", str(FULL_IN * 72)], check=True)
    if not pdf.exists():
        raise RuntimeError(f"Word did not write {pdf}")
    import pymupdf as fitz

    pages = BUILD / "pages"
    pages.mkdir(exist_ok=True)
    for old in pages.glob("*.png"):
        old.unlink()
    doc = fitz.open(pdf)
    for i, page in enumerate(doc):
        page.get_pixmap(dpi=80).save(pages / f"page-{i + 1:02d}.png")
    print(f"{pdf.relative_to(ROOT)}: {doc.page_count} pages")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--pdf", action="store_true")
    parser.add_argument("--values", action="store_true", help="print the computed values and exit")
    args = parser.parse_args()
    if args.values:
        print(json.dumps(values(), indent=1, default=str))
        return
    build(pdf=args.pdf)


if __name__ == "__main__":
    main()
