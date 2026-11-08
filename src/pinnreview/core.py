"""Phase 3 — core methodological set: year-stratified screening for IC4 (protocol 5.2, made operational).

    python -m pinnreview.core queue v3     # randomised per-year screening queues + quotas
    python -m pinnreview.core status v3    # included vs quota per year, next records to screen
    python -m pinnreview.core finalise v3  # core-set file, PRISMA counts and IC4 prevalence per year

Core set = in-scope seed papers (purposive) + year-stratified random Tier 1 papers screened in queue order
until each year's quota of IC4-included papers is met. Screening decisions are read from
``data/processed/g3/screening_{v}.csv`` (columns ``decision`` y/n and ``reason``).
"""

import argparse
import json

import numpy as np
import pandas as pd

from pinnreview import biblio
from pinnreview.paths import PROCESSED
from pinnreview.relevance import load_texts
from pinnreview.validation import EXCEL_ENCODING, SEED

G3 = PROCESSED / "g3"
CORE_TOTAL = 250
REASONS = {
    "INC_METHOD": "proposes or modifies a method in the taxonomy",
    "INC_EVAL": "empirically compares or benchmarks existing methods",
    "INC_THEORY": "theoretical analysis of PINN/operator training or approximation",
    "INC_SURVEY": "survey or tutorial of methods (coded, flagged as secondary)",
    "EC4_APPLICATION": "applies an off-the-shelf method without methodological contribution or comparison",
    "EC1_NO_DE": "no differential-equation structure (Tier 1 false positive)",
    "OFF_TOPIC": "not about neural DE methods (Tier 1 false positive)",
    "EC2_NOTICE": "editorial, publisher's note or special-issue preface",
}


def seed_ids(version):
    seeds = pd.read_csv(PROCESSED / "seed_set_resolved.csv")
    seeds = seeds[seeds["role"] == "seed"]
    corpus = pd.read_csv(PROCESSED / f"corpus_{version}.csv.gz")
    canonical = {v: row.id for row in corpus.itertuples() for v in row.version_ids.split(";")}
    out = {}
    for row in seeds.itertuples():
        versions = (row.version_ids if isinstance(row.version_ids, str) else row.openalex_id).split(";")
        out[row.key] = next((canonical[v] for v in versions if v in canonical), versions[0])
    return out


def quotas(df, n_random):
    counts = df["year"].value_counts().sort_index()
    raw = counts / counts.sum() * n_random
    q = np.floor(raw).astype(int)
    for y in (raw - q).sort_values(ascending=False).index[: n_random - q.sum()]:
        q[y] += 1
    return q


def queue(version):
    df = biblio.load(version)
    seeds = seed_ids(version)
    pool = df[~df["id"].isin(set(seeds.values()))]
    q = quotas(pool, CORE_TOTAL - len(seeds))
    texts = load_texts(version).set_index("id")
    rows = []
    for year, group in pool.groupby("year"):
        order = group.sample(frac=1, random_state=SEED + int(year))
        for rank, r in enumerate(order.itertuples()):
            rows.append({"year": year, "rank_in_year": rank, "id": r.id, "title": r.title,
                         "abstract": texts.at[r.id, "abstract"], "decision": "", "reason": "", "screener": ""})
    G3.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(G3 / f"screening_{version}.csv", index=False, encoding=EXCEL_ENCODING)
    meta = {"seeds_in_core": len(seeds), "random_quota_total": int(q.sum()), "quotas": {int(k): int(v) for k, v in q.items()}}
    (G3 / f"design_{version}.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return meta


def status(version, show=None):
    meta = json.loads((G3 / f"design_{version}.json").read_text(encoding="utf-8"))
    s = pd.read_csv(G3 / f"screening_{version}.csv", encoding=EXCEL_ENCODING, dtype={"decision": str, "reason": str})
    out = []
    for year, quota in meta["quotas"].items():
        g = s[s["year"] == int(year)].sort_values("rank_in_year")
        done = g[g["decision"].isin(["y", "n"])]
        inc = int((done["decision"] == "y").sum())
        out.append({"year": int(year), "quota": quota, "included": inc, "screened": len(done),
                    "complete": inc >= quota, "next_rank": int(len(done))})
    return pd.DataFrame(out)


def finalise(version):
    meta = json.loads((G3 / f"design_{version}.json").read_text(encoding="utf-8"))
    s = pd.read_csv(G3 / f"screening_{version}.csv", encoding=EXCEL_ENCODING, dtype={"decision": str, "reason": str})
    st = status(version)
    if not st["complete"].all():
        raise SystemExit("quotas not met:\n" + st.to_string(index=False))
    screened = s[s["decision"].isin(["y", "n"])]
    # records after the quota was met in a year are not part of the stratified sample
    keep = []
    for year, g in screened.groupby("year"):
        g = g.sort_values("rank_in_year")
        cum = (g["decision"] == "y").cumsum()
        stop = cum[cum >= meta["quotas"][str(year)]].index.min()
        keep.append(g.loc[: stop] if pd.notna(stop) else g)
    sample = pd.concat(keep)
    seeds = seed_ids(version)
    core = pd.concat([
        pd.DataFrame({"id": list(seeds.values()), "seed_key": list(seeds.keys()), "stratum": "seed"}),
        sample[sample["decision"] == "y"][["id", "year", "reason"]].assign(stratum="random"),
    ], ignore_index=True)
    core.to_csv(G3 / f"core_{version}.csv", index=False)
    prev = sample.groupby("year").apply(lambda g: (g["decision"] == "y").mean(), include_groups=False)
    tier1_counts = biblio.load(version)["year"].value_counts()
    weighted = float((prev * tier1_counts.reindex(prev.index)).sum() / tier1_counts.reindex(prev.index).sum())
    prisma = {
        "screened_title_abstract": int(len(sample)),
        "included_random": int((sample["decision"] == "y").sum()),
        "excluded_by_reason": sample.loc[sample["decision"] == "n", "reason"].value_counts().to_dict(),
        "seeds_added": len(seeds),
        "core_total": int(len(core)),
        "ic4_prevalence_by_year": {int(k): float(v) for k, v in prev.items()},
        "ic4_prevalence_tier1_weighted": weighted,
        "estimated_ic4_papers_in_tier1": int(round(weighted * len(biblio.load(version)))),
    }
    (PROCESSED / "snapshots" / f"core_{version}.json").write_text(json.dumps(prisma, indent=2), encoding="utf-8")
    return prisma


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("action", choices=["queue", "status", "finalise"])
    parser.add_argument("version")
    args = parser.parse_args()
    if args.action == "queue":
        print(json.dumps(queue(args.version), indent=2))
    elif args.action == "status":
        print(status(args.version).to_string(index=False))
    else:
        print(json.dumps(finalise(args.version), indent=2))


if __name__ == "__main__":
    main()
