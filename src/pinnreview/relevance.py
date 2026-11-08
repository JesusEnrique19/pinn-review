"""Tier 1 relevance classifier (protocol deviation of 2026-10-30, Option A of Gate G1).

    python -m pinnreview.relevance texts v3     # title + abstract per paper -> data/interim (not versioned)
    python -m pinnreview.relevance sample v3    # new training and independent test samples to label
    python -m pinnreview.relevance embed v3     # SPECTER2 embeddings of corpus and labelled records
    python -m pinnreview.relevance select v3    # cross-validated model comparison on training labels
    python -m pinnreview.relevance final v3     # fit, fix threshold, evaluate on test set, score corpus

The test set is never used for training, model selection or threshold choice. Human labels from the
audit files take precedence over LLM labels wherever both exist.
"""

import argparse
import gzip
import json
import math

import numpy as np
import pandas as pd

from pinnreview import retrieval
from pinnreview.paths import INTERIM, PROCESSED
from pinnreview.validation import EXCEL_ENCODING, G1, SEED, abstract_text

MODEL_NAME = "allenai/specter2_base"
TARGET_RECALL = 0.95
TRAIN_FILES = ["precision_v1.csv", "holdout_v1.csv", "precision_v3_new.csv", "relevance_train_{v}.csv"]
TEST_FILE = "relevance_test_{v}.csv"
AUDIT_FILES = ["audit_v1.csv", "audit_relevance_{v}.csv"]


def texts_file(version):
    return INTERIM / f"texts_{version}.jsonl.gz"


def build_texts(version):
    """One text per deduplicated paper: canonical title, and the first available abstract among its versions."""
    raw = {w["id"]: w for w in retrieval.iter_works(version)}
    corpus = pd.read_csv(PROCESSED / f"corpus_{version}.csv.gz")
    INTERIM.mkdir(parents=True, exist_ok=True)
    with gzip.open(texts_file(version), "wt", encoding="utf-8") as out:
        for row in corpus.itertuples():
            versions = [raw[i] for i in row.version_ids.split(";") if i in raw]
            abstract = next((t for t in (abstract_text(v.get("abstract_inverted_index")) for v in versions) if t), "")
            out.write(json.dumps({"id": row.id, "version_ids": row.version_ids, "title": row.title or "",
                                  "abstract": abstract, "venue": row.venue if isinstance(row.venue, str) else "",
                                  "type": row.type, "year": row.year}) + "\n")
    return len(corpus)


def load_texts(version):
    with gzip.open(texts_file(version), "rt", encoding="utf-8") as fh:
        return pd.DataFrame([json.loads(line) for line in fh])


def version_map(texts):
    """Map every OpenAlex version id to its canonical corpus id."""
    return {v: row.id for row in texts.itertuples() for v in row.version_ids.split(";")}


def labelled_ids(version):
    ids = set()
    for name in TRAIN_FILES + [TEST_FILE]:
        path = G1 / name.format(v=version)
        if path.exists():
            ids |= set(pd.read_csv(path, encoding=EXCEL_ENCODING)["id"])
    return ids


def sample(version, n_train=250, n_test=200):
    texts = load_texts(version)
    vmap = version_map(texts)
    taken = {vmap.get(i, i) for i in labelled_ids(version)}
    pool = texts[~texts["id"].isin(taken)].sample(frac=1, random_state=SEED)
    cols = ["id", "year", "type", "venue", "title", "abstract"]
    for name, part in [(TEST_FILE, pool.iloc[:n_test]), ("relevance_train_{v}.csv", pool.iloc[n_test:n_test + n_train])]:
        out = part[cols].assign(in_scope="", confidence="", labeller="", notes="")
        out.to_csv(G1 / name.format(v=version), index=False, encoding=EXCEL_ENCODING)
    return n_train, n_test


def _overrides(version):
    """{id: (label, source)} from audit files: a human label wins over an LLM second-pass label."""
    overrides = {}
    for name in AUDIT_FILES:
        path = G1 / name.format(v=version)
        if not path.exists():
            continue
        audit = pd.read_csv(path, encoding=EXCEL_ENCODING, dtype={"human_label": str, "llm2_label": str})
        for row in audit.itertuples():
            if getattr(row, "human_label", None) in ("y", "n"):
                overrides[row.id] = (row.human_label, "human")
            elif getattr(row, "llm2_label", None) in ("y", "n"):
                overrides[row.id] = (row.llm2_label, "llm_second_pass")
    return overrides


def load_labels(version, files):
    """Labelled records (id, text, y, source) from the given g1 files; audit labels override first-pass LLM labels."""
    overrides = _overrides(version)
    frames = []
    for name in files:
        path = G1 / name.format(v=version)
        if not path.exists():
            continue
        df = pd.read_csv(path, encoding=EXCEL_ENCODING, dtype={"in_scope": str})
        df["label"] = [overrides.get(i, (lab, ""))[0] for i, lab in zip(df["id"], df["in_scope"])]
        df["human"] = [overrides.get(i, ("", ""))[1] == "human" for i in df["id"]]
        df["second_pass"] = [overrides.get(i, ("", ""))[1] == "llm_second_pass" for i in df["id"]]
        df = df[df["label"].isin(["y", "n"])]
        frames.append(df.assign(source=name.format(v=version)))
    df = pd.concat(frames, ignore_index=True).drop_duplicates("id")
    df["abstract"] = df["abstract"].astype(object).where(df["abstract"].notna(), "")
    df["y"] = (df["label"] == "y").astype(int)
    return df


def to_text(title, abstract, sep=" [SEP] "):
    return f"{title or ''}{sep}{abstract or ''}"


def embeddings_file(version):
    return INTERIM / f"specter2_{version}.npz"


def embed(version, batch_size=32, chunk=1000):
    """Encode in chunks saved under data/interim/specter2_chunks; an interrupted run resumes where it stopped."""
    import time

    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(MODEL_NAME, device="cpu")
    model.max_seq_length = 256
    texts = load_texts(version)
    labelled = load_labels(version, TRAIN_FILES + [TEST_FILE])
    extra = labelled[~labelled["id"].isin(set(texts["id"]))]
    ids = list(texts["id"]) + list(extra["id"])
    docs = [to_text(t, a, model.tokenizer.sep_token) for t, a in zip(texts["title"], texts["abstract"])]
    docs += [to_text(t, a, model.tokenizer.sep_token) for t, a in zip(extra["title"], extra["abstract"])]
    chunk_dir = INTERIM / f"specter2_chunks_{version}"
    chunk_dir.mkdir(parents=True, exist_ok=True)
    start = time.monotonic()
    for k in range(0, len(docs), chunk):
        part = chunk_dir / f"{k:06d}.npy"
        if not part.exists():
            np.save(part, model.encode(docs[k:k + chunk], batch_size=batch_size, normalize_embeddings=True).astype(np.float32))
        done = min(k + chunk, len(docs))
        print(f"{done}/{len(docs)} encoded, {time.monotonic() - start:.0f}s", flush=True)
    vectors = np.concatenate([np.load(chunk_dir / f"{k:06d}.npy") for k in range(0, len(docs), chunk)])
    np.savez_compressed(embeddings_file(version), ids=np.array(ids), vectors=vectors)
    return len(ids)


def _features(version, df):
    """SPECTER2 vectors for the rows of ``df`` (by canonical id, falling back to the record's own id)."""
    data = np.load(embeddings_file(version))
    index = {i: k for k, i in enumerate(data["ids"])}
    vmap = version_map(load_texts(version))
    rows = [index.get(vmap.get(i, i), index.get(i)) for i in df["id"]]
    if any(r is None for r in rows):
        raise KeyError("labelled records without embeddings; re-run `embed`")
    return data["vectors"][rows]


def _models():
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline

    return {
        "tfidf_lr": lambda: make_pipeline(TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True),
                                          LogisticRegression(C=4.0, class_weight="balanced", max_iter=2000)),
        "specter2_lr": lambda: LogisticRegression(C=1.0, class_weight="balanced", max_iter=2000),
    }


def threshold_for_recall(y, scores, target=TARGET_RECALL):
    """Highest threshold whose recall on (y, scores) is >= target."""
    pos = np.sort(scores[y == 1])
    k = int(math.floor((1 - target) * len(pos)))
    return pos[k] if len(pos) else 0.5


def _subset(X, idx):
    return X[idx] if isinstance(X, np.ndarray) else [X[i] for i in idx]


def _oof_scores(version, df, name, folds=5, repeats=3):
    """Outer repeated CV; the threshold of each outer fold comes from inner-CV scores of its training part."""
    from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedKFold, cross_val_predict

    X = _features(version, df) if name.startswith("specter2") else [to_text(t, a) for t, a in zip(df["title"], df["abstract"])]
    y = df["y"].to_numpy()
    rskf = RepeatedStratifiedKFold(n_splits=folds, n_repeats=repeats, random_state=SEED)
    inner = StratifiedKFold(n_splits=3, shuffle=True, random_state=SEED)
    results = []
    for train, test in rskf.split(np.zeros(len(y)), y):
        X_train, y_train = _subset(X, train), y[train]
        inner_scores = cross_val_predict(_models()[name](), X_train, y_train, cv=inner, method="predict_proba")[:, 1]
        thr = threshold_for_recall(y_train, inner_scores)
        model = _models()[name]().fit(X_train, y_train)
        results.append((y[test], model.predict_proba(_subset(X, test))[:, 1], thr))
    return results


def select(version, models=None):
    from sklearn.metrics import average_precision_score, roc_auc_score

    df = load_labels(version, TRAIN_FILES)
    report = {"n": len(df), "positives": int(df["y"].sum()), "models": {}}
    for name in models or _models():
        folds = _oof_scores(version, df, name)
        prec = [((s >= t) & (y == 1)).sum() / max((s >= t).sum(), 1) for y, s, t in folds]
        rec = [((s >= t) & (y == 1)).sum() / max((y == 1).sum(), 1) for y, s, t in folds]
        report["models"][name] = {
            "roc_auc": round(float(np.mean([roc_auc_score(y, s) for y, s, _ in folds])), 3),
            "average_precision": round(float(np.mean([average_precision_score(y, s) for y, s, _ in folds])), 3),
            "precision_at_target_recall": round(float(np.mean(prec)), 3),
            "recall_at_threshold": round(float(np.mean(rec)), 3),
        }
    report["selected"] = max(report["models"], key=lambda m: report["models"][m]["average_precision"])
    (PROCESSED / "snapshots" / f"relevance_selection_{version}.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def wilson(k, n, z=1.96):
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return (round(centre - half, 3), round(centre + half, 3))


def final(version):
    from sklearn.model_selection import StratifiedKFold, cross_val_predict

    selection = json.loads((PROCESSED / "snapshots" / f"relevance_selection_{version}.json").read_text(encoding="utf-8"))
    name = selection["selected"]
    train = load_labels(version, TRAIN_FILES)
    test = load_labels(version, [TEST_FILE])
    texts = load_texts(version)

    def feats(frame):
        if name.startswith("specter2"):
            return _features(version, frame)
        return [to_text(t, a) for t, a in zip(frame["title"], frame["abstract"])]

    y = train["y"].to_numpy()
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    oof = cross_val_predict(_models()[name](), feats(train), y, cv=cv, method="predict_proba")[:, 1]
    thr = float(threshold_for_recall(y, oof))
    model = _models()[name]().fit(feats(train), y)

    s_test = model.predict_proba(feats(test))[:, 1]
    yt = test["y"].to_numpy()
    tp, fp = int(((s_test >= thr) & (yt == 1)).sum()), int(((s_test >= thr) & (yt == 0)).sum())
    fn = int(((s_test < thr) & (yt == 1)).sum())
    base = int(yt.sum())

    corpus_scores = model.predict_proba(feats(texts))[:, 1]
    tier1 = pd.DataFrame({"id": texts["id"], "relevance": corpus_scores.round(4), "tier1": corpus_scores >= thr})
    tier1.to_csv(PROCESSED / f"tier1_{version}.csv.gz", index=False, compression="gzip")
    report = {
        "model": name,
        "threshold": round(thr, 4),
        "train_n": len(train), "test_n": len(test),
        "test_prevalence": round(base / len(test), 3),
        "test_precision": round(tp / max(tp + fp, 1), 3), "test_precision_ci95": wilson(tp, tp + fp),
        "test_recall": round(tp / max(tp + fn, 1), 3), "test_recall_ci95": wilson(tp, tp + fn),
        "test_human_labels": int(test["human"].sum()),
        "corpus_n": len(texts), "tier1_n": int(tier1["tier1"].sum()),
    }
    (PROCESSED / "snapshots" / f"relevance_final_{version}.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def audit(version, high_fraction=0.15):
    """Human audit of the test labels: every low-confidence label plus a random share of high-confidence ones."""
    test = pd.read_csv(G1 / TEST_FILE.format(v=version), encoding=EXCEL_ENCODING)
    low = test[test["confidence"] == "low"]
    high = test[test["confidence"] == "high"].sample(frac=high_fraction, random_state=SEED)
    out = pd.concat([low, high])[["id", "year", "type", "venue", "title", "abstract", "in_scope", "confidence", "notes"]]
    out = out.rename(columns={"in_scope": "llm_label", "notes": "llm_notes"}).assign(human_label="", human_notes="")
    out.to_csv(G1 / f"audit_relevance_{version}.csv", index=False, encoding=EXCEL_ENCODING)
    return out["confidence"].value_counts().to_dict()


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("action", choices=["texts", "sample", "embed", "select", "final", "audit"])
    parser.add_argument("version")
    parser.add_argument("--models", nargs="+", choices=list(_models()), help="restrict `select` to these models")
    args = parser.parse_args()
    if args.action == "texts":
        print(f"{build_texts(args.version)} texts -> {texts_file(args.version)}")
    elif args.action == "sample":
        print("train, test =", sample(args.version))
    elif args.action == "embed":
        print(f"{embed(args.version)} embeddings -> {embeddings_file(args.version)}")
    elif args.action == "audit":
        print(audit(args.version))
    elif args.action == "select":
        print(json.dumps(select(args.version, args.models), indent=2))
    else:
        print(json.dumps(final(args.version), indent=2))


if __name__ == "__main__":
    main()
