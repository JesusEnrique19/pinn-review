"""Phase 3 — open-access full texts of the core set (downloads authorised by the author on 2026-10-30).

    python -m pinnreview.fulltext download v3   # arXiv PDF first, else OpenAlex best open-access PDF
    python -m pinnreview.fulltext extract v3    # PDF -> plain text (PyMuPDF)

PDFs go to data/raw/fulltext (not versioned) and text to data/interim/fulltext. The manifest
data/processed/g3/fulltext_{v}.csv records source, URL, HTTP status, size and SHA-256 of every file.
Papers without an open-access version are coded from the abstract only (STATUS, Phase 3).
"""

import argparse
import hashlib
import time

import pandas as pd
import requests

from pinnreview import retrieval
from pinnreview.core import G3
from pinnreview.dedup import arxiv_id
from pinnreview.paths import INTERIM, PROCESSED, RAW

PDF_DIR = RAW / "fulltext"
TEXT_DIR = INTERIM / "fulltext"
HEADERS = {"User-Agent": "pinnreview/0.1 (systematic review; polite, rate-limited)"}
ARXIV_DELAY_S = 3.0   # arXiv asks automated clients for at most one request every 3 seconds
OTHER_DELAY_S = 1.0


def short(work_id):
    return work_id.rsplit("/", 1)[-1]


def sources(version):
    core = pd.read_csv(G3 / f"core_{version}.csv")
    corpus = pd.read_csv(PROCESSED / f"corpus_{version}.csv.gz").set_index("id")
    raw = {w["id"]: w for w in retrieval.iter_works(version)}
    rows = []
    for cid in core["id"]:
        vids = corpus.at[cid, "version_ids"].split(";") if cid in corpus.index else [cid]
        versions = [raw[v] for v in vids if v in raw]
        arx = next((a for a in (arxiv_id(v) for v in versions) if a), None)
        pdf = next((u for u in (((v.get("best_oa_location") or {}).get("pdf_url")) for v in versions) if u), None)
        if arx:
            rows.append({"id": cid, "source": "arxiv", "url": f"https://arxiv.org/pdf/{arx}"})
        elif pdf:
            rows.append({"id": cid, "source": "openalex_oa", "url": pdf})
        else:
            rows.append({"id": cid, "source": "none", "url": None})
    return pd.DataFrame(rows)


def download(version):
    PDF_DIR.mkdir(parents=True, exist_ok=True)
    manifest_path = G3 / f"fulltext_{version}.csv"
    src = sources(version)
    done = pd.read_csv(manifest_path) if manifest_path.exists() else pd.DataFrame(columns=["id"])
    ok = set(done.loc[done.get("status", pd.Series(dtype=object)) == "ok", "id"]) if len(done) else set()
    rows = [r for r in done.to_dict("records") if r["id"] in ok]
    for r in src.to_dict("records"):
        if r["id"] in ok:
            continue
        if r["source"] == "none":
            rows.append({**r, "status": "no_open_access", "http": None, "bytes": 0, "sha256": None})
            continue
        time.sleep(ARXIV_DELAY_S if r["source"] == "arxiv" else OTHER_DELAY_S)
        path = PDF_DIR / f"{short(r['id'])}.pdf"
        try:
            resp = requests.get(r["url"], headers=HEADERS, timeout=90, allow_redirects=True)
            body = resp.content
            is_pdf = resp.ok and body[:5] == b"%PDF-"
            if is_pdf:
                path.write_bytes(body)
            rows.append({**r, "status": "ok" if is_pdf else ("not_pdf" if resp.ok else "http_error"),
                         "http": resp.status_code, "bytes": len(body) if is_pdf else 0,
                         "sha256": hashlib.sha256(body).hexdigest() if is_pdf else None})
        except requests.RequestException as exc:
            rows.append({**r, "status": f"error:{type(exc).__name__}", "http": None, "bytes": 0, "sha256": None})
        pd.DataFrame(rows).to_csv(manifest_path, index=False)
    out = pd.DataFrame(rows)
    out.to_csv(manifest_path, index=False)
    return out["status"].value_counts().to_dict()


def extract(version):
    import pymupdf

    TEXT_DIR.mkdir(parents=True, exist_ok=True)
    manifest = pd.read_csv(G3 / f"fulltext_{version}.csv")
    stats = {"ok": 0, "failed": 0}
    for r in manifest[manifest["status"] == "ok"].to_dict("records"):
        out = TEXT_DIR / f"{short(r['id'])}.txt"
        if out.exists():
            stats["ok"] += 1
            continue
        try:
            with pymupdf.open(PDF_DIR / f"{short(r['id'])}.pdf") as doc:
                out.write_text("\n".join(page.get_text() for page in doc), encoding="utf-8")
            stats["ok"] += 1
        except Exception:  # corrupt or encrypted PDFs are counted, not fatal
            stats["failed"] += 1
    return stats


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("action", choices=["download", "extract"])
    parser.add_argument("version")
    args = parser.parse_args()
    print(download(args.version) if args.action == "download" else extract(args.version))


if __name__ == "__main__":
    main()
