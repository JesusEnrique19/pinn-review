"""Write a clean copy of the repository for public release (GitHub, then Zenodo).

The copy holds the tracked files of HEAD with three changes, so that nothing is redistributed that the project
may not share: publisher abstracts are blanked in every CSV (they are re-fetched from OpenAlex by work id),
the manuscript sources and internal notes are left out until the paper is published, and the git history is
not copied (it carries the abstracts and personal e-mail addresses). Usage:

    python scripts/make_public_release.py [OUT_DIR]      (default: release/pinn-review-public)
"""
import shutil
import subprocess
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "release" / "pinn-review-public"
EXCLUDE = ("paper/manuscript.md", "paper/G5_checklist.md", "paper/discusion_conclusiones_es.md", "STATUS.md")
ABSTRACT_COLS = ("abstract", "abstract_s2")


def main():
    files = subprocess.run(["git", "ls-files"], cwd=ROOT, capture_output=True, text=True, check=True).stdout.split("\n")
    if OUT.exists():
        shutil.rmtree(OUT)
    blanked = 0
    for rel in filter(None, files):
        if rel in EXCLUDE:
            continue
        src, dst = ROOT / rel, OUT / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        if rel.endswith((".csv", ".csv.gz")):
            head = pd.read_csv(src, nrows=0, encoding="utf-8-sig").columns
            cols = [c for c in ABSTRACT_COLS if c in head]
            if cols:
                df = pd.read_csv(src, encoding="utf-8-sig", dtype=str, keep_default_na=False)
                df[cols] = ""
                df.to_csv(dst, index=False, encoding="utf-8-sig")
                blanked += 1
                continue
        shutil.copy2(src, dst)
    (OUT / "paper" / "README.md").parent.mkdir(exist_ok=True)
    (OUT / "paper" / "README.md").write_text(
        "The manuscript sources are released after publication. `references.yaml` lists the verified references.\n",
        encoding="utf-8")
    (OUT / "data" / "processed" / "ABSTRACTS.md").write_text(
        "Publisher abstracts are not redistributed: the `abstract` columns of the CSV files are empty. Every row keeps\n"
        "its OpenAlex work id, so the abstracts can be re-fetched from the OpenAlex API (`pinnreview.retrieval`).\n",
        encoding="utf-8")
    print(f"{len(files)} tracked files copied to {OUT}; abstracts blanked in {blanked} CSV files")


if __name__ == "__main__":
    main()
