"""Phase 3 — objective evidence indicators from full texts, and coding dossiers for the LLM coder.

    python -m pinnreview.evidence v3

For every core paper with extracted text, the body (text before the last "References" heading) is searched
with fixed regular expressions for each quality-rubric signal. Each indicator keeps up to two snippets
(±160 characters) so that the final codebook value (Q1–Q8) can be checked against the source.
Outputs: data/processed/g3/evidence_{v}.csv (counts and flags, versioned) and
data/interim/dossiers_{v}.jsonl (snippets + contribution sentences, not versioned: copyrighted text).
"""

import argparse
import json
import re

import pandas as pd

from pinnreview.core import G3
from pinnreview.fulltext import TEXT_DIR, short
from pinnreview.paths import INTERIM
from pinnreview.relevance import load_texts

PATTERNS = {
    "code_link": r"(github\.com|gitlab\.com|bitbucket\.org|zenodo\.org|huggingface\.co|codeocean\.com)/[\w\-./]+",
    "code_statement": r"(code|implementation|software|scripts?)[^.\n]{0,60}(publicly|openly|freely)?\s*available|available\s+(at|on|from)\s+(https?://|github)",
    "on_request": r"(upon|on)\s+(reasonable\s+)?request",
    "seeds": r"random\s+seeds?|independent\s+(runs|trials|realizations)|\b\d+\s+(runs|trials|repetitions|realizations|seeds)\b|different\s+(random\s+)?initiali[sz]ations",
    "dispersion": r"±|\\pm|standard\s+deviation|\bstd\b|confidence\s+interval|error\s+bars?",
    "classical_solver": r"finite[\s-]+element|finite[\s-]+difference|finite[\s-]+volume|spectral\s+method|\bFEM\b|\bFDM\b|\bFVM\b|COMSOL|OpenFOAM|ANSYS|Abaqus|FEniCS|Chebfun|Runge[\s–-]+Kutta|ode45|\bDedalus\b",
    "solver_comparison": r"(compared?|comparison|versus|vs\.?|outperform\w*|faster than|speed[\s-]?up)[^.\n]{0,80}(finite[\s-]+element|finite[\s-]+difference|finite[\s-]+volume|FEM|FDM|FVM|numerical\s+solver|traditional\s+(numerical\s+)?(method|solver)s?|classical\s+(numerical\s+)?(method|solver)s?)",
    "cost": r"wall[\s-]?clock|training\s+time|computational\s+(time|cost)|CPU\s+time|GPU\s+hours?|run[\s-]?time|seconds\s+per|inference\s+time",
    "hardware": r"NVIDIA|\bGPU\b|\bA100\b|\bV100\b|\bH100\b|\bRTX\b|\bTPU\b|Intel\s+(Xeon|Core)",
    "metric_rel_l2": r"relative\s+L\s?_?\{?2\}?|relative\s+L2|L2\s+relative|rel\.?\s*L2|relative\s+(l2[\s-])?error",
    "metric_other": r"\bMSE\b|mean\s+squared\s+error|\bRMSE\b|\bMAE\b|L\s?∞|L_\{?\\infty|maximum\s+(absolute\s+)?error",
    "tuning": r"hyper[\s-]?parameter\s+(search|tuning|optimi[sz]ation|selection)|grid\s+search|random\s+search|Bayesian\s+optimi[sz]ation\s+of\s+hyper|Optuna",
    "limitations": r"limitations?|fail(s|ed|ure)\s+(to|mode)|does\s+not\s+converge|future\s+work|drawbacks?|shortcomings?",
}
CONTRIBUTION = re.compile(r"[^.\n]*\b(we\s+propose|we\s+introduce|we\s+present|we\s+develop|this\s+(paper|work|study)\s+(proposes|introduces|presents|develops)|our\s+contributions?)\b[^.]*\.", re.I)
REFS = re.compile(r"\n\s*(References|REFERENCES|Bibliography|BIBLIOGRAPHY)\s*\n")


def body_of(text):
    hits = list(REFS.finditer(text))
    return text[: hits[-1].start()] if hits else text


def snippets(regex, text, k=2, width=160):
    out = []
    for m in re.finditer(regex, text, flags=re.I):
        s = max(0, m.start() - width)
        out.append(re.sub(r"\s+", " ", text[s:m.end() + width]))
        if len(out) >= k:
            break
    return out


def run(version):
    core = pd.read_csv(G3 / f"core_{version}.csv")
    manifest = pd.read_csv(G3 / f"fulltext_{version}.csv").set_index("id")
    texts = load_texts(version).set_index("id")
    rows, dossiers = [], []
    for cid in core["id"]:
        path = TEXT_DIR / f"{short(cid)}.txt"
        has_text = path.exists()
        body = body_of(path.read_text(encoding="utf-8")) if has_text else ""
        row = {"id": cid, "full_text": has_text, "source": manifest.at[cid, "source"] if cid in manifest.index else "none",
               "body_chars": len(body)}
        dossier = {"id": cid, "full_text": has_text,
                   "title": texts.at[cid, "title"] if cid in texts.index else None,
                   "abstract": texts.at[cid, "abstract"] if cid in texts.index else None,
                   "contribution": [re.sub(r"\s+", " ", m.group(0)).strip() for m in CONTRIBUTION.finditer(body)][:4],
                   "evidence": {}}
        for name, regex in PATTERNS.items():
            n = len(re.findall(regex, body, flags=re.I)) if has_text else 0
            row[name] = n
            if n:
                dossier["evidence"][name] = snippets(regex, body)
        rows.append(row)
        dossiers.append(dossier)
    out = pd.DataFrame(rows)
    out.to_csv(G3 / f"evidence_{version}.csv", index=False)
    with open(INTERIM / f"dossiers_{version}.jsonl", "w", encoding="utf-8") as fh:
        for d in dossiers:
            fh.write(json.dumps(d) + "\n")
    with_text = out[out["full_text"]]
    return {"core": len(out), "with_full_text": int(len(with_text)),
            **{f"share_{k}": round(float((with_text[k] > 0).mean()), 3) for k in PATTERNS}}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("version")
    args = parser.parse_args()
    print(json.dumps(run(args.version), indent=2))


if __name__ == "__main__":
    main()
