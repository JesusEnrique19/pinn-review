"""Print sentences that report errors for the benchmark problems in the original method papers.

Used once, before any benchmark run, to fill protocol/replication_anchors.csv from the sources
(not from memory). Usage: python scripts/g4/anchor_snippets.py
"""
import re

from pinnreview.fulltext import TEXT_DIR

SOURCES = {
    "krishnapriyan2021 (W3197732874)": ("W3197732874", r"convection|β\s*=\s*30|beta\s*=\s*30"),
    "wang2021gradpath (W3200673624)": ("W3200673624", r"helmholtz"),
    "wang2022ntk (W3205073660)": ("W3205073660", r"poisson|wave"),
    "wu2023sampling (W4307154444)": ("W4307154444", r"allen|burgers"),
}
ERR = re.compile(r"error|L2|l2|accuracy", re.I)
NUM = re.compile(r"\d\.\d+\s*[×x]\s*10|e-0?\d|\d+(\.\d+)?\s*%|10\s*[−-]\s*\d")

for name, (sid, topic) in SOURCES.items():
    text = re.sub(r"\s+", " ", (TEXT_DIR / f"{sid}.txt").read_text(encoding="utf-8"))
    sents = [s for s in re.split(r"(?<=[.!?])\s+", text) if re.search(topic, s, re.I) and ERR.search(s) and NUM.search(s)]
    print(f"\n### {name}: {len(sents)} candidate sentences")
    for s in sents[:7]:
        print("  -", s[:330])
