"""LLM title/abstract screening decisions for the core set (IC4), keyed by (year, rank_in_year).

Rule: include (y) papers that propose or modify a method, empirically compare methods, give theory, or
survey methods; exclude (n) off-the-shelf applications without methodological contribution (EC4), records
without DE structure or neural networks (EC1/OFF_TOPIC) and notices (EC2). Screener: LLM (Claude).

    python scripts/g3/screen_labels.py
"""
import pandas as pd

from pinnreview.core import G3, REASONS
from pinnreview.validation import EXCEL_ENCODING

M, E, T, S = "INC_METHOD", "INC_EVAL", "INC_THEORY", "INC_SURVEY"
A, X, O, N = "EC4_APPLICATION", "EC1_NO_DE", "OFF_TOPIC", "EC2_NOTICE"
Y = {M, E, T, S}

DECISIONS = {
    2017: [M],
    2018: [M],
    2019: [M, M, X, X, M],
    2020: [X, X, M, M, M, M, E, M],
    2021: [A, A, M, A, A, M, M, M, O, M, X, M, X, M, T, E, S, M],
    2022: [A, M, M, M, T, A, M, M, N, E, A, A, M, O, A, M, M, M, N, T, M, A, M, M, M, M, A, A, A, M, M],
    2023: [M, M, X, A, O, M, E, M, O, M, A, X, M, M, M, X, M, M, M, S, A, E, E, M, O, M, M, X, M, M, M, A, E,
           A, X, A, M, M, A, M, M, M, M, X, X, M],
    2024: [M, O, A, A, X, M, A, M, M, A, M, A, M, T, O, M, M, M, M, M, S, M, S, X, X, M, X, X, M, M, M, M, M, M,
           A, A, A, M, M, O, M, M, X, M, A, M, A, M, M, O, M, S, M, M, M, M, M, M, M, A, M, A, M, M, X, M, M, X,
           M, M, A, A, E, X, A, A, A, M],
    2025: [M, M, M, M, M, X, M, A, O, E, M, A, A, M, A, M, M, A, X, M, M, M, A, A, M, M, M, A, M, M, O, M, O, M,
           X, A, A, M, O, M, O, A, M, M, E, M, M, A, S, M, M, M, O, A, O, A, M, M, M, A, M, E, M, A, M, M, A, M,
           A, M, X, E, O, A, E, O, O, M, A, A, M, A, M, M, M, M, A, M, S, A, M, A, M, E, M, A, M, X, A, X, A, M,
           A, X, M, E, M, A, A, X, A, M, M, X, E, M, T, M, A, O,
           M, T, M, M, M, A, A, M, A, M, M, M, M, M],
}


def apply():
    path = G3 / "screening_v3.csv"
    df = pd.read_csv(path, encoding=EXCEL_ENCODING, dtype={"decision": str, "reason": str, "screener": str})
    for year, reasons in DECISIONS.items():
        for rank, reason in enumerate(reasons):
            assert reason in REASONS, reason
            idx = df.index[(df["year"] == year) & (df["rank_in_year"] == rank)]
            df.loc[idx, ["decision", "reason", "screener"]] = ["y" if reason in Y else "n", reason, "LLM (Claude)"]
    df.to_csv(path, index=False, encoding=EXCEL_ENCODING)


if __name__ == "__main__":
    apply()
