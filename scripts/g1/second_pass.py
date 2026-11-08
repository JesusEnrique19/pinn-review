"""Second-pass LLM review of the audit samples (not a human audit; see protocol/deviations.md).

Procedure: every record re-read with its full OpenAlex abstract, or the Semantic Scholar abstract/TLDR
when OpenAlex has none, applying IC1/EC1/IC2 and the fixed rules below. The decision is written to
``llm2_label`` with a reason when it differs from the first pass. ``human_label`` stays empty.

Rules: (1) physics constraints that are algebraic (force balance, empirical laws, circuit gains) do not
count as differential-equation structure; (2) supervised NNs that approximate the solution profile of a
DE are in scope, regressions on scalar outputs are not; (3) book chapters are excluded by IC2;
(4) ML regression on simulation outputs (material properties, coefficients) is out of scope.

    python scripts/g1/second_pass.py audit_v1
    python scripts/g1/second_pass.py audit_relevance_v3
"""
import sys

import pandas as pd

from pinnreview.validation import EXCEL_ENCODING, G1

# index -> (new label, reason); indices not listed keep the first-pass label
CHANGES = {
    "audit_v1": {
        5: ("n", "physics enters as algebraic force/COP balance, no DE structure (rule 1)"),
        9: ("y", "full abstract: physics-informed proxy of reservoir dynamics (PDE surrogate)"),
        12: ("n", "full abstract: ML regression of material properties from phase-field outputs (rule 4)"),
        32: ("n", "full abstract: stability-margin auxiliary tasks, explicitly not trajectory/DE based (rule 1)"),
        50: ("n", "book chapter, excluded by IC2 (rule 3)"),
        53: ("y", "full abstract: wavelet-based physics-informed NN solves the transformed flow equations"),
    },
    "audit_relevance_v3": {
        10: ("y", "full abstract: network learns geodesic flows governed by nonlinear DEs (solution operator)"),
        18: ("n", "physics enters as a precomputed transmission coefficient; no DE in loss or architecture (rule 1)"),
        25: ("n", "LSTM forecasts parameter time series; the PDE model is not part of the network (rule 4)"),
        51: ("n", "transformer regresses implosion features from simulation outputs (rule 4)"),
        53: ("y", "full abstract: Hermia fouling laws (differential equations) embedded in the ML model"),
        55: ("y", "full abstract: CNN combined with PINN residuals for flow-field reconstruction"),
        72: ("y", "full abstract: Newton-Euler dynamics residuals embedded in the PINN loss"),
    },
}


def main(name):
    path = G1 / f"{name}.csv"
    df = pd.read_csv(path, encoding=EXCEL_ENCODING, dtype={"human_label": str, "human_notes": str})
    changes = CHANGES[name]
    df["llm2_label"] = [changes.get(i, (lab, ""))[0] for i, lab in zip(df.index, df["llm_label"])]
    df["llm2_reason"] = [changes.get(i, ("", ""))[1] for i in df.index]
    df.to_csv(path, index=False, encoding=EXCEL_ENCODING)
    flips = df["llm2_label"] != df["llm_label"]
    print(name, "changed:", int(flips.sum()), "of", len(df),
          "| low:", int((flips & (df["confidence"] == "low")).sum()), "/", int((df["confidence"] == "low").sum()),
          "| high:", int((flips & (df["confidence"] == "high")).sum()), "/", int((df["confidence"] == "high").sum()))


if __name__ == "__main__":
    main(sys.argv[1])
