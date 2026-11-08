"""LLM labels for the random sample of the v3-only stratum (records v3 adds over v1), pending human audit."""
import pandas as pd

from pinnreview.validation import EXCEL_ENCODING, G1

L = {
    0: ("n", "h", ""), 1: ("n", "h", ""), 2: ("n", "l", "numerical electromigration solver; NN role unclear"),
    3: ("n", "l", "AI hydrology review"), 4: ("n", "l", "EM-informed generative model for RF sensing"), 5: ("n", "h", ""),
    6: ("n", "l", "battery state estimation"), 7: ("n", "l", "physics-informed GRU for road noise"),
    8: ("y", "l", "physics-informed autoencoder for thin-shell analysis"), 9: ("y", "l", "DNN surrogate of shock/boundary-layer flow"),
    10: ("y", "h", ""), 11: ("n", "h", ""), 12: ("n", "h", ""), 13: ("n", "l", "classification of control strategies"),
    14: ("y", "h", "spiking NN solving boundary value problems"), 15: ("n", "h", ""), 16: ("n", "h", ""), 17: ("n", "h", ""),
    18: ("n", "h", ""), 19: ("n", "h", ""), 20: ("n", "l", "physics-informed transformer for trajectories"), 21: ("n", "h", ""),
    22: ("y", "h", ""), 23: ("n", "h", ""), 24: ("y", "h", "integral equations, EM scattering"), 25: ("y", "h", ""),
    26: ("y", "l", "glacier model with physics-informed NN emulator"), 27: ("n", "l", ""), 28: ("n", "l", ""), 29: ("n", "l", ""),
    30: ("n", "l", "battery state of charge"), 31: ("y", "l", "physics-informed RNN for nonlinear ODE systems"),
    32: ("n", "l", "perfusion kinetic model likely algebraic"), 33: ("n", "l", ""), 34: ("n", "l", "power flow (algebraic)"),
    35: ("n", "h", ""), 36: ("n", "l", "physical image-formation model"), 37: ("n", "l", ""), 38: ("n", "h", ""),
    39: ("n", "l", ""), 40: ("n", "l", ""), 41: ("n", "l", "fatigue life regression"), 42: ("n", "h", ""), 43: ("n", "l", ""),
    44: ("y", "l", "physics-informed LSTM for bubble dynamics ODE"), 45: ("y", "h", ""), 46: ("y", "l", "physics-guided DL for CO2 migration fields"),
    47: ("n", "l", "compact circuit models"), 48: ("n", "h", ""), 49: ("n", "h", ""), 50: ("n", "h", ""), 51: ("y", "h", ""),
    52: ("y", "l", "DL + physics-based inversion of self-potential data"), 53: ("n", "h", ""), 54: ("n", "l", "battery state of charge"),
    55: ("n", "l", "DL regression of scalar permeability"), 56: ("y", "l", "recurrent nets identifying heat-exchanger dynamics"),
    57: ("n", "h", ""), 58: ("n", "l", "physics-informed diffusion model for radio maps"), 59: ("n", "h", ""),
}

path = G1 / "precision_v3_new.csv"
df = pd.read_csv(path, encoding=EXCEL_ENCODING, dtype={"in_scope": str, "labeller": str, "notes": str})
assert sorted(L) == list(df.index)
df["in_scope"] = [L[i][0] for i in df.index]
df["confidence"] = ["high" if L[i][1] == "h" else "low" for i in df.index]
df["labeller"] = "LLM (Claude), pending human audit"
df["notes"] = [L[i][2] for i in df.index]
df.to_csv(path, index=False, encoding=EXCEL_ENCODING)
print(df["in_scope"].value_counts().to_dict())
