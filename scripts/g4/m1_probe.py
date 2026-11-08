"""Diagnostic of the M1 collapse on P4 (relative error 1.0 in every seed of the grid), run after the grid results.

Trains P4 with learning-rate annealing for a few steps and prints the boundary weight and both losses, for two
update conventions: 'paper' (lambda <- 0.1 lambda + 0.9 lambda_hat over all parameters, as in pinn.py) and
'prev' (lambda <- 0.9 lambda + 0.1 lambda_hat over weight matrices only). Not part of the pre-registered analysis;
it only explains a result. Output: results/bench/diagnostics/m1_probe_<variant>.txt

    python scripts/g4/m1_probe.py paper 1500
"""
import sys

import numpy as np
import torch

from pinnreview.bench import pinn
from pinnreview.bench.problems import make
from pinnreview.paths import RESULTS


def main():
    variant, steps = sys.argv[1], int(sys.argv[2])
    torch.set_num_threads(1)
    torch.manual_seed(0)
    gen = torch.Generator().manual_seed(0)
    prob = make("P4")
    model = pinn.Model(prob, exact=False)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    lam, lines = 1.0, []
    for step in range(1, steps + 1):
        a, b = pinn.uniform(prob, pinn.N_RES, gen)
        lres = (pinn.residual(model, a, b) ** 2).mean()
        lbc = pinn.condition_losses(model, gen)["bc"]
        if step % 10 == 1:
            ps = [p for name, p in model.named_parameters() if variant == "paper" or name.endswith("weight")]
            gr = torch.autograd.grad(lres, ps, retain_graph=True, allow_unused=True)
            gb = torch.autograd.grad(lbc, ps, retain_graph=True, allow_unused=True)
            hat = max(float(g.abs().max()) for g in gr if g is not None) / (
                np.mean([float(g.abs().mean()) for g in gb if g is not None]) + 1e-12)
            lam = 0.1 * lam + 0.9 * hat if variant == "paper" else 0.9 * lam + 0.1 * hat
        loss = lres + lam * lbc
        opt.zero_grad()
        loss.backward()
        opt.step()
        if step in (1, 11, 101) or step % 250 == 0:
            lines.append(f"{variant} step={step} lambda_bc={lam:.3g} loss_res={float(lres):.3g} loss_bc={float(lbc):.3g}")
            print(lines[-1], flush=True)
    out = RESULTS / "bench" / "diagnostics" / f"m1_probe_{variant}.txt"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
