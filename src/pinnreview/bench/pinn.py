"""PINN training with methods M0-M5 under equal budgets (protocol Sections 9.2-9.3).

M0 vanilla, unit loss weights (Raissi et al. 2019).
M1 learning-rate annealing weights, alpha = 0.9, updated every 10 steps (Wang, Teng and Perdikaris 2021).
M2 NTK-trace weights, updated every 1000 steps from 128 points per term (Wang, Yu and Perdikaris 2022).
M3 residual-based adaptive distribution, k = 1, c = 1, pool 20 000, resampled every 2000 steps (Wu et al. 2023).
M4 causal weights over 32 time bins, epsilon = 1 (Wang, Sankaran and Perdikaris 2024); time-dependent only.
M5 exact initial/boundary conditions by ansatz + periodic Fourier embedding (Sukumar and Srivastava 2022;
   Dong and Ni 2021).
"""

import json
import math
import time

import numpy as np
import torch
from torch import nn

from pinnreview.bench.problems import NU_BURGERS, make

BETA = 30.0
WIDTH, DEPTH = 64, 4
STEPS, N_RES, N_BC, N_IC = 20_000, 4096, 256, 256


class MLP(nn.Module):
    def __init__(self, d_in):
        super().__init__()
        layers, d = [], d_in
        for _ in range(DEPTH):
            layers += [nn.Linear(d, WIDTH), nn.Tanh()]
            d = WIDTH
        layers.append(nn.Linear(d, 1))
        self.net = nn.Sequential(*layers)
        for m in self.net:
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, z):
        return self.net(z)


class Model(nn.Module):
    """u(a, b) for (a, b) = (x, t) or (x, y); method M5 adds the exact-constraint ansatz."""

    def __init__(self, prob, exact):
        super().__init__()
        self.p, self.exact = prob, exact
        self.lo = torch.tensor(prob.lo, dtype=torch.float32)
        self.hi = torch.tensor(prob.hi, dtype=torch.float32)
        periodic_embed = exact and prob.periodic
        self.net = MLP(3 if periodic_embed else 2)

    def features(self, a, b):
        za = 2 * (a - self.lo[0]) / (self.hi[0] - self.lo[0]) - 1
        zb = 2 * (b - self.lo[1]) / (self.hi[1] - self.lo[1]) - 1
        if self.exact and self.p.periodic:
            w = 2 * math.pi / float(self.hi[0] - self.lo[0])
            return torch.cat([torch.cos(w * a), torch.sin(w * a), zb], dim=1)
        return torch.cat([za, zb], dim=1)

    def forward(self, a, b):
        n = self.net(self.features(a, b))
        if not self.exact:
            return n
        name = self.p.name
        if name == "P1":
            return torch.sin(a) + b * n
        if name == "P2":
            return a**2 * torch.cos(math.pi * a) + b * n
        if name == "P3":
            return -torch.sin(math.pi * a) + b * (1 - a**2) * n
        return (1 - a**2) * (1 - b**2) * n  # P4


def grad(u, z):
    return torch.autograd.grad(u, z, torch.ones_like(u), create_graph=True)[0]


def residual(model, a, b):
    a.requires_grad_(True)
    b.requires_grad_(True)
    u = model(a, b)
    ua, ub = grad(u, a), grad(u, b)
    name = model.p.name
    if name == "P1":
        return ub + BETA * ua
    if name == "P2":
        return ub - 1e-4 * grad(ua, a) + 5 * u**3 - 5 * u
    if name == "P3":
        return ub + u * ua - NU_BURGERS * grad(ua, a)
    uaa, ubb = grad(ua, a), grad(ub, b)
    q = (1 - math.pi**2 - 16 * math.pi**2) * torch.sin(math.pi * a) * torch.sin(4 * math.pi * b)
    return uaa + ubb + u - q


def uniform(prob, n, gen):
    lo, hi = torch.tensor(prob.lo), torch.tensor(prob.hi)
    z = lo + (hi - lo) * torch.rand(n, 2, generator=gen)
    return z[:, :1].float(), z[:, 1:].float()


def condition_residuals(model, gen, n_bc=N_BC, n_ic=N_IC):
    """Per-point boundary and initial residual vectors (empty for M5, whose ansatz satisfies them exactly)."""
    p = model.p
    if model.exact:
        return {}
    out = {}
    if p.time_dependent:
        x = p.lo[0] + (p.hi[0] - p.lo[0]) * torch.rand(n_ic, 1, generator=gen)
        t0 = torch.zeros_like(x)
        u0 = {"P1": torch.sin(x), "P2": x**2 * torch.cos(math.pi * x), "P3": -torch.sin(math.pi * x)}[p.name]
        out["ic"] = model(x, t0) - u0
        t = p.hi[1] * torch.rand(n_bc, 1, generator=gen)
        left, right = torch.full_like(t, p.lo[0]), torch.full_like(t, p.hi[0])
        if p.periodic:
            left.requires_grad_(True)
            right.requires_grad_(True)
            ul, ur = model(left, t), model(right, t)
            out["bc"] = torch.cat([ul - ur, grad(ul, left) - grad(ur, right)])
        else:
            out["bc"] = torch.cat([model(left, t), model(right, t)])
    else:
        s = -1 + 2 * torch.rand(n_bc, 1, generator=gen)
        edges = [(torch.full_like(s, -1.0), s), (torch.full_like(s, 1.0), s), (s, torch.full_like(s, -1.0)), (s, torch.full_like(s, 1.0))]
        out["bc"] = torch.cat([model(a, b) for a, b in edges])
    return out


def condition_losses(model, gen):
    return {k: (v**2).mean() for k, v in condition_residuals(model, gen).items()}


def causal_residual_loss(r, t, bins=32, eps=1.0):
    idx = torch.clamp((t.squeeze(1) * bins).long(), max=bins - 1)
    per_bin = torch.zeros(bins).index_add(0, idx, r.squeeze(1) ** 2) / torch.bincount(idx, minlength=bins).clamp(min=1)
    w = torch.exp(-eps * torch.cumsum(per_bin, 0) + eps * per_bin).detach()  # sum over earlier bins only
    return (w * per_bin).mean()


def ntk_trace(model, values):
    """Mean diagonal of the NTK block of a term: mean over points of ||d value_i / d theta||^2."""
    params = list(model.parameters())
    flat = values.reshape(-1)
    total = 0.0
    for v in flat:
        # parameters absent from a term (e.g. the output bias in a pure-derivative residual) contribute zero
        g = torch.autograd.grad(v, params, retain_graph=True, allow_unused=True)
        total += sum(float((gi**2).sum()) for gi in g if gi is not None)
    return total / len(flat)


def evaluate(model):
    A, B, U = model.p.grid
    a = torch.tensor(A.reshape(-1, 1), dtype=torch.float32)
    b = torch.tensor(B.reshape(-1, 1), dtype=torch.float32)
    with torch.no_grad():
        pred = torch.cat([model(a[i:i + 20000], b[i:i + 20000]) for i in range(0, len(a), 20000)]).numpy().ravel()
    u = U.reshape(-1)
    return float(np.linalg.norm(pred - u) / np.linalg.norm(u))


def train(problem, method, seed, steps=STEPS, log_every=500):
    torch.manual_seed(seed)
    gen = torch.Generator().manual_seed(seed)
    prob = make(problem)
    if method == "M4" and not prob.time_dependent:
        raise ValueError("M4 applies to time-dependent problems only")
    model = Model(prob, exact=(method == "M5"))
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    sched = torch.optim.lr_scheduler.StepLR(opt, step_size=2000, gamma=0.9)
    lam = {"res": 1.0, "bc": 1.0, "ic": 1.0}
    rad_points = None
    history = []
    start = time.perf_counter()
    for step in range(1, steps + 1):
        if method == "M3":
            if rad_points is None or step % 2000 == 1:
                pa, pb = uniform(prob, 20_000, gen)
                r = residual(model, pa, pb).detach().abs().squeeze(1)
                prob_w = r / r.mean() + 1.0
                pick = torch.multinomial(prob_w / prob_w.sum(), N_RES, replacement=False, generator=gen)
                rad_points = (pa[pick].detach(), pb[pick].detach())
            a, b = rad_points[0].clone(), rad_points[1].clone()
        else:
            a, b = uniform(prob, N_RES, gen)
        r = residual(model, a, b)
        terms = {"res": causal_residual_loss(r, b) if method == "M4" else (r**2).mean()}
        terms.update(condition_losses(model, gen))

        if method == "M1" and step % 10 == 1 and len(terms) > 1:
            # lambda_hat_k = max|grad L_res| / mean|grad L_k|; lambda_k <- (1 - alpha) lambda_k + alpha lambda_hat_k
            params = list(model.parameters())
            # allow_unused: a residual without an undifferentiated u (P1 advection) does not use the output bias
            g_res = torch.autograd.grad(terms["res"], params, retain_graph=True, allow_unused=True)
            max_res = max(float(g.abs().max()) for g in g_res if g is not None)
            for k in terms:
                if k == "res":
                    continue
                g_k = torch.autograd.grad(terms[k], params, retain_graph=True, allow_unused=True)
                mean_k = float(np.mean([float(g.abs().mean()) for g in g_k if g is not None]))
                lam[k] = 0.1 * lam[k] + 0.9 * max_res / (mean_k + 1e-12)
        if method == "M2" and step % 1000 == 1 and len(terms) > 1:
            # lambda_k = sum_j tr(K_j) / tr(K_k), traces estimated on 128 residual and 64 condition points
            pa, pb = uniform(prob, 128, gen)
            traces = {"res": ntk_trace(model, residual(model, pa, pb))}
            for k, v in condition_residuals(model, gen, n_bc=32, n_ic=64).items():
                traces[k] = ntk_trace(model, v)
            total = sum(traces.values())
            lam = {k: total / max(v, 1e-12) for k, v in traces.items()}

        loss = sum(lam[k] * v for k, v in terms.items())
        opt.zero_grad()
        loss.backward()
        opt.step()
        sched.step()
        if step % log_every == 0 or step == steps:
            history.append({"step": step, "loss": float(loss), **{k: float(v) for k, v in terms.items()},
                            "seconds": time.perf_counter() - start})
    wall = time.perf_counter() - start
    return {"problem": problem, "method": method, "seed": seed, "steps": steps, "rel_l2": evaluate(model),
            "wall_seconds": wall, "final_weights": lam, "history": history,
            "torch_threads": torch.get_num_threads(), "torch_version": torch.__version__}


def main():
    import argparse

    from pinnreview.paths import RESULTS

    ap = argparse.ArgumentParser()
    ap.add_argument("problem", choices=["P1", "P2", "P3", "P4"])
    ap.add_argument("method", choices=["M0", "M1", "M2", "M3", "M4", "M5"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--steps", type=int, default=STEPS)
    ap.add_argument("--tag", default="")
    args = ap.parse_args()
    out = train(args.problem, args.method, args.seed, args.steps)
    path = RESULTS / "bench" / f"{args.problem}_{args.method}_s{args.seed}{args.tag}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(json.dumps({k: out[k] for k in ("problem", "method", "seed", "steps", "rel_l2", "wall_seconds")}))


if __name__ == "__main__":
    main()
