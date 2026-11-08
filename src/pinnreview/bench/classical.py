"""Classical baselines C (protocol 9.2): resolution sweeps giving error-versus-wall-clock curves on the same CPU.

P1 Fourier pseudo-spectral + RK4; P2 Fourier ETDRK4 (Kassam-Trefethen); P3 Fourier pseudo-spectral ETDRK4 with
2/3 dealiasing (u0 = -sin(pi x) is odd and 2-periodic, so u(+-1, t) = 0 is preserved); P4 second-order
5-point finite differences with a sparse direct solver. Errors are relative L2 against the references in
problems.py, on the solver's own grid points.

    python -m pinnreview.bench.classical
"""

import json
import time
from functools import lru_cache

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import spsolve

from pinnreview.bench.problems import NU_BURGERS, allen_cahn_reference, burgers_exact
from pinnreview.paths import RESULTS


def etdrk4_coeffs(L, dt, m=32):
    r = np.exp(1j * np.pi * (np.arange(1, m + 1) - 0.5) / m)
    LR = dt * L[:, None] + r[None, :]
    Q = dt * np.real(np.mean((np.exp(LR / 2) - 1) / LR, axis=1))
    f1 = dt * np.real(np.mean((-4 - LR + np.exp(LR) * (4 - 3 * LR + LR**2)) / LR**3, axis=1))
    f2 = dt * np.real(np.mean((2 + LR + np.exp(LR) * (-2 + LR)) / LR**3, axis=1))
    f3 = dt * np.real(np.mean((-4 - 3 * LR - LR**2 + np.exp(LR) * (4 - LR)) / LR**3, axis=1))
    return np.exp(dt * L), np.exp(dt * L / 2), Q, f1, f2, f3


def p1(n, beta=30.0, nt=101):
    x = 2 * np.pi * np.arange(n) / n
    k = np.fft.fftfreq(n, d=1 / n)
    # RK4 is stable for imaginary eigenvalues |lambda dt| <= 2.8; lambda_max = beta * n / 2. Use half of that,
    # and a step count that is a multiple of the output interval so saved times match the evaluation grid.
    dt_max = 0.5 * 2.8 / (beta * n / 2)
    per_out = int(np.ceil(1.0 / (nt - 1) / dt_max))
    steps = per_out * (nt - 1)
    dt = 1.0 / steps
    rhs = lambda v: -beta * 1j * k * v
    v = np.fft.fft(np.sin(x))
    out = []
    for s in range(steps + 1):
        if s % per_out == 0:
            out.append(np.real(np.fft.ifft(v)))
        if s == steps:
            break
        a = rhs(v); b = rhs(v + dt / 2 * a); c = rhs(v + dt / 2 * b); d = rhs(v + dt * c)
        v = v + dt / 6 * (a + 2 * b + 2 * c + d)
    U = np.array(out).T
    T = np.linspace(0, 1, nt)
    exact = np.sin(x[:, None] - beta * T[None, :])
    return float(np.linalg.norm(U - exact) / np.linalg.norm(exact))


@lru_cache(maxsize=None)
def _allen_ref():
    return allen_cahn_reference(n=512, dt=1e-4, nt_out=101)[2]


@lru_cache(maxsize=None)
def _burgers_ref(n, nt=101):
    x = -1 + 2 * np.arange(n) / n
    sel = slice(None, None, max(1, n // 64))
    return burgers_exact(x[sel][:, None], np.linspace(0, 1, nt)[None, :])


def p2(n, dt):
    ref = _allen_ref()  # cached: computed during the untimed warm-up run
    _, _, u = allen_cahn_reference(n=n, dt=dt, nt_out=101)
    stride = 512 // n
    ref_sub = ref[::stride][: u.shape[0]]
    return float(np.linalg.norm(u - ref_sub) / np.linalg.norm(ref_sub))


def p3(n, dt, nt=101):
    x = -1 + 2 * np.arange(n) / n
    k = np.pi * np.fft.fftfreq(n, d=1 / n)
    L = -NU_BURGERS * k**2
    E, E2, Q, f1, f2, f3 = etdrk4_coeffs(L, dt)
    mask = np.abs(np.fft.fftfreq(n, d=1 / n)) < n / 3   # 2/3 dealiasing
    N = lambda v: -0.5j * k * np.fft.fft(np.real(np.fft.ifft(v * mask)) ** 2) * mask
    v = np.fft.fft(-np.sin(np.pi * x))
    steps = int(round(1.0 / dt))
    every = steps // (nt - 1)
    out = [np.real(np.fft.ifft(v))]
    for s in range(1, steps + 1):
        Nv = N(v); a = E2 * v + Q * Nv; Na = N(a); b = E2 * v + Q * Na; Nb = N(b)
        c = E2 * a + Q * (2 * Nb - Nv); Nc = N(c)
        v = E * v + Nv * f1 + 2 * (Na + Nb) * f2 + Nc * f3
        if s % every == 0:
            out.append(np.real(np.fft.ifft(v)))
    U = np.array(out).T
    sel = slice(None, None, max(1, n // 64))  # exact solution on a subset of grid points (cached, untimed)
    exact = _burgers_ref(n, nt)
    return float(np.linalg.norm(U[sel] - exact) / np.linalg.norm(exact))


def p4(n, k=1.0):
    h = 2.0 / n
    xi = np.linspace(-1, 1, n + 1)[1:-1]
    X, Y = np.meshgrid(xi, xi, indexing="ij")
    m = n - 1
    D = sparse.diags([np.ones(m - 1), -2 * np.ones(m), np.ones(m - 1)], [-1, 0, 1]) / h**2
    I = sparse.identity(m)
    A = sparse.kron(D, I) + sparse.kron(I, D) + k**2 * sparse.identity(m * m)
    exact = np.sin(np.pi * X) * np.sin(4 * np.pi * Y)
    q = (k**2 - np.pi**2 - 16 * np.pi**2) * exact
    u = spsolve(A.tocsc(), q.ravel()).reshape(m, m)
    return float(np.linalg.norm(u - exact) / np.linalg.norm(exact))


SWEEPS = {
    "P1": [((n,), p1) for n in (16, 32, 64, 128, 256)],
    "P2": [((n, dt), p2) for n, dt in ((32, 1e-2), (64, 5e-3), (128, 1e-3), (256, 5e-4))],
    "P3": [((n, dt), p3) for n, dt in ((128, 1e-3), (256, 5e-4), (512, 2e-4), (1024, 1e-4))],
    "P4": [((n,), p4) for n in (16, 32, 64, 128, 256)],
}


def main(repeats=3):
    rows = []
    for problem, sweep in SWEEPS.items():
        for args, fn in sweep:
            fn(*args)  # untimed warm-up: fills the reference caches so timings cover the solver only
            times = []
            for _ in range(repeats):
                t0 = time.perf_counter()
                err = fn(*args)
                times.append(time.perf_counter() - t0)
            rows.append({"problem": problem, "resolution": list(args), "rel_l2": err, "wall_seconds": float(np.median(times))})
            print(rows[-1])
    out = RESULTS / "bench" / "classical.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rows, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
