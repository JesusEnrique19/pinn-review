"""Benchmark problems P1-P4 (protocol Section 9.1) with reference solutions on evaluation grids.

P1 convection u_t + beta u_x = 0, x in [0, 2pi], t in [0, 1], u0 = sin x, periodic; exact sin(x - beta t).
P2 Allen-Cahn u_t - 1e-4 u_xx + 5u^3 - 5u = 0, x in [-1, 1], t in [0, 1], u0 = x^2 cos(pi x), periodic;
   reference by Fourier pseudo-spectral ETDRK4 (Kassam and Trefethen 2005).
P3 Burgers u_t + u u_x - (0.01/pi) u_xx = 0, x in [-1, 1], t in [0, 1], u0 = -sin(pi x), u(+-1, t) = 0;
   exact Cole-Hopf solution evaluated by a dense trapezoidal rule (see burgers_exact; Gauss-Hermite was inaccurate).
P4 Helmholtz u_xx + u_yy + k^2 u = q, (x, y) in [-1, 1]^2, k = 1, exact u = sin(pi x) sin(4 pi y), Dirichlet 0.
"""

from dataclasses import dataclass

import numpy as np

NU_BURGERS = 0.01 / np.pi


@dataclass
class Problem:
    name: str
    lo: tuple          # lower corner of (x, t) or (x, y)
    hi: tuple
    time_dependent: bool
    periodic: bool
    grid: tuple        # evaluation grid: (X, T or Y, U) arrays of equal shape


def u0_allen(x):
    return x**2 * np.cos(np.pi * x)


def allen_cahn_reference(n=512, dt=1e-4, nt_out=101):
    """ETDRK4 for u_t = 1e-4 u_xx + 5u - 5u^3 on a periodic grid of [-1, 1)."""
    x = -1 + 2 * np.arange(n) / n
    k = np.pi * np.fft.fftfreq(n, d=1 / n)          # wavenumbers for period 2
    L = -1e-4 * k**2 + 5.0
    E, E2 = np.exp(dt * L), np.exp(dt * L / 2)
    m = 32
    r = np.exp(1j * np.pi * (np.arange(1, m + 1) - 0.5) / m)
    LR = dt * L[:, None] + r[None, :]
    Q = dt * np.real(np.mean((np.exp(LR / 2) - 1) / LR, axis=1))
    f1 = dt * np.real(np.mean((-4 - LR + np.exp(LR) * (4 - 3 * LR + LR**2)) / LR**3, axis=1))
    f2 = dt * np.real(np.mean((2 + LR + np.exp(LR) * (-2 + LR)) / LR**3, axis=1))
    f3 = dt * np.real(np.mean((-4 - 3 * LR - LR**2 + np.exp(LR) * (4 - LR)) / LR**3, axis=1))
    N = lambda v: np.fft.fft(-5.0 * np.real(np.fft.ifft(v)) ** 3)
    v = np.fft.fft(u0_allen(x))
    steps = int(round(1.0 / dt))
    every = steps // (nt_out - 1)
    out = [np.real(np.fft.ifft(v))]
    for s in range(1, steps + 1):
        Nv = N(v)
        a = E2 * v + Q * Nv
        Na = N(a)
        b = E2 * v + Q * Na
        Nb = N(b)
        c = E2 * a + Q * (2 * Nb - Nv)
        Nc = N(c)
        v = E * v + Nv * f1 + 2 * (Na + Nb) * f2 + Nc * f3
        if s % every == 0:
            out.append(np.real(np.fft.ifft(v)))
    t = np.linspace(0, 1, nt_out)
    # append the periodic endpoint x = 1 for evaluation
    U = np.array(out).T
    U = np.vstack([U, U[:1]])
    xs = np.append(x, 1.0)
    X, T = np.meshgrid(xs, t, indexing="ij")
    return X, T, U


def burgers_exact(x, t, n_eta=8001, width=12.0):
    """Cole-Hopf solution of viscous Burgers with u0 = -sin(pi x) (Basdevant et al. 1986).

    u = -int sin(pi(x-eta)) f(x-eta) G(eta) d eta / int f(x-eta) G(eta) d eta, with
    f(y) = exp(-cos(pi y) / (2 pi nu)) and G the heat kernel of width sqrt(4 nu t). The integrand varies on
    a scale ~ 1/(pi^2 / (2 pi nu)) much finer than the kernel, so Gauss-Hermite quadrature is inaccurate;
    a dense trapezoidal rule over +-width kernel widths is used instead.
    """
    x, t = np.broadcast_arrays(np.asarray(x, float), np.asarray(t, float))
    out = np.empty(x.shape)
    flat_x, flat_t, flat_o = x.ravel(), t.ravel(), out.ravel()
    for i, (xi, ti) in enumerate(zip(flat_x, flat_t)):
        if ti == 0:
            flat_o[i] = -np.sin(np.pi * xi)
            continue
        s = np.sqrt(4 * NU_BURGERS * ti)
        eta = np.linspace(-width * s, width * s, n_eta)
        expo = -eta**2 / s**2 - np.cos(np.pi * (xi - eta)) / (2 * np.pi * NU_BURGERS)
        f = np.exp(expo - expo.max())
        flat_o[i] = -np.trapezoid(np.sin(np.pi * (xi - eta)) * f, eta) / np.trapezoid(f, eta)
    return out


def make(name):
    if name == "P1":
        x, t = np.linspace(0, 2 * np.pi, 256), np.linspace(0, 1, 101)
        X, T = np.meshgrid(x, t, indexing="ij")
        return Problem("P1", (0.0, 0.0), (2 * np.pi, 1.0), True, True, (X, T, np.sin(X - 30.0 * T)))
    if name == "P2":
        X, T, U = allen_cahn_reference()
        return Problem("P2", (-1.0, 0.0), (1.0, 1.0), True, True, (X, T, U))
    if name == "P3":
        x, t = np.linspace(-1, 1, 256), np.linspace(0, 1, 101)
        X, T = np.meshgrid(x, t, indexing="ij")
        return Problem("P3", (-1.0, 0.0), (1.0, 1.0), True, False, (X, T, burgers_exact(X, T)))
    if name == "P4":
        x = np.linspace(-1, 1, 201)
        X, Y = np.meshgrid(x, x, indexing="ij")
        return Problem("P4", (-1.0, -1.0), (1.0, 1.0), False, False, (X, Y, np.sin(np.pi * X) * np.sin(4 * np.pi * Y)))
    raise ValueError(name)


def helmholtz_source(x, y, k=1.0):
    u = np.sin(np.pi * x) * np.sin(4 * np.pi * y)
    return (k**2 - np.pi**2 - 16 * np.pi**2) * u
