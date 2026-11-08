import numpy as np

from pinnreview.bench.problems import allen_cahn_reference, burgers_exact, helmholtz_source


def test_burgers_initial_condition_and_boundaries():
    x = np.linspace(-1, 1, 11)
    assert np.allclose(burgers_exact(x, 0.0), -np.sin(np.pi * x))
    assert np.allclose(burgers_exact(np.array([-1.0, 1.0]), 0.5), 0.0, atol=1e-8)


def test_burgers_shock_slope_matches_literature():
    # Basdevant et al. (1986): the maximum slope |u_x| = 152.00516 is reached at x = 0, t = 1.6037/pi
    h = 1e-6
    t = 1.6037 / np.pi
    slope = (burgers_exact(h, t) - burgers_exact(-h, t)) / (2 * h)
    assert abs(slope - (-152.00516)) < 1e-3


def test_allen_cahn_reference_converges_in_time_step():
    _, _, u_fine = allen_cahn_reference(n=256, dt=5e-5, nt_out=11)
    _, _, u_coarse = allen_cahn_reference(n=256, dt=2e-4, nt_out=11)
    rel = np.linalg.norm(u_fine - u_coarse) / np.linalg.norm(u_fine)
    assert rel < 1e-3
    assert np.allclose(u_fine[:, 0], u_fine[:, 0])  # finite values


def test_helmholtz_source_satisfies_pde():
    x, y, h = 0.3, -0.2, 1e-4
    u = lambda a, b: np.sin(np.pi * a) * np.sin(4 * np.pi * b)
    lap = (u(x + h, y) + u(x - h, y) + u(x, y + h) + u(x, y - h) - 4 * u(x, y)) / h**2
    assert abs(lap + u(x, y) - helmholtz_source(x, y)) < 1e-3 * abs(helmholtz_source(x, y)) + 1e-3
