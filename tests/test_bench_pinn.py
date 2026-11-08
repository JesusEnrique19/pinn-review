import math

import pytest

torch = pytest.importorskip("torch")

from pinnreview.bench.pinn import train  # noqa: E402


@pytest.mark.parametrize("problem,method", [("P1", m) for m in ("M0", "M1", "M2", "M3", "M4", "M5")]
                         + [("P4", m) for m in ("M0", "M1", "M2", "M3", "M5")])
def test_every_method_trains_on_cheap_problems(problem, method):
    # P1's residual does not use the output bias; M1 once failed on it (unused gradient)
    out = train(problem, method, seed=0, steps=12, log_every=6)
    assert math.isfinite(out["rel_l2"]) and len(out["history"]) == 2
    assert all(math.isfinite(v) for v in out["final_weights"].values())
