import re

from pinnreview.evidence import PATTERNS, body_of


def hit(name, text):
    return bool(re.search(PATTERNS[name], text, flags=re.I))


def test_body_stops_at_last_references_heading():
    text = "Intro\nReferences to prior work are many.\nResults\n\nReferences\n[1] Raissi github.com/maziarraissi/PINNs"
    body = body_of(text)
    assert "Results" in body and "github.com" not in body


def test_patterns_hit_typical_phrases():
    assert hit("code_link", "Code: https://github.com/user/repo.")
    assert hit("code_statement", "The code is publicly available at our repository.")
    assert hit("on_request", "Data are available upon reasonable request.")
    assert hit("seeds", "Results are averaged over 5 runs with different random seeds.")
    assert hit("dispersion", "error 1.2e-3 ± 0.4e-3")
    assert hit("classical_solver", "reference solution from a finite-difference scheme")
    assert hit("solver_comparison", "PINN is 10x faster than the finite element solver")
    assert hit("cost", "the wall-clock time is 20 minutes")
    assert hit("metric_rel_l2", "relative L2 error of 1e-3")
    assert hit("tuning", "a grid search over learning rates")


def test_patterns_avoid_obvious_false_positives():
    assert not hit("seeds", "the seed of the idea")
    assert not hit("classical_solver", "a fine element of style")
    assert not hit("hardware", "the GPUs-free text mentions graphics")
