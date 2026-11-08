import math

import pytest

from pinnreview.codebook import cohen_kappa, field_kappa, load_codebook, validate_record


@pytest.fixture(scope="module")
def codebook():
    return load_codebook()


def valid_record():
    return {
        "openalex_id": "https://openalex.org/W0000000001",
        "coder": "JA",
        "contribution_type": ["method"],
        "method_family": ["loss_balancing"],
        "failure_mode_targeted": ["gradient_imbalance"],
        "pde_class": ["elliptic"],
        "problem_type": ["forward"],
        "application_domain": "generic_benchmark",
        "benchmark_pdes": "Helmholtz",
        "error_metric": "rel_l2",
        "best_reported_error": 1e-3,
        "quality": {
            "q1_strong_baseline": "no",
            "q2_cost_parity": "partial",
            "q3_seeds_variance": "yes",
            "q4_code_available": "yes",
            "q5_tuning_parity": "no",
            "q6_metric_defined": "yes",
            "q7_generalisation_test": "na",
            "q8_limitations": "partial",
        },
    }


def test_every_field_is_operationally_defined(codebook):
    for name, spec in codebook["fields"].items():
        assert spec.get("definition"), name
        assert spec["type"] in {"string", "text", "multi", "single", "number"}, name
        if spec["type"] in {"multi", "single"}:
            assert spec["values"], name
            assert all(desc for desc in spec["values"].values()), name
    for item, spec in codebook["quality"]["items"].items():
        assert spec.get("definition"), item


def test_valid_record_passes(codebook):
    assert validate_record(valid_record(), codebook) == []


@pytest.mark.parametrize(
    "mutate, expected",
    [
        (lambda r: r.pop("method_family"), "method_family: missing"),
        (lambda r: r.update(method_family=["magic"]), "unknown value 'magic'"),
        (lambda r: r.update(method_family="loss_balancing"), "expected a list"),
        (lambda r: r.update(application_domain="mars"), "application_domain: unknown"),
        (lambda r: r.update(best_reported_error=float("nan")), "finite number"),
        (lambda r: r["quality"].pop("q3_seeds_variance"), "q3_seeds_variance: missing"),
        (lambda r: r["quality"].update(q1_strong_baseline="maybe"), "unknown value 'maybe'"),
        (lambda r: r.update(typo_field=1), "not in codebook"),
    ],
)
def test_invalid_records_are_rejected(codebook, mutate, expected):
    record = valid_record()
    mutate(record)
    errors = validate_record(record, codebook)
    assert any(expected in e for e in errors), errors


def test_cohen_kappa_known_values():
    assert cohen_kappa(list("aabb"), list("aabb")) == 1.0
    # Classic 2x2 example: p_o = 0.7, p_e = 0.5 -> kappa = 0.4
    a = ["y"] * 5 + ["n"] * 5
    b = ["y"] * 4 + ["n"] + ["y"] * 2 + ["n"] * 3
    assert math.isclose(cohen_kappa(a, b), 0.4)


def test_field_kappa_multi_valued(codebook):
    r1, r2 = valid_record(), valid_record()
    r3, r4 = valid_record(), valid_record()
    r3["method_family"] = ["adaptive_sampling"]
    r4["method_family"] = ["adaptive_sampling"]
    assert field_kappa([r1, r3], [r2, r4], "method_family", codebook) == 1.0
