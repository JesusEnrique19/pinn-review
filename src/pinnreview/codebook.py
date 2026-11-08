"""Codebook loading, record validation and inter-coder reliability (protocol Sections 6–7)."""

import math

import yaml

from pinnreview.paths import PROTOCOL

CODEBOOK_FILE = PROTOCOL / "codebook.yaml"


def load_codebook(path=CODEBOOK_FILE):
    with open(path, encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def validate_record(record, codebook):
    """Return a list of error messages; an empty list means the record is valid."""
    errors = []
    for name, spec in codebook["fields"].items():
        value = record.get(name)
        if value is None or value == "" or value == []:
            if spec.get("required"):
                errors.append(f"{name}: missing required field")
            continue
        kind = spec["type"]
        allowed = set(spec.get("values", {}))
        if kind == "multi":
            if not isinstance(value, list):
                errors.append(f"{name}: expected a list")
                continue
            errors += [f"{name}: unknown value {v!r}" for v in value if v not in allowed]
            if len(set(value)) != len(value):
                errors.append(f"{name}: duplicate values")
        elif kind == "single":
            if value not in allowed:
                errors.append(f"{name}: unknown value {value!r}")
        elif kind == "number":
            if not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(value):
                errors.append(f"{name}: expected a finite number")
        elif kind in ("string", "text"):
            if not isinstance(value, str):
                errors.append(f"{name}: expected a string")

    quality = codebook["quality"]
    allowed = set(quality["values"])
    for item in quality["items"]:
        value = (record.get("quality") or {}).get(item)
        if value is None:
            errors.append(f"quality.{item}: missing")
        elif value not in allowed:
            errors.append(f"quality.{item}: unknown value {value!r}")

    known = set(codebook["fields"]) | {"quality", "notes"}
    errors += [f"{name}: field not in codebook" for name in record if name not in known]
    return errors


def cohen_kappa(a, b):
    """Cohen's kappa for two equal-length sequences of categorical labels."""
    if len(a) != len(b) or not a:
        raise ValueError("sequences must be non-empty and of equal length")
    n = len(a)
    labels = set(a) | set(b)
    observed = sum(x == y for x, y in zip(a, b)) / n
    expected = sum((a.count(c) / n) * (b.count(c) / n) for c in labels)
    if expected == 1:
        return 1.0
    return (observed - expected) / (1 - expected)


def field_kappa(records_a, records_b, field, codebook):
    """Kappa for one codebook field between two coders over the same records.

    Multi-valued fields are scored per value as binary present/absent decisions and averaged
    over the values that either coder used.
    """
    spec = codebook["fields"][field]
    if spec["type"] != "multi":
        return cohen_kappa([r.get(field) for r in records_a], [r.get(field) for r in records_b])
    used = {v for r in records_a + records_b for v in (r.get(field) or [])}
    kappas = [
        cohen_kappa([v in (r.get(field) or []) for r in records_a], [v in (r.get(field) or []) for r in records_b])
        for v in sorted(used)
    ]
    return sum(kappas) / len(kappas) if kappas else float("nan")
