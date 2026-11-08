"""Minimal Semantic Scholar Graph API client with on-disk caching.

Used to fill abstracts missing from OpenAlex (G1 second-pass review) and, in Phase 2, reference lists
for papers whose OpenAlex record has none (protocol 4.1.1). An optional key is read from
``S2_API_KEY``; without it the shared public rate limit applies, so requests are spaced and retried.
"""

import hashlib
import json
import os
import time

import requests

from pinnreview.paths import CACHE

BASE_URL = "https://api.semanticscholar.org/graph/v1"
CACHE_DIR = CACHE / "s2"
MIN_INTERVAL_S = 1.1

_last_request = 0.0


class S2Error(RuntimeError):
    pass


def get(path, params=None, retries=6):
    """GET ``BASE_URL/path``; returns the JSON body, or None for 404 (unknown paper)."""
    global _last_request
    params = dict(params or {})
    key = hashlib.sha256(json.dumps([path, params], sort_keys=True).encode()).hexdigest()
    cache_file = CACHE_DIR / f"{key}.json"
    if cache_file.exists():
        return json.loads(cache_file.read_text(encoding="utf-8"))
    headers = {"x-api-key": os.environ["S2_API_KEY"]} if os.environ.get("S2_API_KEY") else {}
    for attempt in range(retries):
        wait = MIN_INTERVAL_S - (time.monotonic() - _last_request)
        if wait > 0:
            time.sleep(wait)
        _last_request = time.monotonic()
        try:
            resp = requests.get(f"{BASE_URL}/{path}", params=params, headers=headers, timeout=60)
        except requests.RequestException as exc:
            raise S2Error(f"{type(exc).__name__} on {path}") from None
        if resp.status_code == 429 or resp.status_code >= 500:
            time.sleep(min(60, 2 ** (attempt + 1)))
            continue
        if resp.status_code == 404:
            body = None
        elif resp.ok:
            body = resp.json()
        else:
            raise S2Error(f"HTTP {resp.status_code} on {path}: {resp.text[:200]}")
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        cache_file.write_text(json.dumps(body), encoding="utf-8")
        return body
    raise S2Error(f"HTTP {resp.status_code} on {path} after {retries} attempts")


def batch(ids, fields, retries=6):
    """POST /paper/batch for up to 500 ids ("DOI:...", "ARXIV:..."); cached per request; None for unknown ids."""
    global _last_request
    key = hashlib.sha256(json.dumps(["batch", ids, fields], sort_keys=True).encode()).hexdigest()
    cache_file = CACHE_DIR / f"{key}.json"
    if cache_file.exists():
        return json.loads(cache_file.read_text(encoding="utf-8"))
    headers = {"x-api-key": os.environ["S2_API_KEY"]} if os.environ.get("S2_API_KEY") else {}
    for attempt in range(retries):
        wait = MIN_INTERVAL_S - (time.monotonic() - _last_request)
        if wait > 0:
            time.sleep(wait)
        _last_request = time.monotonic()
        try:
            resp = requests.post(f"{BASE_URL}/paper/batch", params={"fields": fields}, json={"ids": ids},
                                 headers=headers, timeout=120)
        except requests.RequestException as exc:
            raise S2Error(f"{type(exc).__name__} on paper/batch") from None
        if resp.status_code == 429 or resp.status_code >= 500:
            time.sleep(min(60, 2 ** (attempt + 2)))
            continue
        if not resp.ok:
            raise S2Error(f"HTTP {resp.status_code} on paper/batch: {resp.text[:200]}")
        body = resp.json()
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        cache_file.write_text(json.dumps(body), encoding="utf-8")
        return body
    raise S2Error(f"HTTP {resp.status_code} on paper/batch after {retries} attempts")


def paper_by_doi(doi, fields):
    doi = doi.lower().removeprefix("https://doi.org/")
    return get(f"paper/DOI:{doi}", {"fields": fields})


def paper_by_title(title, fields):
    """Best title match; None if Semantic Scholar finds nothing."""
    body = get("paper/search/match", {"query": title, "fields": fields})
    data = (body or {}).get("data") or []
    return data[0] if data else None
