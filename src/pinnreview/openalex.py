"""Minimal OpenAlex client with on-disk caching.

Every response is cached under ``data/cache/openalex`` keyed by the request, so re-running the
pipeline against the same query is deterministic and does not hit the API again.
"""

import hashlib
import json
import os
import time

import requests

from pinnreview.paths import CACHE

BASE_URL = "https://api.openalex.org"
CACHE_DIR = CACHE / "openalex"
MIN_INTERVAL_S = 0.15  # stay well under the 10 requests/second limit

_last_request = 0.0


class OpenAlexError(RuntimeError):
    pass


def _auth_params():
    params = {}
    if key := os.environ.get("OPENALEX_API_KEY"):
        params["api_key"] = key
    if mailto := os.environ.get("OPENALEX_MAILTO"):
        params["mailto"] = mailto
    return params


def get(endpoint, params=None, use_cache=True, retries=4):
    """GET ``BASE_URL/endpoint`` and return the decoded JSON body."""
    global _last_request
    params = dict(params or {})
    cache_key = hashlib.sha256(json.dumps([endpoint, params], sort_keys=True).encode()).hexdigest()
    cache_file = CACHE_DIR / f"{cache_key}.json"
    if use_cache and cache_file.exists():
        return json.loads(cache_file.read_text(encoding="utf-8"))

    for attempt in range(retries):
        wait = MIN_INTERVAL_S - (time.monotonic() - _last_request)
        if wait > 0:
            time.sleep(wait)
        _last_request = time.monotonic()
        try:
            resp = requests.get(f"{BASE_URL}/{endpoint}", params={**params, **_auth_params()}, timeout=60)
        except requests.RequestException as exc:
            raise OpenAlexError(f"{type(exc).__name__} on {endpoint}") from None
        if resp.status_code == 429 or resp.status_code >= 500:
            time.sleep(2**attempt)
            continue
        if not resp.ok:
            # Never use raise_for_status(): its message contains the full URL, including the API key
            raise OpenAlexError(f"HTTP {resp.status_code} on {endpoint}: {resp.text[:300]}")
        body = resp.json()
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        cache_file.write_text(json.dumps(body), encoding="utf-8")
        return body
    raise OpenAlexError(f"HTTP {resp.status_code} on {endpoint} after {retries} attempts: {resp.text[:300]}")


def search_works(text, per_page=10, select=None):
    params = {"search": text, "per_page": per_page}
    if select:
        params["select"] = ",".join(select)
    return get("works", params)["results"]
