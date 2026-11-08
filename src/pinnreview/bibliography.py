"""IEEE-style reference list built from OpenAlex metadata (protocol Section 10: no hand-typed references).

Cite keys resolve in this order:
1. seed-set keys -> the OpenAlex work matched in Phase 0 (data/processed/seed_set_resolved.csv);
2. paper/references.yaml -> ``openalex`` id, ``doi`` or ``arxiv`` id, checked against ``title`` (fuzzy >= 90);
   entries without any identifier (old books, reports) give the full ``ieee`` string and are flagged ``manual``.
Every resolved title is compared with the expected one, so a wrong identifier fails the build.
Venue, volume, issue, pages and year come from the publisher's Crossref record when the work has a (non-arXiv)
DOI: OpenAlex gives the online-first year and drops article numbers. Responses are cached in data/cache/crossref.
Where OpenAlex lacks the published version (conference proceedings, a journal DOI attached to another record) or
has a wrong name, an entry may add ``published`` (venue fields) and/or ``authors``; both need a ``source`` saying
where they were verified. Seed keys may appear in references.yaml with these override fields only.
"""

import hashlib
import json
import re

import pandas as pd
import requests
import yaml
from rapidfuzz import fuzz

from pinnreview import openalex
from pinnreview.paths import CACHE, PROCESSED, ROOT

EXTRA_FILE = ROOT / "paper" / "references.yaml"
PARTICLES = {"de", "van", "von", "der", "den", "da", "di", "du", "le", "la", "del", "dos"}
MONTHLESS_TYPES = {"preprint", "posted-content"}
# OpenAlex venue names that need the conventional form
VENUES = {"neural information processing systems": "Advances in Neural Information Processing Systems"}


class ReferenceError(RuntimeError):
    pass


def initials(given):
    parts = []
    for word in given.split():
        sub = [w for w in word.split("-") if w]
        parts.append("-".join(f"{w[0]}." for w in sub))
    return " ".join(parts)


def ieee_name(display_name):
    """'Maziar Raissi' -> 'M. Raissi'; 'Tim De Ryck' -> 'T. De Ryck'; 'G.E. Karniadakis' -> 'G. E. Karniadakis'."""
    tokens = display_name.replace(".", ". ").split()
    if "," in display_name:  # 'Surname, Given'
        surname, given = (x.strip() for x in display_name.split(",", 1))
        return f"{initials(given.replace('.', '. '))} {surname}".strip()
    if len(tokens) == 1:
        return tokens[0]
    i = len(tokens) - 1
    while i > 1 and tokens[i - 1].lower() in PARTICLES:
        i -= 1
    return f"{initials(' '.join(tokens[:i]))} {' '.join(tokens[i:])}".strip()


def author_list(names):
    names = [ieee_name(n) for n in names]
    if len(names) > 6:
        return f"{names[0]} et al."
    if len(names) <= 2:
        return " and ".join(names)
    return ", ".join(names[:-1]) + ", and " + names[-1]


def _arxiv_id(work):
    for loc in work.get("locations") or []:
        url = (loc.get("landing_page_url") or "") + " " + (loc.get("pdf_url") or "")
        if m := re.search(r"arxiv\.org/(?:abs|pdf)/([0-9]{4}\.[0-9]{4,5}|[a-z\-]+/[0-9]{7})", url):
            return m.group(1)
    doi = (work.get("doi") or "").lower()
    if m := re.search(r"10\.48550/arxiv\.(.+)$", doi):
        return m.group(1)
    return None


def _strip_tex(title):
    """Remove TeX font commands such as '$mathrm{FAIR}$' (with backslash) found in some OpenAlex titles."""
    for cmd in ("mathrm", "mathbf", "mathit"):
        title = title.replace("$" + chr(92) + cmd + "{", "")
    return title.replace("}$", "")


OVERRIDE_FIELDS = {"published", "authors", "source"}
CROSSREF_CACHE = CACHE / "crossref"


def crossref(doi):
    """Crossref work record (cached), or None if the DOI is unknown to Crossref."""
    path = CROSSREF_CACHE / f"{hashlib.sha256(doi.lower().encode()).hexdigest()}.json"
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    r = requests.get(f"https://api.crossref.org/works/{doi}", timeout=30,
                     headers={"User-Agent": "pinnreview reference builder (reproducible review)"})
    msg = r.json()["message"] if r.status_code == 200 else None
    CROSSREF_CACHE.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(msg), encoding="utf-8")
    return msg


def published_from_crossref(msg):
    """``published`` fields from a Crossref record: print year, article number, proceedings volume."""
    year = next(((msg.get(k) or {}).get("date-parts", [[None]])[0][0] for k in ("published-print", "issued")
                 if (msg.get(k) or {}).get("date-parts", [[None]])[0][0]), None)
    venue = (msg.get("container-title") or [""])[0]
    pub = {"kind": "conference" if msg.get("type") == "proceedings-article" else "journal", "venue": venue,
           "volume": msg.get("volume"), "issue": msg.get("issue"), "year": year, "doi": msg.get("DOI")}
    if m := re.fullmatch(r"(Advances in Neural Information Processing Systems) (\d+)", venue):
        pub["venue"], pub["volume"] = m.group(1), m.group(2)
    page = msg.get("page") or ""
    if "-" in page:
        pub["first_page"], pub["last_page"] = page.split("-", 1)
    elif msg.get("article-number") or page:  # a single "page" is an article number (MDPI, Springer, Elsevier)
        pub["article"] = msg.get("article-number") or page
    return pub


def _pages(fp, lp):
    if lp and lp != fp:
        return f"pp. {fp}–{lp}"
    # article numbers are long or alphanumeric; a short number is a single page
    return f"p. {fp}" if str(fp).isdigit() and len(str(fp)) <= 4 else f"Art. no. {fp}"


def _locator(pub):
    if pub.get("first_page"):
        return [_pages(pub["first_page"], pub.get("last_page"))]
    return [f"Art. no. {pub['article']}"] if pub.get("article") else []


def format_published(authors, title, pub):
    """Reference from a verified ``published`` override: kind 'conference' or 'journal'."""
    if pub["kind"] == "conference":
        parts = [f"in *{pub['venue']}*"] + ([f"vol. {pub['volume']}"] if pub.get("volume") else [])
        parts += [str(pub["year"])] + _locator(pub)
    else:
        parts = [f"*{pub['venue']}*"]
        parts += [f"{lab} {pub[k]}" for k, lab in (("volume", "vol."), ("issue", "no.")) if pub.get(k)]
        parts += _locator(pub) + [str(pub["year"])]
    ref = f"{authors}, “{title},” {', '.join(parts)}"
    return ref + (f", doi: {pub['doi']}." if pub.get("doi") else ".")


def format_work(work, published=None, authors=None):
    """One IEEE reference string (without the [n] label) from an OpenAlex work, with optional verified overrides."""
    # Names as printed on the paper; OpenAlex's disambiguated display names are sometimes garbled
    names = authors or [a.get("raw_author_name") or a["author"]["display_name"] for a in work.get("authorships") or []]
    authors = author_list(names)
    title = (work.get("title") or work.get("display_name") or "").rstrip(".")
    title = _strip_tex(title)
    if published and "kind" not in published:  # published version identified by its DOI only
        msg = crossref(published["doi"])
        if msg is None or fuzz.ratio(msg["title"][0].lower(), title.lower()) < 90:
            raise ReferenceError(f"published DOI {published['doi']} does not match '{title[:60]}'")
        published = published_from_crossref(msg)
    if published:  # the published title may differ slightly from the preprint's
        return format_published(authors, published.get("title", title), published)
    doi = (work.get("doi") or "").replace("https://doi.org/", "")
    if doi and not doi.lower().startswith("10.48550") and (msg := crossref(doi)):
        return format_published(authors, title, published_from_crossref(msg))
    year = work.get("publication_year")
    source = ((work.get("primary_location") or {}).get("source") or {})
    venue = source.get("display_name") or ""
    venue = VENUES.get(venue.lower(), venue)
    biblio = work.get("biblio") or {}
    doi = (work.get("doi") or "").replace("https://doi.org/", "")
    arxiv = _arxiv_id(work)
    if not venue or "arxiv" in venue.lower() or work.get("type") in MONTHLESS_TYPES:
        tail = f"arXiv preprint arXiv:{arxiv}, {year}" if arxiv else f"preprint, {year}"
        return f"{authors}, “{title},” {tail}."
    parts = [f"*{venue}*"]
    if biblio.get("volume"):
        parts.append(f"vol. {biblio['volume']}")
    if biblio.get("issue"):
        parts.append(f"no. {biblio['issue']}")
    if biblio.get("first_page"):
        parts.append(_pages(biblio["first_page"], biblio.get("last_page")))
    parts.append(str(year))
    ref = f"{authors}, “{title},” {', '.join(parts)}"
    if doi and not doi.lower().startswith("10.48550"):
        ref += f", doi: {doi}"
    return ref + "."


def _check(key, work, expected):
    got = (work.get("title") or "").lower()
    expected = [expected] if isinstance(expected, str) else (expected or [])
    if expected and max(fuzz.ratio(e.lower(), got) for e in expected) < 90:
        raise ReferenceError(f"{key}: identifier resolves to '{got[:80]}', expected '{expected[0][:80]}'")


def _work(entry):
    if oa := entry.get("openalex"):
        return openalex.get(f"works/{oa.rsplit('/', 1)[-1]}")
    if doi := entry.get("doi"):
        return openalex.get(f"works/https://doi.org/{doi}")
    if arxiv := entry.get("arxiv"):
        return openalex.get(f"works/https://doi.org/10.48550/arXiv.{arxiv}")
    return None


class Bibliography:
    def __init__(self):
        seeds = pd.read_csv(PROCESSED / "seed_set_resolved.csv")
        meta = pd.read_csv(ROOT / "protocol" / "seed_set.csv").set_index("key")
        self.entries = {r.key: {"openalex": r.openalex_id,
                                "title": [t for t in (meta.at[r.key, "title"], meta.at[r.key, "alt_title"]) if isinstance(t, str)]}
                        for r in seeds.itertuples()}
        extra = yaml.safe_load(EXTRA_FILE.read_text(encoding="utf-8")) or {}
        for key, entry in extra.items():
            if key in self.entries:  # seed key: only verified overrides may be added
                if set(entry) - OVERRIDE_FIELDS:
                    raise ReferenceError(f"{key}: seed keys accept only {sorted(OVERRIDE_FIELDS)}")
                self.entries[key].update(entry)
            else:
                self.entries[key] = entry
            if ({"published", "authors"} & set(entry)) and not entry.get("source"):
                raise ReferenceError(f"{key}: override without a verification source")
        self.order = []

    def cite(self, key):
        if key not in self.entries:
            raise ReferenceError(f"unknown cite key '{key}'")
        if key not in self.order:
            self.order.append(key)
        return self.order.index(key) + 1

    def render(self, key):
        entry = self.entries[key]
        if "ieee" in entry:
            return entry["ieee"], True
        work = _work(entry)
        _check(key, work, entry.get("title"))
        return format_work(work, entry.get("published"), entry.get("authors")), False

    def rendered(self):
        """[(key, text, manual)] in citation order."""
        return [(k, *self.render(k)) for k in self.order]
