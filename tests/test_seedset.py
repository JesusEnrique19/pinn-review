from pinnreview.seedset import REQUIRED_COLUMNS, ROLES, load_seed_set, match_entry


def test_seed_set_integrity():
    seed = load_seed_set()
    assert list(seed.columns[: len(REQUIRED_COLUMNS)]) == REQUIRED_COLUMNS
    assert seed["key"].is_unique
    assert set(seed["role"]) <= ROLES
    assert seed[REQUIRED_COLUMNS].notna().all().all()
    # In-scope seeds must fall inside the protocol time window (Section 4.2)
    seeds = seed[seed["role"] == "seed"]
    assert seeds["year"].between(2017, 2025).all()
    assert len(seeds) >= 50


def _work(wid, title, author, cited=0, year=2021):
    return {
        "id": wid,
        "display_name": title,
        "authorships": [{"author": {"display_name": author}}],
        "cited_by_count": cited,
        "publication_year": year,
    }


ENTRY = {"title": "Physics-informed machine learning for fluids", "first_author": "Smith", "year": 2021}


def test_subset_title_is_not_a_match():
    candidates = [_work("W1", "Physics-informed machine learning", "Jane Smith")]
    _, score, versions = match_entry(ENTRY, candidates)
    assert versions == []
    assert score < 90


def test_duplicate_records_are_kept_as_versions_and_most_cited_wins():
    candidates = [
        _work("W1", "Physics-informed machine learning for fluids", "Smith, Jane", cited=10, year=2020),
        _work("W2", "Physics-Informed Machine Learning for Fluids", "Jane Smith", cited=500),
        _work("W3", "Something else entirely", "Jane Smith", cited=9999),
    ]
    best, score, versions = match_entry(ENTRY, candidates)
    assert best["id"] == "W2"
    assert score == 100
    assert {v["id"] for v in versions} == {"W1", "W2"}


def test_alt_title_is_used():
    entry = {**ENTRY, "alt_title": "Learning fluids with physics"}
    best, score, versions = match_entry(entry, [_work("W9", "Learning fluids with physics", "Smith")])
    assert best["id"] == "W9" and len(versions) == 1
