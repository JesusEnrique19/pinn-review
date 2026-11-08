from pinnreview.dedup import arxiv_id, deduplicate


def work(wid, title, author="Jane Smith", year=2021, doi=None, wtype="article", cited=0, url=None, refs=()):
    return {
        "id": wid,
        "doi": doi,
        "display_name": title,
        "publication_year": year,
        "type": wtype,
        "authorships": [{"author": {"display_name": author}}],
        "cited_by_count": cited,
        "referenced_works": list(refs),
        "locations": [{"landing_page_url": url}] if url else [],
        "primary_location": {"source": {"display_name": "Venue", "id": "S1"}},
    }


def test_preprint_and_journal_version_merge_and_keep_published_metadata():
    pre = work("W1", "Fourier neural operator for parametric PDEs", year=2020, wtype="preprint", cited=50, refs=["R1"])
    pub = work("W2", "Fourier Neural Operator for Parametric PDEs.", year=2021, cited=900, refs=["R1", "R2"])
    records, log, repeated = deduplicate([pre, pub])
    assert len(records) == 1
    row = records.iloc[0]
    assert row["id"] == "W2" and row["year"] == 2020 and row["cited_by_count"] == 950
    assert row["n_versions"] == 2 and row["n_references"] == 2
    assert repeated == 0


def test_subset_title_by_same_author_is_not_merged():
    a = work("W1", "Physics-informed machine learning")
    b = work("W2", "Physics-informed machine learning for turbulent channel flows")
    records, _, _ = deduplicate([a, b])
    assert len(records) == 2


def test_same_title_different_author_or_distant_year_not_merged():
    a = work("W1", "A survey of neural operators", author="Jane Smith", year=2021)
    b = work("W2", "A survey of neural operators", author="Li Wei", year=2021)
    c = work("W3", "A survey of neural operators", author="Jane Smith", year=2024)
    records, _, _ = deduplicate([a, b, c])
    assert len(records) == 3


def test_doi_arxiv_and_repeated_ids():
    a = work("W1", "Title one", doi="https://doi.org/10.1/x")
    b = work("W2", "Completely different title", author="Other Person", doi="https://doi.org/10.1/X")
    c = work("W3", "Title three", url="https://arxiv.org/abs/2010.08895v3")
    d = work("W4", "Another name for three", author="Other Person", doi="https://doi.org/10.48550/arxiv.2010.08895")
    records, log, repeated = deduplicate([a, b, c, d, a])
    assert repeated == 1
    assert len(records) == 2
    assert set(log["rule"]) == {"doi", "arxiv"}
    assert arxiv_id(c) == "2010.08895"
