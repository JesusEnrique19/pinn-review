import yaml

from pinnreview import retrieval
from pinnreview.validation import abstract_text


def test_every_query_version_is_well_formed():
    with open(retrieval.QUERIES_FILE, encoding="utf-8") as fh:
        queries = yaml.safe_load(fh)
    for version, q in queries.items():
        assert q["source"] == "openalex", version
        assert q["years"] == [2017, 2025], version
        assert q["terms"], version
        for term in q["terms"]:
            # commas separate OpenAlex filters and would silently split the query
            assert "," not in term, (version, term)
            assert term.count("(") == term.count(")"), (version, term)
            assert term.count('"') % 2 == 0, (version, term)


def test_v1_extends_v0():
    v0, v1 = retrieval.load_query("v0"), retrieval.load_query("v1")
    assert v1["terms"][: len(v0["terms"])] == v0["terms"]


def test_search_filter():
    q = {"field": "title_and_abstract.search", "years": [2017, 2025], "terms": ['"a b"', "C"]}
    assert retrieval.search_filter(q) == 'title_and_abstract.search:"a b" OR C,publication_year:2017-2025'


def test_abstract_text_rebuilds_word_order():
    assert abstract_text({"networks": [1], "Neural": [0], "PDEs.": [3], "solve": [2]}) == "Neural networks solve PDEs."
    assert abstract_text(None) == ""
