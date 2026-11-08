from pinnreview.bibliography import author_list, format_published, ieee_name, published_from_crossref
from pinnreview.manuscript import Writer, parse, resolve_refs, sci


class FakeBib:
    def __init__(self):
        self.order = []

    def cite(self, key):
        if key not in self.order:
            self.order.append(key)
        return self.order.index(key) + 1


def test_ieee_names():
    assert ieee_name("Maziar Raissi") == "M. Raissi"
    assert ieee_name("Tim De Ryck") == "T. De Ryck"
    assert ieee_name("G.E. Karniadakis") == "G. E. Karniadakis"
    assert ieee_name("Jean-Luc Picard") == "J.-L. Picard"
    assert author_list(["A B", "C D", "E F"]) == "A. B, C. D, and E. F"
    assert author_list([f"Ann Author{i}" for i in range(7)]) == "A. Author0 et al."


def test_citations_are_numbered_by_first_appearance_and_ranged():
    w = Writer.__new__(Writer)
    w.bib = FakeBib()
    assert w.cite(["a"]) == "[1]"
    assert w.cite(["b", "c", "d"]) == "[2]–[4]"
    assert w.cite(["a", "c"]) == "[1], [3]"
    assert w.cite(["e", "b"]) == "[2], [5]"


def test_cross_references_follow_ieee_style():
    text = "# Intro {#sec:intro}\n\nSee @fig:a, @tbl:t and @sec:m.\n\n![Cap](fig_x){#fig:a wide}\n\n" \
           "# Methods {#sec:m}\n\nTable: Caption {#tbl:t}\n| A | B |\n|---|---|\n| 1 | @fig:a |\n"
    blocks = resolve_refs(parse(text))
    assert blocks[1] == ("p", "See Fig. 1, Table I and Section II.")
    fig = blocks[2]
    assert fig[0] == "fig" and fig[1] == "fig_x" and fig[4] is True
    tbl = blocks[4]
    assert tbl[2] == [[" A ", " B "], [" 1 ", " Fig. 1 "]]


def test_smart_quotes():
    assert Writer.smart_quotes('uses "physics-informed" and Bradford\'s law') == \
        "uses “physics-informed” and Bradford’s law"


def test_crossref_record_gives_print_year_article_number_and_proceedings_volume():
    journal = {"type": "journal-article", "container-title": ["J. X"], "volume": "378", "page": "686-707",
               "published-print": {"date-parts": [[2019, 2]]}, "issued": {"date-parts": [[2018]]}, "DOI": "10.1/a"}
    pub = published_from_crossref(journal)
    assert (pub["year"], pub["first_page"], pub["last_page"]) == (2019, "686", "707")
    online_only = {"type": "journal-article", "container-title": ["Sci. Rep."], "volume": "9", "issue": "1",
                   "article-number": "5233", "issued": {"date-parts": [[2019]]}, "DOI": "10.1/b"}
    assert format_published("A. B", "T", published_from_crossref(online_only)) ==         "A. B, “T,” *Sci. Rep.*, vol. 9, no. 1, Art. no. 5233, 2019, doi: 10.1/b."
    neurips = {"type": "proceedings-article", "container-title": ["Advances in Neural Information Processing Systems 37"],
               "page": "1-9", "issued": {"date-parts": [[2024]]}, "DOI": "10.1/c"}
    assert format_published("A. B", "T", published_from_crossref(neurips)) ==         "A. B, “T,” in *Advances in Neural Information Processing Systems*, vol. 37, 2024, pp. 1–9, doi: 10.1/c."


def test_sci_uses_markup_exponents_and_decimals_in_the_middle_range():
    assert sci(0.0631) == "6.3 × 10^−2^"
    assert sci(1.9e-9) == "1.9 × 10^−9^"
    assert sci(145283.9) == "1.5 × 10^5^"
    assert sci(0.989) == "0.99"
    assert sci(1.00002) == "1.00"
