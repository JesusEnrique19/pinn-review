from pinnreview import openalex, retrieval

v1, v3 = retrieval.load_query("v1"), retrieval.load_query("v3")


def n(q):
    return openalex.get("works", {"filter": retrieval.search_filter(q), "per_page": 1})["meta"]["count"]


expr = f"({' OR '.join(v3['terms'])}) AND NOT ({' OR '.join(v1['terms'])})"
print("v3-only:", n({**v3, "terms": [expr]}))
print("v3 total:", n(v3))
core = {**v3, "terms": v3["terms"][:8]}
print("v3 Block A only (specific names):", n(core))
