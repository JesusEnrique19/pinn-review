import pandas as pd

from pinnreview.bench.analysis import cd_groups, time_to_accuracy
from pinnreview.bench.timing import active_seconds


def test_active_seconds_removes_suspension_gap():
    hist = [{"step": 500 * i, "seconds": 30.0 * i} for i in range(1, 11)]
    for h in hist[5:]:
        h["seconds"] += 40_000  # machine suspended between steps 2500 and 3000
    active, gaps = active_seconds(hist)
    assert gaps == 1
    assert abs(active - 300.0) < 1e-6


def test_time_to_accuracy_interpolates_in_log_log():
    curve = pd.DataFrame({"wall_seconds": [0.01, 0.1, 1.0], "rel_l2": [1e-1, 1e-3, 1e-5]})
    assert abs(time_to_accuracy(curve, 1e-2) - 10 ** -1.5) < 1e-9
    assert time_to_accuracy(curve, 1e-6) is None          # never reached
    assert time_to_accuracy(curve, 0.5) == 0.01            # reached at the cheapest resolution (upper bound)


def test_cd_groups_are_maximal_non_significant_runs():
    ranks = {"A": 1.0, "B": 1.5, "C": 2.2, "D": 4.0}
    assert cd_groups(ranks, 1.3) == [["A", "B", "C"]]
    assert cd_groups(ranks, 1.0) == [["A", "B"], ["B", "C"]]
    assert cd_groups(ranks, 0.1) == []
