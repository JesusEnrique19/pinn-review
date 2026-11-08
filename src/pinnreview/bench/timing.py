"""Active training time robust to machine suspension (incident of 2026-10-30/31).

The grid's wall clock kept running while the computer was suspended, inflating ``wall_seconds`` of runs that
spanned the suspension (one 500-step interval of 47 088 s instead of ~30 s). Active time is recomputed from the
logged history: any interval longer than ``factor`` times the median interval is replaced by the median.
"""

import numpy as np


def active_seconds(history, factor=5.0):
    s = np.array([0.0] + [h["seconds"] for h in history])
    steps = np.array([0] + [h["step"] for h in history])
    per_step = np.diff(s) / np.maximum(np.diff(steps), 1)
    med = float(np.median(per_step))
    per_step = np.where(per_step > factor * med, med, per_step)
    return float((per_step * np.diff(steps)).sum()), int((np.diff(s) / np.maximum(np.diff(steps), 1) > factor * med).sum())
