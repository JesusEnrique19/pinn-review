"""Run the full benchmark grid (protocol 9): P1-P3 x M0-M5, P4 x {M0, M1, M2, M3, M5}, seeds 0-4.

Resumable: a run whose JSON exists in results/bench is skipped. Jobs are ordered seed-major so every
(problem, method) cell gets a first result early. Workers run in parallel with 2 torch threads each.

    python scripts/g4/run_grid.py [--workers 3]
"""
import argparse
import json
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

from pinnreview.paths import RESULTS

OUT = RESULTS / "bench"
PROBLEMS = {"P1": ["M0", "M1", "M2", "M3", "M4", "M5"], "P2": ["M0", "M1", "M2", "M3", "M4", "M5"],
            "P3": ["M0", "M1", "M2", "M3", "M4", "M5"], "P4": ["M0", "M1", "M2", "M3", "M5"]}
SEEDS = range(5)


def run(job):
    import torch

    torch.set_num_threads(2)
    from pinnreview.bench.pinn import train

    problem, method, seed = job
    out = train(problem, method, seed)
    (OUT / f"{problem}_{method}_s{seed}.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    return job, out["rel_l2"], out["wall_seconds"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=3)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    jobs = [(p, m, s) for s in SEEDS for p, ms in PROBLEMS.items() for m in ms
            if not (OUT / f"{p}_{m}_s{s}.json").exists()]
    log = OUT / "progress.log"
    with open(log, "a", encoding="utf-8") as fh:
        fh.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} start: {len(jobs)} jobs, {args.workers} workers\n")
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(run, j): j for j in jobs}
        for done, fut in enumerate(as_completed(futures), 1):
            job, err, wall = fut.result()
            with open(log, "a", encoding="utf-8") as fh:
                fh.write(f"{time.strftime('%H:%M:%S')} {done}/{len(jobs)} {job} rel_l2={err:.3e} {wall / 60:.1f} min\n")


if __name__ == "__main__":
    main()
