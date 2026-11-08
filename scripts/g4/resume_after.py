"""Wait for a running grid process to exit, then run the grid again (it resumes: only missing runs are done).

Holds the same per-process keep-awake request as keep_awake.py for the whole time. Used on 2026-11-01 to
re-run the P1-M1 jobs that failed before the allow_unused fix, which the already-running workers could not load.

    python scripts/g4/resume_after.py <pid>
"""
import ctypes
import subprocess
import sys
import time
from pathlib import Path

from keep_awake import ES_CONTINUOUS, ES_SYSTEM_REQUIRED, alive

GRID = Path(__file__).with_name("run_grid.py")


def main():
    pid = int(sys.argv[1])
    ctypes.windll.kernel32.SetThreadExecutionState(ES_CONTINUOUS | ES_SYSTEM_REQUIRED)
    while alive(pid):
        time.sleep(60)
    subprocess.run([sys.executable, str(GRID)], cwd=GRID.parents[2])
    ctypes.windll.kernel32.SetThreadExecutionState(ES_CONTINUOUS)


if __name__ == "__main__":
    main()
