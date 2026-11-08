"""Keep Windows from idle-sleeping while a given process (the benchmark grid) is alive.

Uses SetThreadExecutionState(ES_CONTINUOUS | ES_SYSTEM_REQUIRED), a per-process power request that changes
no system setting and is released automatically when this script exits. It does not prevent sleep from
closing the lid or choosing Sleep manually.

    python scripts/g4/keep_awake.py <pid>
"""
import ctypes
import subprocess
import sys
import time

ES_CONTINUOUS, ES_SYSTEM_REQUIRED = 0x80000000, 0x00000001


def alive(pid):
    out = subprocess.run(["tasklist", "/FI", f"PID eq {pid}"], capture_output=True, text=True).stdout
    return str(pid) in out


def main():
    pid = int(sys.argv[1])
    while alive(pid):
        ctypes.windll.kernel32.SetThreadExecutionState(ES_CONTINUOUS | ES_SYSTEM_REQUIRED)
        time.sleep(60)
    ctypes.windll.kernel32.SetThreadExecutionState(ES_CONTINUOUS)


if __name__ == "__main__":
    main()
