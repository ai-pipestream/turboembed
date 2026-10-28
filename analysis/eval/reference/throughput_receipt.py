"""Time a command N times and write a throughput receipt.

Usage: throughput_receipt.py OUT.json SENTENCES [--runs N] -- CMD...

Runs CMD N times (default 20, the full receipt form of #1606),
timing each run's wall clock and reading /proc/loadavg before and
after it, and writes one JSON receipt: the exact command, the git
commit of the current directory, the host's core count, every run's
seconds and loads, and min, mean, p50 and p99 seconds (nearest-rank
percentiles) with sentences per second at the median. SENTENCES is
the gold sentence count the command processes per run; it is the
denominator, nothing else is inferred from it.

The timing is the whole command, model load included, because that
is what a user of the command waits for; a receipt that wants
inference-only timing says so in its record note instead. A run
that exits nonzero stops the receipt: partial timing lists are not
written. Receipts are only comparable at like host load, which is
why every run's 1-minute load average is recorded rather than
summarized away.
"""

import json
import math
import os
import statistics
import subprocess
import sys
import time

SCHEMA = "turboembed.throughput-receipt.v1"


def load1():
    try:
        with open("/proc/loadavg", encoding="utf-8") as f:
            return float(f.read().split()[0])
    except (OSError, ValueError, IndexError):
        return None


def percentile(sorted_values, q):
    rank = max(1, math.ceil(q / 100 * len(sorted_values)))
    return sorted_values[rank - 1]


def summarize(seconds):
    ordered = sorted(seconds)
    return {
        "min": ordered[0],
        "mean": statistics.fmean(ordered),
        "p50": percentile(ordered, 50),
        "p99": percentile(ordered, 99),
        "max": ordered[-1],
    }


def main():
    args = sys.argv[1:]
    if "--" not in args or len(args) < 4:
        raise SystemExit(__doc__)
    out, sentences = args[0], int(args[1])
    rest = args[2:]
    runs = 20
    if rest[:1] == ["--runs"]:
        runs = int(rest[1])
        rest = rest[2:]
    if rest[:1] != ["--"] or len(rest) < 2:
        raise SystemExit(__doc__)
    cmd = rest[1:]
    if os.path.exists(out):
        raise SystemExit(f"{out} exists; refusing to overwrite a receipt")
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=False
    ).stdout.strip() or None
    timed = []
    for i in range(runs):
        before = load1()
        start = time.perf_counter()
        done = subprocess.run(cmd, capture_output=True, text=True, check=False)
        seconds = time.perf_counter() - start
        if done.returncode != 0:
            raise SystemExit(
                f"run {i + 1} of {runs} exited {done.returncode}:\n{done.stderr[-2000:]}"
            )
        timed.append({"seconds": seconds, "load1_before": before, "load1_after": load1()})
        print(f"run {i + 1}/{runs}: {seconds:.3f}s", file=sys.stderr, flush=True)
    stats = summarize([t["seconds"] for t in timed])
    receipt = {
        "schema": SCHEMA,
        "command": cmd,
        "commit": commit,
        "sentences_per_run": sentences,
        "runs": timed,
        "seconds": stats,
        "sentences_per_second_at_p50": sentences / stats["p50"],
        "host_cores": os.cpu_count(),
        "unix_time": int(time.time()),
    }
    with open(out, "w", encoding="utf-8") as f:
        json.dump(receipt, f, indent=2)
        f.write("\n")
    print(f"wrote {out}", file=sys.stderr)


if __name__ == "__main__":
    main()
