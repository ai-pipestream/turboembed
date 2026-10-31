"""Time a command N times and write a throughput receipt.

Usage: throughput_receipt.py OUT.json SENTENCES [--runs N]
       [--output FILE] -- CMD...

Runs CMD N times (default 20, the full receipt form of #1606),
timing each run's wall clock and reading /proc/loadavg before and
after it, and writes one JSON receipt: the exact command, the git
commit of the current directory and its working-tree state (dirty
or not, with SHA-256 digests of `git status --porcelain` and of
`git diff HEAD`, so a run from uncommitted sources is disclosed as
one and its state is pinned by digest), the host's core count,
every run's seconds and loads, and min, mean, p50 and p99 seconds
(nearest-rank percentiles) with sentences per second at the median.
SENTENCES is the gold sentence count the command processes per
run; it is the denominator, nothing else is inferred from it.

With --output FILE, FILE (the file the command writes its result
to, overwritten on each run as the peer commands do) is hashed
after every run and the digest is kept per run, so a receipt can
show whether every run produced the same bytes; without it a
receipt says nothing about outputs at all.

The timing is the whole command, model load included, because that
is what a user of the command waits for; a receipt that wants
inference-only timing says so in its record note instead. A run
that exits nonzero stops the receipt: partial timing lists are not
written. Receipts are only comparable at like host load, which is
why every run's 1-minute load average is recorded rather than
summarized away.
"""

import hashlib
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


def git_bytes(*argv):
    done = subprocess.run(["git", *argv], capture_output=True, check=False)
    return done.stdout if done.returncode == 0 else None


def working_tree():
    """The current directory's commit and uncommitted-state digests.

    `commit` alone once let a receipt name a commit whose tree did
    not hold the harness being timed; `dirty` and the two digests
    say when the sources were not that commit's, and pin what the
    state was. Outside a git checkout every field is None/False
    rather than guessed.
    """
    commit = git_bytes("rev-parse", "HEAD")
    if commit is None:
        return {"commit": None, "dirty": False, "status_sha256": None, "diff_sha256": None}
    status = git_bytes("status", "--porcelain=v1") or b""
    diff = git_bytes("diff", "--binary", "HEAD") or b""
    return {
        "commit": commit.decode("utf-8", "replace").strip() or None,
        "dirty": bool(status.strip()),
        "status_sha256": hashlib.sha256(status).hexdigest(),
        "diff_sha256": hashlib.sha256(diff).hexdigest(),
    }


def file_sha256(path):
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


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
    output = None
    while rest[:1] in (["--runs"], ["--output"]):
        if len(rest) < 2:
            raise SystemExit(__doc__)
        if rest[0] == "--runs":
            runs = int(rest[1])
        else:
            output = rest[1]
        rest = rest[2:]
    if rest[:1] != ["--"] or len(rest) < 2:
        raise SystemExit(__doc__)
    cmd = rest[1:]
    if os.path.exists(out):
        raise SystemExit(f"{out} exists; refusing to overwrite a receipt")
    tree = working_tree()
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
        entry = {"seconds": seconds, "load1_before": before, "load1_after": load1()}
        if output is not None:
            try:
                entry["output_sha256"] = file_sha256(output)
            except OSError as e:
                raise SystemExit(f"run {i + 1} of {runs}: cannot hash --output {output}: {e}")
        timed.append(entry)
        print(f"run {i + 1}/{runs}: {seconds:.3f}s", file=sys.stderr, flush=True)
    stats = summarize([t["seconds"] for t in timed])
    receipt = {
        "schema": SCHEMA,
        "command": cmd,
        "commit": tree["commit"],
        "working_tree": {
            "dirty": tree["dirty"],
            "status_sha256": tree["status_sha256"],
            "diff_sha256": tree["diff_sha256"],
        },
        "sentences_per_run": sentences,
        "output_file": output,
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
