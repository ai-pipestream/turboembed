#!/usr/bin/env python3
"""Run the bounded Intel native/ABI pilot and retain each case's raw samples."""
import argparse
import json
import subprocess
import time
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", required=True, type=Path)
    parser.add_argument("--bundle", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    benchmark = args.benchmark.resolve(strict=True)
    bundle = args.bundle.resolve(strict=True)
    args.output.mkdir(parents=True, exist_ok=False)
    deadline = time.monotonic() + 1800
    for batch in (1, 8, 32):
        for sequence in (32, 128, 256):
            for pattern in ("full", "mixed"):
                name = f"b{batch}-s{sequence}-{pattern}"
                print(f"START {name}", flush=True)
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise SystemExit("30-minute platform budget exhausted")
                command = [str(benchmark), str(bundle), str(batch), str(sequence), pattern,
                           "10", str((args.output / f"{name}.json").resolve())]
                with (args.output / f"{name}.log").open("w") as log:
                    subprocess.run(command, stdout=log, stderr=subprocess.STDOUT,
                                   check=True, timeout=remaining)
                case = json.loads((args.output / f"{name}.json").read_text())
                before, after = case["abi_before_timing"], case["abi_after_timing"]
                if any(before[key] != after[key] for key in ("input_write_bytes", "output_read_bytes")):
                    raise SystemExit(f"{name}: unexpected transfer during prepared execution")
                if any(item["p50_ratio"] > 1.05 or item["throughput_ratio"] < 0.95
                       for item in case["repeats"]):
                    raise SystemExit(f"{name}: predeclared native overhead budget exceeded")
                ratios = [round(item["p50_ratio"], 4) for item in case["repeats"]]
                print(f"FINISH {name}: p50 ratios {ratios}", flush=True)


if __name__ == "__main__":
    main()
