#!/usr/bin/env python3
"""Run the bounded, predeclared Java binding pilot on an isolated SSH directory."""
import argparse
import json
from pathlib import Path
import shlex
import subprocess
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("host", "directory", "sdk", "bundle", "java", "output"):
        parser.add_argument("--" + name, required=True)
    args = parser.parse_args()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()

    def remote(arguments):
        remaining = 600 - (time.monotonic() - started)
        if remaining <= 0:
            raise TimeoutError("ten-minute pilot deadline exceeded")
        return subprocess.run(
            ["ssh", "-o", "BatchMode=yes", "-o", "StrictHostKeyChecking=yes",
             "-o", "ConnectTimeout=5", args.host,
             shlex.join(["timeout", "--signal=TERM", str(max(1, int(remaining))), *arguments])],
            check=True, capture_output=True, text=True, timeout=remaining + 5)

    directory = args.directory.rstrip("/")
    classpath = ":".join(directory + "/" + name for name in (
        ".", "turboembed-api-0.1.0-SNAPSHOT.jar", "turboembed-ffm-0.1.0-SNAPSHOT.jar"))
    for case, (batch, sequence) in enumerate(((1, 32), (8, 128), (32, 256))):
        for language in (("native", "java") if case % 2 == 0 else ("java", "native")):
            name = f"{language}-{batch}-{sequence}"
            print("running " + name, flush=True)
            if language == "native":
                result = remote([directory + "/binding_benchmark", args.bundle, str(batch), str(sequence)])
                data = result.stdout
            else:
                filename = directory + "/" + name + ".json"
                result = remote([args.java, "--enable-native-access=ALL-UNNAMED", "-cp", classpath,
                                 "BindingBenchmark", args.sdk, args.bundle, str(batch), str(sequence), filename])
                data = remote(["cat", filename]).stdout
            json.loads(data)
            (output / (name + ".json")).write_text(data)
            (output / (name + ".stderr.log")).write_text(result.stderr)
        native = json.loads((output / f"native-{batch}-{sequence}.json").read_text())
        java = json.loads((output / f"java-{batch}-{sequence}.json").read_text())
        for n, j in zip(native["repeats"], java["repeats"], strict=True):
            delta50 = j["bridge"]["p50_ns"] - n["bridge"]["p50_ns"]
            delta99 = j["bridge"]["p99_ns"] - n["bridge"]["p99_ns"]
            print(f"{batch}x{sequence} repeat {n['repeat']}: bridge p50 +{delta50}ns, p99 +{delta99}ns", flush=True)
            if delta50 > 5000 or delta99 > 20000:
                raise RuntimeError("predeclared Java bridge budget exceeded")
    print(f"all bridge gates passed in {time.monotonic() - started:.1f}s", flush=True)


if __name__ == "__main__":
    main()
