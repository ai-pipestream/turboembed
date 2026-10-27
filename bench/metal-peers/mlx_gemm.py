"""MLX's matrix product timed alone at the bare shapes of gemm_time.swift, for
the GEMM receipt of docs/metal.md (The linear layers alone against MLX):
x [M, k] times w [k, n] in float16, as mlx_bench.py's linear layers compute
them, with contiguous [k, n] weights and Q, K and V as one
product of n = 3h. MLX's steel GEMM sums float16 products in float32, as
our F32 SIMD-group matrices do, and writes float16.

Arguments: the hidden widths (default 384 768). Refuses an MLX other than
VERSION. Prints a first line with the MLX version and the SHA-256 of
mlx/lib/libmlx.dylib and mlx/lib/mlx.metallib, then one JSON line a shape,
each figure one product per mx.eval, from the host. Each is TIMES timed
after one untimed: least, mean, median and nearest-rank p99 in ms,
with every raw sample retained. No batched graph is divided by an assumed
product count: graph deduplication and scheduling need separate evidence.
"""

import hashlib
import json
import math
import os
import statistics
import sys
import time

VERSION = "0.29.3"
TIMES = 20


def stats(v):
    samples = list(v)
    if not samples or any(not math.isfinite(x) or x <= 0 for x in samples):
        raise ValueError("timing samples must be finite, positive and nonempty")
    s = sorted(samples)
    return {
        "least": s[0],
        "mean": statistics.mean(s),
        "median": statistics.median(s),
        "p99": s[math.ceil(0.99 * len(s)) - 1],
        "samples": samples,
    }


def sha256(path):
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def main():
    import mlx.core as mx

    if mx.__version__ != VERSION:
        raise SystemExit(f"MLX {mx.__version__} is not {VERSION}")
    lib = os.path.join(os.path.dirname(mx.__file__), "lib")
    print(
        json.dumps(
            {
                "side": "mlx",
                "mlx": mx.__version__,
                "libmlx_sha256": sha256(os.path.join(lib, "libmlx.dylib")),
                "metallib_sha256": sha256(os.path.join(lib, "mlx.metallib")),
                "schema": 2,
                "schedule": "single",
                "input_dtype": "float16",
                "accumulator_dtype": "float32",
                "output_dtype": "float16",
                "weight_layout": "contiguous-k-n",
                "warmup": 1,
                "times": TIMES,
            }
        ),
        flush=True,
    )
    mx.random.seed(0)
    hiddens = [int(a) for a in sys.argv[1:]] or [384, 768]
    for h in hiddens:
        shapes = [("qkv", 3 * h, h), ("attn_out", h, h), ("ffn_in", 4 * h, h), ("ffn_out", h, 4 * h)]
        for m in [128, 512, 1024, 4096, 16384]:
            for shape, n, k in shapes:
                x = (mx.random.normal((m, k)) / 16).astype(mx.float16)
                w = (mx.random.normal((k, n)) / 16).astype(mx.float16)
                mx.eval(x, w)

                def timed():
                    t0 = time.perf_counter()
                    mx.eval(x @ w)
                    return (time.perf_counter() - t0) * 1000

                timed()
                single = [timed() for _ in range(TIMES)]
                print(
                    json.dumps(
                        {
                            "side": "mlx",
                            "mode": "bare",
                            "hidden": h,
                            "shape": shape,
                            "m": m,
                            "n": n,
                            "k": k,
                            "flops": 2 * m * n * k,
                            "single_host": stats(single),
                        }
                    ),
                    flush=True,
                )


if __name__ == "__main__":
    main()
