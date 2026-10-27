"""MLX's matrix product timed alone at the bare shapes of gemm_time.swift, for
the GEMM receipt of docs/metal.md (The linear layers alone against MLX):
x [M, k] times w [k, n] in float16, as mlx_bench.py's linear layers compute
them (the weights transposed once, so `x @ w`), with Q, K and V as one
product of n = 3h. MLX's steel GEMM sums float16 products in float32, as
our F32 SIMD-group matrices do, and writes float16.

Arguments: the hidden widths (default 384 768). Refuses an MLX other than
VERSION. Prints a first line with the MLX version and the SHA-256 of
mlx/lib/libmlx.dylib and mlx/lib/mlx.metallib, then one JSON line a shape,
each figure two ways, as gemm_time.swift gives ours: "single", one product
per mx.eval, from the host; and "batched", REPS products of the same x and
w in one mx.eval graph, divided by REPS. The batched products are
independent, so MLX may overlap them on the GPU, where ours run one after
another. Each is TIMES timed after one untimed: least, mean, median and
p99 (the largest of TIMES) in ms.
"""

import hashlib
import json
import os
import sys
import time

import mlx.core as mx

VERSION = "0.29.3"
REPS = 20
TIMES = 20


def stats(v):
    s = sorted(v)
    return {
        "least": round(s[0], 4),
        "mean": round(sum(s) / len(s), 4),
        "median": round(s[len(s) // 2], 4),
        "p99": round(s[-1], 4),
    }


def sha256(path):
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def main():
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
                "reps": REPS,
                "times": TIMES,
            }
        ),
        flush=True,
    )
    # GEMM_ONLY=<hidden>,<m>,<shape> times that one shape alone, as
    # gemm_time.swift reads it, so a driver can interleave the two.
    only = os.environ.get("GEMM_ONLY", "").split(",") if os.environ.get("GEMM_ONLY") else None
    hiddens = [int(only[0])] if only else [int(a) for a in sys.argv[1:]] or [384, 768]
    for h in hiddens:
        shapes = [("qkv", 3 * h, h), ("attn_out", h, h), ("ffn_in", 4 * h, h), ("ffn_out", h, 4 * h)]
        for m in [int(only[1])] if only else [128, 512, 1024, 4096, 16384]:
            for shape, n, k in shapes:
                if only and shape != only[2]:
                    continue
                x = (mx.random.normal((m, k)) / 16).astype(mx.float16)
                w = (mx.random.normal((k, n)) / 16).astype(mx.float16)
                mx.eval(x, w)

                def timed(count):
                    t0 = time.perf_counter()
                    mx.eval(*[x @ w for _ in range(count)])
                    return (time.perf_counter() - t0) * 1000 / count

                timed(1)
                single = [timed(1) for _ in range(TIMES)]
                timed(REPS)
                batched = [timed(REPS) for _ in range(TIMES)]
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
                            "batched_host": stats(batched),
                        }
                    ),
                    flush=True,
                )


if __name__ == "__main__":
    main()
