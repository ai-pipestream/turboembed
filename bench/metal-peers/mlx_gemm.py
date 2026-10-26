"""MLX's matrix product timed alone at the shapes of gemm_time.swift, for the
GEMM receipt of docs/metal.md (The F16 tile): x [M, k] times w [k, n] in
float16, as mlx_bench.py's linear layers compute them (the weights
transposed once, so `x @ w`), with Q, K and V as one product of n = 3h.

Arguments: the hidden widths (default 384 768). For each shape it evaluates
REPS products in one graph, TIMES times after one untimed, and prints one
JSON line: the least and the median time per product in ms from the host,
and the TFLOPS at the least. The products are independent, so MLX may
overlap them; the per-product time is the graph's time over REPS.
"""

import json
import sys
import time

import mlx.core as mx

REPS = 20
TIMES = 7


def main():
    hiddens = [int(a) for a in sys.argv[1:]] or [384, 768]
    for h in hiddens:
        shapes = [("qkv", 3 * h, h), ("attn_out", h, h), ("ffn_in", 4 * h, h), ("ffn_out", h, 4 * h)]
        for m in [128, 512, 1024, 4096, 16384]:
            for shape, n, k in shapes:
                x = (mx.random.normal((m, k)) / 16).astype(mx.float16)
                ws = [(mx.random.normal((k, n)) / 16).astype(mx.float16) for _ in range(REPS)]
                mx.eval(x, *ws)
                ms = []
                for i in range(TIMES + 1):
                    t0 = time.perf_counter()
                    mx.eval(*[x @ w for w in ws])
                    t = (time.perf_counter() - t0) * 1000 / REPS
                    if i > 0:
                        ms.append(t)
                ms.sort()
                print(
                    json.dumps(
                        {
                            "side": "mlx",
                            "mlx": mx.__version__,
                            "hidden": h,
                            "shape": shape,
                            "m": m,
                            "n": n,
                            "k": k,
                            "least_ms": round(ms[0], 4),
                            "median_ms": round(ms[len(ms) // 2], 4),
                            "tflops": round(2 * m * n * k / ms[0] / 1e9, 3),
                        }
                    ),
                    flush=True,
                )


if __name__ == "__main__":
    main()
