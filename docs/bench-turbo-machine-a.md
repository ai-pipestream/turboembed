# Turbo bench — Machine A (CUDA)

Hostnames stay out of this file (Machine A only).

SOLIDIFY (7) latency / traffic receipt for **TurboEmbed MiniLM** and
**TurboRerank MiniLM-L6** on CUDA. This is the NVIDIA slice of the
A/B/C bench called out in [`grpc-output-scratch.md`](grpc-output-scratch.md).
It is not the Intel remote-USM probe and not the Apple Metal GELU page.

## What is measured

After load + `BENCH_WARMUP` (default 32) untimed forwards:

| metric | TurboEmbed MiniLM | TurboRerank MiniLM-L6 |
|---|---|---|
| Workload | `embed_one("minilm", "hello world")` mean+L2 | `score` of the Berlin relevant pair (one CE forward) |
| p50 / p99 | nearest-rank over `BENCH_ITERS` (default 200, **N≥100**) | same |
| token H2D | `turboembed_ort_cuda_forward_h2d_bytes` | `turbo_buffer_cuda_forward_h2d_bytes` |
| hidden D2H | `turboembed_ort_d2h_bytes` (DEVICE mean+L2) | 0 — activations stay DEVICE; only the 4-byte CLS logit comes back |
| allocs/forward | arena + ORT `gpu_external_alloc` + CUDA forward allocs | arena + CUDA forward allocs |

Goldens (must stay in band or `pass=false`):

- Embed: `testdata/e2e/goldens/nvidia/minilm.json` (`hello world` + `parity:*`, cosine ≥ 0.99)
- Rerank: `testdata/reference_rerank/ms_marco_minilm_l6_berlin.json` (HF Identity logits, atol 2e-3)

Fixed paths must report **0** token H2D, **0** hidden D2H, **0** allocs/forward.
The receipt fails if those counters are non-zero.

Latency is `std::time::Instant` around the real ABI call (tokenize +
device forward). Release build (`cargo run --release`). Do not invent
microseconds.

## Commands

```bash
make bench-machine-a
make bench-turbo MACHINE=A
# optional:
make bench-machine-a BENCH_WARMUP=32 BENCH_ITERS=200
```

`make bench-machine-a` fetches MiniLM-L6 CE weights if missing, sets
`LD_LIBRARY_PATH` for the CUDA 13 ORT bundle, runs the two phase
binaries, and merges `testdata/receipts/bench/machine-a-cuda.json`.

`MACHINE=B` / `MACHINE=C` on this target exit 1 — those hosts write
their own receipts.

## This run (RTX 4080 SUPER)

From `make bench-machine-a` (warmup 32, N=200, release). Same file as the
receipt — re-run the target instead of editing.

| engine | p50 | p99 | token H2D | hidden D2H | allocs/fwd | band |
|---|---|---|---|---|---|---|
| TurboEmbed MiniLM (`hello world`) | 1430 µs | 1584 µs | 0 | 0 | 0 | cosine ≥ 0.999999 vs nvidia MiniLM |
| TurboRerank MiniLM-L6 (Berlin pair) | 676 µs | 684 µs | 0 | 0 | 0 | HF logits, max abs 9.5e-7 |

## Receipt

[`testdata/receipts/bench/machine-a-cuda.json`](../testdata/receipts/bench/machine-a-cuda.json)

Do not hand-edit numbers. Re-run the Make target on Machine A.

`host` is historical provenance (map through the README Machine chart).
Docs use **Machine A** only.

## Not this item

- gRPC `output_scratch` (SOLIDIFY 6) — [`grpc-output-scratch.md`](grpc-output-scratch.md)
- Intel remote-USM / FP32 IR — [`intel-remote-usm-machine-b.md`](intel-remote-usm-machine-b.md)
- Apple Metal Hart GELU — [`apple-turborerank-metal-gelu-machine-c.md`](apple-turborerank-metal-gelu-machine-c.md)
