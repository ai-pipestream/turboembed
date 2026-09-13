# FINAL SOLIDIFY bench — Machine B (OpenVINO GPU)

Live TurboEmbed + TurboRerank latency and honesty gates on the Intel
GPU host. Same gates as Machine A (p50/p99, H↔D/USM bytes honesty,
`allocs/forward == 0`, Berlin + embed goldens in band). Receipt:
[`testdata/receipts/bench/machine-b-ov.json`](../testdata/receipts/bench/machine-b-ov.json).

Hostnames stay out of this file (Machine B only).

## Done criteria

| # | Gate | Proof |
|---|---|---|
| 1 | p50 / p99 measured | `CLOCK_MONOTONIC` around `turboembed_embed_one("minilm", "hello world")` and `turborerank_score` (Berlin 3-doc). Defaults: 32 warmup + 128 measured. `BENCH_WARMUP` / `BENCH_ITERS` override. |
| 2 | H↔D / USM bytes honesty | Explicit `turbo_buffer_ze_memcpy` DEVICE legs are counted (`turbo_buffer_ze_xfer_*`). Steady-state embed/score must see **0**. Remote OCL `USM_USER_BUFFER` wrap of ZE SHARED is **unavailable**; `ov::Tensor` host wrap means the GPU plugin may copy. The receipt reports `wrapped_input_bytes` (and hidden wrap / memcpy). It does **not** claim H2D=0. |
| 3 | allocs/forward == 0 | After warmup, one more embed / score sees `turbo_buffer_alloc_counter() == 0`. |
| 4 | Berlin + embed goldens in band | Rerank: max abs logit err `< 2e-3`, cosine `> 0.999`, relevance order. Embed: cosine vs `testdata/e2e/goldens/{intel,nvidia}/minilm.json` `≥ 0.99`. |

`make bench-machine-b-ov` writes the receipt from this process and
exits 2 if any gate fails. Do not hand-edit latency or cosine fields.

## Commands

```bash
source /work/opt/openvino_genai/setupvars.sh   # or the host OpenVINO setupvars
make bench-machine-b-ov
# optional:
BENCH_WARMUP=16 BENCH_ITERS=64 make bench-machine-b-ov
```

## What this is not

Machine A CUDA / Machine C Metal benches. Invented p50/p99. A claimed-zero
H2D while remote USM wrap is still size-0. NPU.
