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

## Live measurement (this host)

Receipt `git_sha` `cfeaf1fe98218c09fe3892d33b7ae98765063016`,
`measured_at_utc` `2026-09-13T15:03:52Z`. GPU
`Intel(R) Graphics [0xe223] (dGPU)`. 32 warmup + 128 measured.

| engine | p50 | p99 | allocs/forward | goldens |
|---|---|---|---|---|
| TurboEmbed MiniLM `hello world` | **909.341 µs** | **926.499 µs** | 0 | cosine vs intel/nvidia goldens **0.9999997616** |
| TurboRerank MiniLM-L6 Berlin (3 docs) | **8145.887 µs** | **8237.993 µs** | 0 | max abs **1.43e-6**, cosine **1.0** |

USM honesty on that run:

| | embed | rerank |
|---|---|---|
| explicit ZE H2D / D2H | 0 / 0 | 0 / 0 |
| `remote_ocl_usm_wrap` | false | false |
| wrapped input bytes / call | 3072 (1×256×3×i32) | 18432 (3×512×3×i32) |
| used tokens | — | 34 / 18 / 22 |
| hidden wrap / memcpy | 393216 / 0 | — |

Plugin host-tensor ingest is reported. H2D is not claimed zero.

## Commands

```bash
source /work/opt/openvino_genai/setupvars.sh   # or the host OpenVINO setupvars
make bench-machine-b-ov
# optional:
BENCH_WARMUP=16 BENCH_ITERS=64 make bench-machine-b-ov
```

## What this is not

Machine A CUDA bench is a sibling (`docs/bench-turbo-machine-a.md`,
`make bench-machine-a`). Machine C Metal bench is still that host.
Invented p50/p99. A claimed-zero H2D while remote USM wrap is still
size-0. NPU.
