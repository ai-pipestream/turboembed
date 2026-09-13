# turbo_buffer ZE USM arena — Machine B LIVE

Evidence that SOLIDIFY (1) is real on the Intel GPU host: Level Zero
USM **HOST / SHARED / DEVICE** rent/return through
[`include/turbo_buffer.h`](../include/turbo_buffer.h), TurboRerank
OpenVINO GPU tokens rented as **SHARED**, and Berlin HF scores inside
the golden band. No silent CPU remap.

Hostnames stay out of this file (Machine B only).

## Done criteria

| # | Gate | Proof |
|---|---|---|
| 1 | ZE USM HOST/SHARED/DEVICE rent/return is real | `zeMemGetAllocProperties` via `turbo_buffer_ze_query`. DEVICE round-trip uses `zeCommandListAppendMemoryCopy`, not `memcpy`. Re-rent of the same shapes after return has `turbo_buffer_alloc_counter() == 0`. |
| 2 | TurboRerank OV path rents the arena | `turborerank_buffer_alloc(OPENVINO_GPU)` and engine work/scratch tokens query as SHARED. `owns()` on the engine arena. After warmup, `forward` / `score` increment 0 allocs for those slots. |
| 3 | Berlin HF within band | logits vs `testdata/reference_rerank/ms_marco_minilm_l6_berlin.json` (`atol` 2e-3, cosine > 0.999). Receipts under `testdata/receipts/turborerank/`. |
| 4 | No silent CPU remap | PINNED on a ZE arena is `NOT_IMPLEMENTED`. SHARED/DEVICE without a GPU is `UNAVAILABLE`, not HOST. OV GPU `forward` of a CPU buffer is `INTERNAL` and names SHARED. GPU create without GPU still fails loud. |

## Commands

```bash
make turbo-buffer-tests              # includes ZE HOST/SHARED/DEVICE LIVE
make turbo-buffer-intel-receipt      # testdata/receipts/turbo_buffer/intel-ze.json
make test-turborerank-intel          # OV GPU/CPU Berlin + receipts
```

`TURBO_BUFFER_ZE` is compile-gated with Level Zero (`libze_loader` +
`ze_api.h`). A binary without L0 returns `NOT_IMPLEMENTED` and never
opens a CPU arena as a stand-in.

## What is not this item

Items 2–7 of SOLIDIFY are out of scope here. CUDA PINNED/DEVICE is
Machine A. Metal SHARED is Machine C.
