# FINAL SOLIDIFY bench receipts

One JSON per Machine. Written by a **live** `make bench-machine-*` on
that host. Not a copy of a prior cosine receipt. Not invented p50/p99.

| file | Machine | memory path | command |
|---|---|---|---|
| `machine-c-metal.json` | Machine C (Apple M2) | Metal **SHARED** (`MTLResourceStorageModeShared`) | `make bench-machine-c` — **LIVE** |
| `machine-a-cuda.json` | Machine A | CUDA PINNED mapped + DEVICE | `make bench-machine-a` (when measured) |
| `machine-b-ze.json` | Machine B | ZE **SHARED** USM | `make bench-machine-b` (when measured) |

## Gates (same on every Machine)

Both **TurboEmbed** and **TurboRerank** must pass:

1. **p50 / p99 measured** on the live device after warmup. Do not copy
   timings from another host or an older receipt.
2. **allocs/forward == 0** after warmup (`turbo_buffer_alloc_counter`).
3. **Goldens in band** — TurboRerank Berlin (`atol` 2e-3) and TurboEmbed
   MiniLM cosine floors for that Machine.
4. **Memory-path honesty** — the named device placement (SHARED here).
   No silent CPU fallback. `AUTO` resolves to the host GPU.

`pass=false` is an honest fail. Do not hand-edit a passing receipt.
Do not delete another Machine's file when refreshing this one.

Machine C proof: [`docs/apple-solidify-bench-machine-c.md`](../../../docs/apple-solidify-bench-machine-c.md).
