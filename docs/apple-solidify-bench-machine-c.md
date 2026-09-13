# FINAL SOLIDIFY bench (Machine C) — Metal

Live p50/p99 plus the same honesty gates as the SOLIDIFY arena / GELU
work. TurboEmbed and TurboRerank both run on **Metal SHARED**. This is
not a copied cosine receipt and not a CPU fallback.

| Field | Value |
|---|---|
| Host | Machine C, Apple M2, Metal |
| Command | `make bench-machine-c` |
| Receipt | `testdata/receipts/bench/machine-c-metal.json` |
| Memory | `turbo_buffer` Metal **SHARED** (`MTLResourceStorageModeShared`) |
| Embed goldens | vs nvidia ≥ 0.97, vs apple ≥ 0.99 (`testdata/e2e/goldens`) |
| Berlin | `testdata/reference_rerank/ms_marco_minilm_l6_berlin.json` (`atol` 2e-3) |

## Gates

| Gate | What fails it |
|---|---|
| p50 / p99 measured | Empty sample list, `p50 == 0`, or `p99 < p50`. Do not invent µs. |
| allocs/forward == 0 | Any `turbo_buffer_alloc_counter` increment after warmup on embed or score. |
| Berlin in band | Metal logits outside `2e-3` / cosine ≤ 0.999 / ranking broken. |
| Embed goldens in band | MiniLM mean+L2 cosine below the floors (BERT pooler / 8-d FNV). |
| SHARED, no CPU fallback | `metal_owns` / `metal_lookup` miss; `AUTO` not METAL; tokens on a private MTL buffer. `make turborerank-tests-nometal` must still fail Metal create loud. |

Timings are unary after warmup: TurboEmbed `turboembed_embed("minilm", "hello world")`, TurboRerank Berlin 3-doc `score` (`IDENTITY`). Defaults: `BENCH_WARMUP=32`, `BENCH_ITERS=200`. Override those env vars if you need a longer sample — do not drop below a real distribution.

GELU stays the HF erf form on the Hart software special
(`docs/apple-turborerank-metal-gelu-machine-c.md`). That is not a host
`erff` fallback.

## Commands

```bash
make bench-machine-c              # nometal fail-loud + measure + write receipt
make bench-machine-c-rerank       # TurboRerank slice only
make metal-erf-probe              # MSL still has no erf()
```

`make bench-machine-c` is the fact of record. Re-running it overwrites
the receipt with newly measured numbers.
