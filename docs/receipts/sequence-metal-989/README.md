# BundleSequenceScorer on Metal (#989)

Code: origin/main at branch time, the commit each run in `tests.log` names,
with its full command; this receipt adds no code. Device: Apple M2 through
the Metal backend, F32 at both precisions, release builds, with
`TURBO_TEST_REQUIRE_METAL=1`, so a run that did not reach the device fails.

## Parity

Twenty runs at each of `TURBO_TEST_PRECISION` `model` and `exact` of

```
TURBO_TEST_DEVICE=metal TURBO_TEST_REQUIRE_METAL=1 TURBO_TEST_PRECISION=model \
  cargo test --release --features turbo/metal -p turbo-analysis-embed --test sequence_device -- --nocapture
```

40 test binaries, every one passing, each printing the precision it ran at
(`f32 (model)` or `f32 (exact)`) and the same numbers as every other run,
the per-text lines (the CPU's label and margin) included:

| Scorer | Largest logit difference from the CPU | Labels the same as the CPU's |
|---|---|---|
| `BundleSequenceScorer::score_chunks` | 5.7e-6 (bound 1e-4) | all seven texts; billing, hardware and search all picked |

Every text's CPU margin (2.3e-3 to 0.053) is over twice the bound, so the
test held each text's Metal label to the CPU's. The bundle and its head are
the test's own (docs/receipts/sequence-levelzero-989 says how).

These are correctness runs, not timings. The M2 is a desktop in use, and a
GLiNER2 parity run shared it: the one-minute load was between 6.9 and 28.5
(the log records it before each run). F16 is not covered: every run is F32.
