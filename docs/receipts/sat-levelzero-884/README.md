# SaT on Level Zero: twenty parity runs per precision (#884)

Base: `29214c70` (origin/main at branch time); no code changes. Device:
Intel Arc Pro B70 through the Level Zero backend, f32 at both
precisions, release builds, with `TURBO_TEST_REQUIRE_LEVELZERO=1`, so a
run that did not reach the device fails. `parity.log` is the run, its
bundle and fixture paths shortened to their names; the B70 had no other
compute client and its CI runner containers no cargo, test or nextest
process when it began. The run paused from 23:21 to 00:13 UTC while
another receipt's timings held the card, and resumed.

Twenty runs at each of `TURBO_TEST_PRECISION` `model` and `exact` of

```
TURBO_TEST_DEVICE=levelzero TURBO_TEST_REQUIRE_LEVELZERO=1 TURBO_TEST_PRECISION=model \
  TURBO_TEST_SAT_BUNDLE=<sat-1l-sm> TURBO_TEST_SAT_FIXTURES=<its three wtpsplit fixtures> \
  cargo test --release --locked -p turbo-sat --features turbo/levelzero --test device --test parity \
  -- --nocapture --test-threads=1
```

80 test binaries (`tests/device.rs` and `tests/parity.rs`, each run),
every one passing, each line printing the precision it ran at
(`f32 (model)` or `f32 (exact)`), and every repeat printing the same
numbers. `a_hailo_session_refuses_sat` is ignored here, as it needs a
Hailo host.

Bundles: `testdata/tiny-sat-bundle` (2 layers, width 32) and
`segment-any-text/sat-1l-sm` at `00a783439c201b20ee91ee1624dc817f1bf0e578`,
the bundle docs/receipts/sat-884.md names (manifest `c3a0dbcc`;
`turbo-bundle verify` exits 0), with the wtpsplit fixtures made from the
same weights (`t0.25-s16-b62`, `t0.25-s64-b510`, `t0.5-s7-b30`).

`tests/device.rs`, the device against the CPU, largest probability
difference at both precisions (bound 1e-4); no token crossed the
threshold on one side only:

| Bundle | (0.5, 16, 62) | (0.5, 7, 30) | (0.25, 62, 62) | Packed, max_batch | Packed, 3 rows |
|---|---:|---:|---:|---:|---:|
| tiny | 8.9e-7 | 6.0e-7 | 8.9e-7 | 6.9e-7 (284 windows) | 5.1e-7 |
| sat-1l-sm | 1.1e-6 | 1.1e-6 | 1.1e-6 | 1.1e-6 (208 windows) | 1.2e-6 |

`tests/parity.rs`, the device against wtpsplit (`PROB_TOLERANCE` 1e-3),
all six tests at both precisions, largest probability difference:

| Fixture | Tiny bundle | sat-1l-sm |
|---|---:|---:|
| stride 16, block 62 | 3.2e-4 | 4.2e-4 (threshold 0.25) |
| stride 7, block 30 | 2.5e-4 | 3.7e-4 (threshold 0.25) |
| stride 62, block 62 | 1.2e-4 (one character near the threshold) | |
| stride 64, block 510 | | 1.6e-4 |

The lookahead test checks, on a device, that a model of limited
lookahead is refused. The host's one-minute load was between 1.1 and
42.9 over the run (the log records it before each); these are
correctness runs, not timings. `fastest` was not run here: the tiny
fixtures fail it against wtpsplit's 1e-3 on every device
(docs/receipts/sat-884.md).

## Not done

- Timings: the SaT `speed` test on this card follows as its own
  receipt.
