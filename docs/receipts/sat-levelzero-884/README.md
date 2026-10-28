# SaT on Level Zero: twenty parity runs per precision (#884)

Base: `29214c70` (origin/main at branch time); no code changes. Device:
Intel Arc Pro B70 through the Level Zero backend, f32 at both
precisions, release builds, with `TURBO_TEST_REQUIRE_LEVELZERO=1`, so a
run that did not reach the device fails. `parity.log` is the run, its
bundle and fixture paths shortened to their names; the B70 had no other
compute client and its CI runner containers no cargo, test or nextest
process when it began, on 26 Oct at 22:52 UTC. The run paused from 26
Oct 23:21 to 27 Oct 00:13 UTC while another receipt's timings held the
card, and resumed, ending at 00:29 UTC.

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
| stride 7, block 30 | 2.5e-4 | 3.7e-4 (threshold 0.5) |
| stride 62, block 62 | 1.2e-4 (one character near the threshold) | |
| stride 64, block 510 | | 1.6e-4 |

The lookahead test checks, on a device, that a model of limited
lookahead is refused. The host's one-minute load was between 1.1 and
42.9 over the run (the log records it before each); these are
correctness runs, not timings. `fastest` was not run here against
wtpsplit: on CUDA and on Level Zero the tiny fixtures fail it against
wtpsplit's 1e-3 (docs/receipts/sat-884.md); on Metal (#1450) their
largest differences, 3.3e-4, 1.2e-4 and 2.5e-4, are within it.

## Timings

`timings.log` is the run: twenty release runs of `tests/speed.rs` for
each bundle at each of `model` and `fastest`, 80 in all, on 28 Oct from
00:27 UTC, at `1cc14855` (origin/main `e6a9693f` with a docs and
records commit; SaT's code is main's):

```
TURBO_TEST_DEVICE=levelzero TURBO_TEST_REQUIRE_LEVELZERO=1 TURBO_TEST_PRECISION=model \
  TURBO_SPEED_RUNS=7 [TURBO_SAT_BUNDLE=<sat-1l-sm>] \
  cargo test --release --locked -p turbo-sat --features turbo/levelzero --test speed -- --ignored --nocapture
```

Each run alternates seven passes on the CPU (at `model`, the host's 32
threads) with seven on the B70 at the tier named, over the same 36
documents, and gives the median of each side's seven; the table is over
the twenty runs' medians (min, mean, p50, p99), in ms a pass. Before
each run the B70 had no other compute client and its CI runner
containers no build or test process; during each, no other xe client
had compute engine time. The host's one-minute load, before and after
each run, was 0.9 to 7.9 for the tiny bundle and 1.0 to 30.3 for
`sat-1l-sm`, whose runs met other work on the host for part of the
time, which shows in the CPU's wider spread at `model`.

| Bundle | Tier | CPU: min, mean, p50, p99 | B70: min, mean, p50, p99 | CPU p50 over B70 p50 (runs) | B70 tokens/s, p50 |
|---|---|---|---|---|---|
| tiny (16464 tokens, block 62, stride 62) | `model` | 5.6, 5.9, 5.8, 6.6 | 5.4, 5.5, 5.5, 5.7 | 1.05x (1.02 to 1.18) | 2,988,000 |
| tiny | `fastest` | 5.7, 6.3, 6.2, 7.3 | 5.4, 5.5, 5.5, 5.8 | 1.12x (1.04 to 1.30) | 2,999,000 |
| sat-1l-sm (11612 tokens, block 510, stride 64) | `model` | 195.5, 239.2, 231.1, 321.0 | 61.0, 66.5, 66.2, 70.8 | 3.49x (3.02 to 4.74) | 175,000 |
| sat-1l-sm | `fastest` | 193.4, 197.1, 197.9, 202.9 | 33.7, 34.7, 34.8, 37.2 | 5.71x (5.22 to 5.87) | 334,000 |

Above 1 the B70 is the faster, by that factor. Both sides found the
same sentences in every run: 9220 (tiny) and 676 (`sat-1l-sm`). On the
tiny model of two layers of width 32 the device is mostly per-run
overhead; `sat-1l-sm` is the one that says what the card does. These
supersede docs/receipts/sat-884.md's Level Zero speed rows, taken while
a desktop client held the render engine about 68 percent busy.

## Not done

- Timings on a quiet host for `sat-1l-sm` at `model`: its host load
  reached 30.3.
