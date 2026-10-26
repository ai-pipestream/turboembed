# Narrow attention over SIMD groups on the M2 (#1406)

Branch head before this receipt: `442ba1f6b9c9539ce1799b4a421d85fc37ba365a`; main: `9552e94d6`
(the branch's base). Apple M2, release builds, `TURBO_TEST_REQUIRE_METAL=1`.

## Repeat runs

`runs.log` is 20 rounds of `heads_past_64_or_not_a_multiple_of_8_run_f16_attention`,
the branch and then main in each round. Each run's line records the UTC time, the
1, 5 and 15-minute load averages, and the GPU's "Device Utilization %" just before it.
The command was:

```
TURBO_TEST_REQUIRE_METAL=1 <metal_f16 test binary> heads_past_64_or_not_a_multiple_of_8_run_f16_attention --nocapture
```

- branch: 20 of 20 passed. 0 failed. The heads of 136 ran their largest shape every time
  with no error passed over.
- main: 20 of 20 passed. 0 runs stopped. Main's test passes over the interactivity stop
  and prints "stopped by macOS"; no run printed it.

The 1-minute load before the runs was 5.6 to 17.1, and the GPU 39 to 74% busy.

Main did not hit the stop in these 20 rounds. The stop depends on how busy the desktop
holds the GPU. Under heavier load it was hit, and those runs are the red:

- at `d0aae695` (#1365, before main was merged into it), 4 of 6 runs of the same test
  with the GPU 90 to 99% busy;
- on main `9552e94d6`, 2 of 4 runs of the timing harness in PR #1426's body, with the
  1-minute load 7.6 to 26.

At this head the branch was not stopped in any run: those 20, the 8 consecutive
`metal_f16` runs at 98% GPU, and the 4 harness runs.

## Byte identity

`core/tests/metal_narrow_bits.rs` (ignored) writes the vectors of two-layer BERTs whose
heads run the narrow kernel (hidden 272 with heads of 136, 104 with heads of 26, 768 with
heads of 128) at EXACT and FASTEST, at 8 x 128 and 32 x 256, as F32 bits. Built at main and
at the branch:

```
METAL_BITS_OUT=<file> cargo test --release -p turbo --features metal --test metal_narrow_bits -- --ignored
```

The two files are identical, SHA-256
`2eba5d8948e82f988e987f46b48988a0a7dd3423731ccc6bd799ff6d641a3389`.
