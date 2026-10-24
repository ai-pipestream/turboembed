# Native Hailo receipt validation test

These saved results are from source revision
`6e0c8f4bd90ebae54ee859090c4d0b54191fc232`, on 2026-10-24.
They establish execution of the acceptance and rejection test on the two
devices below. They are separate from the timing records at
`94a833f60389b886d6ce103fe2ee2a08063ee75d` and do not claim a hardware
rerun of subsequent documentation changes or main merges.

| Architecture label | Device name | Device class | Runtime |
|---|---|---|---|
| pi5-hailo8 | Hailo-8 | DEVICE_NPU | HailoRT 4.23.0 |
| pi5-hailo10h | Hailo-10H | DEVICE_NPU | HailoRT 5.1.1 |

The names and classes are recorded by the paired library runs in
[hailo8.json](hailo8.json) and [hailo10h.json](hailo10h.json). The test
output below reports the assertion result, not a device enumeration.

## Invocation and required inputs

Reproduction command, with private paths supplied through environment
variables:

```sh
cargo test --locked --release -p turbo-bench --features hailo \
  --test hailo_native -- --ignored --nocapture
```

Set `TURBO_HAILO_BUNDLE` to the corresponding deployed MiniLM bundle and
`TURBO_HAILO_RECEIPT` to the native runner JSON for that bundle. This test
uses these two variables, not `TURBO_TEST_BUNDLE`. Device `hailo` and
`TURBO_PRECISION_MODEL` are fixed in the test's `Plan`; it does not read
`TURBO_TEST_DEVICE` or `TURBO_TEST_PRECISION`. No
`TURBO_TEST_REQUIRE_*` variable is used: `--ignored` explicitly opts in,
both input variables are required, and a missing device or failed run
panics through `unwrap`. There is no successful skip in the test body.

## Saved output

pi5-hailo8:

```text
running 1 test
test native_receipt_matches_the_real_library_and_rejects_changed_inputs ... ok
test result: ok. 1 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.92s
```

pi5-hailo10h:

```text
running 1 test
test native_receipt_matches_the_real_library_and_rejects_changed_inputs ... ok
test result: ok. 1 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.83s
```

Each run asserts a minimum cosine of 0.93, 4096 computed positions, and
acceptance of the matching native receipt. It then rejects 13 altered
receipt fields, including device, bundle and HEF identity, rows, shape,
case coverage, CLI provenance, sample duration and throughput. These are
negative-input assertions within a passing test, not a claimed red run
against the previous implementation. An earlier invocation failed because
its receipt file was absent; that setup failure is not correctness proof.
