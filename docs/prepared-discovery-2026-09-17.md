# Prepared device discovery validation, 2026-09-17

This receipt covers the additive device-discovery symbols
(`turboembed_prepared_v1_device_count`, `turboembed_prepared_v1_device_info`)
and a CPU re-validation of the prepared execution path on a GPU-less host. It
does not exercise an Intel GPU: GPU discovery, GPU parity, OpenCL results, and
the native-overhead pilot are outside this receipt and remain covered by the
Machine B (`krick-1`) receipts and their pending re-runs. No artifact was
published or deployed.

## Environment and model

- Host: hosted Linux x86_64 VM, Ubuntu 24.04.4, kernel 6.12.94, no GPU
  (`/dev/dri` absent). CPU reported by OpenVINO as `Intel(R) Xeon(R) Processor`.
- Compiler/toolchain: GCC 13.3, CMake 3.28, glibc 2.39.
- OpenVINO archive `openvino_toolkit_ubuntu24_2026.3.1.22476.56d9685302d_x86_64`;
  runtime build string `2026.3.1-22476-759c5a6ab8c-releases/2026/3`, the same
  build recorded in the [2026-09-14 SDK receipt](intel-prepared-sdk-2026-09-14.md).
- Model: `sentence-transformers/all-MiniLM-L6-v2`, revision
  `1110a243fdf4706b3f48f1d95db1a4f5529b4d41`; source ONNX SHA-256
  `6fd5d72fe4589f189f8ebc006442dbb529bb7ce38f8082112682524616046452` (matches
  the 2026-09-14 receipt). The bundle was provisioned with
  `prepare-native-bundle.py` and the packaged exporter on this host.

## What ran and passed

- SDK, contract-test, and consumer-example builds from
  [`docs/native-sdk.md`](native-sdk.md) commands (the contract executable was
  built but not run: it requires the Intel GPU and fails loud without one).
- Rust, with `--features prepared` against the installed SDK:
  - `device_discovery_and_explicit_selection` (ignored test, run explicitly):
    enumeration returned the single CPU entry with resolved identity and
    capabilities matching an explicitly created context; selecting an absent
    GPU ordinal returned `TE_UNAVAILABLE` with no CPU fallback.
  - `prepared_cpu_reference_execution` (ignored test, run explicitly): explicit
    CPU text execution matched the repository ORT CUDA MiniLM fixture
    (`testdata/reference_embeddings/ort_cuda_minilm_short.json`) within the
    repository parity gates (max 5e-4, RMSE 1e-4); a padded `[2,32]`
    mixed-length batch reproduced both single-row outputs; out-of-limit slot
    shapes failed loud.
  - `prepared_v1_layout_matches_c_header` plus a C11 probe of
    `te_device_info` (size 408, alignment 8, field offsets matching the Rust
    declaration) and the module's compile-fail doc tests.
- The packaged consumer example, with `LD_LIBRARY_PATH` unset: explicit CPU
  printed the discovered CPU device and passed text/prepared parity; default
  GPU selection failed with `TE_UNAVAILABLE`
  ("requested device is unavailable; CPU is never an automatic fallback").
- `cargo fmt --all -- --check`,
  `cargo clippy --locked --workspace --all-targets -- -D warnings`, and the
  ordinary parallel `cargo test --locked --workspace`
  (299 passed, 0 failed, 8 ignored; the prepared feature is off by default).

## Hardware-unverified until re-run on Machine B

- GPU device enumeration (ordinal ordering with a real GPU present, device
  name, OpenCL driver string, `TE_CAP_OPENCL_RESULT`).
- The discovery section of `prepared_contract_test` and the ignored
  `machine_b_gpu_discovery_receipt` Rust test.
- Discovery adds symbols only; the execution path is unchanged from the
  [2026-09-14 performance pilot](intel-prepared-performance-2026-09-14.md),
  whose measurements remain the recorded native-overhead evidence.
