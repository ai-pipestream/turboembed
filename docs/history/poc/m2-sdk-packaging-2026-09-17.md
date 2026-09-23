# M2 SDK packaging validation (CPU path), 2026-09-17

This receipt covers the roadmap M2 packaging work: the scripted release
archive, manifest-pinned model provisioning, and the clean-consumer acceptance
run. It was produced on a GPU-less build VM, so it validates the complete
install and explicit-CPU path only. Intel GPU coverage remains gated on the
Machine B (the Intel Arc B70 host) reference machine, as listed at the end.
No artifact was
published to a hosted registry.

## Environment

- Host: cloud build VM, Linux x86_64, Ubuntu 24.04.4 LTS, glibc 2.39,
  4 vCPU, no GPU, no OpenCL ICD beyond the loader.
- Toolchain: GCC 13.3.0 (`CC=gcc CXX=g++`; the VM's `c++` alternative points
  at clang and was not used), CMake 3.28.3, Python 3.12.3.
- OpenVINO runtime: archive distribution
  `openvino_toolkit_ubuntu24_2025.3.0.19807.44526285f24_x86_64.tgz`
  (SHA-256 `de0d5e16b161efea013a5c017e3b2bce1191ca009a1947d392ccab8ed9d0f6e4`),
  reporting `2025.3.0-19807-44526285f24-releases/2025/3`. The GPU-qualified
  baseline on the B70 host remains `2026.3.1-22476-759c5a6ab8c` per the
  [2026-09-14 receipt](intel-prepared-sdk-2026-09-14.md); this run additionally
  demonstrates the packaging path against the runtime pinned for hosted CI.
- Source: this branch at the M1 merge base `3492119`, plus the M2 packaging
  changes under review.

## Release archive

`scripts/make-sdk-release.sh --openvino-root <extracted archive>` produced
`turboembed-prepared-sdk-1.0.0-linux-x86_64.tar.gz` from a clean staging
prefix (29 files, 11 symlinks). The dynamic-symbol gate passed: `nm -D`
reported exactly the 19 `turboembed_prepared_v1_*` functions declared by
`include/turboembed_prepared.h`, all bound to the `TURBOEMBED_PREPARED_1` ELF
version, and the list was installed as
`share/turboembed/exported-symbols.txt`. `share/turboembed/sdk-manifest.json`
recorded the version, source commit, OpenVINO build string, and SHA-256 of
every installed file.

## Model provisioning through the committed manifest

`cargo run -p inferstream-fetch -- --prepared minilm` downloaded the pinned
`sentence-transformers/all-MiniLM-L6-v2` revision
`1110a243fdf4706b3f48f1d95db1a4f5529b4d41` into `models/prepared-src/minilm/`.
All four files hashed identically to the values recorded independently in the
[2026-09-14 receipt](intel-prepared-sdk-2026-09-14.md) (source ONNX
`6fd5d72f…`, tokenizer `be50c362…`, config `953f9c0d…`, model card
`dcd602d2…`), confirming the committed pins against live upstream. Re-running
the fetch reported all files cached and verified; `--verify-only` passed.

`scripts/provision-minilm-bundle.sh` re-verified the sources against
`models/manifests/prepared-sources.json`, then produced a bundle whose
`bundle.json` records the pinned identity, revision, license, source hash, and
converter build `2025.3.0-19807-44526285f24-releases/2025/3`. The 8 Python
provisioning unit tests passed
(`python3 -m unittest scripts.tests.test_prepare_native_bundle`).

## Clean-consumer acceptance (`cpu-only` mode)

`scripts/sdk-consumer-acceptance.sh <archive> <bundle> cpu-only` passed in a
fresh temporary directory using `env -i PATH=/usr/bin:/bin` for every consumer
configure, build, and run:

- Archive SHA-256 and the full `sdk-manifest.json` file/symlink set verified.
- The installed C example configured and built as a separate CMake project
  against the extracted prefix only (`find_package(TurboEmbedPrepared)`).
- `ldd` resolved `libturboembed_prepared.so.1`, `libopenvino.so.2530`, and
  `libtbb.so.12` from the extracted prefix; the OpenCL ICD loader, C++
  runtime, and glibc resolved from the operating system; nothing unresolved.
- Explicit CPU run embedded text and pinned prepared tokens, 384 dimensions,
  norm `1.000000`, `prepared/text=PASS`.
- Device policy: the example's default GPU selection failed loudly with
  status 4 (`requested device is unavailable; CPU is never an automatic
  fallback`), exit code 1. No silent CPU fallback occurred.
- A bundle copy with one flipped byte in `openvino_model.bin` was rejected at
  model load (integrity error), exit code 1.
- Bounded wall-clock check: five repeated CPU example runs (discovery, model
  load, one text execute, four prepared executes each) completed in
  0.90–1.00 s per run against a 180 s finish line on this 4-vCPU VM. These are
  machine-local measurements, not the roadmap's GPU performance budget.

## Rust bindings against the packaged library

With `TURBOEMBED_PREPARED_SDK` pointing at the extracted archive and
`TURBOEMBED_PREPARED_BUNDLE` at the provisioned bundle,
`cargo test -p turboembed --features prepared --test prepared_sdk -- --ignored
device_discovery_and_explicit_selection prepared_cpu_reference_execution`
passed (2 tests). This includes numerical parity of the packaged CPU path
against the committed cross-runtime reference fixture and the explicit
absent-GPU failure check.

## Workspace checks

- `cargo test --locked --workspace` (default parallel): see the PR validation
  summary for the counts recorded on this branch.
- `cargo fmt --all -- --check` and default-feature workspace Clippy: recorded
  in the same summary.

## Remaining Machine B (the Intel Arc B70 host) gates

Everything GPU-specific in this packaging remains hardware-unverified until
run on the reference Intel Battlemage host:

1. Build the release archive there against the qualified `2026.3.1` runtime
   and re-run `scripts/sdk-consumer-acceptance.sh <archive> <bundle> gpu`
   (default GPU selection must succeed, including the leased-OpenCL example
   path).
2. Re-run the native contract executable (`-DTE_BUILD_CONTRACT_TEST=ON`)
   against a bundle provisioned by `scripts/provision-minilm-bundle.sh`.
3. Re-run the ignored Rust suites `--test prepared_sdk -- --ignored`
   (all five tests) and refresh `machine_b_gpu_discovery_receipt`.
4. Record the GPU wall-clock and overhead numbers under the roadmap's
   performance budget; the CPU timings above do not stand in for them.
