# CUDA provider

`turbo-provider-cuda` runs text embedding, reranking, sequence classification,
and token classification on NVIDIA GPUs through the ONNX Runtime CUDA
execution provider, with the provider's own CUDA kernels for pooling, L2
normalization, sigmoid, and softmax. See the crate documentation in
`src/lib.rs` for the data path and the bundle contract, and
`PLAN.md` section 10 (P3) for scope.

## What it needs to build

- A CUDA toolkit with `nvcc` and `cuda_runtime.h` (`CUDA_PATH`, else
  `/usr/local/cuda`, else `/usr`). The kernels are compiled for
  `TURBO_CUDA_ARCHS` (default `87;89`: Jetson Orin and Ada) plus PTX for the
  last architecture listed.
- `libcudart.so` from that toolkit. The provider's own kernels link the
  runtime of the toolkit that compiled them (`libcudart.so.12` for the
  12.4 toolkit on the RTX 4080 SUPER host), while the ONNX Runtime execution
  provider it
  loads needs the CUDA 13 user-space libraries below; both must be present
  on the consumer machine, and `ldd` on the packaged library names the
  first.
- Network access on the first build: with the default `download` feature
  the `ort` crate downloads the ONNX Runtime 1.28 CUDA 13 bundle for
  `x86_64-unknown-linux-gnu` into `~/.cache/ort.pyke.io`. There is no
  prebuilt CUDA bundle for `aarch64-unknown-linux-gnu`; a Jetson build
  passes `--no-default-features` and points `ORT_LIB_LOCATION` at a
  directory holding `lib/libonnxruntime.so` plus the CUDA execution provider
  libraries (ONNX Runtime 1.17 or newer; the Orin Nano uses 1.24.0), with
  `TURBO_CUDA_ARCHS=87`.

```bash
cargo build -p turbo-provider-cuda --release
```

The crate is a workspace member, so `cargo build --workspace` needs the
toolkit too. CI excludes it (`--exclude turbo-provider-cuda`) because the
hosted runners have no CUDA.

## What it needs at runtime

- `libturbo_provider_cuda.so` (this crate).
- `libonnxruntime_providers_cuda.so` and `libonnxruntime_providers_shared.so`
  next to it. With the `copy-dylibs` feature (on by default here) cargo links
  them into `target/<profile>/` so tests find them; a packaged install ships
  them alongside the provider.
- The CUDA 13 user-space libraries the execution provider links: cuBLAS,
  cuBLASLt, cuDNN 9, NVRTC, and the CUDA 13 runtime. Either put them on the
  loader path or name their directory with the context option `cuda_lib_dir`
  or the environment variable `TURBO_CUDA_LIB_DIR`; the provider preloads
  every `lib*.so*` there before creating a session. A missing library is
  `TURBO_E_DEVICE_UNAVAILABLE` at context creation, never a fallback to the
  CPU execution provider. On the RTX 4080 SUPER host these come from the
  NVIDIA PyPI wheels unpacked into `.libs/nvidia/lib`.

## Running the crate tests

```bash
TURBO_CUDA_LIB_DIR=$PWD/.libs/nvidia/lib TURBO_LIVE_BUNDLE=~/opt/bundles/minilm-onnx \
cargo test -p turbo-provider-cuda
```

`tests/kernels.rs` checks the pooling, sigmoid, and softmax kernels against
sequential CPU references; `tests/provider.rs` loads the provider by path
and checks devices, capability cells, buffers, imports, and session
counters; `tests/wordpiece.rs` checks word spans under truncation. Each
test skips with a printed reason when the device, the library, or the
bundle is absent.

## Running the live tests

```bash
TURBO_LIVE_LIB=$PWD/target/debug/libturbo_provider_cuda.so \
TURBO_LIVE_PROVIDER=cuda TURBO_LIVE_ORDINAL=0 \
TURBO_CUDA_LIB_DIR=$PWD/.libs/nvidia/lib \
TURBO_LIVE_BUNDLE=~/opt/bundles/minilm-onnx \
TURBO_LIVE_RERANK_BUNDLE=~/opt/bundles/rerank-onnx \
TURBO_LIVE_CLASSIFY_BUNDLE=~/opt/bundles/sst2-onnx \
TURBO_LIVE_NER_BUNDLE=~/opt/bundles/ner-onnx \
cargo test -p turbo-conformance --test live_embed --test live_tasks -- --test-threads=1
```

`crates/turbo-conformance/tests/live_cuda.rs` is this provider's own live
file; add `--test live_cuda` to the line above to run it. It covers the four
bundle kinds on one context with the placement each stage really ran at, the
error paths (an over-long pair, a token id outside the vocabulary, batch and
sequence limits, a task written to the wrong model kind, a `prompt_role` the
bundle has no prefix for), the H2D/D2H byte accounting across repeated runs,
two models running concurrently on one device, and the refusal to resolve a
device ordinal that does not exist. See `docs/testing.md`'s "Live provider
tests" section.

Bundles come from `turbo-bundle import` (see `docs/bundles.md`). Results for
the RTX 4080 SUPER host are in
`testdata/receipts/turbo/cuda-2026-09-21.json`.

## On a Jetson Orin Nano

The Jetson has no prebuilt ONNX Runtime CUDA bundle for
`aarch64-unknown-linux-gnu`, so the provider builds against JetPack's own
CUDA and a locally staged ONNX Runtime 1.24.0. This is the environment every
command on the Orin Nano runs under:

```bash
export PATH=$HOME/.cargo/bin:/usr/local/cuda/bin:$PATH
export ORT_LIB_LOCATION=$HOME/ort-1.24/lib   # holds libonnxruntime.so and the CUDA EP libraries
export ORT_PREFER_DYNAMIC_LINK=1
export CUDA_PATH=/usr/local/cuda
export TURBO_CUDA_ARCHS=87                   # Orin only
export LD_LIBRARY_PATH=$HOME/ort-1.24/lib:/usr/local/cuda/lib64
export TURBO_CUDA_LIB_DIR=/usr/local/cuda/lib64
```

Every `cargo` command then adds `--no-default-features` so the `ort` crate
does not try to download a bundle that does not exist for this target:

```bash
cargo build --release -p turbo-provider-cuda --no-default-features
cargo test -p turbo-provider-cuda --no-default-features
```

A non-login `ssh` shell does not have cargo on `PATH`, hence the first line.
Results for the Orin Nano are in
`testdata/receipts/turbo/bench/cuda-orin-nano-*.json` and
`compare-cuda-orin-nano-embed-2026-09-22.json`; `docs/testing.md`'s "The CUDA
provider on the Jetson Orin Nano" section lists what passes there.

## Status

Every cell is `EXPERIMENTAL`. Precision matches the FP32 reference to cosine
1.000 and the matched-native benchmark is now recorded on both machines
(1.04x to 2.64x on the RTX 4080 SUPER and 0.96x to 1.04x on the Orin Nano,
of ONNX Runtime CUDA alone, both SUPPORTED), but no conformance or precision
receipt file is committed for the Orin Nano yet, so the cells stay `EXPERIMENTAL` until one is.

Sessions of one model share the ONNX Runtime session under a lock
(`ort::Session::run_binding` takes the session exclusively), so two Turbo
sessions on the same model run one at a time rather than concurrently as
`PLAN.md` section 4 describes; the alternative, one ONNX Runtime session per
Turbo session, would give up weight sharing. `user_compute_stream` import
(`TURBO_CAP_EXTERNAL_QUEUE`) and CUDA graphs are not implemented.
