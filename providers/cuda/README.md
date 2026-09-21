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
- `libcudart.so` from that toolkit.
- Network access on the first build: the `ort` crate downloads the ONNX
  Runtime 1.28 CUDA 13 bundle for `x86_64-unknown-linux-gnu` into
  `~/.cache/ort.pyke.io`. There is no prebuilt CUDA bundle for
  `aarch64-unknown-linux-gnu`; Jetson builds point `ORT_LIB_LOCATION` at a
  local ONNX Runtime build (see `jetson/`, planned).

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
  CPU execution provider. On `krick` these come from the NVIDIA PyPI wheels
  unpacked into `.libs/nvidia/lib`.

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

Bundles come from `turbo-bundle import` (see `docs/bundles.md`). Results for
`krick` (RTX 4080 SUPER) are in `testdata/receipts/turbo/cuda-2026-09-21.json`.

## Status

Every cell is `EXPERIMENTAL`: precision matches the FP32 reference to cosine
1.000, but the matched-native benchmark and the Jetson receipt required for
`SUPPORTED` are not recorded yet. Sessions of one model share the ONNX
Runtime session under a lock, so two Turbo sessions on the same model run
one at a time; `user_compute_stream` import (`TURBO_CAP_EXTERNAL_QUEUE`) and
CUDA graphs are not implemented.
