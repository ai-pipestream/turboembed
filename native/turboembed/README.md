# native/turboembed

C++ implementation of the frozen C ABI in [`include/turboembed.h`](../../include/turboembed.h).

Default compile is a **linkable stub**: deterministic `mock-embed` on
**explicit** `CPU` / `OPENVINO_CPU` / `MOCK` only. Catalog aliases
(`minilm`, `bge-*`, …) return `NOT_IMPLEMENTED`. GPU requests without a
compiled provider (`METAL`, `TENSORRT`, OpenVINO NPU; `CUDA` /
`OPENVINO_GPU` / `AUTO` when the matching feature is off) fail at
`turboembed_engine_create` with `UNAVAILABLE` — they never fall back to
CPU or the 8-d FNV mock. `AUTO` is host-default GPU, not "CPU if GPU is
down".

`--features ort-cuda` (Rust crate) defines `TURBOEMBED_ORT_CUDA`. Catalog
aliases such as `minilm` then call Rust hooks that load ONNX Runtime.
`TURBOEMBED_DEVICE_CUDA` / AUTO use the CUDA EP (`error_on_failure`) and
IoBinding device buffers — never a silent CPU fallback.
`TURBOEMBED_DEVICE_CPU` is an explicit CPU EP path (same ONNX, same mean+L2).
See `docs/turboembed.md`.

`--features genai` (Rust crate) also compiles `src/genai.cpp` and defines
`TURBOEMBED_GENAI`. Catalog aliases such as `minilm` then construct
`ov::genai::TextEmbeddingPipeline(models_path, device, config)` with
`device` `"GPU"` or `"CPU"` (official sample strings) and call
`embed_documents`. A GPU request never compiles `"CPU"`. No OVMS. No Python.

On macOS the Rust crate **does not** link this stub — it links
`libTurboEmbed.dylib` (Swift MLX). See `docs/turboembed-swift.md`.

## Build the static stub (no Rust)

```bash
make turboembed-stub
# writes native/turboembed/build/libturboembed.a
```

Or by hand:

```bash
c++ -std=c++17 -fPIC -O2 -I include \
  -c native/turboembed/src/stub.cpp \
  -o native/turboembed/build/stub.o
ar rcs native/turboembed/build/libturboembed.a native/turboembed/build/stub.o
```

Shared object (optional):

```bash
c++ -std=c++17 -fPIC -shared -O2 -I include \
  native/turboembed/src/stub.cpp \
  -o native/turboembed/build/libturboembed.so
```

## Build via the Rust crate (preferred)

```bash
cargo test -p turboembed
```

`crates/turboembed/build.rs` compiles `src/stub.cpp` **on non-macOS**
(and `src/genai.cpp` when `--features genai`) with the `cc` crate. On
macOS the crate links `libTurboEmbed.dylib` (Swift MLX) and never this
stub. Needs a C++17 compiler (`g++` / `clang++`) on Linux. GenAI builds
also need OpenVINO + OpenVINO GenAI on the loader path
(`source /work/opt/openvino_genai/setupvars.sh`).

```bash
cargo test -p turboembed --features genai
# or: make test-turboembed-intel
```
