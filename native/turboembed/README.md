# native/turboembed

C++ implementation of the frozen C ABI in [`include/turboembed.h`](../../include/turboembed.h).

Default no-feature compile is **mock smoke**: deterministic `mock-embed`
on **explicit** `CPU` / `OPENVINO_CPU` / `MOCK` only. Catalog aliases
(`minilm`, `bge-*`, …) return `NOT_IMPLEMENTED`. GPU requests without a
compiled provider (`METAL`, `TENSORRT`, OpenVINO NPU; `CUDA` /
`OPENVINO_GPU` / `AUTO` when the matching feature is off) fail at
`turboembed_engine_create` with `UNAVAILABLE` — they never fall back to
CPU or the 8-d FNV mock. `AUTO` is host-default GPU, not "CPU if GPU is
down". Real providers arrive via Cargo features (`ort-cuda`, `genai`)
or the Apple `libTurboEmbed.dylib` — this default link is not the
product.

`--features ort-cuda` (Rust crate) defines `TURBOEMBED_ORT_CUDA`. Catalog
aliases such as `minilm` then call Rust hooks that load ONNX Runtime.
`TURBOEMBED_DEVICE_CUDA` / AUTO use the CUDA EP (`error_on_failure`) and
IoBinding on `turbo_buffer` PINNED mapped tokens + DEVICE hidden,
then DEVICE mean+L2 into a mapped PINNED result row —
never a silent CPU fallback. `TURBOEMBED_DEVICE_CPU` is an explicit
CPU EP path (same ONNX, same mean+L2, HOST arena). See `docs/turboembed.md`.

`--features genai` (Rust crate) also compiles `src/genai.cpp` and defines
`TURBOEMBED_GENAI`. Catalog aliases such as `minilm` then construct
`ov::genai::Tokenizer` + `CompiledModel` on `"GPU"`, `"CPU"`, or `"NPU"`
and rent token/result slabs from turbo_buffer (ZE SHARED on GPU).
`embed_documents` is not used — that API private-allocs every call.
A GPU / NPU request never compiles `"CPU"` and never opens a CPU arena.
NPU create fails loud when the plugin is missing (Machine B: no Intel
NPU). No OVMS. No Python. See `docs/turboembed-genai-ze-machine-b.md`.

On macOS the Rust crate **does not** link this default C++ object — it
links `libTurboEmbed.dylib` (Swift MLX). See `docs/turboembed-swift.md`.

## Build the mock-smoke static lib (no Rust)

```bash
make turboembed-stub
# writes native/turboembed/build/libturboembed.a
```

`stub.cpp` alone is not a complete library: it depends on the `turbo_buffer`
arena and the native WordPiece tokenizer (plus vendored utf8proc). The Make
target compiles and archives all required objects; see the `turboembed-stub`
recipe in the root [`Makefile`](../../Makefile) for the exact object list if
you need to reproduce it in another build system.

## Build via the Rust crate (preferred)

```bash
cargo test -p turboembed
```

`crates/turboembed/build.rs` compiles `src/stub.cpp` **on non-macOS**
(and `src/genai.cpp` when `--features genai`) with the `cc` crate. On
macOS the crate links `libTurboEmbed.dylib` (Swift MLX) and never this
default C++ object. Needs a C++17 compiler (`g++` / `clang++`) on Linux.
GenAI builds also need OpenVINO + OpenVINO GenAI on the loader path
(`source /work/opt/openvino_genai/setupvars.sh`).

```bash
cargo test -p turboembed --features genai
# or: make test-turboembed-intel
```
