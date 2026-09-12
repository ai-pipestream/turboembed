# native/turboembed

C++ implementation of the frozen C ABI in [`include/turboembed.h`](../../include/turboembed.h).

Default compile is a **linkable stub**: deterministic `mock-embed` plus
`NOT_IMPLEMENTED` for catalog aliases / provider registration.

`--features genai` (Rust crate) also compiles `src/genai.cpp` and defines
`TURBOEMBED_GENAI`. Catalog aliases such as `minilm` then construct
`ov::genai::TextEmbeddingPipeline(models_path, device, config)` with
`device` `"GPU"` or `"CPU"` (official sample strings) and call
`embed_documents`. A GPU request never compiles `"CPU"`. No OVMS. No Python.
Apple still exports the same header from Swift (`docs/turboembed-swift.md`).

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

`crates/turboembed/build.rs` compiles `src/stub.cpp` (and `src/genai.cpp`
when `--features genai`) with the `cc` crate. Needs a C++17 compiler
(`g++` / `clang++`). GenAI builds also need OpenVINO + OpenVINO GenAI
on the loader path (`source /work/opt/openvino_genai/setupvars.sh`).

```bash
cargo test -p turboembed --features genai
# or: make test-turboembed-intel
```
