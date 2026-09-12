# native/turboembed

C++ implementation of the frozen C ABI in [`include/turboembed.h`](../../include/turboembed.h).

Today this is a **linkable stub**: deterministic `mock-embed` plus
`NOT_IMPLEMENTED` for catalog aliases / provider registration. Enough for
the Rust crate to call. Later this directory grows the nvidia (ORT) and
intel (OpenVINO GenAI) providers; Apple keeps the same header and exports
the symbols from Swift (`docs/turboembed-swift.md`).

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
and links `libturboembed_stub.a`. On macOS the crate links
`libTurboEmbed.dylib` (Swift MLX) and never this stub — see
`docs/turboembed-swift.md`. Needs a C++17 compiler (`g++` / `clang++`)
on Linux.
