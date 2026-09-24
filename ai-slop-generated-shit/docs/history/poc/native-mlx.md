# Native MLX (Apple) — legacy Rust FFI notes

The **supported** Mac serve path is the all-Swift gRPC server. See
[`docs/swift-apple.md`](swift-apple.md).

This page describes the **legacy** Rust `inferstream-apple` binary, which
links mlx-swift through a C ABI. It is kept for CI type-checking only.

```
inferstream-apple (legacy Rust)
  └─ crates/backend-apple
       └─ FFI (native/mlx-engine/include/mlx_engine.h)
            └─ libMlxEngine.dylib  (Swift, type: dynamic)
                 ├─ mlx-swift          Apple MLX (C++/Metal)
                 ├─ mlx-swift-lm       MLXLLM + MLXEmbedders
                 └─ swift-transformers tokenizer.json loader
```

`crates/backend-apple/build.rs` runs `swift build -c release` in
`native/mlx-engine` on macOS and links `libMlxEngine.dylib` with an rpath
into `.build/release`. It then runs `native/mlx-engine/build-metallib.sh`
(POSIX + `xcrun metal` / `metallib`, no Python) so `mlx.metallib` sits
next to the dylib. MLX's C++ `current_binary_dir()` (`dladdr` on the Cmlx
image) loads that file — SwiftPM does not emit it for a dylib consumed
from Rust. Linux CI skips the Swift build; the crate compiles as a stub
and every call is `Unavailable`.

On a Mac whose `xcode-select` points at Command Line Tools, `build-metallib.sh`
sets `DEVELOPER_DIR` to `/Applications/Xcode.app/Contents/Developer` (Xcode 26
also needs `xcodebuild -downloadComponent MetalToolchain` once). `fence.metal`
is skipped when the Metal dialect is older than MLX expects; the other
default kernels still load.

Weights are **local directories** (`models/mlx/<alias>/`) produced by
`cargo xtask fetch --mlx` against `models/manifests/mlx.json` (pinned HF
revision + SHA-256). The runtime does not download and does not import
Python.

Tokenize / Detokenize use the Rust `tokenizers` crate and
`tokenizer_dir` / `tokenizer.json`.

Engine-side decode speed is reported on the final StreamInfer chunk as
`decode_tokens_per_second` (generation tokens / generate-loop seconds
inside mlx-swift-lm, not gRPC wall-clock).
