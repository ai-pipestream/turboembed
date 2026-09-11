# Native MLX (Apple)

`inferstream-apple` runs **in-process native MLX** on Metal. There is no
Python interpreter on Embed, Tokenize, or StreamInfer.

## How it is linked

```
inferstream-apple (Rust)
  └─ crates/backend-apple
       └─ FFI (native/mlx-engine/include/mlx_engine.h)
            └─ libMlxEngine.dylib  (Swift, type: dynamic)
                 ├─ mlx-swift          Apple MLX (C++/Metal)
                 ├─ mlx-swift-lm       MLXLLM + MLXEmbedders
                 └─ swift-transformers tokenizer.json loader
```

`crates/backend-apple/build.rs` runs `swift build -c release` in
`native/mlx-engine` on macOS and links `libMlxEngine.dylib` with an rpath
into `.build/release`. Linux CI skips the Swift build; the crate compiles
as a stub and every call is `Unavailable`.

Weights are **local directories** (`models/mlx/<alias>/`) produced by
`cargo xtask fetch --mlx` against `models/manifests/mlx.json` (pinned HF
revision + SHA-256). The runtime does not download and does not import
Python.

Tokenize / Detokenize use the Rust `tokenizers` crate and
`tokenizer_dir` / `tokenizer.json`.

Engine-side decode speed is reported on the final StreamInfer chunk as
`decode_tokens_per_second` (generation tokens / generate-loop seconds
inside mlx-swift-lm, not gRPC wall-clock).
