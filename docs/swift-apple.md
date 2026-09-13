# All-Swift Apple server

`inferstream-apple` on macOS is a **native Swift gRPC server**. It implements
the same wire contract as the Rust nvidia/intel binaries and loads the same
logical aliases (`minilm`, `default-llm`, `qwen-0.5b`, `qwen-7b`, …).

There is **no Rust serve process**, **no Rust→Swift FFI**, and **no Python**
on this path.

```
clients (grpcurl / OIP / inferstream.v1)
        │
        ▼
inferstream-apple          Swift executable (grpc-swift 2)
  ├─ inference.GRPCInferenceService     proto/open_inference_grpc.proto
  ├─ inferstream.v1.InferstreamService  proto/inferstream_extension.proto
  ├─ catalog aliases                    config/catalog.toml + config/apple.toml
  ├─ TurboEmbedBackend (catalog embeds) include/turboembed.h → libTurboEmbed.dylib
  └─ MlxBackend (LLM generate / tokenize)
       ├─ mlx-swift / mlx-swift-lm      Metal
       └─ swift-transformers            tokenizer.json
```

Catalog embed aliases (`minilm`, … — any `mlx` row with `pooling`) call the
same TurboEmbed C ABI as nvidia/intel. LLM aliases stay on `MlxBackend`
so `ModelStreamInfer` is unchanged. Missing Metal fails at engine create.

The legacy Rust binary (`crates/arch-apple`, FFI to
`native/mlx-engine/libMlxEngine.dylib`) is type-check only. See
`native/mlx-engine/LEGACY.md`.

## Proto sharing

`proto/` is the single source of truth. Rust (`crates/protocol/build.rs`)
compiles those files with tonic. The Swift package copies them into
`swift/Sources/InferstreamApple/Protos/` so the grpc-swift
`GRPCProtobufGenerator` plugin can see them (the plugin only scans the
target directory):

```bash
./scripts/sync-proto.sh          # refresh Swift copies after editing proto/
./scripts/sync-proto.sh --check  # CI / Make: fail if they drifted
```

Do not edit the copies under `swift/` by hand.

## Build and run

On a Mac with Xcode / Swift 6.2+ and the fetched MLX weights:

```bash
./scripts/setup-mlx.sh                 # make fetch-mlx + qwen tokenizer.json
make apple                             # swift build + mlx.metallib next to the binary
./swift/.build/release/inferstream-apple --config config/apple.toml
```

`make apple` runs `scripts/build-apple-metallib.sh` after the Swift link.
Cmlx is statically linked, so MLX's `dladdr` looks next to
`inferstream-apple` for `mlx.metallib`. Without that file the process
exits at Engine ping (`Failed to load the default metallib`). The
`fence` kernel is skipped when the host Metal dialect is older than MLX
expects — the other default kernels still load.

Default listen is `0.0.0.0:8461`. Bearer token in `config/apple.toml` is
`change-me` (or `INFERSTREAM_API_KEYS`).

End-to-end Metal smoke (ListModels, Tokenize, Embed, ModelStreamInfer):

```bash
make smoke-apple                       # or ./scripts/smoke-apple.sh
```

The smoke script asserts the running process is the Swift binary, does not
link `libMlxEngine.dylib`, and has no Python image.

## Config

`config/apple.toml` is the same TOML shape the Rust servers use:

```toml
listen = "0.0.0.0:8461"
serve = ["minilm", "bge-small", "default-llm", "qwen-0.5b"]

[auth]
mode = "bearer"
bearer_tokens = ["change-me"]
```

`serve` aliases expand from `config/catalog.toml` using the `[models.<alias>.apple]`
tables (`backend = "mlx"`, `path = "models/mlx/<alias>"`). Point
`catalog = "/path/to/catalog.toml"` to override the matrix without rebuilding.

## Engine-side tok/s

The final `ModelStreamInfer` chunk sets `decode_tokens_per_second` from
mlx-swift-lm's generate-loop counter (generation tokens / engine seconds),
not gRPC wall-clock. That is the number `scripts/smoke-apple.sh` prints as
`engine_decode_tps`.
