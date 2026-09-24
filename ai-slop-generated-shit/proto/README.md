# Shared inferstream proto (source of truth)

These two files are the **only** wire-contract source:

| file | package | service |
|---|---|---|
| `open_inference_grpc.proto` | `inference` | KServe OIP V2 `GRPCInferenceService` + Triton-shaped `ModelStreamInfer` |
| `inferstream_extension.proto` | `inferstream.v1` | `InferstreamService` (Tokenize / Detokenize / Embed / EmbedStream / ListModels / Rerank) |

Do not fork these per language. Consumers:

- **Rust** (nvidia / intel / mock): `crates/protocol/build.rs` compiles `../../proto` with tonic.
- **Swift** (Apple): `scripts/sync-proto.sh` copies them into `swift/Sources/InferstreamApple/Protos/` for the grpc-swift `GRPCProtobufGenerator` plugin (the plugin can only see files inside the target). The copies are derived; edit **this** directory.

```bash
# After editing a .proto:
./scripts/sync-proto.sh          # refresh the Swift copies
# Rust picks up proto/ on the next cargo build (rerun-if-changed).
```
