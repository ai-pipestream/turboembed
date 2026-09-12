# All-Swift inferstream-apple Metal smoke (krickert-mac)

Host: Darwin arm64, Metal GPU. Serve path is the Swift executable
`swift/.build/release/inferstream-apple` (grpc-swift 2 + in-process
mlx-swift). No Rust process, no `libMlxEngine.dylib`, no Python.

## Process proof

```
ps: ./swift/.build/release/inferstream-apple --config config/apple.toml
children: none
otool -L: Metal / Foundation / libSystem — no libpython, no libMlxEngine
boot log: mlx device=Device(gpu, 0) metal=true
```

## RPCs

| RPC | result |
|---|---|
| ListModels | `minilm`, `default-llm`, `qwen-0.5b`, `mock-embed` ready, `hasTokenizer=true` |
| Tokenize minilm | `[CLS] hello world [SEP]` ids `101,7592,2088,102` |
| Detokenize minilm | `hello world` |
| Tokenize default-llm | 2 tokens (`14990,1879`) from MLX `tokenizer.json` |
| Embed minilm | **dim=384**, L2 = **0.999998** |
| ModelStreamInfer `default-llm` | tokens=10, final=true, **engine_decode_tps=194.4** (cold) / **216.0** (warm) |
| ModelStreamInfer `qwen-0.5b` | tokens=10, final=true, **engine_decode_tps=194.9** (same 0.5B weights) |

Engine-side tok/s is mlx-swift-lm generate-loop tokens / seconds on the
final chunk (`decode_tokens_per_second`), same contract as the legacy
Rust FFI path (~219 t/s in `docs/apple-native-mlx-smoke-krickert-mac.md`).
Warm in-process Swift is **216 t/s** — competitive, no FFI hop.

`qwen-7b` was not fetched on this host (needs ~24 GB). Catalog alias is
wired; add it to `serve` after `make fetch-mlx ALIASES=qwen-7b`.
