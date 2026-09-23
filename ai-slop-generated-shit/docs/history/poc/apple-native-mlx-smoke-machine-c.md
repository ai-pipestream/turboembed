# Apple native MLX smoke (Machine C)

Host: Darwin arm64, Metal GPU. Runtime is in-process Swift MLX
(`libMlxEngine.dylib` + `mlx.metallib`). No Python interpreter.

## Live crate tests

```
cargo test -p inferstream-backend-apple --features mlx-live -- --ignored --nocapture
```

| test | result |
|---|---|
| `ping_reports_metal_device` | `Device(gpu, 0)` metal=true |
| `embed_minilm_batch_shapes_and_normalization` | 2×384, L2 ≈ 1.0 |
| `stream_generate_small_lm` | **193.8 tok/s** engine-side (17 tokens) |

## gRPC smoke (`scripts/smoke-apple.sh`)

| RPC | result |
|---|---|
| ListModels | `minilm`, `default-llm`, `qwen-0.5b` ready, `hasTokenizer=true` |
| Tokenize / Detokenize | Rust `tokenizers` crate (`[CLS] hello world [SEP]`) |
| Embed minilm | dim=384, L2=0.999998, load log `Device(gpu, 0)` |
| ModelStreamInfer `default-llm` | tokens=10, final=true, **engine_decode_tps=219.6** |

## No Python

- `ps` command line of `inferstream-apple`: no `python`
- `pgrep -P` children: none
- `otool -L`: `@rpath/libMlxEngine.dylib`, libiconv, libSystem — no libpython
- `vmmap`: no `Python.framework` / `libpython`

`rg -n python` is clean under `crates/backend-apple`, `crates/arch-apple`,
`crates/xtask`, `native/mlx-engine`, and `Makefile` (Make never invokes
`python3`). Fetch is `cargo xtask`. `scripts/smoke-apple.sh` mentions
`python` only in the proof greps above. NVIDIA `scripts/fetch-runtime-libs.sh`
still uses a venv to unpack CUDA wheels — that is not the Apple path.

The earlier Python-bridge write-up (`docs/apple-llm-metal-smoke-machine-c.md`)
is history only.
