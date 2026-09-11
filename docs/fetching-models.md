# Fetching model artifacts (reproducible, SHA-256 verified)

All model artifacts the built-in catalog (`config/catalog.toml`) expects
are fetched by **`cargo xtask`** (Rust) against committed JSON manifests.
Make never invokes `python3`.

- **Tool:** `cargo xtask` (`crates/xtask`). Thin wrappers:
  `scripts/fetch-embedding-models.sh`, `scripts/fetch-llm-models.sh`.
- **Embedding manifest:** `models/manifests/embeddings.json` — ONNX +
  tokenizer.json + config.json at a pinned HF revision, SHA-256 per file.
- **LLM manifest:** `models/manifests/llms.json` — official Qwen GGUF +
  instruct `tokenizer.json`. `default-llm` is `alias_of` `qwen-0.5b`.
- **MLX manifest:** `models/manifests/mlx.json` — Apple native-MLX
  safetensors + config + tokenizer into `models/mlx/<alias>/`.
- **OVMS manifest:** `models/manifests/ovms-embeddings.json` — verify-only
  for pre-exported OpenVINO IR (export is OpenVINO's own toolchain).

Weights are never committed.

## Fetching

```bash
make fetch-embeddings                        # everything in the embedding manifest
make fetch-embeddings ALIASES=minilm,mpnet   # a subset
cargo xtask fetch --embeddings bge-m3

make fetch-llms                              # qwen-0.5b + qwen-7b GGUF + tokenizers
make fetch-llms ALIASES=qwen-0.5b
cargo xtask fetch --llms default-llm         # same files as qwen-0.5b

make fetch-mlx                               # Apple native MLX weights
make fetch-mlx ALIASES=minilm,qwen-0.5b
```

Fetches are idempotent: a file already on disk with a matching SHA-256 is
skipped. A hash mismatch after download is a hard error.

## Verifying (offline)

```bash
make verify-embeddings
make verify-llms
make verify-mlx
make verify-embeddings-intel                 # pre-exported IR on the Intel host
```

## Updating a manifest (maintainers)

1. Add the alias to `crates/xtask/src/sources.rs` (and the catalog).
2. Re-pin:

```bash
cargo xtask update-manifest --embeddings --all
cargo xtask update-manifest --llms --all
cargo xtask update-manifest --mlx --all
```

3. Review the JSON diff and commit it.

Intel OVMS IR re-export is **not** invoked from Make (no Python). Use
OpenVINO's official conversion tools, then `cargo xtask verify --ovms`.

## Per-arch coverage

| Arch | What fetch produces |
|---|---|
| nvidia | ONNX under `models/onnx/<alias>/`, GGUF under `models/gguf/<alias>/` |
| apple | MLX safetensors under `models/mlx/<alias>/`; LLM Tokenize uses `models/gguf/<alias>/tokenizer.json` |
| intel embeddings | Verify pre-exported IR under `--out` (default `/work/models/ovms-embedder`) |
