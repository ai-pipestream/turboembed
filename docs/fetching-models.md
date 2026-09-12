# Fetching model artifacts (reproducible, SHA-256 verified)

All model artifacts the built-in catalog (`config/catalog.toml`) expects on
the **nvidia** arch are fetched by a Rust binary against a committed
manifest, so every byte that lands on a host is revision-pinned and
hash-verified:

- **Tool:** `cargo run -p inferstream-fetch` (also `make fetch-*` /
  `scripts/fetch-embedding-models.sh` / `scripts/fetch-llm-models.sh`).
  HTTPS to Hugging Face; no extra interpreter. `make fetch-llms` /
  `verify-llms` / `list-llms` and `scripts/smoke-llms.sh` must not need
  python3. `scripts/fetch-llms.sh` is an optional curl + sha256sum
  fallback that reads the same manifest.
- **Embedding manifest:** `models/manifests/embeddings.json` — for every
  alias, the Hugging Face repo, the **exact commit revision** (never a
  floating branch), and per-file **SHA-256 + size** for `onnx/model.onnx`
  (plus the `onnx/model.onnx_data` external-weights sidecar where the repo
  splits it out — `bge-m3`, `e5-large`), `tokenizer.json`, and `config.json`.
- **LLM manifest:** `models/manifests/llms.json` — official Qwen GGUF files
  (Qwen2.5-0.5B Q8_0; Qwen2.5-7B Q5_K_M as two shards) plus the matching
  instruct `tokenizer.json` from a second pinned repo. `default-llm` is
  `alias_of` `qwen-0.5b` (same files).
- **Destination:** `models/onnx/<alias>/…` (embeddings) and
  `models/gguf/<alias>/…` (LLMs) relative to the repo root — the paths the
  catalog's nvidia fetch entries point at. The binaries are **never
  committed** to git; only the manifests, fetcher crate, Makefile, and docs
  are.
- **MLX manifest:** `models/manifests/mlx.json` — Apple native-MLX
  safetensors + config + tokenizer into `models/mlx/<alias>/`. Fetched by
  `cargo xtask fetch --mlx` / `make fetch-mlx` (Rust; no Python). Make
  never invokes `python3`.

## Fetching

```bash
make fetch-embeddings                        # everything in the embedding manifest
make fetch-embeddings ALIASES=minilm,mpnet   # a subset
cargo run -p inferstream-fetch -- bge-m3     # direct invocation, same thing
make list-embeddings                         # aliases, repos, pinned revisions

make fetch-llms                              # qwen-0.5b + qwen-7b GGUF + tokenizers
make fetch-llms ALIASES=qwen-0.5b            # smoke-sized 0.5B only (~650 MiB + tokenizer)
cargo run -p inferstream-fetch -- --llms default-llm   # same files as qwen-0.5b (alias_of)
make list-llms

make fetch-ov-genai                          # Intel in-process GenAI OV-format dirs
make fetch-ov-genai ALIASES=minilm,bge-base
make list-ov-genai

make fetch-mlx                               # Apple native MLX weights
make fetch-mlx ALIASES=minilm,qwen-0.5b
cargo xtask fetch --mlx minilm

make fetch-corpus                            # Tiny Shakespeare + STS pairs
make verify-corpus                           # offline hash check (sts-pairs is committed)
cargo run -p inferstream-fetch -- --corpus --list
```

## E2E / bring-up auto-fetch

The live harness fetches the smoke set automatically so a worker that is
missing weights can still `make e2e-<arch>`:

```bash
make e2e-nvidia                  # FETCH=1 default: ONNX minilm + 0.5B GGUF, then the suite
make e2e-intel FETCH=1           # --ov-genai minilm + GGUF
make e2e-apple                   # cargo xtask fetch --mlx + tokenizer.json
make e2e-nvidia FETCH=0          # skip download
make fetch-e2e-nvidia            # artifacts only (`--fetch-only`); no gRPC
scripts/ensure-models.sh nvidia  # same --fetch-only wrapper (cargo only; no Python)
cargo run -p inferstream-e2e -- --target intel --fetch-only --only minilm
```

`make e2e-mock` never fetches. `--fetch` is idempotent: matching SHA-256
on disk is a skip. Per-target alias tables and `FETCH=all` / `FETCH=serve`
are documented in [`docs/e2e.md`](e2e.md). `FETCH_CORPUS=1` additionally
pulls the soak/STS text corpus (`models/manifests/corpus.json`); default
CI leaves this off.

Fetches are **idempotent**: a file already on disk with a matching SHA-256
is skipped; a stale or tampered file is re-downloaded. Downloads are written
atomically (temp file + rename) and hashed while streaming; a post-download
hash or size mismatch is a **hard error** — the bad file is deleted and the
tool exits non-zero, telling you to re-pin with `--update-manifest` if
upstream legitimately changed.

After fetching embeddings, add the aliases to `serve` in `config/nvidia.toml`,
restart, and smoke with `scripts/smoke-embeddings.sh <host:port> <bearer-token>`.
After fetching LLMs, add `qwen-0.5b` / `qwen-7b` to `serve` (nvidia `default-llm`
already points at the krick GGUF) and smoke with
`scripts/smoke-llms.sh <host:port> <bearer-token>` — Tokenize + a short
`ModelStreamInfer`. Live GPU is the acceptance path; the smoke scripts talk
to an already-running server. Bring-up on the worker can use
`make fetch-e2e-<arch>` / `make e2e-<arch>` (`FETCH=1`) so missing files
are pulled first.

## Verifying (offline)

```bash
make verify-embeddings                       # every embedding alias
make verify-embeddings ALIASES=bge-m3        # subset
cargo run -p inferstream-fetch -- --all --verify-only
make verify-llms                             # every LLM alias
make verify-llms ALIASES=qwen-0.5b
cargo run -p inferstream-fetch -- --llms --all --verify-only
```

No network: checks that every manifest file exists on disk with a matching
SHA-256 and exits non-zero listing anything missing or mismatched. Suitable
for provisioning checks and CI (`make test-fetch` / `cargo test -p inferstream-fetch`
exercises the same verify path on offline fixtures).

(This replaces the interim `SHA256SUMS.models` / `sha256sum -c` flow — that
file folded into `models/manifests/embeddings.json`; every hash and pinned
revision it recorded was cross-checked identical before removal. The
manifest additionally records the `minilm` artifacts that krick serves from
the TEI HF cache, at the same pinned snapshot `1110a243…`.)

## Updating the manifest (maintainers)

When adding an alias or deliberately moving to newer upstream artifacts:

1. If it's a new embedding alias, add it to `ONNX_REPOS` in
   `crates/fetch/src/lib.rs` (and `MLX_REPOS` if the apple arch serves it)
   and to `config/catalog.toml`. For an LLM, add it to `LLM_SOURCES` /
   `LLM_ALIASES` / `LLM_MLX_REPOS` instead.
2. Re-pin and re-hash:

   ```bash
   make update-embedding-manifest ALIASES=<alias>   # or omit ALIASES for all
   # hash without keeping ~9 GB on disk:
   cargo run -p inferstream-fetch -- --all --update-manifest --no-store
   make update-llm-manifest ALIASES=<alias>         # GGUF + tokenizer
   cargo run -p inferstream-fetch -- --llms --all --update-manifest --no-store
   cargo run -p inferstream-fetch -- --ov-genai --all --update-manifest --no-store
   ```

   This resolves each repo's current `main` commit, downloads every file at
   that **pinned** revision, computes SHA-256, and rewrites the matching
   manifest (a subset update preserves the other entries).
3. Review the manifest diff — a changed hash means upstream changed the
   artifact; make sure that's expected — and commit it. Never commit the
   GGUF / ONNX weights themselves.
4. `make test-fetch` must pass: it checks both manifests structurally,
   checks them against `config/catalog.toml` (every nvidia ORT alias that
   points into `models/onnx/` and every nvidia llama-cpp alias that points
   into `models/gguf/` must be fetchable at exactly the catalog's paths),
   and exercises verify/idempotence/tamper-detection on offline fixtures.

## Per-arch coverage and gaps

| arch | status |
|---|---|
| **nvidia (ORT embeddings)** | **Fully pinned + hashed** — all 13 embedding aliases (`minilm`, `minilm-l12`, `mpnet`, `bge-small/base/large/m3`, `e5-small/base/large`, `gte-small/base`, `nomic-embed-text`). Note: the built-in catalog resolves `minilm` on krick to the TEI HF-cache path, but the manifest fetches the **same pinned revision** (`1110a243…`) into `models/onnx/minilm/` for hosts without that cache — point a catalog copy at it. |
| **nvidia (llama.cpp LLMs)** | **Fully pinned + hashed** — `qwen-0.5b` (Q8_0, ~644 MiB) and `qwen-7b` (official Q5_K_M split into two shards, ~5.1 GiB). `default-llm` on nvidia uses the GGUF already on krick (`/work/models/gguf/qwen2.5-0.5b-instruct-q8_0.gguf`); `make fetch-llms ALIASES=qwen-0.5b` puts the same pin into `models/gguf/qwen-0.5b/` for other hosts. The 7B catalog path is the first shard; llama.cpp loads the second from the same directory. |
| **apple (MLX)** | Runtime-fetched by design: the MLX backend downloads HF repos into the HF cache on first use (4-bit conversions can't be pre-fetched as single files the same way). Each manifest's `mlx_repos` section records each alias's repo and the **expected pinned revision** for auditability; re-pin with `--update-manifest` / `--llms --update-manifest`. LLM Tokenize on apple uses the fetched `tokenizer.json` (`models/gguf/<alias>/`), which `scripts/setup-mlx.sh` also writes for `qwen-0.5b` and `qwen-7b`. |
| **intel (OpenVINO GenAI embeddings)** | **Pinned + hashed** — `models/manifests/ov-genai-embeddings.json`. `make fetch-ov-genai` downloads OV-format dirs into `models/ov/<alias>/` (no Python). Official / first-party HF IR where it exists; tokenizer IR is required at load (`openvino_tokenizer.xml`). Aliases without a public tokenizer IR (`bge-small`, `bge-large`, `nomic-embed-text`, and ST-style model-only repos) need that pair from a one-off export in `contrib/offline-once/` (historical IR tooling; not invoked by Make). Walkthrough: `docs/intel-genai-embed.md`. OVMS gRPC is out of scope. |
| **intel (llama.cpp LLMs)** | **Same GGUF fetch as nvidia** — `make fetch-llms` lands Qwen2.5-0.5B Q8_0 and Qwen2.5-7B-Instruct Q5_K_M into `models/gguf/<alias>/`. Catalog intel entries are **in-process SYCL** (`path`, no `endpoint`). The host `vlm-server` on `:8085` is not on this path. |

## Manifest format

```json
{
  "schema_version": 1,
  "models": {
    "<alias>": {
      "repo": "org/name",
      "revision": "<40-hex HF commit>",
      "dest": "models/onnx/<alias>",
      "files": [
        {"path": "onnx/model.onnx", "sha256": "<64-hex>", "size": 123}
      ]
    }
  },
  "mlx_repos": {
    "<alias>": {"repo": "org/name", "revision": "<40-hex HF commit>"}
  }
}
```

LLM manifests add an optional `tokenizer` object (second HF repo + revision
+ files, landing in the same `dest`) and `alias_of` for logical names that
share a family (`default-llm` → `qwen-0.5b`):

```json
{
  "schema_version": 1,
  "models": {
    "default-llm": {"alias_of": "qwen-0.5b"},
    "qwen-0.5b": {
      "repo": "Qwen/Qwen2.5-0.5B-Instruct-GGUF",
      "revision": "<40-hex HF commit>",
      "dest": "models/gguf/qwen-0.5b",
      "files": [
        {"path": "qwen2.5-0.5b-instruct-q8_0.gguf", "sha256": "<64-hex>", "size": 123}
      ],
      "tokenizer": {
        "repo": "Qwen/Qwen2.5-0.5B-Instruct",
        "revision": "<40-hex HF commit>",
        "files": [{"path": "tokenizer.json", "sha256": "<64-hex>", "size": 123}]
      }
    }
  }
}
```

Download URLs are constructed as
`https://huggingface.co/<repo>/resolve/<revision>/<path>` — the revision in
the URL is the pinned commit, so the fetch is reproducible even if upstream
`main` moves. Never edit hashes by hand; always regenerate with
`--update-manifest` so hash, size, and revision stay consistent.
