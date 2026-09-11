# Fetching model artifacts (reproducible, SHA-256 verified)

All model artifacts the built-in catalog (`config/catalog.toml`) expects on
the **nvidia** arch are fetched by one Python script against a committed
manifest, so every byte that lands on a host is revision-pinned and
hash-verified:

- **Script:** `scripts/fetch_models.py` (stdlib only — no `huggingface_hub`
  needed; `scripts/fetch-embedding-models.sh` survives as a thin wrapper).
- **Manifest:** `models/manifests/embeddings.json` — for every alias, the
  Hugging Face repo, the **exact commit revision** (never a floating
  branch), and per-file **SHA-256 + size** for `onnx/model.onnx` (plus the
  `onnx/model.onnx_data` external-weights sidecar where the repo splits it
  out — `bge-m3`, `e5-large`), `tokenizer.json`, and `config.json`.
- **Destination:** `models/onnx/<alias>/…` relative to the repo root — the
  paths the catalog's nvidia entries point at. The binaries are **never
  committed** to git; only the manifest, script, Makefile, and docs are.

## Fetching

```bash
make fetch-embeddings                        # everything in the manifest
make fetch-embeddings ALIASES=minilm,mpnet   # a subset
scripts/fetch_models.py bge-m3               # direct invocation, same thing
make list-embeddings                         # aliases, repos, pinned revisions
```

Fetches are **idempotent**: a file already on disk with a matching SHA-256
is skipped; a stale or tampered file is re-downloaded. Downloads are written
atomically (temp file + rename) and hashed while streaming; a post-download
hash or size mismatch is a **hard error** — the bad file is deleted and the
script exits non-zero, telling you to re-pin with `--update-manifest` if
upstream legitimately changed.

After fetching, add the aliases to `serve` in `config/nvidia.toml`, restart,
and smoke with `scripts/smoke-embeddings.sh <host:port> <bearer-token>`.

## Verifying (offline)

```bash
make verify-embeddings                       # every alias
make verify-embeddings ALIASES=bge-m3        # subset
scripts/fetch_models.py --all --verify-only  # same, direct
```

No network: checks that every manifest file exists on disk with a matching
SHA-256 and exits non-zero listing anything missing or mismatched. Suitable
for provisioning checks and CI.

(This replaces the interim `SHA256SUMS.models` / `sha256sum -c` flow — that
file folded into `models/manifests/embeddings.json`; every hash and pinned
revision it recorded was cross-checked identical before removal. The
manifest additionally records the `minilm` artifacts that krick serves from
the TEI HF cache, at the same pinned snapshot `1110a243…`.)

## Updating the manifest (maintainers)

When adding an alias or deliberately moving to newer upstream artifacts:

1. If it's a new alias, add it to `ONNX_REPOS` in `scripts/fetch_models.py`
   (and `MLX_REPOS` if the apple arch serves it) and to
   `config/catalog.toml`.
2. Re-pin and re-hash:

   ```bash
   make update-embedding-manifest ALIASES=<alias>   # or omit ALIASES for all
   # add --no-store via direct invocation to hash without keeping ~9 GB:
   scripts/fetch_models.py --all --update-manifest --no-store
   ```

   This resolves each repo's current `main` commit, downloads every file at
   that **pinned** revision, computes SHA-256, and rewrites
   `models/manifests/embeddings.json` (a subset update preserves the other
   entries).
3. Review the manifest diff — a changed hash means upstream changed the
   artifact; make sure that's expected — and commit it.
4. `make test-fetch` must pass: it checks the manifest structurally, checks
   it against `config/catalog.toml` (every nvidia ORT alias that points into
   `models/onnx/` must be fetchable at exactly the catalog's paths), and
   exercises verify/idempotence/tamper-detection on offline fixtures.

## Per-arch coverage and gaps

| arch | status |
|---|---|
| **nvidia (ORT)** | **Fully pinned + hashed** — all 13 embedding aliases (`minilm`, `minilm-l12`, `mpnet`, `bge-small/base/large/m3`, `e5-small/base/large`, `gte-small/base`, `nomic-embed-text`). Note: the built-in catalog resolves `minilm` on krick to the TEI HF-cache path, but the manifest fetches the **same pinned revision** (`1110a243…`) into `models/onnx/minilm/` for hosts without that cache — point a catalog copy at it. |
| **apple (MLX)** | Runtime-fetched by design: the MLX bridge downloads HF repos through `huggingface_hub` into the HF cache on first use (4-bit conversions can't be pre-fetched as single files the same way). The manifest's `mlx_repos` section records each alias's repo and the **expected pinned revision** for auditability; re-pin with `--update-manifest`. |
| **intel (OVMS)** | Nothing to fetch here: OVMS IR pipelines are provisioned on the Model Server host (converted IR + tokenizer DAG), not downloaded by this repo. The walkthrough is `docs/adding-ovms-embedding-pipelines.md`; a hashed IR-artifact manifest can be added later once the conversion pipeline is scripted. |

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

Download URLs are constructed as
`https://huggingface.co/<repo>/resolve/<revision>/<path>` — the revision in
the URL is the pinned commit, so the fetch is reproducible even if upstream
`main` moves. Never edit hashes by hand; always regenerate with
`--update-manifest` so hash, size, and revision stay consistent.
