# E2E harness data

- `matrix.json` — built-in alias × arch table (dims + required flags). The
  binary embeds this file; `--matrix` / `INFERSTREAM_E2E_MATRIX` overrides it.
  `--fetch` / `FETCH=1` uses the same table to decide which SHA-256-pinned
  artifacts to materialize (`minilm` + `default-llm` / `qwen-0.5b` by
  default; `--fetch-all` takes every alias on the target).
- `goldens/<arch>/<alias>.json` — optional reference embeddings. Same schema
  as `testdata/reference_embeddings/` for the first vector, plus a parity
  `items[]` array when captured by `--parity-goldens --parity-write`.
  Missing files are ignored; when present the regular suite requires
  cosine ≥ 0.99 against the first Embed vector. Cross-arch compare:
  `docs/e2e-parity.md`.
- Corpus used for parity / soak lives under `testdata/corpus/` (micro
  fixtures committed; full Tiny Shakespeare via `make fetch-corpus`).
