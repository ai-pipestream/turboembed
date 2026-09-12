# E2E harness data

- `matrix.json` — built-in alias × arch table (dims + required flags). The
  binary embeds this file; `--matrix` / `INFERSTREAM_E2E_MATRIX` overrides it.
- `goldens/<arch>/<alias>.json` — optional reference embeddings. Same schema
  as `testdata/reference_embeddings/`. Missing files are ignored; when
  present the harness requires cosine ≥ 0.99 against the first Embed vector.
