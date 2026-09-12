# Per-arch embedding goldens

`inferstream-e2e --parity-goldens --parity-write` writes
`<arch>/<alias>.json` here (see `docs/e2e-parity.md`).

Missing files are not an error for the regular suite. When a file is
present, both the suite (first vector, cosine ≥ 0.99) and
`--parity-goldens` (all item ids) compare against it.

Do not commit GPU dumps until they were captured on the named host
(krick / krick-1 / krickert-mac) with the catalog pooling for that alias.

**nvidia / krick (2026-09-12):** `minilm.json` + `bge-small.json` committed.
Self-replay cosine **1.0000**. See `nvidia/README.md` and
`docs/nvidia-e2e-parity-goldens-krick.md`.
