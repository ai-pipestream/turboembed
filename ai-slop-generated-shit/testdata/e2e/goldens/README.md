# Per-arch embedding goldens

`inferstream-e2e --parity-goldens --parity-write` writes
`<arch>/<alias>.json` here (see `docs/e2e-parity.md`).

Missing files are not an error for the regular suite. When a file is
present, both the suite (first vector, cosine ≥ 0.99) and
`--parity-goldens` (all item ids) compare against it.

Do not commit GPU dumps until they were captured on the named host
(Machine A / Machine B / Machine C) with the catalog pooling for that alias.

**nvidia / Machine A (2026-09-12):** `minilm.json` + `bge-small.json` committed.
Self-replay cosine **1.0000**. See `nvidia/README.md` and
`docs/nvidia-e2e-parity-goldens-machine-a.md`.

**intel / Machine B (2026-09-12):** `minilm.json` + `bge-small.json` +
`mpnet.json` from in-process OpenVINO GenAI on GPU. Self-replay on the
capture host **1.0000**. nvidia dump ↔ intel dump minilm worst-pair
**0.999999310862** (213 shared ids); bge-small **0.999998043062**. See
`intel/README.md` and `docs/intel-e2e-parity-goldens-machine-b.md`.

**apple / Machine C (2026-09-12):** `minilm.json` + `bge-small.json`
committed after the mean+L2 / FP-catalog fix. vs nvidia: MiniLM min
**0.9795** / mean **0.9997**; BGE-small min **0.9938** / mean **0.9999**.
See `apple/README.md`.
