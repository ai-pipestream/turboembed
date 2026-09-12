# Cross-architecture embedding parity

Same catalog alias + same text must produce **nearly identical vectors** on
nvidia (ORT), intel (OpenVINO GenAI), and apple (MLX). The harness is
`inferstream-e2e` — it never starts GPUs. Point it at servers that are
already up, or at dumps captured earlier.

Pooling is sent explicitly on `Embed` from the catalog family convention
(BGE = **CLS**, MiniLM / MPNet / E5 / GTE / Nomic = **mean**) so Apple
rows that omit `pooling` still get the same request as ORT / GenAI.

## Cosine thresholds (defaults)

| pair | min cosine | why this number |
|---|---|---|
| same arch (live vs golden on that host) | **0.99** | ORT MiniLM vs TEI on krick measured **0.999998** (`testdata/reference_embeddings/README.md`). Replay of a golden captured on the same engine must stay in that band. |
| nvidia ORT FP32 ↔ intel GenAI | **0.99** | Same MiniLM family, mean pool, L2. Intel IR is often FP16; MiniLM still lands ≥ 0.99 on the FP path. A miss prints the worst text id and both scores. |
| any pair that includes **apple** | **0.97** | Catalog MiniLM is `mlx-community/all-MiniLM-L6-v2-4bit`; BGE-small is `bge-small-en-v1.5-4bit`. **4-bit vs FP32 cannot honestly be gated at 0.99.** 0.97 is the floor we still fail on a broken tokenizer/pooling mismatch. |
| mock ↔ mock (CI) | **0.99** | Deterministic backend; used only to exercise the harness. |

Honest gaps:

- **Apple 4-bit vs nvidia/intel FP** will often sit in 0.97–0.99. That is
  quantization, not a routing bug. Maximize MiniLM **FP** parity
  (nvidia ↔ intel) first.
- **Different quant** (GGUF Q8 vs MLX 4-bit, INT8 ORT, …) is not claimed
  at 0.99. Do not add those aliases to the default parity set.
- **`mpnet`** has no apple catalog row (`NotAvailableOnArch`). Cross
  compares nvidia ↔ intel only when both serve it.
- **`bge-small`** is CLS on every arch that serves it; intel still needs
  a GenAI IR (not in the public fetch set today) so the case skips until
  the host actually lists it.

Override is not offered as a silent weaken: if you need a looser gate for
an experiment, capture dumps and inspect the printed min/mean rather than
shipping a lower default.

## Capture goldens on one arch

On the host (or any client that can reach it):

```bash
# nvidia / krick
make e2e-parity-goldens TARGET=nvidia WRITE=1
# writes testdata/e2e/goldens/nvidia/minilm.json (and bge-small / mpnet if served)

# intel / krick-1
make e2e-parity-goldens TARGET=intel WRITE=1 INFERSTREAM_E2E_INTEL_ADDR=krick-1:8461

# apple / krickert-mac
make e2e-parity-goldens TARGET=apple WRITE=1 INFERSTREAM_E2E_APPLE_ADDR=krickert-mac:8461
```

Or the binary:

```bash
cargo run -p inferstream-e2e -- --parity-goldens --parity-write \
  --target nvidia --addr krick:8461 --token change-me
```

Replay later (same host or a dump checked into git):

```bash
make e2e-parity-goldens TARGET=nvidia          # compare, do not overwrite
cargo run -p inferstream-e2e -- --parity-goldens --target nvidia --addr krick:8461
```

Dump schema (also satisfies the regular suite's `text` / `vector` / `dim`
golden loader for the first item):

```json
{
  "schema_version": 1,
  "arch": "nvidia",
  "alias": "minilm",
  "pooling": "mean",
  "normalize": true,
  "dim": 384,
  "text": "hello world",
  "vector": [0.01, "..."],
  "items": [{"id": "parity:short", "text": "hello world", "vector": ["..."]}]
}
```

## Three-way live compare

All three servers up:

```bash
INFERSTREAM_E2E_NVIDIA_ADDR=krick:8461 \
INFERSTREAM_E2E_INTEL_ADDR=krick-1:8461 \
INFERSTREAM_E2E_APPLE_ADDR=krickert-mac:8461 \
  make e2e-parity
```

Mix live peers and saved dumps (capture on a laptop, compare later):

```bash
cargo run -p inferstream-e2e -- --parity-cross \
  --peer nvidia=krick:8461 \
  --dump intel=testdata/e2e/goldens/intel \
  --dump apple=testdata/e2e/goldens/apple
```

`make e2e-parity` with **no** addrs and **no** `DUMP_*` prints a skip line
and exits 0 — CI cloud must not start remote GPUs.

```bash
make e2e-parity DUMP_NVIDIA=testdata/e2e/goldens/nvidia \
                DUMP_INTEL=testdata/e2e/goldens/intel
```

## Text set

The same ids are embedded on every arch:

1. Built-in parity prompts (`hello world`, unicode, MiniLM-shaped sentences)
2. STS-style pairs from `testdata/corpus/sts-pairs.jsonl` (96 committed
   pairs; unit tests use `fixtures/sts-micro.jsonl`)
3. Stable sentence ids from the Tiny Shakespeare **excerpt**
4. When `make fetch-corpus` has landed `tiny-shakespeare.txt`, the first
   `--soak-limit` (default 24) extra sentence units

Chunk ids: `tiny-shakespeare:p0000`, `tiny-shakespeare:p0000:s0001` —
position in the SHA-pinned file, so Embed batches repeat.

```bash
make fetch-corpus                 # optional; ~1.1 MiB Shakespeare
make e2e-nvidia FETCH_CORPUS=1    # e2e optional path; CI stays FETCH_CORPUS=0
```

## Aliases

Default: `minilm` (required on all three arches), plus `bge-small` and
`mpnet` when `ListModels` has them.

```bash
cargo run -p inferstream-e2e -- --parity-cross --only minilm \
  --peer nvidia=krick:8461 --peer intel=krick-1:8461
```
