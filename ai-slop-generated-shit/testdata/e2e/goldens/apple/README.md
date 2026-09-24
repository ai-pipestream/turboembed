# Apple goldens (Machine C)

Captured **2026-09-12** on **Machine C** (Apple Silicon / Metal) after
the MiniLM pooling fix. Origin branch
`ai-pipestream/apple-minilm-parity-81a5`.

| file | alias | pooling | dim | items |
|---|---|---|---|---|
| `minilm.json` | `minilm` | mean + L2 | 384 | 213 |
| `bge-small.json` | `bge-small` | CLS + L2 | 384 | 213 |

`mpnet` is not on the apple catalog. Item count is 213 (committed STS +
excerpt + built-in prompts) — nvidia's 237 includes 24 extra soak
sentences from fetched Tiny Shakespeare.

## Server

- binary: `swift/.build/release/inferstream-apple` (mlx-swift, Metal)
- config: `config/apple.toml` (`serve = ["minilm", "bge-small", "default-llm", "qwen-0.5b"]`)
- listen: **`0.0.0.0:8461`**
- bearer: **`change-me`**
- weights: FP `sentence-transformers/all-MiniLM-L6-v2` @ `1110a243` and
  `BAAI/bge-small-en-v1.5` @ `5c38ec7c` (`models/mlx/<alias>/`)

## Before / after vs nvidia dumps

nvidia goldens: `testdata/e2e/goldens/nvidia/` (Machine A ORT CUDA, 2026-09-12).

| pair | when | min cosine | mean cosine | n |
|---|---|---|---|---|
| apple↔nvidia `minilm` | **before** (4-bit + BERT pooler + LayerNorm) | **-0.1404** | **-0.0085** | 237 |
| apple↔nvidia `minilm` | **after** (FP + mean + L2) | **0.9795** | **0.9997** | 213 |
| apple↔nvidia `bge-small` | after (FP + first-token CLS + L2) | **0.9938** | **0.9999** | 213 |

The MiniLM min is three copies of the same Japanese STS line
(`sts-0056:a` / `0057:a` / `0058:a`: `東京の朝は通勤客で混雑する。`).
Every other overlapping text is **≥ 0.999999**. That is CJK WordPiece
UNK handling on an English MiniLM, not pooling or weight mismatch.

## Known-stale entries (2026-09-17)

The three `minilm.json` entries above are **stale relative to the current
native tokenizer** and are kept unmodified as the honest 2026-09-12
capture record:

- This capture predates commit `78a88d8` (2026-09-14, "validate native
  BERT tokenizers and match reference token IDs"), which made native
  WordPiece tokenize the CJK line the same way as the HF reference
  tokenizer instead of emitting `[UNK]`.
- Comparing the committed dumps offline (no GPU involved), the three
  entries disagree with the nvidia golden for the identical text at
  cosine **0.979456**; every other overlapping item agrees at
  ≥ 0.999999.
- The 2026-09-17 Machine C live run confirmed the flip: live↔nvidia min
  **0.999999** across all 237 items (CJK included), while live↔apple
  bottomed at **0.979436** on exactly these three entries.

The `mlx-live` tests (`apple_minilm_metal.rs`, `apple_solidify_bench.rs`)
therefore detect golden entries whose cross-golden cosine vs nvidia falls
below the 0.99 apple floor, and gate the live vector on those items
against the **nvidia** golden at ≥ 0.999 instead — a tighter bar than the
entry it replaces. Exempted items are listed in the receipts under
`stale_apple_golden_items`, and at most 5% of the set may be exempted.
A future recapture of these dumps on Machine C (post-`78a88d8` tokenizer)
should make the exemption list empty again.

## Replay

```
make e2e-parity-goldens TARGET=apple INFERSTREAM_E2E_APPLE_ADDR=127.0.0.1:8461
make e2e-parity \
  DUMP_NVIDIA=testdata/e2e/goldens/nvidia \
  DUMP_APPLE=testdata/e2e/goldens/apple
```
