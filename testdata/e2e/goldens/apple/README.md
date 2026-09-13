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

## Replay

```
make e2e-parity-goldens TARGET=apple INFERSTREAM_E2E_APPLE_ADDR=127.0.0.1:8461
make e2e-parity \
  DUMP_NVIDIA=testdata/e2e/goldens/nvidia \
  DUMP_APPLE=testdata/e2e/goldens/apple
```
