# Intel goldens (Machine B)

Captured **2026-09-12** on **Machine B** (Battlemage) via in-process
OpenVINO GenAI `TextEmbeddingPipeline` on **GPU**. See
`docs/intel-e2e-parity-goldens-machine-b.md` and
`testdata/receipts/turboembed/intel-minilm.json` for the TurboEmbed
C ABI MiniLM replay (cosine ≥ 0.99).

| file | alias | pooling | dim | items |
|---|---|---|---|---|
| `minilm.json` | `minilm` | mean + L2 | 384 | 213 |
| `bge-small.json` | `bge-small` | CLS + L2 | 384 | 213 |
| `mpnet.json` | `mpnet` | mean + L2 | 768 | 213 |

Item count is 213 (committed STS + excerpt + built-in prompts). nvidia's
237 includes 24 extra soak sentences from fetched Tiny Shakespeare.

Dump-vs-dump vs `testdata/e2e/goldens/nvidia/` (no GPU): MiniLM worst-pair
**0.999999310862**; BGE-small worst-pair **0.999998043062**. `mpnet` has
no nvidia dump. Live `bge-small` / `mpnet` still skip until Machine B
lists a GenAI IR (`docs/e2e-parity.md`).
