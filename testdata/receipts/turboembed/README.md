# TurboEmbed live receipts

JSON written by a **live** `cargo test -p turboembed --features genai`
on the named host. Not mocks. Not OVMS.

| file | host | engine | device |
|---|---|---|---|
| `intel-minilm.json` | krick-1 (Battlemage) | `ov::genai::TextEmbeddingPipeline` | **GPU** |
| `intel-minilm-cpu.json` | krick-1 | same pipeline, `"CPU"` device string | **CPU** |

Same MiniLM IR (`models/ov/minilm`). Cosine vs
`testdata/e2e/goldens/{intel,nvidia}/minilm.json` stays ≥ 0.99 on **both**
devices (CPU is the same IR, not a mock). GPU create/load still fails if
the GPU plugin is missing — that path never compiles `"CPU"`.

Fields: `device`, `cosine` (vs intel + nvidia goldens, floor 0.99),
`sha` (git + IR bins).
