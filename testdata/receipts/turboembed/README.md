# TurboEmbed live receipts

JSON written by a **live** `cargo test -p turboembed` on the named host.
Not mocks. Not OVMS. Not a CPU EP pretending to be CUDA.

| file | host | engine | device | command |
|---|---|---|---|---|
| `nvidia-minilm.json` | krick (RTX 4080 SUPER) | ORT CUDA EP + IoBinding | **CUDA** | `make test-turboembed-nvidia` |
| `intel-minilm.json` | krick-1 (Battlemage) | `ov::genai::TextEmbeddingPipeline` | **GPU** | `make test-turboembed-intel` |
| `intel-minilm-cpu.json` | krick-1 | same pipeline, `"CPU"` device string | **CPU** | `make test-turboembed-intel` |

NVIDIA fields: `device=CUDA`, `dims`, `worst_cosine` vs
`testdata/e2e/goldens/nvidia/minilm.json` (`parity:*` + hello world),
floor 0.99, `pass=true`, git sha, `/proc/self/maps` needles.

Intel: same MiniLM IR (`models/ov/minilm`). Cosine vs
`testdata/e2e/goldens/{intel,nvidia}/minilm.json` stays ≥ 0.99 on **both**
devices (CPU is the same IR, not a mock). GPU create/load still fails if
the GPU plugin is missing — that path never compiles `"CPU"`.
`sha` (git + IR bins).
