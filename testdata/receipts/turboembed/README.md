# TurboEmbed live receipts

JSON written by a **live** `cargo test -p turboembed` on the named host.
Not mocks. Not OVMS. Not a CPU EP pretending to be CUDA. Not BERT pooler.

| file | host | engine | device | command |
|---|---|---|---|---|
| `nvidia-minilm.json` | krick (RTX 4080 SUPER) | ORT CUDA EP + IoBinding | **CUDA** | `make test-turboembed-nvidia` |
| `nvidia-minilm-cpu.json` | krick | ORT CPU EP (explicit `Device::Cpu`) | **CPU** | `make test-turboembed-nvidia` |
| `intel-minilm.json` | krick-1 (Battlemage) | `ov::genai::TextEmbeddingPipeline` | **GPU** | `make test-turboembed-intel` |
| `intel-minilm-cpu.json` | krick-1 | same pipeline, `"CPU"` device string | **CPU** | `make test-turboembed-intel` |
| `intel-npu.json` | krick-1 | **defer** — NPU create fails loud (no plugin) | **NPU** | `make test-turboembed-intel` |
| `apple-minilm.json` | krickert-mac (Apple M2) | mlx-swift `MlxEngine` mean+L2 | **Metal** | `make test-turboembed-apple` |

NVIDIA fields: `device=CUDA`, `dims`, `worst_cosine` vs
`testdata/e2e/goldens/nvidia/minilm.json` (`parity:*` + hello world),
floor 0.99, `pass=true`, git sha, `/proc/self/maps` needles.

Intel: same MiniLM IR (`models/ov/minilm`). Cosine vs
`testdata/e2e/goldens/{intel,nvidia}/minilm.json` stays ≥ 0.99 on **both**
devices (CPU is the same IR, not a mock). GPU create/load still fails if
the GPU plugin is missing — that path never compiles `"CPU"`.
`intel-npu.json` is an honest **fail** receipt (`pass=false`): krick-1 has
no Intel NPU / no `libopenvino_intel_npu_plugin.so`. Do not treat it as
a MiniLM success.

Apple: `Engine::create(Metal|AUTO)` lists catalog `minilm` dim 384 (never
mock-only). `turboembed_embed(minilm)` on `Device(gpu, 0)`. Cosine vs nvidia
goldens ≥ 0.97 (CJK UNK floor) and apple goldens ≥ 0.99. Dim 384,
mean+L2, not BERT `tanh(dense(CLS))`, not 8-d FNV mock.

Policy tests (synthetic device list, no embed):
`cargo test -p turboembed --features genai --test device_policy`.
`device=GPU` + listed `CPU` only → `UNSUPPORTED_DEVICE` (loud fail).
`device=CPU` + listed `CPU,GPU` → `"CPU"`.

Intel receipt fields: `device`, `cosine` (floor 0.99), `sha` (git + IR bins).

Do not hand-edit a passing receipt. Do not delete nvidia/intel/apple files
when refreshing one host.
