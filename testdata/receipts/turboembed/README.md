# TurboEmbed live receipts

JSON written by a **live** `cargo test -p turboembed` on the named Machine.
Not mocks. Not OVMS. Not a CPU EP pretending to be CUDA. Not BERT pooler.
Catalog aliases never become 8-d FNV.

Receipt `host` is historical provenance (the lab box that wrote the file).
Do not treat that string as a public hostname. Map it through the Machine
chart in the root `README.md` using the file prefix / engine:

| files | Machine | Role |
|---|---|---|
| `nvidia-*.json` | **A** | NVIDIA Linux (ORT CUDA / CPU / TensorRT) |
| `intel-*.json` | **B** | Intel Linux (GenAI GPU+CPU; NPU fail-loud) |
| `apple-*.json` | **C** | Apple Silicon Mac (Metal / MLX) |

Do not rewrite committed receipt JSON to invent fields or cosine numbers.
Do not delete a receipt when refreshing another Machine.

Committed set: NVIDIA CUDA + CPU + **TensorRT**, Intel GPU + CPU +
**NPU fail**, Apple Metal.

| file | Machine | engine | device | command |
|---|---|---|---|---|
| `nvidia-minilm.json` | Machine A (RTX 4080 SUPER) | ORT CUDA EP + IoBinding | **CUDA** | `make test-turboembed-nvidia` |
| `nvidia-minilm-cpu.json` | Machine A | ORT CPU EP (explicit `Device::Cpu`) | **CPU** | `make test-turboembed-nvidia` |
| `nvidia-minilm-tensorrt.json` | Machine A (RTX 4080 SUPER) | ORT TensorRT EP | **TENSORRT** | ignored `minilm_ort_tensorrt_matches_golden` |
| `intel-minilm.json` | Machine B (Battlemage) | GenAI Tokenizer + CompiledModel, ZE SHARED USM | **GPU** | `make test-turboembed-intel` |
| `intel-minilm-cpu.json` | Machine B | same path, `"CPU"` + HOST USM | **CPU** | `make test-turboembed-intel` |
| `intel-npu.json` | Machine B | **defer** — NPU create fails loud (no plugin) | **NPU** | `make test-turboembed-intel` |
| `apple-minilm.json` | Machine C (Apple M2) | mlx-swift `MlxEngine` mean+L2 | **Metal** | `make test-turboembed-apple` |

NVIDIA fields: `device=CUDA`, `dims`, `worst_cosine` vs
`testdata/e2e/goldens/nvidia/minilm.json` (`parity:*` + hello world),
floor 0.99, `pass=true`, git sha, `/proc/self/maps` needles.

Intel: same MiniLM IR (`models/ov/minilm`). Cosine vs
`testdata/e2e/goldens/{intel,nvidia}/minilm.json` stays ≥ 0.99 on **both**
devices (CPU is the same IR, not a mock). GPU create/load still fails if
the GPU plugin is missing — that path never compiles `"CPU"`.
`intel-npu.json` is an honest **fail** receipt (`pass=false`): Machine B has
no Intel NPU / no `libopenvino_intel_npu_plugin.so`. A passing NPU
receipt needs a **Core Ultra client NPU** host — not Xeon, not AWS
Inferentia, not this Battlemage box. Do not treat the fail file as a
MiniLM success.

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

**TensorRT:** `nvidia-minilm-tensorrt.json` is a live MiniLM receipt
(`device=TENSORRT`, dim 384, cosine ~1.0, maps `libnvinfer` +
`libonnxruntime_providers_tensorrt`). Needs
`scripts/fetch-runtime-libs.sh nvidia-trt` (not the default CUDA fetch).
