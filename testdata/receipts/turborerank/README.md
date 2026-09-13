# TurboRerank live receipts

| file | Machine | engine | device |
|---|---|---|---|
| `cpu-minilm-l6.json` | any | first-party FP32 MiniLM CE | **CPU** |
| `nvidia-minilm-l6.json` | Machine A | CUDA MiniLM CE (PINNED mapped tokens + DEVICE activations + cuBLASLt GEMM; `h2d_per_row` == 0) | **CUDA** |
| `intel-minilm-l6.json` | Machine B | OpenVINO CompiledModel + turbo_buffer ZE SHARED USM | **OPENVINO_GPU** |
| `intel-cpu-minilm-l6.json` | Machine B | OpenVINO CompiledModel (explicit CPU) | **OPENVINO_CPU** |
| `apple-minilm-l6.json` | Machine C | first-party Metal MiniLM CE (`turbo_buffer` Metal SHARED + device kernels) | **METAL** |

TensorRT / NPU receipts land here when those backends exist —
never invent numbers, never put lab hostnames in docs (Machine A / B / C
only).

Proof: `make test-turborerank` / `make test-turborerank-nvidia` /
`make test-turborerank-intel` / `make test-turborerank-apple` vs
`testdata/reference_rerank/ms_marco_minilm_l6_berlin.json` (HF
`AutoModelForSequenceClassification` on the pinned checkpoint).
gRPC `Rerank` matches the same goldens via
`crates/backend-turborerank` when `--features turborerank` is on.
