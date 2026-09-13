# TurboRerank live receipts

| file | Machine | engine | device |
|---|---|---|---|
| `cpu-minilm-l6.json` | any | first-party FP32 MiniLM CE | **CPU** |
| `nvidia-minilm-l6.json` | Machine A | first-party CUDA MiniLM CE (`cudaHostAlloc` + device kernels) | **CUDA** |
| `intel-minilm-l6.json` | Machine B | OpenVINO CompiledModel + Level Zero USM | **OPENVINO_GPU** |
| `intel-cpu-minilm-l6.json` | Machine B | OpenVINO CompiledModel (explicit CPU) | **OPENVINO_CPU** |

Metal / TensorRT / NPU receipts land here when those backends exist —
never invent numbers, never put lab hostnames in docs (Machine A / B / C
only).

Proof: `make test-turborerank` / `make test-turborerank-nvidia` /
`make test-turborerank-intel` vs
`testdata/reference_rerank/ms_marco_minilm_l6_berlin.json` (HF
`AutoModelForSequenceClassification` on the pinned checkpoint).
