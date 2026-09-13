# TurboRerank live receipts

| file | Machine | engine | device |
|---|---|---|---|
| `cpu-minilm-l6.json` | any | first-party FP32 MiniLM CE | **CPU** |
| `nvidia-minilm-l6.json` | Machine A | first-party CUDA MiniLM CE (`cudaHostAlloc` + device kernels) | **CUDA** |

Metal / OpenVINO / TensorRT receipts land here when those backends exist —
never invent numbers, never put lab hostnames in docs (Machine A / B / C
only).

Proof: `make test-turborerank` / `make test-turborerank-nvidia` vs
`testdata/reference_rerank/ms_marco_minilm_l6_berlin.json` (HF
`AutoModelForSequenceClassification` on the pinned checkpoint).
