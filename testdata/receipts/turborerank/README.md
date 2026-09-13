# TurboRerank live receipts

| file | Machine | engine | device |
|---|---|---|---|
| `cpu-minilm-l6.json` | any | first-party FP32 MiniLM CE | **CPU** |
| `nvidia-minilm-l6.json` | Machine A | CUDA MiniLM CE (PINNED mapped tokens + DEVICE activations + cuBLASLt GEMM; `h2d_per_row` == 0) | **CUDA** |
| `intel-minilm-l6.json` | Machine B | OpenVINO CompiledModel + turbo_buffer ZE SHARED USM; FP32 IR; remote OCL wrap unavailable | **OPENVINO_GPU** |
| `intel-cpu-minilm-l6.json` | Machine B | OpenVINO CompiledModel (explicit CPU); same FP32 IR | **OPENVINO_CPU** |
| `intel-remote-usm-probe.txt` | Machine B | Live `USM_USER_BUFFER` wrap probe (OCL size-0 fail) | **OPENVINO_GPU** |
| `apple-minilm-l6.json` | Machine C | first-party Metal MiniLM CE (`turbo_buffer` Metal SHARED + device kernels) | **METAL** |

TensorRT / NPU receipts land here when those backends exist —
never invent numbers, never put lab hostnames in docs (Machine A / B / C
only).

Proof: `make test-turborerank` / `make test-turborerank-nvidia` /
`make test-turborerank-intel` / `make test-turborerank-apple` vs
`testdata/reference_rerank/ms_marco_minilm_l6_berlin.json` (HF
`AutoModelForSequenceClassification` on the pinned checkpoint).
Machine B latency + USM-byte honesty:
`make bench-machine-b-ov` → `testdata/receipts/bench/machine-b-ov.json`.
gRPC `Rerank` matches the same goldens via
`crates/backend-turborerank` when `--features turborerank` is on.
