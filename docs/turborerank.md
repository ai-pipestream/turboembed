# TurboRerank

Library-first BERT cross-encoder. Frozen C ABI:
[`include/turborerank.h`](../include/turborerank.h). C++ contract:
[`include/reranker.hpp`](../include/reranker.hpp). Architecture and
research notes: [`docs/turborerank-architecture.md`](turborerank-architecture.md).

**CPU** (Phase 1), **CUDA** (Phase 2a, Machine A), **OpenVINO**
(Phase 2b, Machine B — GPU + explicit CPU), and **Metal** (Phase 2c,
Machine C) are live. TensorRT / OpenVINO NPU fail loud (`UNAVAILABLE` /
`NOT_IMPLEMENTED`) and never return a mock relevance score. CUDA
create without a CUDA device, OpenVINO GPU create without a GPU, and
Metal create without Metal, also fail loud — never a silent CPU engine.

```bash
make fetch-rerankers            # SHA-256 pin ms-marco-MiniLM-L6-v2
make turborerank-tests          # C++ buffer / pack / CUDA-if-present
make turborerank-tests-nocuda   # same sources, CUDA create fails loud
make test-turborerank           # fetch + C++ + Rust including live scores
make test-turborerank-nvidia    # Machine A CUDA receipt + HF golden
make convert-rerank-ov          # ONNX → SHA-pinned OpenVINO IR (C++)
make test-turborerank-intel     # Machine B OpenVINO GPU/CPU receipt + HF golden
make test-turborerank-apple     # Machine C Metal receipt + HF golden
cargo test -p turborerank       # ABI + pack (skips live scores if no weights)
```

Model: `cross-encoder/ms-marco-MiniLM-L6-v2` @
`233902d25c440f23af6f7d6e94d2946bac0bee0a` (Apache-2.0).
Alias `ms-marco-minilm-l6`. Manifest: `models/manifests/rerankers.json`.

`turborerank_forward` takes caller-written `[CLS] query [SEP] doc [SEP]`
int32 buffers. CPU: 64-byte `posix_memalign`. CUDA / AUTO-with-CUDA:
`cudaHostAlloc` pinned (caller writes tokens in host-visible pinned
memory). OpenVINO GPU / explicit OV CPU: Level Zero USM, wrapped with
`ov::Tensor(..., usm_pointer)`. Metal / AUTO-with-Metal:
`turbo_buffer` Metal SHARED (`MTLResourceStorageModeShared`; caller
writes unified memory; kernels bind those MTLBuffers via
`turbo_buffer_metal_lookup` — no extra token copy, no private token
alloc). No `std::vector` on that path. Weights are copied once at
load into MTL shared buffers. Machine C proof:
[`docs/apple-turbo-buffer-metal-arena-machine-c.md`](apple-turbo-buffer-metal-arena-machine-c.md).

CUDA compute is **on device**: weights and activations live on the GPU.
GEMM, embeddings, LayerNorm, GELU (erf), attention, pooler, and
classifier are first-party CUDA kernels matching the CPU BERT graph
(same reduction order as `linear_nt`). Each forward copies one packed
int32 row H2D from the pinned workspace (the fast path `cudaHostAlloc`
exists for). This is not a pinned-host CPU interim and not a
word-overlap mock.

`AUTO` on a CUDA host resolves to `TURBORERANK_DEVICE_CUDA`. On an
Intel GPU host without CUDA it resolves to
`TURBORERANK_DEVICE_OPENVINO_GPU`. On Machine C (no CUDA / OpenVINO
GPU) it resolves to `TURBORERANK_DEVICE_METAL`.

Default score is `sigmoid(CLS logit)`; `IDENTITY` returns the raw logit
(sentence-transformers default for this checkpoint).

Receipts: `testdata/receipts/turborerank/cpu-minilm-l6.json`,
`nvidia-minilm-l6.json` (Machine A), `intel-minilm-l6.json` and
`intel-cpu-minilm-l6.json` (Machine B), `apple-minilm-l6.json`
(Machine C). Goldens:
`testdata/reference_rerank/ms_marco_minilm_l6_berlin.json`.

gRPC `Rerank` is a thin façade over this ABI when the arch binary is
built with `--features turborerank` (nvidia / intel) or when the Swift
server on Machine C serves `ms-marco-minilm-l6`. Scores are
`sigmoid(CLS logit)` in **input order**; the RPC layer still sorts and
applies `top_n`. Missing weights or device fail at startup — never
word-overlap. The mock backend keeps word-overlap for explicit
`backend = "mock"` models only.

```bash
# Machine A (NVIDIA): CUDA MiniLM CE on the wire
cargo build -p inferstream-arch-nvidia --release --features "ort-cuda,turborerank"
# then add "ms-marco-minilm-l6" to serve in config/nvidia.toml

# Machine B (Intel): OpenVINO MiniLM CE on the wire
# scripts/build-intel.sh plus --features turborerank; make convert-rerank-ov

# Machine C (Apple): Swift server links libturborerank_apple.a (make apple)
```
