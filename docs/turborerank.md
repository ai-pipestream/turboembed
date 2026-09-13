# TurboRerank

Library-first BERT cross-encoder. Frozen C ABI:
[`include/turborerank.h`](../include/turborerank.h). C++ contract:
[`include/reranker.hpp`](../include/reranker.hpp). Architecture and
research notes: [`docs/turborerank-architecture.md`](turborerank-architecture.md).

**CPU** (Phase 1), **CUDA** (Phase 2a, Machine A), and **OpenVINO**
(Phase 2b, Machine B — GPU + explicit CPU) are live.
`Device::Metal` / TensorRT / OpenVINO NPU fail loud (`UNAVAILABLE` /
`NOT_IMPLEMENTED`) and never return a mock relevance score. CUDA
create without a CUDA device, and OpenVINO GPU create without a GPU,
also fail loud — never a silent CPU engine.

```bash
make fetch-rerankers            # SHA-256 pin ms-marco-MiniLM-L6-v2
make turborerank-tests          # C++ buffer / pack / CUDA-if-present
make turborerank-tests-nocuda   # same sources, CUDA create fails loud
make test-turborerank           # fetch + C++ + Rust including live scores
make test-turborerank-nvidia    # Machine A CUDA receipt + HF golden
make convert-rerank-ov          # ONNX → SHA-pinned OpenVINO IR (C++)
make test-turborerank-intel     # Machine B OpenVINO GPU/CPU receipt + HF golden
cargo test -p turborerank       # ABI + pack (skips live scores if no weights)
```

Model: `cross-encoder/ms-marco-MiniLM-L6-v2` @
`233902d25c440f23af6f7d6e94d2946bac0bee0a` (Apache-2.0).
Alias `ms-marco-minilm-l6`. Manifest: `models/manifests/rerankers.json`.

`turborerank_forward` takes caller-written `[CLS] query [SEP] doc [SEP]`
int32 buffers. CPU: 64-byte `posix_memalign`. CUDA / AUTO-with-CUDA:
`cudaHostAlloc` pinned (caller writes tokens in host-visible pinned
memory). OpenVINO GPU / explicit OV CPU: Level Zero USM, wrapped with
`ov::Tensor(..., usm_pointer)`. No `std::vector` on that path.

CUDA compute is **on device**: weights and activations live on the GPU.
GEMM, embeddings, LayerNorm, GELU (erf), attention, pooler, and
classifier are first-party CUDA kernels matching the CPU BERT graph
(same reduction order as `linear_nt`). Each forward copies one packed
int32 row H2D from the pinned workspace (the fast path `cudaHostAlloc`
exists for). This is not a pinned-host CPU interim and not a
word-overlap mock.

`AUTO` on a CUDA host resolves to `TURBORERANK_DEVICE_CUDA`. On an
Intel GPU host without CUDA it resolves to
`TURBORERANK_DEVICE_OPENVINO_GPU`.

Default score is `sigmoid(CLS logit)`; `IDENTITY` returns the raw logit
(sentence-transformers default for this checkpoint).

Receipts: `testdata/receipts/turborerank/cpu-minilm-l6.json`,
`nvidia-minilm-l6.json` (Machine A), `intel-minilm-l6.json` and
`intel-cpu-minilm-l6.json` (Machine B). Goldens:
`testdata/reference_rerank/ms_marco_minilm_l6_berlin.json`.

gRPC `Rerank` stays on the mock backend until a later façade calls this
ABI. Do not point the RPC at fake MiniLM scores.
