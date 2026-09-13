# native/turborerank

C++ implementation of the frozen C ABI in [`include/turborerank.h`](../../include/turborerank.h)
and the buffer/forward contract in [`include/reranker.hpp`](../../include/reranker.hpp).

**CPU** MiniLM-L6 CE with 64-byte aligned, caller-written token buffers.
**CUDA** (Phase 2a): `cudaHostAlloc` pinned token workspace; device BERT
(first-party CUDA kernels). OpenVINO / Metal / TensorRT create
and `forward` fail loud. No mock relevance scores. CUDA create without a
device fails loud (never silent CPU).

```bash
make turborerank-tests          # buffer / pack / CUDA if nvcc
make turborerank-tests-nocuda   # prove CUDA create fails loud
make fetch-rerankers            # SHA-pin ms-marco-MiniLM-L6-v2
make test-turborerank           # C++ + Rust, including live scores
make test-turborerank-nvidia    # Machine A receipt
```

See [`docs/turborerank-architecture.md`](../../docs/turborerank-architecture.md).
