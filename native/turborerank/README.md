# native/turborerank

C++ implementation of the frozen C ABI in [`include/turborerank.h`](../../include/turborerank.h)
and the buffer/forward contract in [`include/reranker.hpp`](../../include/reranker.hpp).

**CPU** MiniLM-L6 CE with 64-byte aligned, caller-written token buffers.
**CUDA** (Phase 2a): `cudaHostAllocMapped` PINNED token workspace
(kernels read mapped pointers; 0 token-row H2D); device BERT
(cuBLASLt linear layers + first-party attention/LN/GELU). **OpenVINO** (Phase 2b): Level Zero USM
token workspace; `ov::Tensor(..., usm_pointer)` + CompiledModel on GPU
or explicit CPU. **Metal** (Phase 2c, Machine C):
`turbo_buffer` Metal SHARED (`MTLResourceStorageModeShared`) token
workspace; first-party Metal MiniLM CE kernels bind those buffers
via `turbo_buffer_metal_lookup`. TensorRT / NPU create and
`forward` fail loud.
No mock relevance scores. GPU create without that GPU fails loud
(never silent CPU).

```bash
make turborerank-tests          # buffer / pack / CUDA if nvcc
make turborerank-tests-nocuda   # prove CUDA create fails loud
make fetch-rerankers            # SHA-pin ms-marco-MiniLM-L6-v2
make test-turborerank           # C++ + Rust, including live scores
make test-turborerank-nvidia    # Machine A receipt
make convert-rerank-ov          # ONNX → SHA-pinned IR
make test-turborerank-intel     # Machine B OpenVINO receipt
make test-turborerank-apple     # Machine C Metal receipt
make turborerank-tests-nometal  # prove Metal create fails loud
```

See [`docs/turborerank-architecture.md`](../../docs/turborerank-architecture.md).
