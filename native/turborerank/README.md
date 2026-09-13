# native/turborerank

C++ implementation of the frozen C ABI in [`include/turborerank.h`](../../include/turborerank.h)
and the buffer/forward contract in [`include/reranker.hpp`](../../include/reranker.hpp).

Phase 1: **CPU MiniLM-L6 cross-encoder** with 64-byte aligned, caller-written
token buffers. CUDA / OpenVINO / Metal create and `forward` fail loud.
No mock relevance scores.

```bash
make turborerank-tests          # buffer / pack / fail-loud (no weights)
make fetch-rerankers            # SHA-pin ms-marco-MiniLM-L6-v2
make test-turborerank           # C++ + Rust, including live scores
```

See [`docs/turborerank-architecture.md`](../../docs/turborerank-architecture.md).
