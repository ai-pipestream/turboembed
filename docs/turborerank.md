# TurboRerank

Library-first BERT cross-encoder. Frozen C ABI:
[`include/turborerank.h`](../include/turborerank.h). C++ contract:
[`include/reranker.hpp`](../include/reranker.hpp). Architecture and
research notes: [`docs/turborerank-architecture.md`](turborerank-architecture.md).

Phase 1 is **CPU only**. `Device::Cuda` / `Metal` / `Auto` / OpenVINO
fail loud (`UNAVAILABLE` / `NOT_IMPLEMENTED`) and never return a mock
relevance score.

```bash
make fetch-rerankers            # SHA-256 pin ms-marco-MiniLM-L6-v2
make turborerank-tests          # C++ buffer / pack / fail-loud
make test-turborerank           # fetch + C++ + Rust including live scores
cargo test -p turborerank       # ABI + pack (skips live scores if no weights)
```

Model: `cross-encoder/ms-marco-MiniLM-L6-v2` @
`233902d25c440f23af6f7d6e94d2946bac0bee0a` (Apache-2.0).
Alias `ms-marco-minilm-l6`. Manifest: `models/manifests/rerankers.json`.

`turborerank_forward` takes caller-written `[CLS] query [SEP] doc [SEP]`
int32 buffers (64-byte aligned on CPU). No `std::vector` on that path.
Default score is `sigmoid(CLS logit)`; `IDENTITY` returns the raw logit
(sentence-transformers default for this checkpoint).

gRPC `Rerank` stays on the mock backend until a later façade calls this
ABI. Do not point the RPC at fake MiniLM scores.
