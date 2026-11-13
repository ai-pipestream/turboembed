# turbo-client

`turbo-client` (`clients/rust`) is the unified pure-Rust thick client for `turboembed`.

## Execution Modes

- **In-process lexical and NLP (`default-features = false`):** Runs Unicode normalization chains, UAX #29 / whitespace / simple tokenization, Porter and 21-language Snowball stemming, `(pos, pos_len)` lexical lattices (`TokenGraph`), `MappedSurfaceLexicon` stem-to-surface completion, `QuerySpellchecker` contextual bigram spellcheck, `OnePassDocumentNlp` column extraction, and `OnePassQueryAnalyzer` structured filter planning with zero network sockets and bidirectional `ViewId::SOURCE` UTF-8, UTF-16, and code-point offsets.
- **In-process neural inference (`feature = "local-core"`):** Drives `libturbo` (`core/`) and `turbo-analysis-embed` directly in-process for single-pass multi-resolution dense + Matryoshka + `TurboQuant` + sentence range vectors, calibrated `u8`/`u16` SPLADE impact quantization, and calibrated cross-encoder reranking with source-projected semantic highlights.
- **Runtime-scoped pooled gRPC offload (`feature = "grpc"`, optional `feature = "tls"`):** Connects to `turbo-kserve` over HTTP/2 channels scoped to `(tokio::runtime::Id, RemoteEndpointConfig)` with automatic channel eviction on runtime shutdown, bearer authentication, `x-turbo-priority` (`interactive` vs `bulk`) lane headers, `grpc-timeout` deadline propagation, and `EmbedCoalescer` concurrent query micro-batching.

## Index Generation Handshakes

`IndexGenerationHandshake` verifies `AnalysisFingerprint` (domain-separated SHA-256 over the normalizer chain, tokenizer, stemmer, term source, and `UNICODE_VERSION`) and `BundleScoringFingerprint` (domain-separated SHA-256 over `model_id`, `revision`, `manifest_sha256`, `task`, `dim`, `pooling`, and `normalize`) at client build or connect time, returning `ClientError::FailedPrecondition` before indexing or query execution if either contract drifts.
