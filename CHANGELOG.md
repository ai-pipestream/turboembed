# Changelog

What changed in each release of turboembed, newest first. Every pull
request that changes the C ABI, the gRPC services, the server's flags or
variables, the bundle format, or what a release archive or image holds
adds a line under the top heading, in one of **Added**, **Changed**,
**Fixed** or **Breaking**, with its pull request number. What each of
those may contain in a minor or a patch release is in docs/release.md,
Versions and compatibility.

The top heading is the next release. Once it is tagged, a new heading
for the release after it goes above it.

## v0.2.0 (not yet tagged)

### Added

- `DocumentSinkItem.typed` (field 9) carries the item's graph as
  `InspectDocumentGraph` projects it: `--sink-typed-offsets` and
  `--sink-typed-view` (`TURBO_KSERVE_SINK_TYPED_OFFSETS`,
  `TURBO_KSERVE_SINK_TYPED_VIEW`, `SinkConfig::typed`) choose the offset
  mode and view, and without them items are unchanged. `--sink-documents`
  (`TURBO_KSERVE_SINK_DOCUMENTS`) forwards the complete graph of every
  `AnalyzeDocumentProgressive` call, `Server::forward_document` forwards
  one from a program embedding the server, `SinkStats::refused` counts
  graphs refused before the queue, and the sink refuses a projection
  `RoundTripTypedDocument` refuses (docs/grpc-sink.md). (#1787)
- `AnalyzeDocumentProgressive` runs a bundle's learned heads as steps
  (`turbo-tagger:pos`, `turbo-tagger:entities`, `turbo-tagger:dependencies`,
  `turbo-tagger:lemmas`, `turbo-tagger:morphology`, `turbo-tagger:spans`,
  `turbo-tagger:relations`, and `turbo-tagger:heads` for every one the
  bundle provides), the word-level ones from one encoder pass; the
  `annotate-v1` profile runs the part-of-speech, entity, dependency and
  lemma heads, and `BundleCapabilities.head_steps` lists each loaded
  bundle's head steps (docs/grpc-analysis.md). (#1775)
- The C interface's reranking (#759) and extractive QA (#767) tasks.
- In `turbo-kserve`: the analysis service (docs/grpc-analysis.md), the
  search, vocabulary, catalog and sink services, and `grpc.health.v1`;
  per-stream bounds.
- The analysis layers behind the analysis service: normalisation with
  offsets kept across every view, tokenisation, segmentation, chunking,
  and the OpenNLP components ported so far (docs/analysis.md).
- Release archives with SHA-256 sums for Linux aarch64 (cpu) and Windows
  x86_64 (cpu), each with the library, the headers and the server
  (docs/release.md) (#1326).
- This changelog, and the compatibility policy in docs/release.md.
- Versioned `turbo.<name>.v1` gRPC packages, with a `buf breaking`
  check against main in CI (#1342).
- Container images of `turbo-kserve`, CPU and CUDA, pushed to GHCR for
  each tag (docs/release.md) (#1327).
- TLS and mTLS, Prometheus metrics, and gRPC deadlines on the server
  (docs/grpc.md) (#1343).
- The integration guide (docs/integration.md) and server examples with
  Java and Python clients (#1331, #1422).
- Named analysis profiles on `AnalyzeDocumentProgressive`: `default-v1`,
  `retrieval-v1` and `minimal-v1` (docs/grpc-analysis.md) (#1340).
- The ModernBERT and nomic-bert families in the bundle format and
  `turbo_backend.h`: rotary positions, sliding-window attention, gated
  MLPs and bias-free norms, with the core's cos/sin table appended to
  `turbo_backend_model`; no backend runs them yet (#1493).
- The CPU encoder for ModernBERT and nomic-bert: RoPE in the QKV
  product's epilogue over pair-adjacent weights, sliding-window attention
  over each block's band of keys, GeGLU and SwiGLU in one product, held to
  upstream on the sealed tiny bundles (#1544).
- The Level Zero encoder for ModernBERT and nomic-bert, in F32 at MODEL
  and EXACT and with F16 linear layers on the matrix engines at FASTEST:
  RoPE from the core's table, windowed attention, GeGLU and SwiGLU, held
  to the CPU and to upstream's reference vectors (#1461).
- The CUDA encoder for ModernBERT and nomic-bert at MODEL (F32) and
  FASTEST (F16 on the tensor cores): RoPE and the gated MLP in the GEMMs'
  epilogues over pair-adjacent weights, sliding-window attention that
  reads only each block's band of keys, held to the CPU (#1580).
- On cuda, a FASTEST session whose max_seq is over 4096 takes the
  tensor cores' attention with F32 sums of the scores and the output,
  whatever the model's family (BERT and XLM-R too, bge-m3 at 8192 for
  one): the F16 sums went over FASTEST's cosine bound at 8192 tokens.
  That attention is slower; sessions of 4096 or fewer keep the F16 sums
  (#1580).

### Breaking

- The shared analysis layers' ids moved from the `opennlp:` namespace
  to `turbo:`: `turbo:sentences`, `turbo:tokens`, `turbo:subwords`,
  `turbo:entities`, `turbo:pos`, `turbo:lemmas`, `turbo:language`,
  `turbo:embeddings`, `turbo:dependencies`, `turbo:chunk-tags`,
  `turbo:phrases`, `turbo:mentions`, `turbo:speakers` and
  `turbo:relations`; phrase chunks are `turbo:phrase-chunks`, apart
  from the retrieval chunks' `turbo:chunks`. `TypedLayer.id`,
  `LayerInfo`, DocumentGraph bytes and C ABI layer names carry the new
  ids. Readers still accept the old ids: a DocumentGraph, a golden file
  or a progressive step name written with one reads as the new id. The
  document categorizer and dependency parser steps are
  `turbo-doccat:maxent`, `turbo-doccat:maxent-spans`, `turbo-doccat:dl`,
  `turbo-doccat:dl-spans` and `turbo-depparse:parser` (#1780).

### Not in this release yet

Published client stubs (#1297) are still open in the Integration v0
milestone.

## v0.1.0 (2026-10-09)

The first release. Everything below is new.

### Added

- The C interface, `include/turbo/turbo.h` and `turbo_backend.h`: the
  text embedding task; runtimes, devices and their capability cells;
  sessions at the `MODEL`, `FASTEST` and `EXACT` precision tiers; device
  selection from the benchmark records the library carries.
- Backends: cpu everywhere; cuda (NVIDIA, Turing and later), levelzero
  (Intel Arc Xe2), npu (Intel Core Ultra), metal (Apple silicon) and hailo
  (Hailo-8 and Hailo-10H), each a Cargo feature (README.md).
- The bundle format, `bundle_version` 1 (docs/bundle.md), with reference
  vectors a bundle carries and the conformance check against them
  (docs/conformance.md).
- `turbo-kserve`, the gRPC server (docs/grpc.md): the Open Inference
  Protocol, configured by flags or `TURBO_KSERVE_*` variables, with a
  message size limit.
- Release archives with SHA-256 sums: Linux x86_64 (cpu and cuda), with
  the library, the headers and the server, and macOS arm64 (metal and
  cpu), with the library and the headers (docs/release.md).
