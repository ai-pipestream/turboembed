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

## v0.1.0 (not yet tagged)

The first release, cut when the Integration v0 milestone closes: the
first one other ai-pipestream products (protomolt, grparse, the docling
gRPC service) build against. Everything below is new.

### Added

- `AnalyzeDocumentProgressive` runs a bundle's learned heads as steps
  (`turbo-tagger:pos`, `turbo-tagger:entities`, `turbo-tagger:dependencies`,
  `turbo-tagger:lemmas`, `turbo-tagger:morphology`, `turbo-tagger:spans`,
  `turbo-tagger:relations`, and `turbo-tagger:heads` for every one the
  bundle provides), the word-level ones from one encoder pass; the
  `annotate-v1` profile runs the part-of-speech, entity, dependency and
  lemma heads, and `BundleCapabilities.head_steps` lists each loaded
  bundle's head steps (docs/grpc-analysis.md). (#1775)
- The C interface, `include/turbo/turbo.h` and `turbo_backend.h`: the
  text embedding, reranking and extractive QA tasks; runtimes, devices and their capability cells; sessions at the
  `MODEL`, `FASTEST` and `EXACT` precision tiers; device selection from
  the benchmark records the library carries.
- Backends: cpu everywhere; cuda (NVIDIA, Turing and later), levelzero
  (Intel Arc Xe2), npu (Intel Core Ultra), metal (Apple silicon) and hailo
  (Hailo-8 and Hailo-10H), each a Cargo feature (README.md).
- The bundle format, `bundle_version` 1 (docs/bundle.md), with reference
  vectors a bundle carries and the conformance check against them
  (docs/conformance.md).
- `turbo-kserve`, the gRPC server (docs/grpc.md): the Open Inference
  Protocol, the analysis service (docs/grpc-analysis.md), the search,
  vocabulary, catalog and sink services, and `grpc.health.v1`;
  configured by flags or `TURBO_KSERVE_*` variables, with a message size
  limit and per-stream bounds.
- The analysis layers behind the analysis service: normalisation with
  offsets kept across every view, tokenisation, segmentation, chunking,
  and the OpenNLP components ported so far (docs/analysis.md).
- Release archives with SHA-256 sums: Linux x86_64 (cpu and cuda), macOS
  arm64 (metal and cpu), Linux aarch64 (cpu) and Windows x86_64 (cpu),
  each with the library, the headers and the server (docs/release.md).
- This changelog, and the compatibility policy in docs/release.md.
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

### Not in this release yet

Open in the Integration v0 milestone, and listed here as each lands:
versioned `turbo.<name>.v1` gRPC packages and a breaking-change gate
(#1296), published client stubs (#1297), container images (#1298), TLS,
metrics and deadlines (#1300), an integration guide (#1301) and named
analysis profiles (#1302).
