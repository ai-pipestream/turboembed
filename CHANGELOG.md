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

### Not in this release yet

Open in the Integration v0 milestone, and listed here as each lands:
versioned `turbo.<name>.v1` gRPC packages and a breaking-change gate
(#1296), published client stubs (#1297), container images (#1298), TLS,
metrics and deadlines (#1300), an integration guide (#1301) and named
analysis profiles (#1302).
