# OpenNLP integration design (proposal — not shipped)

Status: **design only.** No OpenNLP integration module exists in this
repository, and none has been validated. This document records the design the
roadmap's M6 asks for so implementation can be scoped and reviewed separately.
The M6 acceptance — base OpenNLP runs without the integration installed, an
opt-in provider executes the selected embedding model in process, and model
provenance and source offsets survive the pipeline — has **not** been met, and
nothing below should be read as a support claim.

Why design-only in this change: an honest opt-in module has to execute MiniLM
through the Java API on a host with the packaged native SDK and a provisioned
bundle, under OpenNLP's own build. That is a new Maven module with an
`opennlp-tools` dependency, plus hardware/model validation — out of scope for
the chunker change this document accompanies, and the roadmap forbids shipping
a stub surface as done.

## Constraints (from the roadmap and library design)

- Base OpenNLP must run without this integration installed. GPU runtimes,
  the native library, and model downloads belong to opt-in
  integration/provisioning modules only ([library design](library-design.md#tokenization-chunking-and-future-analysis)).
- The integration consumes the existing Java API (`ai.pipestream.turboembed`,
  the `turboembed-api` module) — it must not bind the C ABI directly, must not
  require a binding generator, and must not load FFM classes on an unsupported
  JVM. Adapter selection happens once at session creation ([Java FFM guide](java-ffm.md)).
- Device policy is preserved: accelerator selection fails loudly when the
  device is missing; CPU is an explicit choice, never a fallback.
- The planned ~50 OpenNLP analysis features keep their own roadmap. This
  integration is embedding/reranking execution only; Java analysis, native
  algorithm ports, and model inference stay separated.

## Proposed shape

One new opt-in Maven module, `turboembed-opennlp` (groupId
`ai.pipestream.turboembed`), depending on `turboembed-api` and
`opennlp-tools`. Nothing in `turboembed-api` or `turboembed-ffm` changes, and
OpenNLP itself takes no dependency in this direction; applications that want
the provider add the module explicitly.

- **Provider.** A `TurboEmbedVectorizer` (name illustrative) implementing the
  OpenNLP-side embedding interface chosen at implementation time, owning a
  `TurboEmbed` session, a loaded `Model`, and an `ExecutionSlot`. It is
  `AutoCloseable` with deterministic release order (result → slot → model →
  session), mirroring the native lifetime rules.
- **Construction, not discovery.** The provider is constructed explicitly
  with a bundle path, an explicit `Device`, and batch/sequence limits. No
  `ServiceLoader` registration in the first version: OpenNLP must behave
  identically with the JAR absent, and silent classpath-driven provider
  selection makes device policy and failure modes ambiguous. A `ServiceLoader`
  facade can be added later if OpenNLP upstream wants one, provided absence
  stays a non-error.
- **Provenance.** The native bundle manifest already pins model id, revision,
  hashes, and license; the provider surfaces `ModelInfo` (including model and
  tokenizer identity) on every result batch so downstream OpenNLP annotations
  can record which checkpoint produced each vector.
- **Source offsets.** Inputs are OpenNLP `Span`s (UTF-16 offsets over the
  caller's document) or plain strings. The provider carries the caller's spans
  through unchanged and attaches them to results; it does not re-segment text.
  If the optional Rust chunker ([chunking guide](chunking.md)) is ever exposed
  to Java, its UTF-8 byte offsets convert to UTF-16 in the Java adapter only
  on request, per the library design. Until then, chunk planning on the Java
  side is the caller's responsibility.
- **Provisioning.** Model download/verification stays in the existing
  manifest/fetch tooling and SDK provisioning scripts ([native SDK guide](native-sdk.md#provision-a-model)).
  The integration module never downloads models at runtime.
- **Reranking** follows the same pattern against the TurboRerank surface once
  that surface passes its own packaging gates; it is not part of the first
  integration milestone.

## Acceptance before any "shipped" claim

1. Base OpenNLP test suite passes with the integration JAR absent.
2. On a host with the packaged SDK and a provisioned MiniLM bundle, the
   provider embeds text and prepared tokens in process on the selected device,
   with parity against the native contract cases (empty text, embedded NUL,
   Unicode, invalid options, close-during-use, multiple engines).
3. A pipeline test shows model identity/revision and the caller's source
   spans intact on the output annotations.
4. Packaging evidence: the module resolves from a clean consumer project, and
   the missing-accelerator error path is demonstrated, not just documented.

Each state — local test, hosted CI, publication — is recorded separately when
the work happens.
