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

### Breaking: PII and the result envelope (#1879)

These change `turbo.result.v1` in place; a client built against v0.1.0
stubs must be regenerated.

- `PiiMention` carries its kind as `kind_ref` (`PiiKindRef`: a stable
  `PiiKindId` or a custom kind index) and its tier as the
  `PiiConfidenceTier` enum `confidence`, and its value as `PiiValue`;
  the string `kind`, `confidence_tier` and clear `normalized` fields
  (1, 2, 3) are deprecated and a mention that sets any of them is
  refused (#1914, #1918).
- `LayerWireStatus` is renumbered with `LAYER_WIRE_STATUS_UNSPECIFIED = 0`,
  and a layer that sends `UNSPECIFIED` is refused (#1912).
- `TypedLayer.status_detail` is replaced by the typed `LayerError`
  (`error`, field 11, with `LayerErrorKind`); the old field is reserved
  (#1912).
- Results are findings only by default: PII values, the source text
  and rewritten text leave the server only when the request's
  `PiiOutputOptions` asks and the server's `--pii-output-allow` allows
  it (#1918).
- `InspectDocumentGraph` refuses a graph whose layers name a view the
  graph does not hold, and validates PII layers and view chains before
  projecting (#1911).

### Added: PII and the result envelope (#1879)

- The result envelope: `InputIdentity`, `ResultIdentity`,
  `LayerProvenance`, `ViewChain`, `EditRun` and `MappingPayload` on
  `TypedDocumentResult`, so a result names what it was computed from
  and how each view maps back to the source (#1912).
- `ProcessingReceipt` and `LayerReceipt` with `ProcessingStatus`,
  `TruncationReason` and the byte ranges each detector covered (#1913).
- Stable PII kind ids (`PiiKindId`), `PiiConfidenceTier`,
  `PiiDetectorRef` and `TypedLayer.custom_pii_kinds` (field 12)
  (#1914).
- C ABI PII rows carry their numeric kind: with
  `turbo_analyzer_desc.flags` bit `TURBO_ANALYZER_PII_KIND_IDS` every
  PII row sets `TURBO_SPAN_PII_KIND` and holds a `TURBO_PII_KIND_*` id,
  or `TURBO_PII_KIND_CUSTOM_BASE` plus a custom-table index, in
  `token_start`. Off by default, so existing callers see the same rows.
  Java `TurboSpanView` and `TurboSpanData` read it as `piiKindCode()`,
  `piiKind()` and `piiCustomKindIndex()` (#1964).
- Rewrites as derived views: the redacted text is a `ViewChain` over
  the source view (`TypedLayer.native_view`, field 10) and every edit is
  a `PiiEditAnnotation` in the `turbo:pii-edits` layer (`pii_edit`,
  field 98) (#1915).
- Candidate decisions and evidence: `PiiCandidateAnnotation`
  (`pii_candidate`, field 99), `PiiDisposition`, `PiiEvidence` (shape,
  checksum, label, deny-list and rule hits) and `PiiScore` (#1916).
- Disclosure options: `PiiOutputOptions` and `PiiValueForm` on
  `InspectDocumentGraph` and on sink items; `--pii-output-allow`
  (`TURBO_KSERVE_PII_OUTPUT_ALLOW`) sets the server ceiling and
  `--sink-pii-output` (`TURBO_KSERVE_SINK_PII_OUTPUT`) what the sink
  forwards (#1918).
- Java conformance smoke for the typed result in `clients/java`
  (`DocumentSmoke`), run in the proto lane (#1919).
- PII value release: per-client grants over mTLS (`--pii-grant`,
  `TURBO_KSERVE_PII_GRANTS`), keyed tokens (`PiiValueToken`,
  `--pii-token-key`, `--pii-token-key-active`,
  `TURBO_KSERVE_PII_TOKEN_KEYS`, `TURBO_KSERVE_PII_TOKEN_KEY_ACTIVE`),
  clear values, and an audit call before any value leaves
  (`turbo.audit.v1.TurboAuditService/RecordPiiAccess`, `--pii-audit-endpoint`,
  `--pii-audit-timeout-ms`, `TURBO_KSERVE_PII_AUDIT_ENDPOINT`,
  `TURBO_KSERVE_PII_AUDIT_TIMEOUT_MS`) (#1920).
- Fragment outcomes: `SpanFragment`, `FragmentMember` and `SpanOutcome`
  say what happened to every piece of a mention a rewrite touched
  (#1921).
- `turbo.result.v1.RedactionPolicy` (`turbo_pii_policy.proto`): typed
  per-kind rules, strategy entries with conditions, allow and deny
  lists, custom identifiers and a canonical policy digest (#1925).
- The `turbo-pii:redact` step and its dry run: a policy chosen per
  request or by name (`--pii-policy`) writes the redacted view and the
  edits layer, and a dry run writes only `PiiPlannedRewrite`
  candidates (#1927).
- `PiiDictionaryRef` resolves through the vocabulary store
  (`--vocabulary-dir`) by artifact id and sha256, in the
  `plain-word-list` and `headword-definition-tsv` formats, with typed
  refusals for a digest mismatch, a wrong format or an unreadable
  artifact. In a dictionary or inline map table, two keys equal once
  trimmed and folded refuse the policy (#1928).
- `PiiValueWithheld.reason` (field 1, `PiiWithheldReason`): each
  withheld mention says why, `NOT_REQUESTED` or `NOT_GRANTED`;
  `UNSPECIFIED` is refused, and `NOT_ALLOWED_BY_POLICY` is defined for a
  per-kind policy ceiling (#1943) but not produced yet. Release tokens are
  bound to the mention's kind, so one value under two kinds gives two
  tokens. Before the graph is read, a values request with no
  `operation_key` is refused with `TURBO_E_INVALID_ARGUMENT`, and one
  from a service granted the form for no kind with
  `TURBO_E_INVALID_STATE` (#1933).
- `PiiAccessMention.rule_id` (6), `policy_revision` (7) and
  `redaction_policy_digest` (8): each released mention names the
  redaction policy rule that covered it, and these feed the audit
  event id (#1928).

### Changed: PII and the result envelope (#1879)

- The release audit refuses to send an event whose value form is
  `UNSPECIFIED` or unknown (#1928).
- docs/grpc.md records the PII and result envelope freeze: what is
  deprecated, what is reserved and which numbers stay unused (#1928).

### Added

- `RegisterModelSource.s3_keys` (field 5, `S3Keys`) and
  `SOURCE_SIGN_IN_S3_KEYS`: an `s3://` model source signs in with its
  own access key pair, region and endpoint instead of the server's
  `AWS_*` environment, and zoo mode takes such a source on AWS
  (`turbo_zoo::SignIn::S3Keys`, `S3Bucket::signed_in`). Zoo mode's
  read-through now fetches through the reading user's model sources
  before the catalog's own, as `InstallModel` does (#1853).
- A pure Java API module shared by the JNI and Panama implementations,
  with typed runtime selection, one adapter package, and shared native
  conformance tests (`clients/java-api`, #1814).

- Morphology rows in the in-process analyzer and the Java bindings
  (`clients/java-panama`, `clients/java-jni`):
  a bundle's morphology head writes `TURBO_ANALYSIS_LAYER_MORPHOLOGY`
  (`TurboLayerKind.MORPHOLOGY`) from the same encoder pass as the other
  word-level heads, one row per token with the feature string as the
  label, and `turbo_analyzer_morphology` (`TurboAnalyzer.morphology`,
  `TurboAnalyzer.morphologyCopied`) tags pre-tokenized tokens (#1812).
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
- `clients/java-jni`, the Java binding over the Java Native Interface for
  Java 21 or newer and for Android, with the Panama binding's API and
  OpenNLP-style adapters: its `native` methods are exported by `libturbo`
  itself, so one library per platform serves both bindings. The release
  gains `turboembed-v<version>-android.tar.gz`, `libturbo.so` for the
  `arm64-v8a` and `x86_64` ABIs at API level 26, and both bindings run the
  conformance suite in `clients/java-conformance` over the sealed
  `testdata` bundles (#1801).
- `AnalyzePipelineRequest.structure_seed`: a document's text and its
  headings, lists, tables and other blocks as a `StructureSeed` protobuf
  message, offsets in the request's unit, built into the same
  `turbo:structure` layer an `initial_graph` carries, so a Java or
  Python client seeds the chunker without a DocumentGraph writer; a
  malformed block is refused naming it (#1777).

### Changed

- The in-process analyzer (`turbo_analyzer_desc`) loads a bundle:
  `bundle_path` names a bundle whose heads write the part-of-speech,
  lemma, dependency and entity layers from one encoder pass a sentence,
  as the Rust annotators write them. The five model path fields are
  retired: they keep their offsets, a `struct_size` of 104 stays accepted
  and names no bundle, and a non-empty value fails with
  `TURBO_E_INVALID_ARGUMENT` naming its field. Dependency rows are the
  new `TURBO_ANALYSIS_LAYER_DEPENDENCIES` layer kind: the relation as the
  label, the head token plus one as the classification, zero for a root.
  The Java Panama binding's `TurboAnalyzer.Config` takes the bundle path
  in place of the model paths, and the binding now requires Java 25 or
  newer, the current long-term release with the final Foreign Function
  and Memory API. The heads' sigmoid and tanh (LSTM adapters, poolers)
  use the cpu backend's own exponential in place of the C library's, so
  a head gives the same bits on every machine (#1779).

### Fixed

- The cpu backend's embed cells set field 9's bit (`layers`) in
  `turbo_capability.options_honored`, which reads `0b1_1111_1111` there,
  and a layer count of 0 or over an encoder's layers is
  `TURBO_E_INVALID_ARGUMENT` naming field 9 on every device, checked
  before whether the device reads the earlier layers; off the CPU it
  was `TURBO_E_UNSUPPORTED_OPTION` (#1851).
- `turbo_analyzer_inspect_graph` refuses a caller's graph holding a
  part-of-speech probability or entity score outside 0..=1, which graph
  import takes, with `TURBO_E_INVALID_ARGUMENT` where it answered
  `TURBO_E_CAPACITY`; every other refusal of what a caller's graph holds
  is `TURBO_E_INVALID_ARGUMENT` too. turbo.h's analyzer comments describe
  the bundle heads in place of the retired model paths, and state the
  rules the calls apply: no input is no rows whatever the bundle,
  `turbo_analyzer_desc.flags` must be 0, a bundle that does not open
  names field 7, and which rows are `TURBO_E_CAPACITY` (#1852).
- `turbo_analyzer_create` decodes `turbo_analyzer_desc` through raw
  field reads after validating `struct_size`, without forming a
  reference to the 120-byte struct over a legacy 104-byte allocation
  from before `bundle_path` was appended (#1833).
- `turbo_model_load` skips an artifact whose backend has no encoder for
  the manifest's family or no attention for its `lookahead` (loader rule
  6), so a bundle with no other is `TURBO_E_BUNDLE_NO_ARTIFACT`, saying
  why each was skipped, before any artifact's file is hashed; it was
  `TURBO_E_UNSUPPORTED` after the weights were verified. The rotary
  families' reason names levelzero among the backends that run them,
  and `bundle/recipes/sat-3l.json` lists `cpu` alone (#1855).

### Breaking

- C ABI `TURBO_PII_CONFIDENCE_LABELLED` moved from `3` to `4` and code `3`
  is now `TURBO_PII_CONFIDENCE_LABELLED_CHECKSUM`, and Java
  `TurboPiiConfidence` has `LABELLED_CHECKSUM(3)` and `LABELLED(4)` (#1899).
- `KindReport.suppressed` and `PiiAuditReport.total_suppressed` were added as
  public fields on `turbo_pii::rewrite::KindReport` and `PiiAuditReport`
  (breaking struct-literal construction outside the crate; `of()` and
  `with_suppressed()` are unaffected; #1898).
- The in-process analyzer withholds PII values by default: a PII row from
  `turbo_analyzer_detect_pii`, `turbo_analyzer_analyze_graph` or
  `turbo_analyzer_inspect_graph` has its kind, tier and ranges, with
  `TURBO_SPAN_VALUE` clear and no value bytes in the string pool.
  `turbo_analyzer_desc.flags` bit `TURBO_ANALYZER_PII_VALUES` (Java
  `TurboAnalyzer.Config.withPiiValues(true)`) opts in; any other bit is
  still refused. `InspectDocumentWindow` and `ProjectChunks` refuse a
  document holding PII with `TURBO_E_INVALID_STATE` unless
  `pii_output` discloses the text of every view the answer shows
  (`DocumentWindowRequest.pii_output` 9, `ProjectChunksRequest.pii_output`
  8; #1929).

- Java callers now import shared values, exceptions and interfaces from
  `ai.pipestream.turboembed`, and adapters from its `opennlp` package.
  Add `turboembed-api` alongside either matching-version bridge jar.
  The duplicated bridge-local value classes and JNI adapter package are
  removed. JNI retains its old exception name as a deprecated subclass
  for existing native libraries; catch the shared exception in new code.
  Bridge-local analyzer, result, session and span-view classes implement
  the common contracts; Panama's buffer callbacks remain concrete-only.
  Native entry points and layouts are unchanged (#1814).


- The model ids layers record (`ModelRef.id`, in `TypedLayer`
  provenance, DocumentGraph bytes and evidence records) moved from the
  `opennlp:` namespace to `turbo:`: `turbo:SentenceDetectorME:<lang>`,
  `turbo:POSTaggerME:<lang>`, `turbo:ChunkerME:<lang>`,
  `turbo:DocumentCategorizerME:<lang>`, `turbo:SentimentME:<lang>`,
  `turbo:SvmDoccatModel:<lang>`, `turbo:CorefAnnotator:<lang>`,
  `turbo:LanguageDetectorME:<lang>`, `turbo:DictionaryLemmatizer:<name>`,
  `turbo:UnigramSegmenter`, `turbo:MecabDictionary`,
  `turbo:MorfologikDictionary`, `turbo:MorfologikDictionaryReader`,
  `turbo:PoliMorfDictionaryReader`, `turbo:FeedforwardPOSModel` and
  `turbo:BilstmPOSModel`. A DocumentGraph or an evidence record written
  with an old id reads as the new id. The JSON-LD export's sentence,
  token and entity objects are typed `turbo:Sentence`, `turbo:Token` and
  `turbo:Entity`, like `turbo:Category` (#1797).
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
