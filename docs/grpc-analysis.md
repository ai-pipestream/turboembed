# The analysis result layer over gRPC

Beside the Open Inference Protocol (docs/kserve.md), `turbo-kserve` serves
`turbo.result.AnalysisService` on the same port: the Stage 5 analysis
result kinds on the wire (epic #151), and a capability call that says what
every loaded bundle is, so a client rejects an incompatible request
before sending it. `server/proto/turbo_analysis.proto` is the crate's
own file; generate a client from it in any language with gRPC.

## The service

| RPC | Answer |
|---|---|
| `Capabilities` | One entry per loaded bundle, and the result-message limits. |
| `LoadBundle` | Serves a bundle under a name; answers with its new entry. |
| `UnloadBundle` | Stops serving a name. |
| `RoundTripSparseTerms` | The message, validated, unchanged. |
| `RoundTripTokenVectors` | The message, validated, unchanged. |
| `RoundTripRerankScores` | The message, validated, unchanged. |
| `RoundTripExtraction` | The message, validated, unchanged. |
| `RoundTripChunkLevels` | The message, validated, unchanged. |
| `RoundTripDocumentGraph` | The document, imported and re-exported, with its layers and views listed. |
| `AnalyzeStream` | Bulk document analysis over one bidirectional stream. |
| `InspectDocumentWindow` | Viewport segments, chips, heatmaps and optional normalization differences. |
| `ProjectChunks` | Vector projection, term stacks and optional drift/strategy comparisons. |
| `ListOutputFormats` | The standard document formatter descriptors. |
| `FormatDocument` | A binary document graph exported in a requested format. |

See [inspection, projection and export RPCs](grpc-inspect.md) for their
request fields, limits and output semantics.

Span offsets cross in the unit a message names (`OffsetMode` on
`ChunkLevelsResult`, `SparseTermsResult`, `TokenVectorsResult`, and
`ExtractionResult`): UTF-8 bytes (the internal offsets), UTF-16 code
units (Java string indices), or Unicode code points, converted through
the view text's checkpoint table (`server/src/offsets.rs`) using
linear-scan column helpers (`OffsetTable::column_to_utf16` and
`OffsetTable::column_to_code_point`). When `text` is omitted on
`SparseTermsResult`, `TokenVectorsResult`, or `ExtractionResult` in
`OFFSET_MODE_UTF8`, spans cross as raw UTF-8 byte offsets; when `text`
is supplied (or when `offsets` is `OFFSET_MODE_UTF16` or
`OFFSET_MODE_CODE_POINT`), every span, occurrence, window, and fragment
is converted and validated against `text`. An offset
that is out of bounds, inverted, or inside a code point or a UTF-16
surrogate pair refuses with `TURBO_E_INVALID_ARGUMENT`; nothing is
snapped to a boundary. `ChunkLevelsResult` carries `turbo:chunks`,
`turbo:levels`, and `opennlp:embeddings` over the chunks in one round
trip, fragments included. `RoundTripDocumentGraph` takes binary
`DocumentGraph` bytes and answers with the same document imported and
re-exported through a `OnceLock<Codecs>` registry
(`server/src/results/document.rs`) initialized with all first-party
workspace `LayerCodec` registrations (`turbo-analysis-layers`,
`turbo-abbrev`, `turbo-chunker`, `turbo-coref`,
`turbo-encoding-layers`, `turbo-entity-linker`, `turbo-geo`,
`turbo-gliner`, `turbo-glossary-layers`, `turbo-group`,
`turbo-keywords`, `turbo-normalize-layers`, `turbo-relation`,
`turbo-schema-extract`, `turbo-stopword-layers`, and
`turbo-tokenize-layers`), listing every layer's standard id (`opennlp:*` or
`turbo:*`), instance, provenance (annotator, version, model, config
hash), item count, and status (`OK`, `SKIPPED`, or `FAILED` with its
detail), and every view likewise. Layers whose codecs the server does
not register are refused by name, never dropped.

The listener serves with TCP_NODELAY set, so a unary call is not held
by Nagle and delayed-ACK interaction.

Every message is the data columns of one in-process result layer in the
layer's own layout, plus the model identity that produced it. Decoding
rebuilds the layer and re-proves its own `check`, so a malformed message
is refused the way the layer refuses a malformed column: a gRPC status
whose turbo-code is `TURBO_E_INVALID_ARGUMENT` (256, `INVALID_ARGUMENT`)
naming the column, never a panic. The RoundTrip rpcs exist so clients and
servers pin the wire format: the answer is the request, validated. They
are the shared layer's round-trip proof, and later stages build their own
rpcs from the same messages.

- `SparseTermsResult`: the `SparseTerms` layer (issue #708): sparse rows
  by column, one entry per (unit, term), ascending term ids, finite
  weights over 0, a source span per input term (expanded terms carry
  none), occurrence spans where the producer recorded them.
- `TokenVectorsResult`: the `Vectors<Subwords>` and `TokenVectorWindows`
  layers (issue #709): one contextual vector per model token
  (little-endian f32, row-major, the way `raw_output_contents` carries
  vectors), the zero rows a producer could make no vector for, and the
  encoder windows on their text view, in text order, non-overlapping,
  that tie each row to its source span.
- `RerankScoresResult`: the `Reranked` layer (issue #707): the source
  layer's items in relevance order, one finite descending score per item,
  the source key's instance name (two instances of one layer index
  different items under one type), and the scoring model's identity, not
  a bare float.
- `ExtractionResult`: the `Answers` layer of extractive QA (issue #745):
  ranked answer spans over a text view with scores, normalized
  probabilities, and the abstention score and its probability; an empty
  result is a calibrated abstention. Answers made of separated pieces
  carry their fragments beside the bounding envelope, so the gap inside
  a discontinuous answer never decodes as answer text.
- `RecordsResult`: the schema-extract `turbo:records` layer: the
  structured records of a schema extraction over one view's text, with
  every span offset in the message's `OffsetMode` (UTF-8 bytes, UTF-16
  units or code points, as `ChunkLevelsResult`). It carries the record
  types, every `record.field` path and every closed choice as named label
  lists; per record its type and instance; per span value its fragments,
  record, path, probability and logit, and the deterministic rows that
  corroborate it (0 glossary, 1 PII, 2 numbers, 3 locations); and per
  closed-choice value its record, path, choice, probability and logit.
  Decoding re-proves the layer's check: a value's path must be a field of
  its record's type, every record has a value, scores lie in [0, 1],
  logits are finite, and no label is listed twice. Support rows index
  layers of the producing document, which the message does not carry, so
  they are checked for order and source only. `RoundTripRecords` answers
  with the message it was given.

## Capabilities

`Capabilities` answers with what a client needs to refuse its own
request, per loaded bundle:

| Field | Meaning |
|---|---|
| `name` | The name requests use. |
| `model` | The bundle's identity: model id, revision, and the manifest's sha256, the contract the load was made against. |
| `task` | The task it serves, the header constant's name (`TASK_EMBED`, `TASK_RERANK`, or `TASK_EXTRACTIVE_QA`). |
| `roles` | The prompt roles it takes: `PROMPT_QUERY` for a non-empty `prefix_query`, `PROMPT_DOCUMENT` for a non-empty `prefix_document`, `PROMPT_NONE` when it carries neither. |
| `widths` | `dim` first, then the `output_dims` a request may name by `output_dim`, ascending. |
| `context_limit` | Tokens per row; a longer row is `TURBO_E_CAPACITY` naming `max_seq`. |
| `batch_limit` | Rows per run; a larger batch is `TURBO_E_CAPACITY` naming `max_batch`. |
| `schema` | The limits an extraction schema may carry, the schema-extract crate's `Limits` (`max_entity_types`, `max_classification_tasks`, `max_labels_per_task`, `max_structures`, `max_fields_per_structure`, `max_text_bytes`, `max_prompt_subwords`, `max_words`); unset until the server loads a bundle whose task reads a schema. |

The limits are the session's: what a request is actually held to, where a
server started below the model's limits. The response's `limits` field
names what one result message may carry (`max_sparse_entries`,
`max_sparse_occurrences`, `max_token_vector_bytes`, `max_rerank_items`,
`max_answers`, `max_record_values` and the rest); a larger message is `TURBO_E_CAPACITY` (771,
`OUT_OF_RANGE`, docs/kserve.md, Errors) naming that limit, and nothing is
truncated.

The listing follows the bundles: `LoadBundle` serves one more and answers
with its entry, `UnloadBundle` stops serving a name (requests holding its
sessions finish; the name is then refused as never served), and the next
`Capabilities` shows the change.

Load and unload are admin calls, and the server does not serve them by
default: start it with `--admin` (docs/grpc.md, Flags) to answer them
(which also allows starting with zero pre-loaded models),
and until then they refuse with `TURBO_E_UNSUPPORTED` (`UNIMPLEMENTED`).
`LoadBundleRequest` accepts either an explicit `device` index or
`select_device = true` (with `device = 0`) to pick a device for the
bundle's task via `turbo_runtime_select`, and any load or session creation
failure returns its mapped gRPC status code with `turbo-code` and
`turbo-field` trailing metadata (docs/kserve.md, Errors).
Even with the flag they take a plaintext, unauthenticated gRPC port into
trust, the same trust every request already carries; put the port behind
what a deployment uses for that (docs/grpc.md, Starting it).

A request is bounded before anything is allocated. `sessions` over
`max_sessions_per_model`, or a registration past `max_loaded_models`,
refuses with `TURBO_E_CAPACITY` naming the limit; both caps are listed
in `CapabilitiesResponse.server`, and a client reads them there before
asking. One request's load, answer, and rollback carry the exact
registration it made: a same-named replacement is a different
registration and is never touched.

## A client call

With [grpcurl](https://github.com/fullstorydev/grpcurl), against a server
started as in docs/grpc.md:

```
grpcurl -plaintext -import-path server/proto -proto turbo_analysis.proto \
    localhost:8081 turbo.result.AnalysisService/Capabilities

grpcurl -plaintext -import-path server/proto -proto turbo_analysis.proto \
    -d '{"name": "more", "bundle": "testdata/tiny-bert-bundle",
         "device": 0, "sessions": 1}' \
    localhost:8081 turbo.result.AnalysisService/LoadBundle

grpcurl -plaintext -import-path server/proto -proto turbo_analysis.proto \
    -d '{"model": {"id": "org/m", "revision": "r1", "sha256": "abc"},
         "row": [0, 2], "term": [3, 9], "weight": [0.5, 1.25]}' \
    localhost:8081 turbo.result.AnalysisService/RoundTripSparseTerms
```

From Rust, the crate's generated client does the same, and
`turbo_kserve::results` converts between the messages and the in-process
layers: `Wire::to_msg` builds a message from a result, `Wire::from_msg`
reads one back with the message's own validation.

## Typed document inspection (issue #1219)

`InspectDocumentGraph` projects a binary graph into `TypedDocumentResult`
for a requested view and offset mode. `RoundTripTypedDocument` validates
and echoes the typed message; it does not reconstruct a native `Document`.

Evidence producers record only annotator and optional model identity.
Their model-table entries use version `"0"`, config hash `0`, and absent
execution metadata for the unrecorded fields. They do not borrow those
fields from another layer with the same annotator/model. Complete layer
provenance remains separately available through the layer and annotation
model references. Identical producer identities share one entry; distinct
model revisions or digests remain distinct.
Optional model and execution record presence is part of that identity:
an absent record is distinct from a present record with empty strings.

`turbo:quotations` uses the additive `quotation` payload (oneof tag 94).
Content is carried by the annotation's spans; optional speaker and cue
spans use the same requested view and offset mode. Kind and native f32
confidence have explicit presence, preserving direct-kind zero and every
confidence bit, including negative zero. The confidence stays in its own
field; no calibrated annotation probability is synthesized.

Content can be discontinuous. Normalization can remove fragments, merge
overlapping projections or collapse a whole item to one empty span.
Present speaker/cue spans can also collapse without becoming absent.
Native record order is retained, including when deletion of an early
fragment moves its visible start beyond a later record. Projected content
does not require exactly one fragment or a new cross-record sort.

`speaker_chain` preserves the native raw contextual row, including zero
and `u32::MAX`. It is not an `indexes_layer` reference, and the typed API
does not invent chain resolution or require a chain layer. Confidence,
kind, payload/scope and projected offsets are validated; invalid nested
offsets are refused with `TURBO_E_INVALID_ARGUMENT`, never snapped.

Projected event arguments and asset payloads also retain their span
presence when text deletion collapses them to an empty span. Their native
source layers still require nonempty spans. Typed validation requires
event argument spans and checks their offsets and anchor references;
asset payload spans and outer spans are required, and the payload must
remain inside its projected outer span, including at an empty boundary.

## Streaming bulk document analysis (issue #722)

`AnalyzeStream` is one bidirectional stream for many documents. The
first message must be an `AnalyzeStreamHeader` naming the ordering,
`ANALYZE_ORDER_INPUT` for the documents' own order or
`ANALYZE_ORDER_COMPLETION` for the order they finish in, plus two flow
control knobs: `max_inflight` (how many documents may be executing or
awaiting their answer at once) and `batch_linger_ms` (how long documents
wait on one model so they can run as one batch). Every knob is capped by
what `Capabilities` advertises under `server.stream`; a larger one is
`TURBO_E_INVALID_ARGUMENT` naming the limit, and a header that is not
first, or an unknown order value, refuses the same way.

After the header the client sends `AnalyzeDocument` messages: a client
correlation `id` (non-empty, carried back on the answer), the served
model's `name`, the `text`, `AnalyzeOptions` (the fields of
`turbo_embed_options`, each 0 for the model's default, plus
`sparse_top_k`, `sparse_floor`, and `token_stride`), `output`
(`ANALYZE_OUTPUT_VECTORS` by default, `ANALYZE_OUTPUT_SPARSE_TERMS` for
learned sparse lexical expansion, or `ANALYZE_OUTPUT_TOKEN_VECTORS` for
per-token contextual vectors and encoder window spans), and `offsets`
(`OFFSET_MODE_UTF8`, `OFFSET_MODE_UTF16`, or `OFFSET_MODE_CODE_POINT` for
span offsets on `SparseTermsResult` and `TokenVectorsResult`). Every
document answers exactly once as an `AnalyzeStreamResponse` tagged with
its id:
- `AnalyzeVectors`, whose `data` is byte for byte the row the
  single-document call's `raw_output_contents` carries for the same text
  and options;
- `SparseTermsResult` (`ANALYZE_OUTPUT_SPARSE_TERMS`, issue #917),
  computed on the model's pooled sessions with its MLM head (`SPLADE`
  max-log pooling) and matching `SparseAnnotator`;
- `TokenVectorsResult` (`ANALYZE_OUTPUT_TOKEN_VECTORS`, issue #917),
  computed on the model's pooled sessions (with optional linear
  `ProjectionHead` for `ColBERT` / late-interaction bundles) and matching
  `VectorAnnotator` (`SubwordVectors` and `TokenVectorWindows`); or
- `AnalyzeFailure` with the turbo status code, field and refusal the
  single-document call would carry.
A failed document never ends the stream; unknown model, unknown option
value, missing MLM head, and an over-limit document all answer this way,
and a batch whose write fails is retried one document at a time so the
failure lands on its document alone.

Documents waiting on the same load with the same output kind, offset
mode, and options gather into one batch up to the bundle's `batch_limit`,
running on one held session; batching never changes a document's answer,
and the batching key is the load's identity, so an unload and a reload
under one name never mix loads in one batch. Every admitted document, a
validation failure included, holds its queue room and window permit
until its answer is sent; queued work is bounded per stream and
server-wide by `ServerLimits.stream` (`max_queued_items_per_stream`,
`max_queued_bytes_per_stream`, and the server-wide pair, the byte caps
reserving each document's own bytes), and a document over a byte cap
refuses at once, naming the limit. A send past the caps is held by
backpressure until the queue drains, never a mid-stream error, so a
fast client that does not read cannot grow the server's memory without
limit. Cancelling the stream (or dropping it) drops the work already
queued for it; a batch already running on a session finishes, but its
answer goes nowhere, and waiting for a session never blocks the
scheduler's cancellation or other models' work.

`Server::stream_stats` reads the model-execution count and the queued
gauges, what the cancellation and backpressure proofs in
`server/tests/stream.rs` count. The speed bar sits in
`server/tests/stream_speed.rs`, release only: the same documents
through one stream must answer at least as fast as the same documents as
concurrent single-document calls on the same server, CPU and thread
count, in alternating rounds (docs/receipts/server-722/README.md).

## Tests

`cargo test -p turbo-kserve` runs everything against the server
in-process on the CPU with `testdata/tiny-bert-bundle`, called over gRPC
with a real client (`server/tests/results.rs`); nothing is mocked. Each result
kind crosses the wire and back unchanged from its in-process layer; the
capability listing matches the manifest and follows load and unload; an
over-limit request is refused with the named limit, for every limit the
listing names. `server/tests/stream.rs` proves the streaming call:
byte-for-byte parity with the single-document call across the harness
texts in both ordering modes, one model execution for eight same-key
documents with unchanged results, the advertised queue caps holding
under a non-reading client, cancellation dropping queued work by the
model-execution count, and per-document failures answered in order.
`server/tests/stream_bounds.rs` proves the reviewers' probes: byte
pressure blocks admission while item permits remain and an oversized
document refuses at once naming the limit; with a window of one, five
hundred validation failures waiting on an in-flight document fill the
queue to its advertised cap and no further; cancelling while another
caller holds the model's only session drains the queue at once; and an
unload plus reload under one name answers the old document from the old
load and refuses the over-length new one, never batched together.
`server/tests/stream_sparse_tokens.rs` proves `ANALYZE_OUTPUT_SPARSE_TERMS`
and `ANALYZE_OUTPUT_TOKEN_VECTORS` over `AnalyzeStream` against in-process
`SparseAnnotator` and `VectorAnnotator` across `SPLADE`, unprojected, and
`ColBERT` projection bundles in all three `OffsetMode`s.
`cargo test --release -p turbo-kserve --test speed --
--ignored` measures the encoding and decoding speed bar against the
current vector and span messages (docs/receipts/server-723/README.md).
