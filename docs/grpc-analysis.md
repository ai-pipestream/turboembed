# The analysis result layer over gRPC

Beside the Open Inference Protocol (docs/kserve.md), `turbo-kserve` serves
`turbo.result.v1.AnalysisService` on the same port: the Stage 5 analysis
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
| `AnalyzeDocumentProgressive` | Progressive server-streaming document analysis by step list or named profile. |
| `InspectDocumentGraph` | Projects a binary document graph into `TypedDocumentResult` for a requested view and offset mode. |
| `RoundTripTypedDocument` | The typed document message, validated, unchanged. |

See [inspection, projection and export RPCs](grpc-inspect.md) for their
request fields, limits and output semantics.

## Progressive document analysis and named profiles (issues #1218, #1302, #1325)

`AnalyzeDocumentProgressive` executes a document analysis pipeline stage by
stage and streams `ProgressiveAnalysisEvent` messages (`step_started`,
`step_finished`, `step_skipped`, `step_failed`, and final `completed` carrying
both `DocumentGraphResponse` and `ProgressiveSummary` timings).

A request may specify either an explicit `steps` list or a named, versioned
`profile` (`AnalyzeDocumentProgressiveRequest.profile`). `Capabilities` lists
every served profile under `CapabilitiesResponse.profiles` (`ProfileCapabilities`:
`name`, `version`, `steps`, `layers`, and `required_bundles`):

| Profile | Version | Steps | Produced Layers | Required Bundles |
|---|---|---|---|---|
| `default-v1` | `"1"` | `turbo-structure:blocks`, `turbo-sentdetect:rules`, `turbo-tokenize:uax29-untyped`, `turbo:structure-boundaries`, `turbo-geo:names`, `turbo:subwords`, `turbo:chunker`, `turbo:level-builder`, `turbo:chunk-vectors` | `turbo:structure`, `Sentences`, `Tokens`, `turbo:boundaries`, `Entities`, `Subwords`, `turbo:chunks`, `turbo:levels`, `Embeddings<turbo:chunks>` | `TASK_EMBED` |
| `retrieval-v1` | `"1"` | `default-v1` steps + `turbo:sparse` | `default-v1` layers + `turbo:sparse-terms<turbo:view>` | `TASK_EMBED`, `embed.sparse` |
| `minimal-v1` | `"1"` | `turbo-structure:blocks`, `turbo-sentdetect:rules`, `turbo-tokenize:uax29-untyped` | `turbo:structure`, `Sentences`, `Tokens` | *(none)* |
| `annotate-v1` | `"1"` | `turbo-structure:blocks`, `turbo-sentdetect:rules`, `turbo-tokenize:uax29-untyped`, `turbo-tagger:pos`, `turbo-tagger:entities`, `turbo-tagger:dependencies`, `turbo-tagger:lemmas` | `turbo:structure`, `Sentences`, `Tokens`, `PosTags`, `Entities`, `Dependencies`, `Lemmas` | `TASK_EMBED`, `head.pos`, `head.entities`, `head.dependencies`, `head.lemmas` |
| `annotate-v2` | `"2"` | `turbo-structure:blocks`, `turbo-sentdetect:rules`, `turbo-tokenize:uax29-untyped`, `turbo-tagger:heads` | `turbo:structure`, `Sentences`, `Tokens`, then the layers of every head step the bundle lists | `TASK_EMBED`, `head.any` |

`annotate-v1` keeps its fixed four heads. `annotate-v2` (#1832) is
capability driven: it runs every head step the selected bundle lists in
`head_steps`, so a partial bundle (a span-classifier-only entity bundle,
say) runs what it carries, and `head.any` is met by any bundle with a
head. When the bundle has a learned sentence head, `turbo-tagger:sentences`
replaces `turbo-sentdetect:rules` in the run. Its `layers` entry lists
what every run writes; the rest follow the bundle's `head_steps`.

A profile's output set never changes within a version. When a profile or
explicit step requires a bundle capability (`TASK_EMBED`, `embed.sparse` or
a `head.*` entry) that is not loaded on the server (or does not match
`req.model` when named), the call refuses before streaming with
`TURBO_E_BUNDLE_NOT_FOUND` (`NOT_FOUND`) naming the missing bundle
requirement.

### Bundle head steps

A `TASK_EMBED` bundle may carry learned heads over its encoder
(docs/bundle.md, Heads). Each head the bundle provides is a step, named
for the layer it writes, and `Capabilities` lists a bundle's steps under
`BundleCapabilities.head_steps`:

| Step | Head | Layer written | Requirement |
|---|---|---|---|
| `turbo-tagger:pos` | `HEAD_TOKEN_TAGGER`, `SCHEME_TAGS`, layer `pos` | `PosTags` | `head.pos` |
| `turbo-tagger:morphology` | `HEAD_TOKEN_TAGGER`, `SCHEME_TAGS`, layer `feats` | `MorphFeatures<Tokens>` (`turbo:feats`) | `head.morphology` |
| `turbo-tagger:entities` | `HEAD_TOKEN_TAGGER`, `SCHEME_BIO` or `SCHEME_BILOU`, layer `entities` | `Entities` | `head.entities` |
| `turbo-tagger:dependencies` | `HEAD_BIAFFINE_PARSER`, layer `dependencies` | `Dependencies<Tokens>` | `head.dependencies` |
| `turbo-tagger:lemmas` | `HEAD_EDIT_TREE_LEMMATIZER`, layer `lemmas` | `Lemmas` | `head.lemmas` |
| `turbo-tagger:spans` | `HEAD_SPAN_CLASSIFIER`, layer `entities` | `Entities` | `head.spans` |
| `turbo-tagger:relations` | `HEAD_RELATION_CLASSIFIER`, layer `relations` | `Relations` and `Edges<Entities, Entities>` | `head.relations` |
| `turbo-tagger:sentences` | `HEAD_SENTENCE_BOUNDARY`, layer `sentences` | `Sentences` | `head.sentences` |
| `turbo-tagger:words` | `HEAD_TOKEN_SPLIT`, layer `words` | `Words` (`turbo:words`) | `head.words` |
| `turbo-tagger:coref` | `HEAD_COREF_RANKER`, layer `coref` | `Groups<Entities>` and `Edges<Entities, Entities>` instance `coref` | `head.coref` |

Each layer is the one the `turbo-tagger` crate's annotator writes
(`MultiTask`, `SentenceTagger`, `WordTagger`, `SpanTagger`,
`RelationTagger`, `CorefTagger`) from the same features; only the step id
differs. Offsets are the tokens' own, so every head layer converts among
UTF-8 bytes, UTF-16 code units and code points through the view's
offset table; a split word is a byte subspan of its token when the
pieces tile it, or an explicit form over the token otherwise.

#### Selection

A step resolves to one head of the bundle by kind, label scheme and
layer: tags for part of speech and morphology, BIO or BILOU for
entities. `step@head` (`turbo-tagger:pos@upos`) selects a head by name.
`head_steps` lists a step when the bundle has at least one head it takes.
The refusals are named in the message:

| Refusal | When | Code |
|---|---|---|
| `requires bundle 'head.*'` | the bundle has no head the step takes | `TURBO_E_BUNDLE_NOT_FOUND` |
| `missing head` | the selected name is no head of the bundle | `TURBO_E_BUNDLE_NOT_FOUND` |
| `ambiguous head` | several heads fit and none is selected (also under `turbo-tagger:heads` and `annotate-v2`) | `TURBO_E_INVALID_ARGUMENT` |
| `unsupported head` | the selected head is of another kind, layer or scheme | `TURBO_E_UNSUPPORTED_TASK` |
| `ambiguous output` | two steps write one output: a step named twice, `turbo-tagger:sentences` beside `turbo-sentdetect:rules`, or `turbo-tagger:entities` beside `turbo-geo:names` | `TURBO_E_INVALID_ARGUMENT` |

No head is skipped and no output overwritten. Where two head steps write
one layer type the output keys are fixed: `turbo-tagger:entities` (or
`turbo-geo:names`) writes the default `Entities`, and `turbo-tagger:spans`
then writes instance `spans`, otherwise the default; relations and
coreference read the default `Entities`. Coreference links are instance
`coref`, so they sit beside the relation step's edges.

#### One encoder read

Every head step of a request decodes from one read of the document by
the bundle's encoder (`HeadGroup`), pooling the states each head asks for
and the most layer taps any head reads. The windows follow one plan:

- With no whole-view head among the steps, the source view's `Tokens` are
  read a sentence at a time (the `Sentences` layer, or the tokens'
  sentence column), each sentence in windows of whole words when it is
  longer than a row. This is the `annotate-v1` read.
- With `turbo-tagger:sentences` or `turbo-tagger:coref`, the view's
  tokens are read once as one sequence, in windows of whole words. Those
  heads decode from it, and every sentence-scoped head decodes from its
  slice for each sentence, as `MultiTask` slices it under a sentence head.
  The tokenizer then reads no sentences, so the learned sentences, which
  read the tokens, run after it; normalization steps are refused beside
  a learned sentence step.

The span, relation and coreference steps decode through the tagger
crate's decoders (`SpanHead`, `RelationHead`, `CorefHead`) and open no
encoder of their own, so adding a compatible head adds no encoder
forward: `server/tests/progressive_heads_rows.rs` counts every row the
process's encoders forward over a whole request
(`turbo_analysis_embed::forwarded_rows`) and finds one head and every
head equal. Steps run after the tokens they read; the relation and
coreference steps after the entities.

`turbo-tagger:heads` in an explicit step list stands for every head step
the chosen bundle provides, and is refused when the bundle carries no
head. A head step over a bundle lacking the head, or `req.model` naming
such a bundle, refuses with `TURBO_E_BUNDLE_NOT_FOUND` naming the
`head.*` requirement; with no `req.model`, the pipeline runs on the
first loaded embed bundle providing every head it needs.

Remaining limits: a step runs over the source view only; under
`annotate-v2` a bundle with two heads for one step is refused as
ambiguous rather than run twice; a learned sentence step cannot follow a
normalizer.

When normalization view steps (`turbo:normalize`, `turbo:normalize:unicode-form`,
`turbo:normalize:whitespace`, `turbo:normalize:clean-text`, and
`turbo-structure:project`) precede tokenization, subword segmentation, chunking,
and embedding steps (`turbo:subwords`, `turbo:chunker`, `turbo:level-builder`,
`turbo:sentence-vectors`, `turbo:chunk-vectors`, `turbo:sparse`), downstream
annotators run on the normalized child view while projecting all span-anchored
layers bi-directionally back to `ViewId::SOURCE` at their default keys so
1-to-1 `Ix<L>` alignment with `Vectors<Sentences>`, `Vectors<Chunks>`, and
`SparseTerms<View>` is preserved across UTF-8, UTF-16, and Unicode code-point
offset modes.

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
`turbo:levels`, and `Embeddings` over the chunks in one round
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
`turbo-tokenize-layers`), listing every layer's standard id,
instance, provenance (annotator, version, model, config
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

## Seeding structure from a parser

A client that already has a document's headings, lists and tables
(from docling, a Markdown or HTML parser, a PDF layout tool) sends them
so the chunker cuts along them. `AnalyzePipelineRequest` takes them two
ways, and both give the same document:

- `initial_graph`: DocumentGraph bytes holding a `turbo:structure`
  layer, which needs the Rust DocumentGraph writer.
- `structure_seed`: a `StructureSeed` message, the text plus one
  `StructureBlock` per block, which any language with protobuf builds.

A block names its `kind` (`StructureBlockKind`), its `start` and `end`
in the request's `offsets` unit (UTF-8 bytes, UTF-16 units as Java
strings index, or code points as Python strings index), its `level`
(a heading's or section's level, a list's, list item's or quote's
nesting, 0 otherwise) and its `parent`, the index of the enclosing
block, unset at the top. The server builds the default
`turbo:structure` layer from the blocks, with provenance
`turbo:structure-seed`, the layer an `initial_graph` written with that
provenance holds; the steps then run as they do over that graph. The
seed carries the text, so `text`, `source_bytes` and `initial_graph`
stay empty beside it.

Blocks are a tree in text order: each starts at or after the block
before it, lies inside its parent, which comes before it, and starts
after its previous sibling ends. A block with an unspecified or unknown
kind, a level over 255, a span outside the text or inside a code point
(or a UTF-16 surrogate pair), a parent at or after it, or a span that
overlaps a sibling or leaves its parent refuses the call before it
streams, `INVALID_ARGUMENT` with `TURBO_E_INVALID_ARGUMENT` and a
message naming the block (`structure_seed block 3 overlaps the block
before it`).

For the text

```
# Café menu

- crème brûlée
- 😀 soup

| a | b |
| ü | é |
```

a heading, a list and a table, in UTF-16 units:

| Block | kind | start | end | level | parent |
|---|---|---|---|---|---|
| 0 | `STRUCTURE_BLOCK_KIND_HEADING` | 0 | 11 | 1 | |
| 1 | `STRUCTURE_BLOCK_KIND_LIST` | 13 | 37 | 1 | |
| 2 | `STRUCTURE_BLOCK_KIND_LIST_ITEM` | 13 | 27 | 1 | 1 |
| 3 | `STRUCTURE_BLOCK_KIND_LIST_ITEM` | 28 | 37 | 1 | 1 |
| 4 | `STRUCTURE_BLOCK_KIND_TABLE` | 39 | 58 | 0 | |
| 5 | `STRUCTURE_BLOCK_KIND_TABLE_ROW` | 39 | 48 | 0 | 4 |
| 6 | `STRUCTURE_BLOCK_KIND_TABLE_CELL` | 41 | 42 | 0 | 5 |
| 7 | `STRUCTURE_BLOCK_KIND_TABLE_ROW` | 49 | 58 | 0 | 4 |
| 8 | `STRUCTURE_BLOCK_KIND_TABLE_CELL` | 51 | 52 | 0 | 7 |

The emoji is two UTF-16 units, so item 3 ends at 37 in UTF-16 units, 36
in code points and 43 in UTF-8 bytes. From Java:

```java
String text = "# Café menu\n\n- crème brûlée\n- 😀 soup\n\n| a | b |\n| ü | é |\n";
int at = text.indexOf("# Café menu");
StructureSeed seed = StructureSeed.newBuilder()
        .setText(text)
        .addBlocks(StructureBlock.newBuilder()
                .setKind(StructureBlockKind.STRUCTURE_BLOCK_KIND_HEADING)
                .setStart(at).setEnd(at + "# Café menu".length()).setLevel(1))
        // ... the list, its items, the table, its rows and cells ...
        .build();
AnalyzePipelineRequest req = AnalyzePipelineRequest.newBuilder()
        .setOffsets(OffsetMode.OFFSET_MODE_UTF16)
        .setProfile("default-v1")
        .setStructureSeed(seed)
        .build();
```

`server/tests/progressive_seed.rs` holds the server to this: the seed
above in each offset unit gives the same graph, byte for byte, as the
same structure sent as `initial_graph`, and every refusal above names
its block. The Java and Python stub clients (`clients/java`,
`clients/python/smoke_test.py`) send a heading and a list as a seed
against the release binary and check an overlapping block is refused.

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
| `head_steps` | The `AnalyzeDocumentProgressive` head steps the bundle's heads serve (`turbo-tagger:pos`, `turbo-tagger:morphology`, `turbo-tagger:entities`, `turbo-tagger:dependencies`, `turbo-tagger:lemmas`, `turbo-tagger:spans`, `turbo-tagger:relations`, `turbo-tagger:sentences`, `turbo-tagger:words`, `turbo-tagger:coref`), in that order; empty for a bundle without heads. |

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
On a server started with a catalog (docs/grpc-catalog.md), a bundle whose
`model.license` is in a class that needs acceptance (docs/zoo.md, licence
classes) loads only for a user who accepted it in the catalog's
acceptance file: `user` (field 12) names that user, and a load without
one, or for a user who has not accepted, refuses with
`TURBO_E_INVALID_STATE` (`FAILED_PRECONDITION`) naming the terms to read.
A `LicenseRef-RestrictedPublic` bundle names no host, so any of the user's
acceptances under that licence counts. A permissive bundle, or one on a
server without a catalog, needs no user.
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
    localhost:8081 turbo.result.v1.AnalysisService/Capabilities

grpcurl -plaintext -import-path server/proto -proto turbo_analysis.proto \
    -d '{"name": "more", "bundle": "testdata/tiny-bert-bundle",
         "device": 0, "sessions": 1}' \
    localhost:8081 turbo.result.v1.AnalysisService/LoadBundle

grpcurl -plaintext -import-path server/proto -proto turbo_analysis.proto \
    -d '{"model": {"id": "org/m", "revision": "r1", "sha256": "abc"},
         "row": [0, 2], "term": [3, 9], "weight": [0.5, 1.25]}' \
    localhost:8081 turbo.result.v1.AnalysisService/RoundTripSparseTerms
```

From Rust, the crate's generated client does the same, and
`turbo_kserve::results` converts between the messages and the in-process
layers: `Wire::to_msg` builds a message from a result, `Wire::from_msg`
reads one back with the message's own validation.

## Typed document inspection (issue #1219)

`InspectDocumentGraph` projects a binary graph into `TypedDocumentResult`
for a requested view and offset mode. `RoundTripTypedDocument` validates
and echoes the typed message; it does not reconstruct a native `Document`.

A layer over another view is projected through the lowest common
ancestor of its view and the requested one, as `Document::map_span`
maps spans, so a span outside the edits between the two keeps its exact
bytes; a detour through the source would widen it to the whole of every
lossy edit above that ancestor (#1906). In debug builds, and so in
tests, the projection runs the `RoundTripTypedDocument` validation on its
own message and refuses, with `TURBO_E_INVALID_ARGUMENT`, a document it
cannot project to a message that validation accepts. Release builds skip
that check, which costs half to three quarters as much as the projection
itself.

A `turbo:pii` payload sits only on a `turbo:pii` layer, has at least one
span, and needs a `PiiKindRef`: a built-in `PiiKindId` (1 to 30) or an
index into the layer's `custom_pii_kinds`, and a `PiiConfidenceTier`
(format 1, checksum 2, labelled checksum 3, labelled 4).
`UNSPECIFIED` (0) and unknown numbers are refused. `custom_pii_kinds` is
empty on every other layer; on `turbo:pii` it holds at most 256 distinct
names of 1 to 64 bytes, none a built-in kind name (#1901). A refusal
names the layer, the row and its wire spans, never the mention's text, a
custom kind name or the refused number. A mention whose
text the requested view deletes keeps an empty span where it was.

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

## PII output (issue #1904)

Detection identifies PII; what a response carries beyond the findings is
an output option. A document holding PII findings is one with a
`turbo:pii` layer instance. Its answers from `InspectDocumentGraph`,
`RoundTripDocumentGraph` and `AnalyzeDocumentProgressive`, and the sink
items about it, carry by default only the findings: each mention's kind,
confidence tier and spans, with layer and view provenance. Every
`PiiMention.value` is `PiiValue.withheld`; a caller that holds the text
slices it with the spans. There is no value and no token.

Graph bytes and source text hold every detected value in clear (the
layer codec writes the normalized column, and the text holds the values
themselves), so a default answer about such a document has no graph
bytes, no `source_text` and an empty `raw_text`. `DocumentGraphResponse`
then carries `findings`, the typed projection onto the source view in
UTF-8 offsets, in place of the graph; progressive step snapshots carry no
graph once the PII step has run. Status details are cut to the failure
kind and the layer or annotator it names, without free text. Other layers
carry the view's words (sentence text, token surfaces, lemmas), so while
the text is withheld they are listed with their status and no
annotations. A typed result is refused on validation when a mention's
value is anything but `withheld` (missing, no form, clear, token or
ciphertext) until keyed release lands (#1909).

`FormatDocument` of such a document needs `full_text`, since every
format carries the document text; no format (JSON-LD, markdown or any
other) writes a mention's value. `PiiMention.normalized` (3) is kept for
wire compatibility, deprecated and never populated.

`PiiOutputOptions` on the request (`pii_output`) turns on more:

- `rewritten_text`: the text of the requested view when that view is a
  PII rewrite view, one a `turbo-pii:` annotator wrote, such as the masked
  view. The text of any other view needs `full_text`.
- `full_text`: source and view texts and graph bytes, as for a document
  without PII.
- `detail`: candidates and evidence detail (#1903).
- `values`: `UNSPECIFIED` or `WITHHELD`. Token, ciphertext and clear
  values need verified identities and grants (#1909); until then they are
  refused with `TURBO_E_UNSUPPORTED_OPTION` (`UNIMPLEMENTED`).

The server caps the options with `--pii-output-allow`
(`TURBO_KSERVE_PII_OUTPUT_ALLOW`): `none`, or a comma-separated list of
`rewritten-text`, `full-text` and `detail`. Unset, all three are allowed,
since a caller of these calls sent the text it asks back. A request for an
option the server does not allow is refused with `TURBO_E_INVALID_STATE`
(`FAILED_PRECONDITION`) naming it; nothing is dropped silently. Sink items
follow `--sink-pii-output` (docs/grpc-sink.md), which must be within the
cap.

In-process callers of the C ABI, JNI and Panama bindings hand the library
their own text. The controls there govern what the library copies out by
default, not protection from a caller who already holds the input. Native
`PiiMention` and the `turbo:pii` value column (`Sensitive`) print
`<withheld>` in place of a value in `Debug`, and Java `TurboSpanData`'s
`toString` does the same.

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
`turbo_embed_options`, each 0 for the model's default, `quantization`
and `int8_ranges` among them, plus `sparse_top_k`, `sparse_floor`, and
`token_stride`), `output`
(`ANALYZE_OUTPUT_VECTORS` by default, `ANALYZE_OUTPUT_SPARSE_TERMS` for
learned sparse lexical expansion, `ANALYZE_OUTPUT_TOKEN_VECTORS` for
per-token contextual vectors and encoder window spans, or
`ANALYZE_OUTPUT_VECTOR_CODES` for the pooled vector's quantized codes), and `offsets`
(`OFFSET_MODE_UTF8`, `OFFSET_MODE_UTF16`, or `OFFSET_MODE_CODE_POINT` for
span offsets on `SparseTermsResult` and `TokenVectorsResult`). Every
document answers exactly once as an `AnalyzeStreamResponse` tagged with
its id:
- `AnalyzeVectors`, whose `data` is byte for byte the row the
  single-document call's `raw_output_contents` carries for the same text
  and options, and whose `quantization` says how: f32, int8 codes or
  packed sign bits (docs/kserve.md, Output). `int8_ranges` is
  `[2, width]` row-major, width the document's `output_dim` or the
  model's `dim`; another length is the document's
  `TURBO_E_INVALID_ARGUMENT` naming field 8, and documents batch together
  only under the same table. A `quantization` other than 0 or
  `TURBO_QUANTIZE_NONE` with a sparse-terms or token-vectors `output`
  is `TURBO_E_UNSUPPORTED_OPTION` naming field 7, as is a codes output
  whose `quantization` names f32;
- `SparseTermsResult` (`ANALYZE_OUTPUT_SPARSE_TERMS`, issue #917),
  computed on the model's pooled sessions with its MLM head (`SPLADE`
  max-log pooling) and matching `SparseAnnotator`;
- `TokenVectorsResult` (`ANALYZE_OUTPUT_TOKEN_VECTORS`, issue #917),
  computed on the model's pooled sessions (with optional linear
  `ProjectionHead` for `ColBERT` / late-interaction bundles) and matching
  `VectorAnnotator` (`SubwordVectors` and `TokenVectorWindows`);
- `VectorCodesResult` (`ANALYZE_OUTPUT_VECTOR_CODES`, issue #1601), the
  pooled vector's codes as their own outcome: the same bytes
  `AnalyzeVectors` carries under `TURBO_QUANTIZE_INT8` (with the
  document's `int8_ranges` table) or `TURBO_QUANTIZE_UBINARY`, so a
  client asks for codes without asking for vectors; or
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
`server/tests/progressive_heads.rs`, `progressive_heads_all.rs` and
`progressive_heads_rows.rs` prove the head steps on sealed tiny bundles:
`annotate-v1` and `annotate-v2` layers field for field against the
tagger crate's annotators (partial bundles, morphology, BIO and span
entities, relations, learned sentences, word splits, coreference) on
Unicode text read in several windows, every head layer's offsets in all
three `OffsetMode`s, the named selection refusals, and encoder forwards
over whole requests.
`cargo test --release -p turbo-kserve --test speed --
--ignored` measures the encoding and decoding speed bar against the
current vector and span messages (docs/receipts/server-723/README.md).
