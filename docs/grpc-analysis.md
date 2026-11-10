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

Converted GLiNER and GLiNER2 packages add two request-driven steps outside the
bundle-head table:

| Step | Request fields | Layers written |
|---|---|---|
| `turbo:gliner` | `model`, `open_span` | `Entities`, `Decisions`, `Evidence<Entities>` |
| `turbo:zero-shot-relations` | `relation_model`, `relations` | `Relations`, `Edges<Entities, Entities>`, `Evidence<Relations>` |

`open_span` carries caller labels, optional parallel aliases or guideline
prompts, threshold, and flat or nested decoding.
`relations` carries labels with optional thresholds, symmetry, domain/range
entity kinds, and sentence scoping. Separate model fields let one request run
GLiNER entities and GLiNER2 relations. `turbo:subwords` uses the GLiNER
package's own tokenizer when present, including normalized-view projection, so
smart chunking works without another embedding bundle.

Remaining limits: a step runs over the source view only; under
`annotate-v2` a bundle with two heads for one step is refused as
ambiguous rather than run twice; a learned sentence step cannot follow a
normalizer; and the rule sentence step after `turbo:normalize:case-fold`
reads the folded view, whose sentences start in lower case, so it finds
no boundary after the first.

`server/tests/full_text_graph.rs` is the in-process acceptance of the
model-free stages: `compile_analyzer` with normalization, structure,
sentences, tokens, abbreviations, terms, stems, stopwords, keywords, the
place-name gazetteer (`turbo-geo:names`), PII, numbers, money, dates and
both boundary layers, over UTF-8 text with a heading, two paragraphs, a
ligature NFKC unfolds, two characters outside the Basic Multilingual
Plane, a combining mark, sharp s and dotted capital I. It checks each
layer's source slices, UTF-16 and code point offsets against counts
made from the text itself (one more UTF-16 unit per surrogate pair) and
their round trips, projection into the normalized and the case-folded
view and back, a byte-identical export, import and re-export,
deterministic runs, empty input and a 600-sentence document. A stage the request cannot run is
refused by name: `turbo:chunker` without a bundle (it reads
`turbo:subwords`), `turbo:gliner` without `open_span`,
`turbo:zero-shot-relations` without relation labels, and an unknown step.
A document over `turbo:zero-shot-relations`' limits (its mention and
pair caps, and the GLiNER2 encoder's `max_sequence` of word pieces)
fails that step with `TURBO_E_CAPACITY`, and the failure message names
the limit and both counts, for example `step 'turbo:zero-shot-relations'
failed: over its encoder max_sequence 512, got 528`; never the text.
The step's layer error, in the step failure, the complete graph and
the receipt, has kind `LAYER_ERROR_KIND_LIMIT` and the same code.

When normalization view steps (`turbo:normalize`, `turbo:normalize:unicode-form`,
`turbo:normalize:case-fold`, `turbo:normalize:whitespace`, `turbo:normalize:clean-text`, and
`turbo-structure:project`) precede tokenization, subword segmentation, chunking,
and embedding steps (`turbo:subwords`, `turbo:chunker`, `turbo:level-builder`,
`turbo:sentence-vectors`, `turbo:chunk-vectors`, `turbo:sparse`), downstream
annotators run on the normalized child view while projecting all span-anchored
layers bi-directionally back to `ViewId::SOURCE` at their default keys so
1-to-1 `Ix<L>` alignment with `Vectors<Sentences>`, `Vectors<Chunks>`, and
`SparseTerms<View>` is preserved across UTF-8, UTF-16, and Unicode code-point
offset modes.

Two of those steps match text as written and so read the cased view: the
rule sentence step (`turbo-sentdetect:rules`) and the place-name step
(`turbo-geo:names`, `turbo:entities`). In a chain with
`turbo:normalize:case-fold` they read the last view written before the
fold, or the source when the fold comes first, and their spans are in
source offsets as always. A full fold also rewrites spellings (`ﬁ` to
`fi`, `ſ` to `s`), which would otherwise turn words the text does not
write into place names. Tokens, subwords and the steps that read them
keep reading the last view, folded. Structure blocks read it too and do
not depend on case. Every other step reads the source.
`server/tests/case_fold_sentences.rs` and
`server/tests/progressive_case_fold_names.rs` hold the two steps to the
results of the same request without the fold.

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
`max_answers`, `max_record_values`, `max_mapping_edits` and the rest); a larger message is `TURBO_E_CAPACITY` (771,
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
bundle's task via `turbo_runtime_select`, and optionally `vocabulary`
(`VocabularyBinding`, field 13, issue #2013; see
[docs/grpc-vocabulary.md](grpc-vocabulary.md)) to bind stored
`stopwords` and `glossary` artifacts (`VocabularyArtifactRef`:
`artifact_id` plus `sha256`) as the bundle's default vocabulary,
advertised on `BundleCapabilities.vocabulary` (field 10). Any load,
vocabulary resolution, or session creation
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
lossy edit above that ancestor (#1906). Every projected `SpanFragment`
says what the projection did to it (#1907, below). In debug builds, and so in
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

A `turbo:pii-edits` layer projects one `PiiEditAnnotation` (oneof arm 98)
an edit: the annotation's spans are the replacement on the requested view;
`source` is the replaced bytes on the parent of the layer's native view in
canonical UTF-8 bytes, whatever the `OffsetMode`; `members` are rows of
the covered `turbo:pii` layer, ascending; `rule` is `default_rule` or a
`PiiKindRef` into the layer's `custom_pii_kinds`, which repeats the
covered pii layer's table; `strategy` is a `PiiStrategy`; and
`policy_revision` is the caller's number. No value is carried. Validation
refuses an edit with no source span, no members, members out of order or
past the covered layer, a missing rule, or an unknown strategy (#1902).
The edits' view is the layer's `native_view` in the mapping, whatever
other views the message has: projected onto that view, the unchanged
stretches between edits are as long in the parent as in the view;
projected onto the source when the source is the parent, every edit's
span is its source span and its members are exactly the mentions under
it; onto any other view only the layer-wide checks apply (#1907).

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
graph once the PII step has run. Layer and view errors are the typed
`LayerError` of every answer (#1908), which never carries free text. Other layers
carry the view's words (sentence text, token surfaces, lemmas), so while
the text is withheld they are listed with their status and no
annotations. A typed result is refused on validation when a mention's
value is missing or has no form, is an empty clear value, a token that is
not 32 bytes or names no key id and version, or a ciphertext that is
empty, names no key id and version, or names a scheme other than
`turbo-pii-aes-256-gcm-v1` (#1885).
The result envelope (#1907) is kept: it carries digests and lengths of
the texts and the edit chain's offsets, never a text.

`FormatDocument` of such a document needs `full_text`, since every
format carries the document text; no format (JSON-LD, markdown or any
other) writes a mention's value. `PiiMention.normalized` (3) is kept for
wire compatibility, deprecated and never populated.

`InspectDocumentWindow` (window text, x-ray `parent_text` and
`child_text`, summary details, heatmaps) and `ProjectChunks` (previews,
term stacks) carry view text too, so for such a document they are
refused with `TURBO_E_INVALID_STATE` (`FAILED_PRECONDITION`) naming the
option unless `pii_output` (`DocumentWindowRequest` 9,
`ProjectChunksRequest` 8) discloses the text of every view the answer
shows: `full_text`, or `rewritten_text` for a PII rewrite view. An x-ray
target view follows the same rule as the main view. Without `full_text`,
`ProjectChunks` term stacks read the vocabulary and term statistics only
when every input the vocabulary's provenance names is of the projected
view: views and tokens directly, any other input (such as linguistic
terms) through its own inputs in turn. A vocabulary with an input of
another view, an input that names none, or no input at all is left out,
and the stacks are built from the projected view's tokens.

`PiiOutputOptions` on the request (`pii_output`) turns on more:

- `rewritten_text`: the text of the requested view when that view is a
  PII rewrite view, one a `turbo-pii:` annotator wrote, such as the masked
  view. The text of any other view needs `full_text`. A rewrite view
  counts only when it rewrote every finding: a `turbo:pii` instance it
  did not read that holds a mention (a second detector run after the
  rewrite) may sit in clear in its text, so that view also needs
  `full_text` (#1932). Findings a parent rewrite view covered, or that a
  rewrite read as a projected copy, count as covered. A layer is judged
  by its lineage (its own view and every layer it read or was made on,
  back to a view): one made from another view's text (the source's
  lemmas, keyword phrases, a `ProjectView` copy of them on the rewrite
  view, linked names joined into entities) is listed bare, and one made
  from the rewrite view alone, such as lemmas aligned to its tokens, is
  kept.
- `full_text`: source and view texts and graph bytes, as for a document
  without PII.
- `detail`: candidates and evidence detail (#1903).
- `values`: `UNSPECIFIED` or `WITHHELD` withholds them. `TOKEN`,
  `CIPHERTEXT` and `CLEAR` are released by `InspectDocumentGraph` only,
  as the next section says. A value form on any other call or on a
  server without PII release, and `CIPHERTEXT` on a server that releases
  no ciphertext, are refused with `TURBO_E_UNSUPPORTED_OPTION`
  (`UNIMPLEMENTED`).
  A token or ciphertext names its key in `key_id` and `key_version`, the
  identity (id, version) of a `PiiPolicyKeyRef`.

The server caps the options with `--pii-output-allow`
(`TURBO_KSERVE_PII_OUTPUT_ALLOW`): `none`, or a comma-separated list of
`rewritten-text`, `full-text`, `detail`, and the value forms `token`,
`ciphertext` and `clear`. Unset, the three text options are allowed, since a caller of
these calls sent the text it asks back, and no value form is. A request for an
option the server does not allow is refused with `TURBO_E_INVALID_STATE`
(`FAILED_PRECONDITION`) naming it; nothing is dropped silently. Sink items
follow `--sink-pii-output` (docs/grpc-sink.md), which must be within the
cap.

In-process callers of the C ABI, JNI and Panama bindings hand the library
their own text. The controls there govern what the library copies out by
default, not protection from a caller who already holds the input. A PII
row withholds the mention's normalized value unless the analyzer was
created with `turbo_analyzer_desc.flags` bit `TURBO_ANALYZER_PII_VALUES`
(Java `TurboAnalyzer.Config.withPiiValues(true)`): without it
`TURBO_SPAN_VALUE` is clear and no byte of the value is in the string
pool. The result's graph is the document as analyzed, the caller's own
text included. Native
`PiiMention` and the `turbo:pii` value column (`Sensitive`) print
`<withheld>` in place of a value in `Debug`, and Java `TurboSpanData`'s
`toString` does the same.

## PII value release (issue #1909)

A calling service is identified by the client certificate its mTLS
connection verified against `--tls-client-ca`, and by nothing else:
request metadata, headers and fields confer nothing. TurboEmbed keeps no
user model; which end user may see what stays with the calling platform,
which decides what to ask for. The identity is the SHA-256 of the leaf
certificate's DER bytes (its fingerprint), so no certificate parser is
needed.

`--pii-grant FINGERPRINT:FORMS:KINDS` (repeated, or
`TURBO_KSERVE_PII_GRANTS` as a `;`-separated list) grants the service
with that fingerprint the forms in `FORMS` (`token`, `ciphertext` and
`clear`, joined by `+`) for the built-in kinds in `KINDS`
(comma-separated names, or `*` for every kind, caller-defined kinds
included). Grants of one fingerprint add up. A grant needs `--tls-cert`,
`--tls-key` and `--tls-client-ca`, and `--pii-audit-endpoint`. A grant
with `ciphertext` also needs a server built with the `pii-reversible`
feature and an encryption key (#1885), as below.

For each mention the forms in effect are the request's form ∩ the forms
`--pii-output-allow` lists ∩ the release ceiling of the request's
redaction policy for the mention's kind ∩ the forms the service is
granted for the mention's kind: a set intersection, never a ranking, in
the one function `release::caps::effective`. A request for a form the
policy does not list is refused with `TURBO_E_INVALID_STATE`
(`FAILED_PRECONDITION`).

Per-kind release ceiling (#1943). `PiiKindRule.release_ceiling` is a
`PiiReleaseCeiling`, a set of `PiiValueForm` values (token, ciphertext,
clear), in a message so that unset and empty differ: unset sets no limit
beyond the grant, and an empty set releases no form for the kind. A
values request names the policy in `InspectDocumentGraphRequest.pii_policy`
(`PiiReleasePolicy`, field 8), as the redact step takes it: `inline`, the
binary `RedactionPolicy` (carried as bytes because
`turbo_pii_policy.proto` imports `turbo_typed_annotations.proto`), or
`name`, a policy loaded with `--pii-policy`. Unset, no ceiling applies and
the release is as before. A selection that does not resolve (no arm, an
unknown name, bytes that are not a `RedactionPolicy`, a policy that does
not convert) is refused with `TURBO_E_INVALID_ARGUMENT` naming field 8
before the graph is read. The policy's digest must then equal the
`policy_digest` of every `turbo:pii-edits` layer of the graph that
records one, and at least one must; otherwise the call is refused with
`TURBO_E_INVALID_ARGUMENT` naming field 8, and the error names no kind
and no value. A custom kind's ceiling is found by the kind's name in the
policy's `custom_kinds`. The caller supplies the graph, so the ceiling
holds a caller to its own policy; it is not a control against a caller
that strips the edits or names no policy, which the grant is.

`--pii-require-recorded-policy` (`TURBO_KSERVE_PII_REQUIRE_RECORDED_POLICY`,
`1`, `true` or `yes`; off by default; needs `--pii-grant`) makes a
recorded ceiling apply to every values request (#1977). With it on, a
values request that names no `pii_policy` is refused with
`TURBO_E_INVALID_ARGUMENT` naming field 8 when any `turbo:pii-edits`
layer of the graph records a policy digest; the error names no kind and
no value, as the digest mismatch does. A graph whose edits record no
digest, or that has none, is released as without the setting. It still
does not hold a caller that strips the edits from the graph: the grant
does.

Why a values request released nothing (#1931). Causes that do not depend
on the document are refused before the graph is read, so they tell the
caller nothing about it:

- No `operation_key`: `TURBO_E_INVALID_ARGUMENT`, naming
  `operation_key` (field 6).
- A service granted the requested form for no kind at all, or a
  connection whose certificate has no grant: `TURBO_E_INVALID_STATE`
  (`FAILED_PRECONDITION`), naming the form and never a kind.

Outcomes that depend on the document are on the answer, per mention.
`PiiValueWithheld.reason` (field 1) says why a value is withheld:
`NOT_REQUESTED` when the request asked for no value, `NOT_GRANTED` when
the service is not granted the form for the mention's kind, and
`NOT_ALLOWED_BY_POLICY` when it is but the release ceiling of the
request's policy for the kind excludes the form (#1943). The server-wide
`--pii-output-allow` never gives a per-mention reason: it refuses the
whole request instead, as above.
`UNSPECIFIED` is never produced and the validator refuses it. A released
mention carries its token or clear arm, and a document with no mentions
has an empty list. A reason names nothing beyond the mention's own kind,
which its findings already show.

Tokens are HMAC-SHA-256 under a registered key, `--pii-token-key
ID:VERSION:PATH` (repeated, or `TURBO_KSERVE_PII_TOKEN_KEYS`), whose 32
to 1024 bytes are read from `PATH` at start. With several keys,
`--pii-token-key-active ID:VERSION` names the one tokens are issued
under. `PiiValueToken` carries the 32-byte token and the key's id and
version; key bytes never appear in a message, an error or a `Debug`
string. The MAC is over the domain `turbo-pii-token-v1`, then the
mention's kind code (`builtin:ID`, or `custom:NAME`) and the value, each
prefixed with its length as 4 big-endian bytes (#1930). Equal values of
one kind under one key give equal tokens; one value under two kinds
gives two, so tokens never show that two kinds share a value. `clear`
carries the normalized value.

Ciphertexts (#1885) are AES-256-GCM under a registered encryption key,
`--pii-encryption-key ID:VERSION:PATH` (the keys a redaction policy's
encryption strategies name, each exactly 32 bytes). With several keys,
`--pii-encryption-key-active ID:VERSION`
(`TURBO_KSERVE_PII_ENCRYPTION_KEY_ACTIVE`) names the one ciphertexts are
made under; it is refused without a grant that includes ciphertext.
`PiiValueCiphertext` carries the scheme `turbo-pii-aes-256-gcm-v1`, the
key's id and version, and as `ciphertext` the UTF-8 text of a
`turbo-pii` AES-256-GCM token: the kind code in capitals (`BUILTIN:3`,
`CUSTOM:BADGE`), `:`, and the unpadded base64url of a 12-byte nonce, the
encrypted normalized value and the 16-byte tag. The associated data binds
the kind code and the context `turbo-kserve.pii-release.v1`, so a holder
of the key decrypts it with `AesGcmEncryptor` under that context, and a
ciphertext moved to another kind fails authentication. Each release
draws a fresh nonce, so, unlike tokens, equal values give different
ciphertexts. A grant with no ciphertext kind gives the policy digest it
gave before ciphertext release.

Before any value leaves, the server records one `PiiAccessEvent` with
`turbo.audit.v1.TurboAuditService.RecordPiiAccess`
(`server/proto/turbo_audit.proto`) at `--pii-audit-endpoint`
(`TURBO_KSERVE_PII_AUDIT_ENDPOINT`) and waits for the acknowledgement,
at most `--pii-audit-timeout-ms` (`TURBO_KSERVE_PII_AUDIT_TIMEOUT_MS`,
5000 by default). A call that fails, times out or acknowledges another
event id fails the request with `TURBO_E_BUSY` (`UNAVAILABLE`) and no
value. A release needs the request's `operation_key` (#1908), as above.
The
event carries the operation key and the attempt id the answer's receipt
names, the input
digest (the source text's SHA-256), the service fingerprint, the forms in effect per kind, the released mentions by layer
instance and index with their kind and form, the policy id and the
SHA-256 of its rules (the server output ceiling and the grant that
allowed the release, and the digest of the redaction policy the request
named for its release ceilings, when it named one), the time and the document id; never a value, token
or text. Each released mention also names the redaction policy that
covers it (#1882 part C): `rule_id`, `policy_revision` and
`redaction_policy_digest` copied from the `turbo:pii-edits` edit of a
named `RedactionPolicy` in the same graph whose members include it, and
empty when no such edit covers it. Its id is the SHA-256 of what a retry
keeps: the service fingerprint, the operation key, the input digest, the
mention set (each mention's redaction policy fields included, or an
explicit absent marker), the forms in effect and the policy digest, so
two releases that differ only in redaction policy have different ids. The attempt is not part of it, so every attempt at one
release has the same id and the audit service keeps one record. The sink drops items when its queue is full, so it
never carries these events. A request that releases nothing records
nothing.

## The result envelope (issue #1907)

Every `TypedDocumentResult` carries, beside the projected layers, what
it was computed from, what it is, how each layer was made, and the view
edit chain, so a consumer can bind the result to its exact input, read
an old result against its original parse, and map any span between any
two views without the server. Every digest is the raw 32 bytes of a
SHA-256 and the proto comment on each field says what it is over; a
digest that is not available is absent, never zero-filled or invented.

- `input` (`InputIdentity`): the SHA-256 and length of the document's
  original bytes (the decoded file bytes, else the UTF-8 of the source
  view) and of the source view's text, plus the caller's `InputBinding`
  (source repository revision, parse artifact id and digest) copied from
  `InspectDocumentGraphRequest.binding` exactly as given. The server
  never invents repository provenance: no binding sent, none returned.
  `AnalyzePipelineRequest.binding` (field 13, issue #1939) is copied the
  same way into the progressive call's receipt and onto its sink item;
  `AnalyzeDocument.binding` on `AnalyzeStream` is below. On every path a
  `parse_artifact_digest` without a `parse_artifact_id` is refused.
- `result` (`ResultIdentity`): the SHA-256 of the exact TGRF bytes
  `DocumentGraph::to_bytes` writes for the document, so `sha256sum` of
  the graph bytes reproduces it, with the graph format version, the
  envelope schema version (1) and `result_id`, the digest in lowercase
  hexadecimal. Protobuf bytes are never digested: they are not canonical
  across languages. The same bytes give the same identity whatever view
  or offsets are asked for. On the `InspectDocumentGraph` path the
  digest is of the server's canonical re-export of the imported graph,
  not of the request bytes as sent: a graph serialised in another order
  or with slack gets the identity of its canonical form, so one
  document has one `result_id` whichever path produced it, and a
  consumer that keeps non-canonical bytes of its own sees a
  `verify_result` mismatch against them. The extra export costs less
  than the import it follows: 6.5 ms for a 4.2 MB graph of 100,000
  tokens (8.3 ms to import), 62 µs for 36 KB, release build, one core.
- `provenance` (`LayerProvenance`, one per view and layer instance in
  document order): the chained provenance digest
  (`turbo:provenance-digest:v2`), the content digest
  (`turbo:layer-content:v2`) where the layer type has one, the snapshot
  of the read layers' content digests (`turbo:upstream-content:v2`), the
  codec schema the layer's graph bytes are in, the layers read and the
  anchor and index instances it is on, and the `model_ref` into
  `models` for the annotator, version, bundle and execution. The FNV
  `config_hash` on that entry stays a local cache key, never an
  identity.
- `mapping` (`MappingPayload`): the inline view tree, `ViewChain` per
  view in id order with its instance, parent, text digest, text length
  and its native alignment to its parent as `EditRun`s. A run is
  `count` consecutive edits of one parent width and one child width,
  each starting where the one before it ends on both sides, so a
  masked view's one edit per character costs one run per stretch of
  equal-width characters; decoding a run gives the native `Edit` list
  byte for byte. Edit coordinates are UTF-8 bytes whatever the result's
  `OffsetMode` is, which applies to annotation spans only. Mapping is
  the stepwise walk from one view up to the lowest common ancestor and
  down to the other, one alignment at a time, with the native
  `Side::Start` and `Side::End` widening; a composed net alignment is
  never substituted, because its empty-span behaviour differs.
  `turbo_kserve::results::typed::chain::WireChain` is the consumer side
  in Rust: it rebuilds the tree from the message and its `map_span`
  reproduces `Document::map_span` exactly for every span it accepts.
  It checks bounds only: the chain carries byte lengths, not every
  view's text, so a span that splits a UTF-8 scalar, which
  `Document::map_span` refuses, is mapped by the wire consumer, and a
  port of it (#1910) need not promise that refusal. The mapping is inline only
  and bounded by `max_mapping_edits` (4,194,304 native edits over every
  view); a document over it is refused with `TURBO_E_CAPACITY`, never
  truncated. Oneof arm 2, `artifact`, is reserved for an immutable
  artifact reference and is not served: the server keeps no artifacts.
- `TypedLayer.native_view`: the view a layer's spans are natively in,
  before projection into `TypedDocumentResult.view`.
- `ScopedRef`: a reference to one annotation that means the same thing
  outside its message: result id, layer, instance, item index, and the
  native view by instance and text digest. `view`, `native_view`,
  annotation indexes and `model_ref` values are message-local.
  `turbo_kserve::results::typed::scoped` builds and resolves them, and
  refuses a reference into another result or a view whose text changed.

`RoundTripTypedDocument` validates the envelope with the rest: both
identities present and whole, `result_id` the hexadecimal of the
content digest, one provenance entry per view and layer whose model
reference resolves, the `views` list and the mapping naming the same
instances with the same parents both ways, the mapping a view tree in
id order whose runs
decode to a valid alignment that turns the parent's length into the
child's, and `raw_text` the text of the mapping's target view by length
and digest. `turbo_kserve::results::typed::identity::verify_input` and
`verify_result` are the consumer checks: a result is refused against a
document or graph bytes that are not the ones it names, so changed input
never inherits old detections.

### Fragment outcomes

Every `SpanFragment` carries a `SpanOutcome`, straight from the native
projection (`Document::map_span_outcome`, `Document::project_fragments`)
and never recomputed on the wire:

- `SPAN_OUTCOME_EXACT`: mapped back along the chain, the fragment is the
  native span, so it covers exactly the bytes the annotation stood for.
  A fragment on the layer's own view (`TypedLayer.native_view` equal to
  the result's `view`) is always exact, and so is a span the message
  does not project, such as a pii edit's `source`.
- `SPAN_OUTCOME_WIDENED`: a lossy edit on the chain widened it to the
  edit's bounds, so it covers more than the native span stood for.
- `SPAN_OUTCOME_DELETED`: a non-empty native span landed on text the
  result's view deleted; the fragment is empty and anchors where the
  text was. A deleted fragment is kept, not dropped: a standalone one
  stays as an empty anchor of its own, so an item with two fragments on
  deleted text has two empty anchors, and one that lands inside or
  touching another projected fragment coalesces into it as a member
  with outcome deleted. Before #1907 a deleted fragment was dropped from
  a multi-fragment span.

Coalescing is a separate fact. Onto another view, projected fragments
that overlap, or an empty one inside or touching another, coalesce into
one fragment whose `members` list the native fragments by ascending
index in the annotation's native fragment list (for a span gathered from
several items, such as a coreference chain or an entity's evidence, the
items' fragments in order), each with its own outcome; the fragment's
outcome is widened when any member is, else exact when any member is,
else deleted. A fragment that is not coalesced carries no members.

A consumer holding the mapping reproduces every outcome with `WireChain`
alone: map the native span to the result's view and back, stepwise along
the chain; it is exact when the round trip gives the native span back,
deleted when a non-empty span lands empty, widened otherwise. The
validator cannot tell exact from widened without the native span, so it
checks what the message shows: an outcome on every fragment, deleted
only on an empty fragment, every fragment exact on the layer's own
view, members absent or at least two and ascending with the fragment's
outcome merged from theirs, no native fragment a member of two
fragments, and an empty fragment in a multi-fragment span only as a
deletion anchor. Cost, release build, one core, 20 KB text, 7 layers,
6,270 annotations, offsets in UTF-16: the outcome is one extra map back
per projected span, 122 ns against 63 ns for the map alone, and 2 bytes
per fragment on the wire (12.5 KB on 201 KB); projection onto the source
1.34 ms, onto a lossy view 2.89 ms.

## The processing receipt (issue #1908)

A `ProcessingReceipt` says what one attempt at one operation did. It is
kept apart from the result: `TypedDocumentResult.receipt` and the
progressive stream's terminal `receipt` event reference the result by
its `ResultIdentity`, and the result identity is the digest of the graph
bytes, so a new attempt or receipt never changes it.

| Field | What it holds |
|---|---|
| `operation_key` | The caller's key from `InspectDocumentGraphRequest.operation_key` or `AnalyzePipelineRequest.operation_key`, as given: at most 128 printable ASCII bytes, empty when the caller has none. Anything else is `TURBO_E_INVALID_ARGUMENT`, and the refusal does not quote it. |
| `attempt_id` | 32 lowercase hexadecimal digits from the server, new on every attempt. |
| `input` | The `InputIdentity` of the exact input. |
| `result` | The `ResultIdentity` produced; absent when the attempt handed over none. |
| `status` | `ProcessingStatus`: UNSPECIFIED (0, never written), COMPLETE, PARTIAL, NOT_ASSESSED, FAILED, CANCELLED. |
| `layers` | One `LayerReceipt` per view and layer: status, item count, assessed source view byte ranges, requested and executed detectors (annotator and version), truncation reasons, sanitized error and the layer's provenance digest, its configuration identity. |
| `config_digest` | SHA-256 of the processing configuration, rule `turbo.processing-config.v1` (`Document::processing_config_digest`): every view and layer entry sorted by (layer id, instance) with its annotator, version, config hash, model identity and read keys, and no text, view, source or content digest. The same configuration over other input has the same digest, so one operation key reused under another configuration is detectable; `input` carries the input digests. Per-layer provenance digests chain the text digests and stay in the envelope. |
| `requested_steps`, `executed_steps` | The pipeline steps asked for and the ones that ran; empty for `InspectDocumentGraph`. |

The document status is CANCELLED when a deadline cancelled the run,
FAILED when any layer failed, NOT_ASSESSED when nothing past the source
view was assessed, PARTIAL when a layer is partial or not assessed or a
requested step did not run, and COMPLETE otherwise. A layer whose
annotator declined the document (error kind DECLINED, such as a
predicate that does not apply) makes the document PARTIAL by design, so
a conditional pipeline often reads PARTIAL: the per-layer kind tells
DECLINED from UPSTREAM and from damage. COMPLETE with an item count of 0
is an explicit "assessed, nothing found". A complete layer assessed its
whole source text unless its producer reports less
(`receipt::Coverage`), which makes it PARTIAL with the truncation
reasons.

Retries keep the operation key and get a new attempt id. The server
binds and echoes the key and keeps no state for it: rejecting a reused
key with different input belongs to the caller, and nothing promises
exactly-once execution.

The progressive stream ends with the receipt: after `complete`, or, when
a deadline cancels the run, as a CANCELLED receipt before the deadline
status. A stream that ends without one (a dropped connection, an
internal error) is incomplete whatever it delivered before;
`turbo_kserve::results::receipt::stream::terminal_receipt` is the
consumer check: the receipt must be whole and self-consistent, and
bound to the caller's operation key, the source text it sent and the
delivered graph; after no graph it must name no result and be CANCELLED
or FAILED. `RoundTripTypedDocument` applies the same receipt checks and
binds the receipt to its envelope: the same input and result, and one
entry per view and layer with the envelope's provenance digest.

`LayerWireStatus` starts at UNSPECIFIED = 0, so an absent status is never
read as OK. `ViewInfo`, `TypedLayer` and `LayerInfo` say why they are not
OK with a `LayerError` (kind, turbo code, the layer it names, the
annotator's name) in place of the old free-text `status_detail`;
`AnalysisStepFailure` carries one too, and its `failure.message` names
the step and the kind only. No receipt or error carries document text,
values, detail strings or debug renderings.

## Redaction policy (issue #1882)

`server/proto/turbo_pii_policy.proto` defines `RedactionPolicy`, the typed
policy the `turbo-pii:redact` step runs under (see "The redact step"
below); `turbo_kserve::pii_policy::RedactionPolicy::from_proto`
converts it into the native rewrite strategies (`turbo_pii::rewrite`)
and detector lists, with a typed `PolicyError` naming the field.

- **Kind rules** (`PiiKindRule`): a `rule_id` (unique across the policy's
  rules, deny lists and custom identifiers), a `PiiKindRef` (a built-in
  `PiiKindId`, or an index into the policy's own `custom_kinds`), a
  `PiiRuleState`, an ordered list of `PiiStrategyEntry`, a per-kind allow
  list and an optional label override. At most one rule per kind.
- **Strategy entries** are tried in order; the first whose conditions all
  hold rewrites the mention. Conditions read #1903 evidence facts, never
  the confidence tier: a passed checksum (optionally of named
  algorithms), a label fact, the detector id and version, a score of a
  kind at or above a threshold (optionally calibrated only).
- **Strategies** (`PiiRewriteStrategy`) are a oneof of typed messages: mask,
  format template, map (inline entries or a registered dictionary, with a
  fallback that is never a map), truncate, last-N, abbreviate, keyed
  token (HMAC under a registered key), SHA-256 (unkeyed, reversible by
  guessing for low-entropy values, never a default), static text,
  pseudonym (per-document numbers, never a default), ZIP Safe Harbor
  (first three digits kept, `000` for a restricted area, the rest masked;
  the bundled HHS 2000 Census table or the caller's own prefixes), age
  Safe Harbor (an age of 89 or under kept, every older age written as
  `90+`) and date (`year`: the year alone; `shift`: moved by fixed
  years, months and days plus an optional offset a registered key
  chooses from a context string, in the form it was written in;
  `relative`: relative to a `yyyy-MM-dd` reference date; each mode
  records its own `PiiStrategy`). A rule for `date` or `currency` turns
  on the opt-in `Dates` or `Money` pack of the `turbo-pii:pii` step.
  Two reversible arms need a server built with the `pii-reversible`
  cargo feature: `aes_256_gcm` (an authenticated AES-256-GCM token of the
  mention as written, bound to the kind and a `context` string; a fresh
  nonce each time, so tokens are not linkable) and `ff1` (FF1
  format-preserving encryption of the mention's digits or letters under
  an alphabet and a `tweak`, every other character kept in place). Each
  names a key registered with `--pii-encryption-key`. A mention FF1
  cannot take, such as an email under the digit alphabet, fails the
  rewrite. A server without the feature does not know either arm, so the
  policy is refused, or the entry masked and flagged under
  `redact_and_flag_unknown`. No rpc decrypts.
- **Default**: a kind with no rule, or no entry that applies, gets
  `default_strategy`, which is a mask or a keyed token only; unset, a
  whole-value `*` mask. The native `RewriteStrategy::default()` is the
  same mask.
- **Lists**: a global allow list (terms, shape or glob patterns), deny
  lists and shape-pattern custom identifiers, each for a custom kind.
  Terms come inline or from a registered dictionary.
- **No paths, no key bytes.** Keys are `PiiPolicyKeyRef` (id and version)
  and dictionaries `PiiDictionaryRef` (vocabulary store artifact id and
  the SHA-256 of its stored bytes), resolved by the server's registries
  through `PolicyResolver`. A reference id is 1 to 128 ASCII letters,
  digits, `.`, `_`, `-` or `:`; one with a path separator, `..` or a
  leading `.` is refused, and a key id of 32 or more hexadecimal digits
  is refused as key material.
- **Refusals**: every enum's UNSPECIFIED (0) and unknown numbers, an
  unknown built-in kind id, and a strategy or condition arm the server
  does not know (or none set). With `redact_and_flag_unknown`, an unknown
  kind's rule is dropped and an entry with an unknown strategy or
  condition becomes an unconditional whole-value mask, each recorded as a
  `PolicyFlag`; nothing is ever passed through.

### Policy digest

A policy's identity is its `id`, `revision` and `PolicyDigest`: the
SHA-256 of its canonical encoding (`turbo_kserve::pii_policy::canonical`).
Protobuf bytes are not canonical, so they are never digested. The
encoding is:

1. The domain tag `turbo.pii-policy.v1` as a string.
2. The policy message. A message is its present fields in ascending
   field-number order, each as the field number (`u32`) then the value.
   A field is present when it is a scalar other than 0, `false` or the
   empty string, a set message field (even an empty message), a non-empty
   repeated or map field, or the set arm of a oneof whatever its value.
   Unknown fields are not encoded.
3. Values, all integers little-endian: `uint32` and enums (their `i32`
   bits) as `u32`; `uint64` as `u64`; `bool` as the byte `1`; `float` as
   the `u32` of its IEEE 754 bits with `-0.0` taken as `0.0`; a string as
   its byte length (`u64`) and bytes; a nested message as its encoding's
   byte length (`u64`) and that encoding.
4. A repeated field is its count (`u64`) and each element in order; order
   is part of the policy. A map field is its count (`u64`) and each
   entry's key and value, sorted by the key's bytes.

So wire field order and map order do not change the digest, a field
added to the schema later leaves every older policy's digest as it was,
and the revision does change it. A kind rule's `release_ceiling` (field
7) is encoded like any message field: unset is absent, and an empty
ceiling is present with no forms, so the two digest differently.
`server/tests/pii_policy/golden.hex` pins the encoding of one policy,
`server/tests/pii_policy/golden_ceiling.hex` that policy with release
ceilings, and `server/tests/pii_policy_digest.rs` their digests,
computed outside the crate with `sha256sum`. The FNV
`config_hash` of a step stays a local cache hash; the policy digest is in
the redact step's cache key and on every edit and decision.

### The risk step

`turbo-pii:risk` (aliases `turbo:pii-risk`, `pii-risk`) runs beside
`turbo-pii:pii` and writes `turbo:pii-risk`: a per-document score, a
`PiiRiskLevel` from `NONE` to `CRITICAL`, and per-kind counts, severities
and suggested rewrites (`TypedAnnotation.pii_risk`, `PiiRiskAnnotation`).
It carries no value, so a findings-only answer keeps it.
`pii_policy::suggested_policy` builds a `RedactionPolicy` from it for the
redact step. docs/pii.md, "Per-document risk", has the tables and cut
points.

### The redact step

`turbo-pii:redact` (aliases `turbo:pii-redact`, `pii-redact`) runs on
`AnalyzeDocumentProgressive` beside `turbo-pii:pii`. Its policy is
`AnalyzePipelineRequest.pii_redaction` (`PiiRedactionSelection`): either
`inline`, a `RedactionPolicy` converted for this request, or `name`, the
id of a policy the server loaded at start. The step refuses to build
without one, and a selection on a pipeline without the step is refused.
An unknown name is refused without echoing it.

- **Named policies**: `--pii-policy PATH` (repeatable) names a file
  holding one binary `RedactionPolicy` message; every file is read and
  converted at start, served by its `id`, and a file that does not decode
  or convert, or repeats an id, fails start-up. Errors name the flag's
  position, never the path, a term or a key.
- **References**: a keyed-token key (`PiiPolicyKeyRef`) resolves through
  the token keys registered with `--pii-token-key` (PII value release);
  an unregistered key refuses the policy. The keys of the `aes_256_gcm`
  and `ff1` arms resolve through `--pii-encryption-key ID:VERSION:PATH`
  (repeated, or `TURBO_KSERVE_PII_ENCRYPTION_KEYS`), whose file must hold
  exactly 32 bytes. That flag needs `--pii-grant`, as the token keys do,
  and a server built with `pii-reversible`; without the feature it fails
  start-up. The two registries are separate: a token key never answers
  for an encryption reference, or the reverse, and an encryption key
  file holding a token key's bytes fails start-up. A dictionary reference
  (`PiiDictionaryRef`) resolves through the vocabulary store
  (`--vocabulary-dir`, docs/grpc-vocabulary.md), pinned by content: the
  stored dictionary must have exactly the reference's SHA-256 and the
  format its use takes, `plain-word-list` for term sources and
  `headword-definition-tsv` (a key, a tab and the replacement) for map
  tables. A digest mismatch, an unknown artifact, another format or a
  server without a store refuses the policy with an error that names the
  reference and never the content. Entries go through the same
  normalization as inline ones (trimmed, and folded unless
  `case_sensitive`), so a dictionary and the same entries inline redact
  alike; in either, two keys that are equal once normalized refuse the
  policy, since which replacement applies would depend on order. Named policies resolve their dictionaries at start
  (a bad reference fails start-up); inline policies resolve per request,
  and resolved dictionaries are cached by artifact id and digest.
  Uploads stay the vocabulary service's admin calls.
- **Detection**: with a policy, `turbo-pii:pii` drops disabled kinds,
  applies the global and per-kind allow lists and label overrides, and
  adds the deny lists and a custom identifier scanner
  (`turbo-pii:policy-identifiers`, versioned by the policy digest) that
  tests each whitespace-separated word, trimmed of surrounding
  punctuation, against the identifier shapes.
- **Output**: the `redacted` view and its `turbo:pii-edits` layer, as
  #1902's `RedactedView` writes them. Each strategy is chosen per mention
  by the policy's rules; overlapping mentions form one edit written by
  the cluster winner's choice. Every `PiiEditAnnotation` carries the
  choosing rule's `rule_id` (empty for the default), the
  `policy_revision` and the 32-byte `policy_digest`. The step's
  `ModelRef` is the policy id, revision and digest, so the digest enters
  every provenance digest it writes and a new digest misses the cache.
- **Disclosure**: the redacted view is a `turbo-pii:` rewrite view, so
  its text leaves only with `PiiOutputOptions.rewritten_text` allowed by
  the server (PII output); the findings-only default shows none. Mention
  values stay withheld unless released through PII value release.
- **Dry run** (`dry_run`): the same rewrite is run and checked, but the
  step writes a `turbo:pii-plan` layer in place of the view and edits:
  one item per mention with the rule, rule id and strategy its edit would
  record. On the wire each kept candidate of `turbo:pii-candidates`
  carries `PiiPlannedRewrite` (rule, rule id, strategy, revision, digest);
  the candidates layer is detail, so a dry run's answer asks for
  `PiiOutputOptions.detail`.
- **Unknown kinds and strategies** are refused when the policy is
  converted; with `redact_and_flag_unknown` they become whole-value masks
  (a `PolicyFlag` each), never a pass-through.

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
per-token contextual vectors and encoder window spans,
`ANALYZE_OUTPUT_VECTOR_CODES` for the pooled vector's quantized codes, or
`ANALYZE_OUTPUT_DOCUMENT_GRAPH` for the progressive document graph
pipeline), and `offsets`
(`OFFSET_MODE_UTF8`, `OFFSET_MODE_UTF16`, or `OFFSET_MODE_CODE_POINT` for
span offsets on `SparseTermsResult` and `TokenVectorsResult`),
optionally `binding` (field 7, issue #1939), the caller's `InputBinding`
of the text: its source revision and parse artifact id and digest, and
optionally `vocabulary` (field 8, issue #2013), a `VocabularyBinding`
(`stopwords` and `glossary` `VocabularyArtifactRef` pairs of
`artifact_id` and `sha256` resolved against the vocabulary store and
cached by digest; see [docs/grpc-vocabulary.md](grpc-vocabulary.md))
overriding any bundle-level binding for this document. The
server echoes `binding` as given on the document's answer
(`AnalyzeStreamResponse.binding`, field 7), failures included, and on
the answer's sink item, so a downstream store joins the result to the
parse artifact; a document that sends none gets none back, and a
`parse_artifact_digest` without a `parse_artifact_id` fails that
document with `TURBO_E_INVALID_ARGUMENT`. Both `binding` and
`vocabulary` count toward the
document's queued bytes; a refusal over the stream's byte cap echoes it
only when its id and binding fit that cap. Every
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
  `TURBO_QUANTIZE_NONE` with a sparse-terms, token-vectors, or
  document-graph `output`
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
  client asks for codes without asking for vectors;
- `DocumentGraphResponse` (`ANALYZE_OUTPUT_DOCUMENT_GRAPH`, field 8,
  issue #2013), the exported document graph with its `layers` (carrying
  `ModelRef { id: artifact_id, revision: format_id, sha256 }` on any
  bound `turbo:stopwords`, `turbo:keywords`, or `turbo:glossary-terms`
  layer) and `findings` (`TypedDocumentResult`); or
- `AnalyzeFailure` with the turbo status code, field and refusal the
  single-document call would carry (including `TURBO_E_INVALID_ARGUMENT`
  when a `vocabulary` reference fails digest or format validation).
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
`server/tests/input_binding.rs` proves the caller's `InputBinding` on
`AnalyzeStream` answers, progressive receipts and sink items (#1939):
copied as given, absent when not sent, the projection and its receipt
agreeing, and the refusals of an orphan digest and a disagreeing item.
`cargo test --release -p turbo-kserve --test speed --
--ignored` measures the encoding and decoding speed bar against the
current vector and span messages (docs/receipts/server-723/README.md).
