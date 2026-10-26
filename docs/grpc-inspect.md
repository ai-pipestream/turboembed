# Document inspection, projection and export over gRPC

`turbo.result.v1.AnalysisService` exposes four unary RPCs for an existing
document graph. Generate clients from
[`turbo_analysis.proto`](../server/proto/turbo_analysis.proto); the service
runs on the [analysis service](grpc-analysis.md) port. These calls inspect
or format supplied results; they do not load a model or run inference.
Graph decoding, inspection, PCA and formatting run on blocking workers.
Format discovery reads the cached formatter registry.

The `graph` field contains the binary `DocumentGraph` encoding, not JSON
or source text. It must be nonempty and no larger than 64 MiB. The server
imports it through its first-party codec registry and validates the
layers and their bindings. Unknown codecs and malformed graphs are
refused. The request's `offsets` selects UTF-8 bytes (default), UTF-16
code units or Unicode code points. Response spans are half-open and
relative to their complete view, not the displayed page.

## InspectDocumentWindow

`DocumentWindowRequest` selects `view` (0 is the source), zero-based
`page`, and `page_size` in UTF-8 bytes. A zero size means 16,000 bytes;
the maximum is 65,536. Nominal page boundaries are snapped down to UTF-8
character boundaries. Very small page sizes can therefore produce an
empty page. `window.byte_start` and `byte_end` remain byte offsets;
`start` and `end` use the requested offset mode. `clamped` means the
nominal page end passed the text end, not that a character boundary was
adjusted. An empty view has one page. An out-of-range page or view is an
error, not a request for the nearest page.

The response contains:

- `summary`: selected-view byte/unit lengths plus graph layer/view metadata.
- `window`: page text and boundaries.
- `segments`: elementary intervals inside the page with overlapping
  annotations, retaining layer, instance and item identity.
- `document_chips`: supported document-level labels/scores.
- `heatmap_lanes`: supported layers over the entire selected view.
- `xray`: present only when `xray_target_view` is set; reports normalization
  edit runs between the requested views, with offsets relative to each
  respective view, plus transformation provenance.

`filter_layers` matches standard layer IDs, not instance names. An empty
list includes all supported positional layers. It filters `segments`
only: chips, heatmaps and summary metadata are not filtered or paginated.

### Heatmaps

Heatmaps come from `turbo_analysis_layers::DocumentHeatmap::build_instances`.
Each lane retains its source instance in its ID, for example `tokens@gold`.
Sources on other views are omitted; category instances follow their recorded
anchor. Incivility and emotion retain targets on the selected view and keep
original source item indices. Failed/skipped sources remain summary metadata.

`heatmap_buckets = 0` tiles the view with exact elementary intervals,
including uncovered gaps and zero-width point segments. A positive value
emits that many raw byte bins with edges `floor(i * byte_length / N)`,
including empty bins and bins on an empty view when a source lane exists.
The maximum is 4,096. Before shared construction, each lane's candidate input
fragments and the aggregate prospective output are checked against
`max_fragments`. The output bound is N per selected lane for minimaps and at
most twice the candidate fragments plus one interval per lane for exact mode.
Other-view sources do not consume this budget. Inputs and outputs are never
truncated to fit.

`byte_start`/`byte_end` preserve the shared bin's exact coverage coordinates,
independent of the request's offset mode. Raw bins can split UTF-8 scalars,
including empty bins inside a scalar, and are not guaranteed text slices.
`start`/`end` are display coordinates: raw boundaries are snapped down to
character boundaries, then converted to the requested offset mode. Display
ranges can coincide or collapse even when the raw bin has coverage. Scores,
counts and primary items belong to the raw bin; display projection does not
resample or reaggregate them.

`count` counts distinct overlapping source items. Multiple fragments of one
item contribute once. Token, entity and PII density all use that count as the
score and the lowest item index as `primary_item`. Scored lanes choose the
score with greatest absolute magnitude, preserving its sign; ties choose the
lowest original item index. Uncovered raw bins have score/count 0 and no
primary item. Scores are neither length-weighted nor averaged. Lane min/max
come directly from shared segment scores.

| Kind | Shared item/coverage semantics |
|---|---|
| `TokenDensity` | Number of covering tokens. |
| `EntityDensity` | Number of covering entities. |
| `PiiDensity` | Number of covering PII mentions. |
| `SentimentPolarity` | Positive minus negative when both labels exist; otherwise a signed best score for exact positive/pos or negative/neg labels. Other labels contribute no sentiment item. |
| `IncivilityScore` | Maximum of the six abuse scores for sentence, chunk or structure-block targets. |
| `EmotionIntensity` | Maximum emotion score for sentence/chunk targets. |
| `ChunkScore` | Best non-abstained `Categories<Chunks>` score, not token count. |

## ProjectChunks

`ProjectChunksRequest` uses default layer instances. The active view is
chosen from chunks, otherwise sentences, otherwise tokens, otherwise the
source. Present default sentences, tokens and subwords must share that
view. Aligned inputs bound to a different instance are refused where
joined; named layers are not silently substituted for default layers.

`pca_dims` accepts 2 or 3; 0 defaults to 2. The projection prefers chunk
vectors and otherwise sentence vectors. Without either, it returns an
empty projection. Present vectors must be finite. The projection is
`turbo_chunker::VectorProjection`, the library's PCA: it centers nonempty
rows, canonicalizes axis signs, and reports explained-variance fractions;
empty rows retain `empty = true` and zero coordinates. In 2D, `z` is zero.
An adjacent cosine exists only when the current and immediately following
rows are nonempty with nonzero norms. PCA refuses more than 512 rows or
2,048 dimensions and refuses coordinates that overflow finite `float`.

Term stacks are built from available sparse terms, otherwise vocabulary
and term statistics, otherwise token counts. `max_stack_segments` is
the retained top-K per stack, default 10 when zero. Omitted terms remain
in the stack totals and `other_*` fields. No supporting layers means no
corresponding stacks. Sparse terms whose spans are in another view than
the projection's, or do not lie on its text, are refused with
`INVALID_ARGUMENT`.

`include_drift_curve` requests sentence-gap cosine/depth/cost information.
Without sentences the field is absent. Without sentence vectors it uses
available boundary costs/reasons, or defaults, with absent cosine values.
`include_strategy_matrix` requests comparisons and requires default
sentences, subwords and boundaries; otherwise the field is absent.

At most 16 `strategy_budgets` are accepted. With a nonempty list, each
entry runs semantic DP chunking with the supplied budget: `max_tokens`
must be positive and a zero `target_tokens` means `max_tokens`. With an
empty list the server compares greedy sentences, structure-only and
semantic DP at max 256, target 192, size weight 1,000 and overhead 2.
The baseline is the document's existing chunks or, without those, the
last computed strategy. Budget entries do not request a matrix when
`include_strategy_matrix` is false.

## ListOutputFormats and FormatDocument

`ListOutputFormats({})` returns descriptors sorted by `format_id` for
`brat`, `conllu`, `csv`, `jsonld`, `markdown`, `tsv` and `warc`.
Use each descriptor's media type, extension, required layers, `binary`
and `lossless` fields instead of guessing from the ID.

`FormatDocument` takes the graph, `format_id`, `offsets`, and an optional
`doc_id`. IDs are trimmed; an empty document ID becomes `doc`. Empty or
unknown format IDs are refused. The response returns the selected
format/media type/extension and raw `content` bytes. Decode text only
for a textual format. Missing required layers and formatter errors are
reported, not converted to another output format.

## Limits and errors

Shared limits are advertised in `CapabilitiesResponse.limits`. Graph
bytes are capped at 64 MiB;
document pages and stack segments at 1,048,576; fragment-based inspection
collections at 4,194,304. Inspection also bounds cumulative combined
segment annotation memberships, not just the segment count. The
RPC-specific page, bucket, PCA and strategy limits are listed above.
Transport decoding limits also apply.

Invalid options, bindings, views and input data report
`TURBO_E_INVALID_ARGUMENT`. A collection over its named limit reports
`TURBO_E_CAPACITY`. Limits are refusals, not truncated successful answers.
Optional fields being absent because their prerequisite layers are absent
is separate from an invalid input or capacity error.
