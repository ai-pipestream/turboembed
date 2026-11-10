# Hybrid search and index lifecycle over gRPC

`turbo.search.v1.TurboSearchService` (`server/proto/turbo_search.proto`)
is served by `turbo-kserve` on the same port as the inference and
analysis services (docs/grpc.md). It manages in-memory hybrid search
indexes (`server/src/search/`), streams documents and `DocumentGraph`s
into them, executes lexical, dense, quantized, sparse, and
late-interaction queries with score or reciprocal-rank fusion, persists
and reloads binary index snapshots, and tracks corpus collection
vocabulary drift over server-streaming watchers.

## The service

| RPC | Mode | Admin | Answer |
|---|---|---|---|
| `ListSearchProviders` | unary | no | The six retrieval backends and query analyzers compiled into the server. |
| `ListIndexes` | unary | no | Every active index's `IndexSummary`, in ascending `index_name` order. |
| `CreateIndex` | unary | `load_from_path` only | Creates an empty index from `spec`, or loads a persisted snapshot from `load_from_path`. |
| `DeleteIndex` | unary | no | Removes an index and any aliases pointing to it. |
| `IndexDocuments` | client stream | no | Ingests up to 4,096 documents into a target index or alias and returns counts and a `VocabularyDriftReport`. |
| `Search` | unary | no | Runs a natural-language and/or `QueryNode` AST query with optional parent expansion and cross-encoder reranking. |
| `PersistIndex` | unary | yes | Atomically writes an index snapshot into the server's snapshot directory. |
| `SealIndex` | unary | no | Marks an index sealed so subsequent `IndexDocuments` calls are refused. |
| `Reindex` | unary | no | Re-indexes retained documents from a source index into a new target index and optionally swaps an alias. |
| `SetAlias` | unary | no | Atomically binds or updates an alias name to an existing concrete index. |
| `DeleteAlias` | unary | no | Removes an alias name. |
| `ListAliases` | unary | no | Every alias and its target `index_name`, in ascending `alias` order. |
| `SetCollection` | unary | no | Creates or updates a corpus collection's baseline terms and `oov_alert_threshold`. |
| `GetCollection` | unary | no | One collection's `CollectionInfo` statistics and top document-frequency terms. |
| `ListCollections` | unary | no | Every collection's `CollectionInfo`, in ascending `collection_name` order. |
| `DeleteCollection` | unary | no | Removes a named collection's state. |
| `WatchCollection` | server stream | no | Streams `CollectionEvent`s when indexing into the collection meets `min_oov_ratio` and the collection's alert threshold. |

## Admin gate

Two operations touch the server's snapshot directory on disk and require
the server to be started with `--admin` (`TURBO_KSERVE_ADMIN=1`):

- `PersistIndex`
- `CreateIndex` when `load_from_path` is non-empty

Without `--admin`, both calls are refused immediately with
`UNIMPLEMENTED` (`TURBO_E_UNSUPPORTED`, `turbo-code` `512`) and touch no
file. Creating an in-memory index (`CreateIndex` with an empty
`load_from_path`) and all other fifteen RPCs are served without
`--admin`.

## Providers

`ListSearchProviders` takes an empty `ListSearchProvidersRequest` and
returns the six compiled providers in fixed order:

| `id` | `kind` | What it does |
|---|---|---|
| `turbo-chunker:bm25-posting` | `lexical` | BM25 inverted posting index with MaxScore pruning and positional phrase slop. |
| `turbo-vector-index:flat-f32` | `dense` | Exact deterministic SIMD `f32` dot and cosine vector index with `Space` verification. |
| `turbo-vector-index:turboquant` | `quantized_dense` | TurboQuant 2-to-4-bit quantized vector index with optional calibration (docs/turboquant.md). |
| `turbo-analysis-layers:sparse-terms` | `sparse` | Learned SPLADE sparse term inverted weight scorer. |
| `turbo-vector-index:token-maxsim` | `late_interaction` | ColBERT-style late-interaction MaxSim scorer with winning subword token attribution. |
| `turbo-query:profile` | `query_analyzer` | Short-query BM25 analyzer and structured temporal, money, quantity, place, glossary, and entity extractor. |

## Indexes and snapshots

### Creating and listing indexes

`CreateIndex` with an empty `load_from_path` builds a new unsealed
`HybridIndex` from `CreateIndexRequest.spec` (`IndexSpec`):

- `index_name` must be non-empty and must not already exist in the
  registry.
- `space` (`SpaceSpec`) is required: `dim` must be greater than 0,
  `model.id` and `model.revision` must be non-empty, `model.sha256` must
  be 64 lowercase ASCII hex digits, and `normalization` must be
  `NORMALIZATION_MODE_NONE` (0) or `NORMALIZATION_MODE_UNIT` (1).
- `metric` must be `DISTANCE_METRIC_COSINE` (0) or `DISTANCE_METRIC_DOT`
  (1).
- `storage` must be `VECTOR_STORAGE_KIND_FLAT_F32` (0) or
  `VECTOR_STORAGE_KIND_TURBO_QUANT` (1). When `TURBO_QUANT` is selected,
  `quant_bits` must be 2, 3, or 4 (0 defaults to 4), and `space.dim`
  must be a multiple of 8.
- When `embed_bundle` is non-empty, it must name a loaded bundle with
  task `TURBO_TASK_EMBED` whose `model_info` (`model_id`, `revision`,
  `manifest_sha256`) matches `space.model` and whose primary `dim` or
  declared `output_dims` includes `space.dim`.

`ListIndexes` returns an `IndexSummary` for every active index in
ascending `index_name` byte order. On a server with no indexes it
returns an empty `indexes` list. Each summary reports `index_name`,
`space`, `metric`, `storage`, `quant_bits` (0 for `FLAT_F32`), `sealed`,
`document_count`, `chunk_count`, `dense_row_count`, `sparse_row_count`,
`token_candidate_count`, and `collection_name`.

`DeleteIndex` removes `index_name` from the registry and prunes any
alias in the alias table whose target was `index_name`. Passing an alias
name to `DeleteIndex` does not resolve the alias; `index_name` must name
a concrete index.

`SealIndex` resolves `index_name` (concrete index or alias), sets
`sealed = true`, and returns the updated `IndexSummary`. Sealing is
idempotent; once sealed, `IndexDocuments` refuses writes to that index
with `FAILED_PRECONDITION` (`TURBO_E_INVALID_STATE`).

### Persisting and reloading snapshots

`PersistIndex` (admin-only) resolves `index_name` (concrete index or
alias) and writes an atomic binary snapshot (`TSRCHIX\x01` header,
encoded `IndexSpec`, seal state, retained documents, chunks, BM25
postings, compiled dense or quantized index, and token index) into the
server's snapshot directory (`Server::set_search_snapshot_dir`,
defaulting to a process-scoped temporary directory). The file is written
to `<path>.tmp_snapshot` and renamed into place.

`CreateIndex` with a non-empty `load_from_path` (admin-only) reads a
snapshot file from the server's snapshot directory, validates that it is
a regular file (symlinks are refused) of at most 128 MiB
(`MAX_SNAPSHOT_BYTES`), decodes the snapshot, optionally overrides the
index name when `spec.index_name` is non-empty, and registers the index
if no index with that name exists.

Both calls confine `path` / `load_from_path` to a single normal file
name: empty names, path separators (`/` or `\`), colons (`:`), NUL
bytes, `.`, and `..` are refused with `INVALID_ARGUMENT`.

### Reindexing

`Reindex` reads the retained documents from `source_index` (concrete
index or alias), creates a new index from `target_spec`, and re-stages
every retained document into the destination index:

- If `target_spec.space` differs from the source index's `SpaceSpec`,
  stored dense and token vectors are stripped and re-computed through
  `target_spec.embed_bundle`. If the source index held dense or token
  vectors and `target_spec.embed_bundle` is empty, `Reindex` refuses the
  call with `INVALID_ARGUMENT` before creating the target index.
- If `target_spec.chunk_budget_tokens` differs from the source index's
  budget and a document was originally ingested without explicit
  `chunks`, the document is re-segmented under the new budget.
- When `swap_alias` is non-empty, `swap_alias` is atomically bound to
  `target_spec.index_name` after the destination index commits.

## Document ingestion

`IndexDocuments` is client-streaming. A stream must carry between 1 and
4,096 (`MAX_STREAM_DOCUMENTS`) `IndexDocumentRequest` messages, and
every message in the stream must name the same non-empty `index_name`
(concrete index or alias). All documents are staged and validated before
any change is committed to the target index:

1. `doc_id` must be non-empty, and either `text` or `graph` (serialized
   `DocumentGraph` bytes, at most `max_graph_bytes` = 64 MiB) must be
   non-empty.
2. When `chunks` is empty, chunks are extracted from `graph` (preferring
   `Levels`, then `Chunks` with `Vectors<Chunks>`,
   `SparseTerms<Chunks>`, and `ChunkCitations`, then `Sentences`, and
   attaching `Subwords` + `Vectors<Subwords>` and `LayoutBoxes` when
   present) or segmented from `text` with `RuleSentenceDetector("en")`
   using `IndexSpec.chunk_budget_tokens` (0 defaults to 64 words).
3. Every chunk `span_start..span_end`, optional
   `parent_start..parent_end`, and `layout_boxes` span must lie within
   the document text on UTF-8 character boundaries. Page citations
   (`has_page_span`, `page_start..=page_end`, and `page_boxes`) must use
   1-based pages, finite coordinates with `x0 <= x1` and `y0 <= y1`, and
   strictly ascending page numbers matching `page_start` and `page_end`.
4. If the index has an `embed_bundle`, any chunk with an empty
   `dense_vector` is embedded on the bundle's session pool with
   `TURBO_PROMPT_DOCUMENT` and the index space's normalization and width.
5. Every non-empty `dense_vector` and `token_vectors` entry must have
   length `space.dim` and contain only finite `f32` values.
6. Each chunk's text is tokenized with `WordTokenizer`, lowercased, and
   stemmed with the English Snowball stemmer for BM25 indexing. After
   all documents in the stream stage cleanly, the BM25, dense, and token
   indexes are rebuilt, collection statistics are updated (under
   `IndexSpec.collection_name` when non-empty, or the resolved
   `index_name` otherwise), and `IndexDocumentsResponse` returns the
   resolved `index_name`, `indexed_documents`, `indexed_chunks`, and the
   batch's `VocabularyDriftReport`.

## Hybrid search

`Search` resolves `index_name` (concrete index or alias) and evaluates up
to `top_k` hits (default 10 when 0, at most `MAX_TOP_K` = 10,000):

- `natural_query`: analyzed by `turbo-query` (`QueryAnalysisProfile`)
  into BM25 query terms (evaluated as a `TERM_MATCH_MODE_ANY` clause)
  and structured narrowing filters (`temporal`, `money`, `quantity`,
  `place`, `glossary`, `entity`), which are returned in
  `extracted_filters` and intersect the candidate set.
- `query_vector` and `query_space`: evaluated as a top-level dense or
  TurboQuant semantic clause against the index's `SpaceSpec`.
- `query_ast` (`QueryNode`):
  - `SemanticClause`: takes `query_vector` with `query_space`, or
    `query_text` when the index configures `embed_bundle` (embedded with
    `TURBO_PROMPT_QUERY`).
  - `TermClause`: BM25 term search with `TERM_MATCH_MODE_ANY` (0) or
    `TERM_MATCH_MODE_ALL` (1).
  - `PhraseClause`: positional BM25 phrase match with `slop` and at most
    64 terms (`MAX_PHRASE_TERMS`).
  - `SparseClause`: dot-product scoring over matching `term_id` (when
    non-zero) or `term_text` entries.
  - `LateInteractionClause`: ColBERT MaxSim scoring over flattened
    `query_token_vectors` (`n_tokens * dim`) verified against
    `query_space`, emitting per-query-token `maxsim` attributions.
  - `FilterClause`: matches `doc_id`, `level_kind`, all `attributes`
    key-value pairs, and case-insensitive `contains_text`. When `query_ast`
    is a top-level `FilterClause` alongside `natural_query` or
    `query_vector`, it acts as a narrowing filter.
  - `BoostClause`: multiplies a child `QueryNode`'s scores and span
    attributions by a finite `boost >= 0.0`.
  - `JoinClause`: combines child `QueryNode`s with `JOIN_OPERATOR_OR` (0)
    or `JOIN_OPERATOR_AND` (1) using `FUSION_STRATEGY_SCORE` (0, sum of
    scores) or `FUSION_STRATEGY_RECIPROCAL_RANK` (1,
    `sum(1 / (rrf_k + rank))`, where `rrf_k <= 0` or non-finite defaults
    to `60.0`).
- When more than one top-level scoring input (`natural_query` terms,
  `query_vector`, `query_ast`) is present, they are combined with
  reciprocal-rank fusion (`k = 60.0`) before narrowing filters are
  applied.
- `rerank_bundle`: when non-empty, cross-encodes the top candidates
  against `natural_query` (or the first non-empty `SemanticClause.query_text`
  in `query_ast`) using the named `TURBO_TASK_RERANK` bundle, replaces
  candidate scores with the reranker logits, and appends a `"rerank"`
  `MatchedSpan`.
- `parent_expansion`: when `enabled` is true, populates `SearchHit.parent_text`
  for chunks with a parent span whose whitespace word count is within
  `max_parent_tokens` (0 allows any length).
- `document_heat`: groups returned hits by `doc_id`, sorted by
  descending `max_score` then ascending `doc_id`, with each document's
  `HeatSegment`s sorted by ascending span start.

## Aliases

Aliases map logical names to concrete index names in the registry:

- `SetAlias` requires both `alias` and `index_name` to be non-empty and
  `index_name` to exist in `registry.indexes` (an alias cannot target
  another alias). If `alias` already exists, it is atomically repointed
  to `index_name`.
- `DeleteAlias` removes `alias` from the alias table, or refuses with
  `NOT_FOUND` when `alias` does not exist.
- `ListAliases` returns all `AliasEntry { alias, index_name }` pairs in
  ascending `alias` byte order. On a server with no aliases it returns
  an empty `aliases` list. Deleting an index via `DeleteIndex` removes
  any aliases pointing to that index.

## Collections and drift watchers

Collections track corpus-wide unit counts, mean chunk length in tokens,
known vocabulary size, document frequencies, and out-of-vocabulary (OOV)
drift across one or more indexes:

- `SetCollection` creates or updates `collection_name` (must be
  non-empty) with `oov_alert_threshold` in `[0.0, 1.0]`. When
  `baseline_terms` is non-empty, each baseline string is analyzed with
  the same tokenizer and English Snowball stemmer as indexed chunks
  (falling back to ASCII lowercase when a non-empty string has no
  alphanumeric tokens), added to `known_terms`, and stored as the
  collection's `baseline_terms`.
- `GetCollection` returns `CollectionInfo` (`collection_name`,
  `total_units`, `mean_length`, `vocabulary_size`,
  `oov_alert_threshold`, and up to 16 `top_terms` ordered by descending
  `document_frequency` then ascending `term`) or refuses with
  `NOT_FOUND`.
- `ListCollections` returns `CollectionInfo` for every collection in
  ascending `collection_name` byte order, including collections created
  explicitly with `SetCollection` and collections created automatically
  when `IndexDocuments` commits into an index (using
  `IndexSpec.collection_name` if set, or the resolved `index_name`
  otherwise). On an empty server it returns an empty `collections` list.
- `DeleteCollection` removes `collection_name` from the collection map,
  or refuses with `NOT_FOUND` when `collection_name` does not exist.
- `WatchCollection` requires a non-empty `collection_name` and opens a
  server-streaming subscription on the registry's broadcast channel for
  events matching `collection_name` whose report has
  `oov_ratio >= max(min_oov_ratio, 0.0)`. An event is broadcast by
  `IndexDocuments` whenever the batch's `oov_ratio` is at least the
  collection's `oov_alert_threshold`.
  - A watcher may subscribe before `collection_name` is created.
  - Calling `DeleteCollection` removes the collection's stored state
    without closing existing `WatchCollection` streams: active watchers
    remain open and receive no events while the collection is absent,
    and if subsequent `IndexDocuments` calls target an index associated
    with `collection_name`, a fresh `CollectionState` (with
    `total_units` reset and `oov_alert_threshold = 0.0`) is created and
    existing watchers receive the new `CollectionEvent`s.

## Errors and status codes

Every error returned by `TurboSearchService` carries the library's
status name in the message and sets `turbo-code` and `turbo-field` in
the gRPC trailing metadata (docs/kserve.md, Errors).

| RPC | Condition | gRPC status | `turbo-code` |
|---|---|---|---|
| `ListSearchProviders` | (none) | `OK` | |
| `ListIndexes` | (none) | `OK` | |
| `CreateIndex` | `load_from_path` is non-empty and `--admin` is not set | `UNIMPLEMENTED` | `512` (`TURBO_E_UNSUPPORTED`) |
| `CreateIndex` | `spec` is missing (when `load_from_path` is empty), `spec.index_name` is empty, `SpaceSpec` is missing or invalid (`dim == 0`, empty model `id`/`revision`, `sha256` not 64 lowercase hex digits, unknown `normalization`), unknown `metric` or `storage`, invalid `TurboQuant` bits or `dim % 8 != 0`, `embed_bundle` is not `TASK_EMBED` or disagrees with `SpaceSpec`, or `load_from_path` has path separators/traversal, is not a regular file, or exceeds 128 MiB | `INVALID_ARGUMENT` | `256` (`TURBO_E_INVALID_ARGUMENT`) |
| `CreateIndex` | `spec.embed_bundle` is not loaded, or `load_from_path` does not exist | `NOT_FOUND` | `1280` (`TURBO_E_BUNDLE_NOT_FOUND`) |
| `CreateIndex` | An index with `index_name` (from `spec` or the loaded snapshot) already exists | `ALREADY_EXISTS` | `261` (`TURBO_E_INVALID_STATE`, `turbo-field` `0`) |
| `CreateIndex` | Snapshot file magic, header, counts, spans, UTF-8, floats, or trailing bytes are invalid | `DATA_LOSS` | `1282` (`TURBO_E_BUNDLE_INTEGRITY`) |
| `CreateIndex` | Filesystem I/O failure reading snapshot file | `INTERNAL` | `1536` (`TURBO_E_INTERNAL`) |
| `DeleteIndex` | `index_name` is empty | `INVALID_ARGUMENT` | `256` (`TURBO_E_INVALID_ARGUMENT`) |
| `DeleteIndex` | No concrete index named `index_name` exists | `NOT_FOUND` | `1280` (`TURBO_E_BUNDLE_NOT_FOUND`) |
| `IndexDocuments` | Empty stream, more than 4,096 documents, stream frames with differing `index_name`, empty `index_name` or `doc_id`, neither `text` nor `graph` set, invalid `DocumentGraph`, out-of-bounds or mid-codepoint spans, invalid `layout_boxes` or `page_boxes`/`page_span`, vector dimension mismatch, or non-finite vector values | `INVALID_ARGUMENT` | `256` (`TURBO_E_INVALID_ARGUMENT`) |
| `IndexDocuments` | Target index/alias does not exist, or the index's `embed_bundle` is not loaded | `NOT_FOUND` | `1280` (`TURBO_E_BUNDLE_NOT_FOUND`) |
| `IndexDocuments` | Target index is sealed (`SealIndex`) | `FAILED_PRECONDITION` | `261` (`TURBO_E_INVALID_STATE`) |
| `IndexDocuments` | `graph` exceeds `max_graph_bytes` (64 MiB), or a chunk exceeds the `embed_bundle` session's `max_seq` | `OUT_OF_RANGE` | `771` (`TURBO_E_CAPACITY`) |
| `IndexDocuments` | All sessions of `embed_bundle` are busy | `UNAVAILABLE` | `770` (`TURBO_E_BUSY`) |
| `IndexDocuments` | Frame or stream exceeds `--max-message-bytes` or `--max-stream-bytes` | `RESOURCE_EXHAUSTED` | (none) |
| `Search` | Empty `index_name`, `top_k > 10_000`, `PhraseClause` over 64 terms, missing `QueryNode.clause`, unknown enum value, negative/non-finite `boost`, missing `BoostClause.child`, `query_vector` without `query_space` or with mismatched `SpaceSpec`/dimension/non-finite values, invalid `SemanticClause` or `LateInteractionClause`, or `rerank_bundle` without query text or naming a non-`TASK_RERANK` bundle | `INVALID_ARGUMENT` | `256` (`TURBO_E_INVALID_ARGUMENT`) |
| `Search` | Target index/alias, `embed_bundle`, or `rerank_bundle` does not exist | `NOT_FOUND` | `1280` (`TURBO_E_BUNDLE_NOT_FOUND`) |
| `Search` | All sessions of `embed_bundle` or `rerank_bundle` are busy | `UNAVAILABLE` | `770` (`TURBO_E_BUSY`) |
| `PersistIndex` | Server was started without `--admin` | `UNIMPLEMENTED` | `512` (`TURBO_E_UNSUPPORTED`) |
| `PersistIndex` | `index_name` is empty, or `path` is empty or contains path separators/traversal | `INVALID_ARGUMENT` | `256` (`TURBO_E_INVALID_ARGUMENT`) |
| `PersistIndex` | Target index/alias does not exist | `NOT_FOUND` | `1280` (`TURBO_E_BUNDLE_NOT_FOUND`) |
| `PersistIndex` | Filesystem I/O failure writing or renaming the snapshot file | `INTERNAL` | `1536` (`TURBO_E_INTERNAL`) |
| `SealIndex` | `index_name` is empty | `INVALID_ARGUMENT` | `256` (`TURBO_E_INVALID_ARGUMENT`) |
| `SealIndex` | Target index/alias does not exist | `NOT_FOUND` | `1280` (`TURBO_E_BUNDLE_NOT_FOUND`) |
| `Reindex` | `source_index` is empty, `target_spec` is missing or invalid, or `target_spec.space` changes on a vector-bearing index without `target_spec.embed_bundle` | `INVALID_ARGUMENT` | `256` (`TURBO_E_INVALID_ARGUMENT`) |
| `Reindex` | `source_index` or `target_spec.embed_bundle` does not exist | `NOT_FOUND` | `1280` (`TURBO_E_BUNDLE_NOT_FOUND`) |
| `Reindex` | `target_spec.index_name` already exists | `ALREADY_EXISTS` | `261` (`TURBO_E_INVALID_STATE`, `turbo-field` `0`) |
| `SetAlias` | `alias` or `index_name` is empty | `INVALID_ARGUMENT` | `256` (`TURBO_E_INVALID_ARGUMENT`) |
| `SetAlias` | Concrete target `index_name` does not exist | `NOT_FOUND` | `1280` (`TURBO_E_BUNDLE_NOT_FOUND`) |
| `DeleteAlias` | `alias` does not exist | `NOT_FOUND` | `1280` (`TURBO_E_BUNDLE_NOT_FOUND`) |
| `ListAliases` | (none) | `OK` | |
| `SetCollection` | `collection_name` is empty, or `oov_alert_threshold` is non-finite or outside `[0.0, 1.0]` | `INVALID_ARGUMENT` | `256` (`TURBO_E_INVALID_ARGUMENT`) |
| `GetCollection` | `collection_name` does not exist | `NOT_FOUND` | `1280` (`TURBO_E_BUNDLE_NOT_FOUND`) |
| `ListCollections` | (none) | `OK` | |
| `DeleteCollection` | `collection_name` does not exist | `NOT_FOUND` | `1280` (`TURBO_E_BUNDLE_NOT_FOUND`) |
| `WatchCollection` | `collection_name` is empty | `INVALID_ARGUMENT` | `256` (`TURBO_E_INVALID_ARGUMENT`) |
