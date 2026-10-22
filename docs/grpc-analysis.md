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

Span offsets cross in the unit a message names (`OffsetMode` on
`ChunkLevelsResult`): UTF-8 bytes (the internal offsets), UTF-16 code
units (Java string indices), or Unicode code points, converted through
the view text's checkpoint table (`server/src/offsets.rs`). An offset
that is out of bounds, inverted, or inside a code point or a UTF-16
surrogate pair refuses with `TURBO_E_INVALID_ARGUMENT`; nothing is
snapped to a boundary. `ChunkLevelsResult` carries `turbo:chunks`,
`turbo:levels`, and `opennlp:embeddings` over the chunks in one round
trip, fragments included. `RoundTripDocumentGraph` takes binary
`DocumentGraph` bytes and answers with the same document imported and
re-exported, listing every layer's standard id (`opennlp:*` or
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
  a discontinuous answer never decodes as answer text. The schema-driven structured
  records of issue #715 have no merged layer yet; when one lands, its
  results join this message or grow one beside it.

## Capabilities

`Capabilities` answers with what a client needs to refuse its own
request, per loaded bundle:

| Field | Meaning |
|---|---|
| `name` | The name requests use. |
| `model` | The bundle's identity: model id, revision, and the manifest's sha256, the contract the load was made against. |
| `task` | The task it serves, the header constant's name (`TASK_EMBED`). |
| `roles` | The prompt roles it takes: `PROMPT_QUERY` for a non-empty `prefix_query`, `PROMPT_DOCUMENT` for a non-empty `prefix_document`, `PROMPT_NONE` when it carries neither. |
| `widths` | `dim` first, then the `output_dims` a request may name by `output_dim`, ascending. |
| `context_limit` | Tokens per row; a longer row is `TURBO_E_CAPACITY` naming `max_seq`. |
| `batch_limit` | Rows per run; a larger batch is `TURBO_E_CAPACITY` naming `max_batch`. |
| `schema` | The limits an extraction schema may carry; unset until a task reads one (issue #715). |

The limits are the session's: what a request is actually held to, where a
server started below the model's limits. The response's `limits` field
names what one result message may carry (`max_sparse_entries`,
`max_sparse_occurrences`, `max_token_vector_bytes`, `max_rerank_items`,
`max_answers`); a larger message is `TURBO_E_CAPACITY` (771,
`OUT_OF_RANGE`, docs/kserve.md, Errors) naming that limit, and nothing is
truncated.

The listing follows the bundles: `LoadBundle` serves one more and answers
with its entry, `UnloadBundle` stops serving a name (requests holding its
sessions finish; the name is then refused as never served), and the next
`Capabilities` shows the change.

Load and unload are admin calls, and the server does not serve them by
default: start it with `--admin` (docs/grpc.md, Flags) to answer them,
and until then they refuse with `TURBO_E_UNSUPPORTED` (`UNIMPLEMENTED`).
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

## Tests

`cargo test -p turbo-kserve` runs everything against the server
in-process on the CPU with `testdata/tiny-bert-bundle`, called over gRPC
with a real client (`server/tests/results.rs`); nothing is mocked. Each result
kind crosses the wire and back unchanged from its in-process layer; the
capability listing matches the manifest and follows load and unload; an
over-limit request is refused with the named limit, for every limit the
listing names. `cargo test --release -p turbo-kserve --test speed --
--ignored` measures the encoding and decoding speed bar against the
current vector and span messages (docs/receipts/server-723/README.md).
