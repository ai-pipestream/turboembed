# The demo page

`turbo-kserve-demo` is a small web page for trying a running
`turbo-kserve` (docs/grpc.md) from a browser: paste a few texts, pick
the served models, send them, and see each model's vectors and the
cosine similarity between the texts, with several models side by side.

It is a client of the server and nothing more. Browsers cannot make
plain gRPC calls, so the demo is a separate binary that serves the page
and answers the page's JSON calls with gRPC calls to the server, the
same calls any client makes. It links no backend and loads no model;
the server is unchanged by it, and it works against a server on any
backend. The page is plain HTML, CSS and JavaScript compiled into the
binary: it loads nothing from any other host, so it works on a machine
with no network beyond the one to the server.

## Starting it

Build both binaries:

```
cargo build --release -p turbo-kserve -p turbo-kserve-demo
```

Start the server with the models to try. To compare tiers, serve one
bundle under one name per tier (docs/grpc.md, Flags):

```
target/release/turbo-kserve --listen 127.0.0.1:8081 \
    --model bundle=/models/bge-small-en-v1.5,device=0,sessions=2 \
    --model name=bge-small-fastest,bundle=/models/bge-small-en-v1.5,device=0,sessions=2,precision=PRECISION_FASTEST \
    --model name=bge-small-exact,bundle=/models/bge-small-en-v1.5,device=0,sessions=2,precision=PRECISION_EXACT
```

Then the demo:

```
target/release/turbo-kserve-demo --server http://127.0.0.1:8081
```

and open `http://127.0.0.1:8080`. The Open Inference Protocol has no
call that lists a server's models, so the demo asks the server's
`AnalysisService/Capabilities` (docs/grpc-analysis.md) and offers every
bundle it lists, in the order served. Naming models overrides the
listing, and the page offers only those:

```
target/release/turbo-kserve-demo --server http://127.0.0.1:8081 \
    --model bge-small-fastest --model bge-small-exact
```

A server without the analysis service offers no names unless given
some. Either way, a field on the page adds any other name the server
serves.

| Flag | Meaning | Absent |
|---|---|---|
| `--listen ADDR:PORT` | Where the page is served. | `TURBO_DEMO_LISTEN`, else `127.0.0.1:8080`. |
| `--server URL` | The server to call, `http://HOST:PORT`. | `TURBO_DEMO_SERVER`, else `http://127.0.0.1:8081`. |
| `--model NAME` | A model name the page offers; repeat for more. | `TURBO_DEMO_MODELS`, comma-separated, else the bundles `Capabilities` lists. |

The demo has no authentication and serves on the loopback address
unless told otherwise. It need not start after the server: it connects
on each call, and the page says when the server is not reachable.

## What the page shows

- The server's name, version, linked backends and readiness
  (`ServerLive`, `ServerReady`, `ServerMetadata`).
- Each offered model with its tier, model id, dimensions, compute type
  and device, from `ModelMetadata`'s properties.
- For each model picked, one `ModelInfer` with every text as one
  batch: the vectors' dimensions, where it ran, the round trip, the
  cosine similarity of every pair of texts, and the first values and
  norm of each vector.
- With more than one model picked, every pair's similarity side by
  side with the spread between the models, and the lowest cosine between
  each model's vector for a text and the first model's. This is where a
  tier's trade-off shows: a device that runs FASTEST in a narrower type
  gives slightly different numbers than at EXACT. The CPU computes every
  tier in F32 (docs/cpu.md), so there the columns are the same.

Optional request parameters are a prompt role (`prompt_role`) and a
token budget per text (`max_tokens` with `truncate` `TRUNCATE_RIGHT`);
everything else is what the bundle says (docs/kserve.md, Parameters). A
refused request shows the server's status, `turbo-code`, field and
message.

The time shown is the round trip the browser saw for one request,
through the demo and the server, with the models called one after
another. It includes the browser, the HTTP and JSON hop through the demo
and the vectors' conversion to JSON, so it is a look at the call, not a
measure of the library; docs/benchmarks.md says how speed is measured.

## The JSON calls

The page uses three calls, which a script can use as well:

| Call | Answer |
|---|---|
| `GET /api/status` | The server address, whether it is reachable, live and ready, its name, version and extensions, each offered model's readiness, and `models_from`: `flags`, `capabilities` or `none`. |
| `GET /api/models/NAME` | `ModelMetadata`: `name`, `versions`, `inputs`, `outputs`, `properties`. |
| `POST /api/infer` | Body `{"model": NAME, "texts": [...], "parameters": {...}}`, a parameter's value a string or an integer. Answers `model`, `version`, `shape`, `vectors` (one array of numbers per text, each the server's float exactly) and `summary`, the response parameters. |

A gRPC error is HTTP 502 with `{"error": {"grpc", "message",
"turbo_code", "turbo_field"}}`; a body the demo cannot read is HTTP 400.

## The gateway calls

These routes are each one gRPC call to the server's
`turbo.result.v1.AnalysisService` (server/proto/turbo_analysis.proto),
`turbo.search.v1.TurboSearchService` (server/proto/turbo_search.proto),
`turbo.vocabulary.v1.TurboVocabularyService`
(server/proto/turbo_vocabulary.proto), or `turbo.training.v1`'s
`TurboCatalogService` and `TurboTrainingService`
(server/proto/turbo_training.proto), for a workbench page or a script.
The page itself does not use them yet. The demo keeps no state between
calls: a graph, an index name or an artifact id is whatever the caller
sends.

Their JSON is the proto message itself: the field names of the .proto,
a missing field its default, a oneof as an object keyed by the field set
(`{"vectors": {...}}`), and an enum as its number. A document's `text`
is a string, and vector `data` (`AnalyzeVectors`, `TokenVectorsResult`,
`ChunkLevelsResult`) is an array of numbers, the server's little-endian
floats in order. A serialized document graph (`graph`,
`initial_graph`), an upload's `source_bytes`, a formatted document's
`content`, a dictionary upload's `chunk` and a download's `data` are
base64 strings, as in the protobuf JSON mapping; base64url and missing
padding are read too. Other `bytes` fields are arrays of byte values.

A `GET` route sends the call's empty request and takes no body. A
`POST` route takes the request message as its body. A route marked
streamed answers one JSON line per message (`application/x-ndjson`), or
one `data:` event per message when the request accepts
`text/event-stream`.

### AnalysisService

| Route | gRPC call |
|---|---|
| `GET /api/capabilities` | `Capabilities`. |
| `POST /api/bundles/load` | `LoadBundle`. An admin call: a server started without `--admin` refuses it. |
| `POST /api/bundles/unload` | `UnloadBundle`, answering `{}`. An admin call too. |
| `POST /api/analyze/stream` | `AnalyzeStream`, streamed. Body `{"header": AnalyzeStreamHeader, "documents": [AnalyzeDocument, ...]}`, the header optional; the demo sends the header and then the documents. |
| `POST /api/analyze/progressive` | `AnalyzeDocumentProgressive`, streamed: `started`, one `layers_ready` per step, then `complete` with the graph. |
| `POST /api/analyze/window` | `InspectDocumentWindow`. |
| `POST /api/analyze/typed` | `InspectDocumentGraph`, answering a `TypedDocumentResult`. |
| `GET /api/analyze/formats` | `ListOutputFormats`. |
| `POST /api/analyze/format` | `FormatDocument`. |
| `POST /api/chunks/project` | `ProjectChunks`. |

### TurboSearchService

| Route | gRPC call |
|---|---|
| `GET /api/search/providers` | `ListSearchProviders`. |
| `GET /api/search/indexes` | `ListIndexes`. |
| `POST /api/search/create` | `CreateIndex`. |
| `POST /api/search/index` | `IndexDocuments`. Body `{"documents": [IndexDocumentRequest, ...]}`, sent in order as the client stream; the first names the index. |
| `POST /api/search/query` | `Search`. |
| `POST /api/search/persist` | `PersistIndex`. Needs the server's snapshot directory. |
| `POST /api/search/seal` | `SealIndex`. |
| `POST /api/search/reindex` | `Reindex`. |
| `POST /api/search/alias` | `SetAlias`. |
| `DELETE /api/search/alias` | `DeleteAlias`. |
| `GET /api/search/aliases` | `ListAliases`. |
| `POST /api/search/delete` | `DeleteIndex`. |
| `POST /api/search/collection` | `SetCollection`. |
| `GET /api/search/collections` | `ListCollections`. |
| `POST /api/search/collection/get` | `GetCollection`. |
| `POST /api/search/collection/delete` | `DeleteCollection`. |
| `POST /api/search/watch` | `WatchCollection`, streamed. The stream has no end; it closes when the caller goes away. |

### TurboVocabularyService

| Route | gRPC call |
|---|---|
| `GET /api/vocabulary/formats` | `ListDictionaryFormats`. |
| `GET /api/vocabulary/dictionaries` | `ListDictionaries`. |
| `GET /api/vocabulary/vocabularies` | `ListVocabularies`. |
| `POST /api/vocabulary/import` | `ImportDictionary`. Body `{"header": ImportDictionaryHeader, "data": base64}`; the demo sends the header and then the data in frames of at most 64 KiB. An admin call. |
| `POST /api/vocabulary/learn` | `LearnVocabulary`. Body `{"header": LearnVocabularyHeader, "documents": [{"text": ...}, ...]}`, the header optional. An admin call. |
| `POST /api/vocabulary/download` | `DownloadVocabulary`, streamed: the descriptor, then the data frames. |
| `POST /api/vocabulary/delete` | `DeleteVocabularyArtifact`. An admin call. |

### TurboCatalogService and TurboTrainingService

| Route | gRPC call |
|---|---|
| `GET /api/catalog/models` | `ListModelCatalog`, unfiltered. |
| `POST /api/catalog/models` | `ListModelCatalog` with the body's `user`, `kind` and `language`. |
| `GET /api/catalog/installed` | `ListInstalledModels`. |
| `POST /api/catalog/accept` | `RecordLicenceAcceptance`. An admin call. |
| `POST /api/catalog/install` | `InstallModel`, streamed. An admin call. |
| `GET /api/training/teachers` | `ListTeachers`. |
| `POST /api/training/static` | `TrainStaticModel`, streamed: the progress stages, then `trained`. An admin call. |
| `GET /api/training/models` | `ListStaticModels`. |
| `POST /api/training/delete` | `DeleteStaticModel`. An admin call. |

The training service has no call that trains a single analysis
component, so there is no route for one.

A gRPC error is the HTTP 502 above, with the `turbo-code` and
`turbo-field` trailers as `turbo_code` and `turbo_field`; a body the
demo cannot read, base64 included, is HTTP 400.

A body is at most 2 MB, axum's default, except on the routes that send
a client stream: `/api/search/index` and `/api/vocabulary/learn` take up
to 256 MiB, the server's dictionary upload cap, and
`/api/vocabulary/import` takes that cap as base64 plus room for the
header. A larger body is HTTP 413 before any call is made. Each message
of the stream is still under the server's own message limit.

A refusal before the stream starts, such as a header over the server's
caps, is the HTTP 502 above. A status that ends a stream after it
started is its last line, `{"error": {...}}` in the same shape, or an
`error` event. A failed document is not an error: it is its own answer,
with a `failure` outcome, and the stream goes on. When the page goes
away, the demo drops the call.

Stream format selection honors case-insensitive `Accept` media ranges,
quality weights and explicit `q=0` exclusions. More specific ranges take
precedence over wildcards. An absent header or wildcard tie selects NDJSON;
an equally preferred explicit SSE range selects SSE. The two formats use
UTF-8. Quoted charset values decode escaped characters; opaque bytes in an
unsupported quoted parameter do not hide later supported media ranges.
If neither format is acceptable, the demo returns HTTP 406 before
opening a gRPC call. Invalid weights and unsupported media parameters do
not match a representation.

## Tests

`cargo test -p turbo-kserve-demo` starts the server in-process on the
CPU with `testdata/tiny-bert-bundle` at all three tiers, starts the demo
in front of it, and makes every call over HTTP: the page and its files,
status and metadata, texts at each tier giving exactly the floats the
server answers over gRPC and the bundle's reference vectors, parameters
and the server's refusals, and a server that is not there. Nothing is
mocked. `demo/tests/gateway.rs` does the same for the gateway routes:

- the capability listing, a stream as lines and as events with a failed
  document among them, its vector equal to `/api/infer`'s, the
  `turbo-code` and `turbo-field` of a refused header and of an admin
  call on a server without them, and an unload and a load;
- a progressive analysis as lines and as events, its graph read back
  through the window, typed, format and projection routes, and an
  unknown step refused;
- an index created, aliased, filled, queried, persisted, sealed,
  reindexed and deleted, a collection set, read, watched while the
  documents arrive, and deleted;
- a dictionary of more than one frame imported and downloaded byte for
  byte, a vocabulary learned, listed, downloaded and deleted, a wrong
  digest refused, and writes refused without `--admin`;
- a static model trained from the tiny bundle, streamed to its trained
  update, listed, refused again by name and deleted, and every catalog
  route refused by a server with no catalog;
- a dictionary upload and a learning text over 2 MB accepted, an index
  batch over 2 MB reaching the server, and a 413 for an analysis body
  over the default.
