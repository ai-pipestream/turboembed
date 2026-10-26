# The demo page

`turbo-kserve-demo` is a small web workbench for trying a running
`turbo-kserve` (docs/grpc.md) from a browser: compare embedding models,
watch a document's analysis arrive step by step, build and search an
index, learn and import vocabularies, and see what the server serves.

It is a client of the server and nothing more. Browsers cannot make
plain gRPC calls, so the demo is a separate binary that serves the page
and bridges the Connect protocol to the server's gRPC: each call the
page makes is the same call, with the same protobuf bytes, any gRPC
client makes. It links no backend, loads no model and decodes no
message; the server is unchanged by it, and it works against a server
on any backend. The page is plain HTML, CSS and JavaScript compiled into
the binary: it loads nothing from any other host, so it works on a
machine with no network beyond the one to the server.

The page is written in TypeScript under `demo/web/src` over the
generated Connect clients (`demo/web/src/gen`) and bundled into
`demo/static/app.js` (Building the page, below).

## No JSON front end

RPC payloads between the page and the server are binary protobuf
through Connect-ES only. The page's clients are generated from the
protos by `@bufbuild/protoc-gen-es` and call the demo with
`@connectrpc/connect-web` using the binary codec: `application/proto`
for unary calls and `application/connect+proto` for streams, each a
`POST /<package>.<Service>/<Method>` that the demo forwards to the
server when the method is on its allowlist. Raw file uploads are an
`HTTP PUT` of binary frames to the demo's upload endpoints, which feed
the client-streaming RPCs, up to the 2 GiB request cap (see Uploads).

JSON is allowed only at the edges where it fits, such as processing a
form. There it stays small and local, and it never carries or mirrors a
proto message. It is never a general front-end layer and never in the
RPC path: there are no `/api/*` JSON, NDJSON or SSE routes, no serde
derives on generated proto types and no hand-written JSON or base64
mapping of messages; GET answers only the static page.

The reason: once JSON is in, agents keep pushing it into the hot path,
and that has forced several rewrites (the latest: #1271, #1370).

`bench/tests/tree.rs` (tree/front_end.rs) enforces the RPC-path part,
where it can be exact: `demo/Cargo.toml` has no `serde` or
`serde_json`, the server has neither outside its tests, no `build.rs`
that compiles the protos adds a serde attribute, and no `"/api/` route
string is in `demo/src`. It does not look at JSON anywhere else, and it
has no pending list: it holds for the whole RPC path.

## The Connect bridge

Each call is a `POST /<package>.<Service>/<Method>`, the gRPC method's
own path, in one of two forms of the Connect protocol, the binary codec
only:

| Content type | Methods | Request | Answer |
|---|---|---|---|
| `application/proto` | unary | the request message | HTTP 200 and the reply message, `application/proto`; or the error below |
| `application/connect+proto` | unary and server-streaming | the request message in one envelope | HTTP 200, `application/connect+proto`: each reply message in an envelope, then the end-of-stream envelope |

An envelope is one flags byte, the length as four bytes big-endian, and
the message. The end-of-stream envelope has flags `0x02` and holds JSON,
by the protocol's definition: `{}` when the call succeeded, else
`{"error": {"code": "not_found", "message": "..."}, "metadata":
{"turbo-code": ["1280"], "turbo-field": ["0"]}}`. A unary error is the
HTTP status of its code (`not_found` 404, `invalid_argument` 400,
`unavailable` 503, and the rest of the protocol's table), the body
`{"code": ..., "message": ...}` as `application/json`, and the server's
`turbo-code` and `turbo-field` as response headers, where Connect
clients read a unary error's metadata. Only the server's `turbo-*`
metadata is passed on. These JSON bodies carry a status, never a
message.

The demo passes the bytes through: a gRPC codec on its side leaves each
message as the bytes it is (`demo/src/codec.rs`), so the request goes to
the server and the reply comes back unchanged, over one HTTP/2 channel
to the server shared by every call. When the page goes away mid-stream,
the demo drops its call and the server sees it cancelled.
`connect-timeout-ms` becomes the gRPC call's deadline.

Refused before any call to the server:

| Request | Answer |
|---|---|
| Any other content type, JSON included, or none | HTTP 415, `accept-post: application/proto, application/connect+proto` |
| A server-streaming method as `application/proto` | HTTP 415 |
| A method the demo knows but does not forward | `permission_denied` |
| A path that is no unary or server-streaming method | `unimplemented` |
| A compressed request or message | `unimplemented` |
| A streaming request that is not exactly one uncompressed envelope | `invalid_argument` |
| A message over 64 MiB, the server's default limit, refused before it is read when Content-Length or the envelope says so | `resource_exhausted` |
| GET on a method's path | HTTP 405, `allow: POST` |

The demo forwards a request message of up to 64 MiB and a reply message
of up to 256 MiB, the server's defaults (`demo/src/limits.rs`,
docs/grpc.md).

GET answers `/`, `/app.js` and `/style.css` only.

### The allowlist

The demo knows every unary and server-streaming method of the server's
services (`demo/src/allow.rs`, checked against `server/proto` by
`demo/tests/methods.rs`). The ones that only read are forwarded by
default; one that changes the server (loading a bundle, an index or
collection write, a licence acceptance, `InstallModel`,
`TrainStaticModel`, a delete) only when named with `--allow`, and the
server still refuses it unless started with `--admin`. A name given to
`--allow` that is not in the table stops the demo at startup. Anything
not in the table is refused `unimplemented` and a write not named is
refused `permission_denied`, both before any bytes go to the server.
The four client-streaming methods are in the table as uploads, all
writes: they are reached only by `HTTP PUT` (see Uploads), and a Connect
`POST` to one is refused `unimplemented`. The gRPC health and reflection
services, the rest of `TurboSinkService` and every bidi method are not
in the table.

### Uploads

A file goes to the server as an `HTTP PUT` to `/upload` followed by the
client-streaming method's own path (`demo/src/upload.rs`,
`demo/src/frames.rs`). Uploads pass the same gate, TLS and allowlist as
any call; all four are writes, so none is forwarded unless named with
`--allow` (by its method path), and the server still applies its own
checks.

| Endpoint | Body after the content type |
|---|---|
| `PUT /upload/turbo.vocabulary.v1.TurboVocabularyService/ImportDictionary` | one `ImportDictionaryHeader` frame, then the dictionary's raw bytes |
| `PUT /upload/turbo.vocabulary.v1.TurboVocabularyService/LearnVocabulary` | one `LearnVocabularyHeader` frame, then one `LearnVocabularyDocument` frame per document |
| `PUT /upload/turbo.search.v1.TurboSearchService/IndexDocuments` | one `IndexDocumentRequest` frame per document |
| `PUT /upload/turbo.sink.v1.TurboSinkService/StreamDocuments` | one `DocumentSinkItem` frame per item |

The content type must be `application/octet-stream`; any other, JSON and
multipart included, is HTTP 415 with `accept: application/octet-stream`.
A frame is the Connect envelope: a flags byte that must be 0, the length
as four bytes big-endian, then the binary protobuf message. The demo
cuts the body into frames as it arrives and sends each to the server as
one message of the stream, wrapping a header or document frame in the
request's `oneof` without decoding it; a dictionary's raw bytes go in 4
MiB chunks. Nothing is buffered beyond one frame or chunk, so the server
starts on the upload before the body has ended.

The answer is HTTP 200, `application/proto`, with the reply message as
bytes. An error has an empty body and its status in headers:
`grpc-status`, `grpc-message` (percent-encoded) and the server's
`turbo-code` and `turbo-field`, with the HTTP status of the code from the
table above. Refused by the demo:

| Request | Answer |
|---|---|
| Not on the allowlist, or a write not named with `--allow` | 403, `permission_denied`, before the server is reached |
| A path that is no upload | 501, `unimplemented` |
| A frame with flags other than 0, or a body that ends inside a frame | 400, `invalid_argument` |
| A frame over 64 MiB less 16 bytes (room for the wrapping), the server's message limit | 429, `resource_exhausted`, as soon as its length is read |
| A `Content-Length` over 2 GiB, or a body whose frames reach the server's 2 GiB stream ceiling (#1414) | 413, `resource_exhausted`, at once or as soon as the frame crossing it is read |
| GET or POST on an upload path | 405, `allow: PUT` |

The 2 GiB cap counts what is forwarded (each message and its five-byte
gRPC prefix), the way the server counts it, so the demo refuses first; a
body near the cap whose frames are wrapped can be refused a few bytes
before 2 GiB of body. The server keeps its own per-call caps where it
holds the input in memory: an imported dictionary of at most 256 MiB and
a learning corpus of at most 1 GiB. A PUT has no deadline.

### The gate

Before the allowlist, each call passes two checks
(`demo/src/gate.rs`):

- Origin. The page is served from the demo's own origin, so a call with
  an `Origin` header that does not match the listener's HTTP/HTTPS scheme
  and the request's `Host` with its effective port is refused
  `permission_denied`, and no CORS header is sent. Origins named with
  `--cors-origin` are let in: their calls get
  `access-control-allow-origin` with that origin and may read
  `turbo-code`, `turbo-field`, `grpc-status` and `grpc-message`, and
  their preflight (`OPTIONS` asking for `POST` or `PUT`) is answered 204
  with `access-control-allow-methods: POST, PUT`. Any other preflight is 403 with no CORS
  headers. Default ports (HTTP 80 and HTTPS 443) are normalized, including
  configured origins; malformed ports and non-HTTP schemes are rejected.
  Forwarded scheme headers do not change this check. There is no wildcard;
  `*` and an origin with a path stop the demo at startup.
- Token. With `--token-file` (or `TURBO_DEMO_TOKEN`) every call must
  carry `authorization: Bearer <token>`; without it, or with another,
  it is refused `unauthenticated` (401). The page's files need no token.
  There is no token by default. The demo will not start on an address
  other than loopback with a write allowed and no token.

With `--tls-cert` and `--tls-key` the demo serves HTTPS (TLS 1.2 and
1.3, HTTP/1.1, the same ring provider as the server) and plain HTTP is
not answered. A certificate or key that does not load stops the demo at
startup. A token sent off loopback without TLS crosses the network in
the clear; the demo says so when it starts.

## Generated clients

The page's protobuf and Connect-ES code is generated, never written by
hand: `scripts/generate-web-stubs.sh` runs `buf generate` with
`server/proto/buf.gen.yaml` and `protoc-gen-es` v2 (`target=ts`) into
`demo/web/src/gen`, one `*_pb.ts` per proto file with its message
schemas and service descriptors. The output is committed; the `proto` CI
job regenerates it and fails on any difference, so a `.proto` change
lands with its regenerated TypeScript. The script needs Node and npm and
installs the exact versions in `demo/web/package-lock.json`
(docs/release.md); no `build.rs` runs it, and `cargo build` needs
neither.

## Building the page

`scripts/build-web.sh` type checks `demo/web` with `tsc`, runs the unit
tests in `demo/web/test` under Node (the page's SHA-256 against Node's,
and the upload bodies decoded frame by frame), and bundles
`demo/web/src/main.ts` with esbuild into `demo/static/app.js`, an
unminified ES module. The bundle is committed, so `cargo build` needs no
Node; the `proto` CI job rebuilds it with `--check` and fails on any
difference. TypeScript 7.0.2 and esbuild 0.28.2 are pinned exactly in
`demo/web/package.json`, beside the generator pins; the page has no UI
framework.

In `demo/web/src`, `rpc.ts` makes the one Connect transport (binary,
POST only, the bearer token from the page's token field) and a client
per service, and `upload.ts` makes the one `fetch`, the upload PUT;
`frames.ts` builds the upload bodies as Blobs, so a picked file is sent
from disk without being read into memory, and `sha256.ts` hashes a
dictionary for its import header. The tree test
(`bench/tests/tree/front_end.rs`) holds the page's own TypeScript to
this: no `JSON.`, `toJson`, `fromJson`, `application/json` or
`connect+json`, no JSON or GET transport, and no `fetch` outside
`upload.ts`.

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

and open `http://127.0.0.1:8080`. To let the page train a static model
on a server started with `--admin`:

```
target/release/turbo-kserve-demo --server http://127.0.0.1:8081 \
    --allow /turbo.training.v1.TurboTrainingService/TrainStaticModel
```

| Flag | Meaning | Absent |
|---|---|---|
| `--listen ADDR:PORT` | Where the page is served. | `TURBO_DEMO_LISTEN`, else `127.0.0.1:8080`. |
| `--server URL` | The server to call, `http://HOST:PORT`. | `TURBO_DEMO_SERVER`, else `http://127.0.0.1:8081`. |
| `--allow METHOD` | Also forward this method that changes the server, `/<package>.<Service>/<Method>`; repeat for more. A name that is not a known method stops the demo at startup. | `TURBO_DEMO_ALLOW`, comma-separated, else the reads only. |
| `--token-file PATH` | A file holding the bearer token every call must carry. | `TURBO_DEMO_TOKEN_FILE`, else the token in `TURBO_DEMO_TOKEN`, else none. |
| `--cors-origin ORIGIN` | Another origin whose pages may call, `http(s)://HOST[:PORT]`; repeat for more. | `TURBO_DEMO_CORS_ORIGINS`, comma-separated, else none. |
| `--tls-cert PEM`, `--tls-key PEM` | Serve HTTPS with this certificate chain and private key; both or neither. | `TURBO_DEMO_TLS_CERT` and `TURBO_DEMO_TLS_KEY`, else plain HTTP. |

The demo serves on the loopback address with no token unless told
otherwise. It need not start after the server: it connects
on each call, and a call while the server is down is `unavailable`.

## What the page shows

The page has a tab per area; leaving a tab cancels its streams and
uploads. Writes (marked below) are refused with `permission_denied`
unless the demo is started with `--allow` for them, and the tab says
so with the call's code, `turbo-code`, field and message, as it does
for any refused call.

- Embed: each served embedding bundle (from `Capabilities`) with its
  tier, model id, dimensions, compute type and device from
  `ModelMetadata`'s properties. For each model picked, one `ModelInfer`
  with every text as one batch: the vectors' dimensions, where it ran,
  the round trip, the cosine similarity of every pair of texts, and the
  first values and norm of each vector. With more than one model picked,
  every pair's similarity side by side with the spread between the
  models, and the lowest cosine between each model's vector for a text
  and the first model's. This is where a tier's trade-off shows: a
  device that runs FASTEST in a narrower type gives slightly different
  numbers than at EXACT. The CPU computes every tier in F32
  (docs/cpu.md), so there the columns are the same. Optional parameters
  are a prompt role (`prompt_role`) and a token budget per text
  (`max_tokens` with `truncate` `TRUNCATE_RIGHT`).
- Analyze: `AnalyzeDocumentProgressive` as it streams, each step with
  its time and layers and any failed step with what it skipped; then
  the finished graph through `InspectDocumentWindow`, the text with its
  annotations marked, a chip per layer to show only some, paging, and
  heatmap lanes; and `FormatDocument` in any format `ListOutputFormats`
  names, shown when it is text and offered as a download.
- Search: `ListIndexes`; `CreateIndex` (write) from an embedding bundle,
  its model, first width, cosine and flat F32 storage; typed paragraphs
  and picked files added with the `IndexDocuments` upload (write); and
  `Search` with each hit's score, signal scores and matched spans, and
  the document heat. `DeleteIndex` is a write.
- Vocabulary: the store's limits and formats, the stored dictionaries
  and vocabularies, `ImportDictionary` and `LearnVocabulary` as uploads
  (writes), `DownloadVocabulary` streamed into a file, and
  `DeleteVocabularyArtifact` (write).
- Server: the server's name, version and readiness (`ServerLive`,
  `ServerReady`, `ServerMetadata`); `Capabilities`' bundles, analysis
  profiles and limits; `LoadBundle` and `UnloadBundle` (writes); and the
  model catalog and installed models when the server keeps a catalog.

The time shown is the round trip the browser saw for one request,
through the demo and the server, with the models called one after
another. It includes the browser and the HTTP hop through the demo, so it is
a look at the call, not a measure of the library; docs/benchmarks.md says how speed is measured.

## Tests

`cargo test -p turbo-kserve-demo` starts the server in-process on the
CPU with `testdata/tiny-bert-bundle`, starts the demo in front of it,
and makes every call over HTTP with Connect frames built by hand from
the server's prost types. Nothing is mocked.

- `demo/tests/unary.rs`: the page's files, and GET refused on a method;
  `ModelInfer` answering exactly the bytes the server answers over gRPC;
  `ModelMetadata`, `Capabilities` and `ServerLive`; the server's
  `not_found` and `out_of_range` with their `turbo-code` and
  `turbo-field` headers; a server that is not there.
- `demo/tests/stream.rs`: `AnalyzeDocumentProgressive` giving the
  server's events; `WatchCollection` giving an event when a document is
  indexed; the server's refusals in the end-of-stream envelope.
- `demo/tests/admin.rs`: `TrainStaticModel` and `InstallModel` (from a
  catalog served from memory) when allowed; a refused field's
  `turbo-code` and `turbo-field`; a page that goes away mid-training,
  after which the server has stopped the training and written nothing.
- `demo/tests/refusals.rs`: 415 for JSON and other content types, the
  allowlist, compression and malformed envelopes.
- `demo/tests/methods.rs`: the method table against the protos.
- `demo/tests/gate.rs`: with the server not there, every write and
  every unknown method refused without it and every read tried; the
  token; a write off loopback without a token refused at startup; the
  origin check, CORS grants and preflights.
- `demo/tests/tls.rs`: a call over TLS with a test CA
  (`demo/tests/tls`, for these tests only), plain HTTP not answered, a
  certificate or key that does not load, and an upload over TLS refused
  by default and forwarded when allowed.
- `demo/tests/upload.rs`: a 7 MB dictionary whose first 4 MiB chunk
  reaches the server before the body is sent, a learning corpus, and a
  batch of 30 documents over 2 MB answered with the binary
  `IndexDocumentsResponse`.
- `demo/tests/upload_refusals.rs`: the allowlist, default-deny writes,
  the token, origins and the PUT preflight, 415 for JSON and multipart,
  compressed and cut-short frames, a frame over the message limit, a
  body over 2 GiB by its `Content-Length`, and the server's refusal with
  its `turbo-code` and `turbo-field` headers. With `TURBO_TEST_2GIB=1` it
  also streams 2 GiB of `StreamDocuments` frames, all accepted, and one
  frame more, refused 413.
