# The demo page

`turbo-kserve-demo` is a small web page for trying a running
`turbo-kserve` (docs/grpc.md) from a browser: paste a few texts, pick
the served models, send them, and see each model's vectors and the
cosine similarity between the texts, with several models side by side.

It is a client of the server and nothing more. Browsers cannot make
plain gRPC calls, so the demo is a separate binary that serves the page
and bridges the Connect protocol to the server's gRPC: each call the
page makes is the same call, with the same protobuf bytes, any gRPC
client makes. It links no backend, loads no model and decodes no
message; the server is unchanged by it, and it works against a server
on any backend. The page is plain HTML, CSS and JavaScript compiled into
the binary: it loads nothing from any other host, so it works on a
machine with no network beyond the one to the server.

The page itself still calls the JSON routes this demo had before #1360
and does not work until #1361 ports it to the generated Connect clients
(`demo/web/src/gen`).

## No JSON front end

RPC payloads between the page and the server are binary protobuf
through Connect-ES only. The page's clients are generated from the
protos by `@bufbuild/protoc-gen-es` and call the demo with
`@connectrpc/connect-web` using the binary codec: `application/proto`
for unary calls and `application/connect+proto` for streams, each a
`POST /<package>.<Service>/<Method>` that the demo forwards to the
server when the method is on its allowlist. Raw file uploads are an
`HTTP PUT` of binary frames to the demo's upload endpoints, which feed
the client-streaming RPCs (#1377), up to the 2 GiB request cap.

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
| A message over 64 MiB, the server's default limit | `resource_exhausted` |
| GET on a method's path | HTTP 405, `allow: POST` |

GET answers `/`, `/app.js` and `/style.css` only.

### The allowlist

The demo knows every unary and server-streaming method of the server's
services (`demo/src/allow.rs`, checked against `server/proto` by
`demo/tests/methods.rs`). The ones that only read are forwarded by
default; one that changes the server (loading a bundle, an index or
collection write, a licence acceptance, `InstallModel`,
`TrainStaticModel`, a delete) only when named with `--allow`, and the
server still refuses it unless started with `--admin`. Client-streaming
and bidi methods are not known: a browser cannot make them (#1377). The
gRPC health service is not forwarded. This list is a fixed default for
now; configuring it, authentication, TLS and CORS are #1373.

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

The demo has no authentication and serves on the loopback address
unless told otherwise. It need not start after the server: it connects
on each call, and a call while the server is down is `unavailable`.

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
