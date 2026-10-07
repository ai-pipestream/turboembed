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

Then the demo, naming the models the page should offer:

```
target/release/turbo-kserve-demo --server http://127.0.0.1:8081 \
    --model bge-small-en-v1.5 --model bge-small-fastest --model bge-small-exact
```

and open `http://127.0.0.1:8080`. The protocol has no call that lists a
server's models, so the page offers the names given here, and a field on
the page adds any other name the server serves.

| Flag | Meaning | Absent |
|---|---|---|
| `--listen ADDR:PORT` | Where the page is served. | `TURBO_DEMO_LISTEN`, else `127.0.0.1:8080`. |
| `--server URL` | The server to call, `http://HOST:PORT`. | `TURBO_DEMO_SERVER`, else `http://127.0.0.1:8081`. |
| `--model NAME` | A model name the page offers; repeat for more. | `TURBO_DEMO_MODELS`, comma-separated, else none. |

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
| `GET /api/status` | The server address, whether it is reachable, live and ready, its name, version and extensions, and each offered model's readiness. |
| `GET /api/models/NAME` | `ModelMetadata`: `name`, `versions`, `inputs`, `outputs`, `properties`. |
| `POST /api/infer` | Body `{"model": NAME, "texts": [...], "parameters": {...}}`, a parameter's value a string or an integer. Answers `model`, `version`, `shape`, `vectors` (one array of numbers per text, each the server's float exactly) and `summary`, the response parameters. |

A gRPC error is HTTP 502 with `{"error": {"grpc", "message",
"turbo_code", "turbo_field"}}`; a body the demo cannot read is HTTP 400.

## Tests

`cargo test -p turbo-kserve-demo` starts the server in-process on the
CPU with `testdata/tiny-bert-bundle` at all three tiers, starts the demo
in front of it, and makes every call over HTTP: the page and its files,
status and metadata, texts at each tier giving exactly the floats the
server answers over gRPC and the bundle's reference vectors, parameters
and the server's refusals, and a server that is not there. Nothing is
mocked.
