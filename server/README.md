# Inferstream

`inferstream` is the Turbo inference server: one binary, two listeners
over one engine.

- **gRPC**: `inference.GRPCInferenceService`, the KServe Open Inference
  Protocol v2 (`ServerLive`, `ServerReady`, `ModelReady`, `ServerMetadata`,
  `ModelMetadata`, `ModelInfer`), generated from the protocol's own
  `open_inference_grpc.proto` (`proto/`); the extension service
  `turbo.inferstream.InferstreamExtension` (`ModelStreamInfer`,
  `RepositoryIndex`, `RepositoryModelLoad`, `RepositoryModelUnload`, from
  `proto/turbo_inferstream.proto`); and server reflection, so `grpcurl`
  and the KServe clients need no proto file.
- **HTTP**: the OIP v2 REST binding under `/v2`, the OpenAI-shaped
  `/v1/embeddings`, `/v1/rerank`, `/v1/chat/completions` (streaming over
  SSE) and `/v1/classify`, and `/info` with the fields
  text-embeddings-inference clients read.

Every model is a bundle on a named provider device with a pool of
fixed-shape sessions. Nothing is truncated unless a request asks for it,
every option is honored exactly or rejected with the Turbo status and the
field index, and a failed model load fails the server rather than leaving
it half ready.

## Run

```sh
cargo build -p turbo-inferstream
target/debug/inferstream \
    --provider-lib target/release/libturbo_provider_cuda.so \
    --provider-lib target/release/libturbo_provider_ggml.so \
    --model name=minilm,bundle=~/opt/bundles/minilm-onnx,provider=cuda \
    --model name=rerank,bundle=~/opt/bundles/rerank-onnx,provider=cuda \
    --model name=qwen,bundle=~/opt/bundles/qwen05-gguf,provider=ggml,generations=2 \
    --http 0.0.0.0:8000 --grpc 0.0.0.0:8001
```

Or a JSON file, `--config server.json`, with `provider_libs` and `models`
(the same keys as the flag). `--http` and `--grpc` also read
`INFERSTREAM_HTTP` and `INFERSTREAM_GRPC` (the flag wins), which is how
the container image binds every interface. `--model` keys: `bundle` and `provider`
(required), `name` (default: the last segment of the bundle's
`model_id`), `ordinal` (default 0), `buckets` (`1x128;8x256`), `sessions`
(per bucket, default 1), `generations` (concurrent generations, default
1). Built-in providers (`mock`, `static`) need no `--provider-lib`; the
mock devices are ordinal 0 (CPU) and 1 (accelerator).

The build needs `protoc` on the path (or `PROTOC=...`); the proto is
compiled at build time by `tonic-prost-build`.

## Session buckets

Turbo sessions have a fixed `(max_batch, max_seq)`. Each model keeps a
pool of sessions per bucket; a request is served by the smallest bucket
whose sequence length fits its longest text (counted by the bundle's
tokenizer) and whose batch fits its rows, split into bucket-sized chunks
when it has more rows than the widest bucket, and padded within the
bucket. A text longer than the longest bucket is rejected with the limit
(`TURBO_E_CAPACITY`, 422, field 2) unless the request sets `truncate` to
`right` or `left`; the default `model` never cuts a text here. Without a
core tokenizer (GGUF bundles) the longest bucket serves every request and
the provider's own length rule applies.

Default buckets: batches 1, 8 and the model's `max_batch`, each at the
model's `max_seq`. Generative models keep no sessions; `generations`
bounds how many run at once, and a client that disconnects mid-stream
cancels its generation on the device.

## Open Inference Protocol v2

Tensor names per model kind (the same as `demo/java-web-spring`, so one
client works against both):

| kind | inputs | outputs |
|---|---|---|
| embedding | `text` BYTES `[n]` | `embeddings` FP32 `[n, dim]` |
| reranker | `query` BYTES `[1]`, `documents` BYTES `[n]` | `scores` FP32 `[n]`, `sorted` INT32 `[k]` |
| classifier | `text` BYTES `[n]` | `scores` FP32 `[n, labels]`, `labels` BYTES `[labels]` |
| token classifier | `text` BYTES `[n]` | `spans` BYTES `[m]` (a JSON object per span: `row`, `byte_start`, `byte_end`, `label`, `score`), `labels` BYTES `[labels]` |
| generative | `prompt` BYTES `[1]`, or `messages` BYTES `[t]` (a JSON `{"role","content"}` per turn) | `text` BYTES `[1]`; `finish_reason`, `prompt_tokens`, `generated_tokens` in the response parameters |
| generic (RUN) | the model's own inputs, typed (FP32, INT32, ...) | the model's own outputs; an output buffer is passed as an input tensor of that name |

Turbo options travel in the request `parameters` map: `truncate`
(`none`, `right`, `left`), `max_tokens`, `prompt_role` (`query`,
`document`), `normalize` (`l2`, `none`), `pooling` (`mean`, `cls`,
`last`), `output_dim`, `top_n`, `raw_scores`, `aggregation`,
`max_new_tokens`, `min_new_tokens`, `temperature`, `top_p`, `top_k`,
`repeat_penalty`, `seed`, `stop`. Response parameters carry `device_ms`
and `placement`. `ServerMetadata.extensions` names this as
`turbo_parameters`.

REST:

```sh
curl -s localhost:8000/v2                          # server metadata
curl -s localhost:8000/v2/health/live; curl -s localhost:8000/v2/health/ready
curl -s localhost:8000/v2/models                   # every model's metadata (a listing extension)
curl -s localhost:8000/v2/models/minilm            # model metadata; /versions/1 is the same model
curl -s localhost:8000/v2/models/minilm/ready
curl -s -X POST localhost:8000/v2/models/minilm/infer -H 'content-type: application/json' -d '{
  "id": "r1", "parameters": {"prompt_role": "query"},
  "inputs": [{"name": "text", "datatype": "BYTES", "shape": [2], "data": ["hello world", "goodbye"]}]}'
```

gRPC, with `grpcurl` (the server publishes reflection, so no proto file
is needed):

```sh
grpcurl -plaintext localhost:8001 list
grpcurl -plaintext -d '{"model_name":"minilm","inputs":[{"name":"text","datatype":"BYTES","shape":[1],"contents":{"bytes_contents":["aGVsbG8="]}}]}' \
    localhost:8001 inference.GRPCInferenceService/ModelInfer
```

## Extensions

`ServerMetadata.extensions` names three, all on the OIP messages:

- `turbo_parameters`: the request and response parameters above.
- `turbo_stream_infer`: `InferstreamExtension/ModelStreamInfer` takes a
  `ModelInferRequest` for a generative model and streams one
  `ModelStreamInferResponse` per chunk, each holding a `ModelInferResponse`
  with the new text in `text`; the last carries `finish_reason`,
  `prompt_tokens` and `generated_tokens` in its parameters, and
  `error_message` is set when the stream ended in an error. A
  non-generative model answers with one response holding the result.
- `turbo_model_repository`: index, load and unload at run time, over gRPC
  (`RepositoryIndex`, `RepositoryModelLoad`, `RepositoryModelUnload`) and
  REST:

```sh
curl -s -X POST localhost:8000/v2/repository/index
curl -s -X POST localhost:8000/v2/repository/models/sst2/load -H 'content-type: application/json' \
    -d '{"bundle": "/opt/bundles/sst2-onnx", "provider": "cuda", "ordinal": 0, "buckets": ["1x128", "8x128"]}'
curl -s -X POST localhost:8000/v2/repository/models/sst2/unload
```

The load body takes the `--model` keys (`bundle` and `provider` required;
`ordinal`, `buckets`, `sessions`, `generations`); the model is ready when
the call returns, a name already served is refused (400), and a bundle
that fails to load answers with its Turbo status (a missing directory is
`TURBO_E_BUNDLE_NOT_FOUND`, 404). An unload removes the name at once;
requests already running finish on the sessions they hold, and the model
is released when the last of them completes. Loads run on a blocking
thread, so serving continues during them.

## Container and KServe

`scripts/inferstream-image.sh` builds `turbo-inferstream:cpu` from
`packaging/inferstream/Dockerfile` (the binary, libturbo and the ggml
provider with llama.cpp's CPU backend on `debian:bookworm-slim`, about
200 MB); `scripts/inferstream-image.sh cuda` builds the same on the
NVIDIA CUDA bases with the ggml CUDA backend. A bundle directory is
mounted and named with `--model`:

```sh
docker run --rm -p 8000:8000 -p 8001:8001 -v ~/opt/bundles/minilm-gguf:/models/minilm:ro \
    turbo-inferstream:cpu --model name=minilm,bundle=/models/minilm,provider=ggml,buckets=1x256
```

`packaging/kserve/` holds a `ClusterServingRuntime` for the image and an
`InferenceService` that points it at a bundle; see the README there.

## RAG demo

`demo/rag/` answers a question over a small corpus in three calls, embed,
rerank and a streamed generation with citations, once through the OpenAI
Python SDK (`rag.py`) and once through KServe's own OIP clients
(`rag_oip.py`, REST or gRPC), against the same server.

`raw_input_contents` is accepted (little-endian rows; BYTES as 4-byte
length-prefixed items). The server has one model version, `1`; another
version is `NOT_FOUND`. Errors are the OIP error object plus `status` (the
`TURBO_E_*` name) and `field`: 400 for a bad argument or enum, 422 for a
capacity limit, 501 for an option the device does not honor, 503 when
busy, 404 for an unknown model; on gRPC, `InvalidArgument`,
`ResourceExhausted`, `Unimplemented`, `Unavailable`, `NotFound`.

## OpenAI-shaped routes

- `POST /v1/embeddings` `{model, input: string | [string], encoding_format?: float | base64, dimensions?, truncate?, prompt_role?, normalize?}` returns the `list` of `embedding` objects with `usage.prompt_tokens` (from the bundle's tokenizer) and a `turbo` object (device, placement, `device_ms`, `dim`).
- `POST /v1/rerank` `{model, query, documents, top_n?, return_documents?, raw_scores?, truncate?}` returns `results` best first with `index` and `relevance_score`, and every score in `turbo.scores`.
- `POST /v1/classify` `{model, inputs: string | [string], raw_scores?, truncate?, aggregation?}` returns, per input, labels with scores best first (a classifier) or entity spans with `entity_group`, `score`, `word`, `start`, `end` (a token classifier).
- `POST /v1/chat/completions` `{model, messages, stream?, max_tokens | max_completion_tokens, temperature, top_p, top_k, seed, stop, n}` returns a `chat.completion`, or with `stream: true` an SSE stream of `chat.completion.chunk` objects ending with `data: [DONE]`; a provider error mid-stream is an `event: error` with the Turbo status. `n` other than 1 is rejected naming the field.
- `GET /v1/models` lists the served models; `GET /info` reports the text-embeddings-inference fields for the first embedding model plus every model with its served limits.

## Tests

```sh
cargo test -p turbo-inferstream
```

The suite starts the server in process: the axum router is driven through
`tower`, and `inference.GRPCInferenceService` through the client generated
from the same proto, over an ephemeral port. Every model is a mock bundle
from `testdata/bundles/mock` on mock ordinal 1, so the suite needs no
hardware and no provider library. `tests/oip_rest.rs` covers the REST
binding, `tests/oip_grpc.rs` the gRPC binding, `tests/ext_grpc.rs` the
extension service and reflection, `tests/repository_rest.rs` the
repository routes, `tests/openai.rs` the OpenAI-shaped routes and
`/info`, and `tests/engine.rs` the bucket chunking, the session pool,
the generation bound and the model loads that must fail. `--model`
parsing is unit-tested in `src/config.rs`.

## Verified

On `krick` (2026-09-22): the six mock bundles over both bindings
(metadata, infer for each kind, the error paths); MiniLM and the ms-marco
reranker through the `cuda` provider on the RTX 4080 SUPER and
Qwen2.5-0.5B through `ggml` (CUDA backend) for embeddings, an over-long
input rejected with the limit and accepted with `truncate`, rerank, chat
and streamed chat, and `ModelInfer` over gRPC. Later the same day: the
reflection listing, a repository load, `ModelReady`, unload and
`NotFound`, and `ModelStreamInfer` with `grpcurl` on the mock bundles;
the RAG demo through the OpenAI SDK and through KServe's REST and gRPC
clients with MiniLM GGUF and Qwen2.5-0.5B on `ggml` (CUDA) and the
reranker on `cuda`; and the `turbo-inferstream:cpu` image serving MiniLM
GGUF on the CPU with `/v1/embeddings` and the reflection listing.
