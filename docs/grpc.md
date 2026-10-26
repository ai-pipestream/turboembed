# The gRPC server

`turbo-kserve` serves turboembed bundles over gRPC with the Open
Inference Protocol (KServe v2, `inference.GRPCInferenceService`), so a
KServe deployment, or any client of that protocol, embeds text through
the library without linking it. On the same port it serves
`turbo.result.v1.AnalysisService`: the analysis result kinds on the wire and
a capability call that lists what every loaded bundle is
(docs/grpc-analysis.md), `turbo.search.v1.TurboSearchService`, hybrid search
and index management, `turbo.training.v1.TurboCatalogService`, the
model catalog: the zoo's artifacts, licence acceptances and verified
installs (docs/grpc-catalog.md), and `turbo.training.v1.TurboTrainingService`,
static models distilled from a served encoder and served as soon as they
are sealed (docs/grpc-training.md). It can forward what its analysis streams
complete to a downstream `turbo.sink.v1.TurboSinkService`, and serve that
service itself (docs/grpc-sink.md). This page says how to run it. What every RPC,
tensor, parameter and status of the Open Inference Protocol means, call
by call against the C interface, is in [kserve.md](kserve.md).

The server is a projection of the library: one `ModelInfer` is one write,
one run and one read on one session, the vectors go to the wire from the
library's own memory, and every error is the library's status code and
message. The backends it runs on are the ones linked into the library it
is built with, so a request answered on a CPU, an NVIDIA GPU, an Intel
GPU, an Apple GPU or a Hailo board gives the same vectors for the same
bundle at the same tier (docs/conformance.md).

## Getting it

The Linux release archive (docs/release.md) carries `bin/turbo-kserve`,
built with the cpu and cuda backends. From the tree:

```
cargo build --release -p turbo-kserve                    # cpu
cargo build --release -p turbo-kserve --features cuda    # cpu and cuda
```

`--features metal`, `levelzero`, `levelzero-onednn`, `npu` and `hailo`
link those backends; each needs what its page under `docs/` says to
build. The cpu backend is always in. The binary is
`target/release/turbo-kserve` and links the library statically: it
needs nothing beside it.

## Starting it

```
turbo-kserve --listen 0.0.0.0:8081 \
    --model bundle=/models/bge-small-en-v1.5,device=0,sessions=4
```

The server answers gRPC as soon as it listens: `ServerLive` is true
while the bundles load, and `ServerReady` turns true once every model
and all its sessions are made. It logs `listening on ADDR:PORT`, then
`ready`. A model that fails to load stops the server with a non-zero
exit, naming the model, the failing call, its status, the field it names
and the library's message.

The connection is plaintext gRPC with no authentication: put it behind
whatever your deployment uses for that, and listen on `0.0.0.0` only
where that is in place.

SIGTERM, what Kubernetes and `docker stop` send, and SIGINT (Ctrl-C)
stop it the same way: it logs `stopping on SIGTERM` or `stopping on
SIGINT`, closes the listener, lets every request in flight finish and
be answered, refuses new connections, logs `stopped` and exits 0. A
signal during load takes effect once the load returns; the library has
no way to stop a load part way.

### Flags

| Flag | Meaning | Absent |
|---|---|---|
| `--listen ADDR:PORT` | Where to answer gRPC, without TLS. | Required. |
| `--model SETTINGS` | One served model; repeat for more. | At least one model (from `--model` or `--bundle-dir`) is required unless `--admin` is set. |
| `--bundle-dir DIR` | Discover immediate child directories containing `manifest.json` and serve each on device 0 with 1 session unless overridden by an explicit `--model` of the same name. | Off. |
| `--max-message-bytes N` | The largest request message read; a larger one is refused with `RESOURCE_EXHAUSTED` before it is read. | 64 MiB. |
| `--admin` | Answer the admin calls: the analysis service's `LoadBundle` and `UnloadBundle` (docs/grpc-analysis.md), the vocabulary store's writes (docs/grpc-vocabulary.md) and the catalog's `RecordLicenceAcceptance` and `InstallModel` (docs/grpc-catalog.md), the training service's `TrainStaticModel` and `DeleteStaticModel` (docs/grpc-training.md), and allow starting with zero models. Without it they refuse with `UNIMPLEMENTED`. | Off. |
| `--vocabulary-dir PATH` | The vocabulary store's directory, made when it does not exist (docs/grpc-vocabulary.md). | No store: it lists nothing and refuses writes with `FAILED_PRECONDITION`. |
| `--catalog-dir PATH` | Serve the model catalog with this zoo home: the cache under `artifacts/`, the acceptance file `acceptances.json`, as the `turbo-zoo` command keeps them (docs/zoo.md). | No catalog; its calls refuse with `UNIMPLEMENTED`. |
| `--catalog-index FILE` | The zoo index the catalog serves. Needs `--catalog-dir`. | The built-in index. |
| `--training-dir PATH` | The directory static models are trained into, one sealed bundle per child, made when it does not exist (docs/grpc-training.md). | No training directory: every training call but `ListTeachers` refuses with `UNIMPLEMENTED`. |
| `--sink-endpoint URI` | Forward every document an `AnalyzeStream` completes to the `TurboSinkService` at this address, such as `http://sink:8081` (docs/grpc-sink.md). | No forwarding. |
| `--sink-queue-items N` | The most documents waiting for the sink; one more is dropped and counted. Needs `--sink-endpoint`. | 1024. |
| `--sink-queue-bytes N` | The most encoded bytes waiting for the sink. Needs `--sink-endpoint`. | 64 MiB. |

`SETTINGS` is comma-separated `key=value` pairs:

| Setting | Meaning | Absent |
|---|---|---|
| `bundle` | The bundle directory (docs/bundle.md). | Required. |
| `device` | A runtime device index, or `select` for the device `turbo_runtime_select` picks for embedding. `select` never picks a CPU; on a host with no other device, name the CPU's index. | Required. |
| `sessions` | How many sessions the model has. One request holds one; a request that finds none idle is refused at once with `UNAVAILABLE`, which a client may retry. | Required, at least 1. |
| `name` | The model name requests use. | The last component of the bundle path. |
| `precision` | The tier every session of this model runs at: `PRECISION_MODEL`, `PRECISION_FASTEST` or `PRECISION_EXACT` (docs/autotune.md). | `PRECISION_MODEL`. |
| `max_batch` | The sessions' batch limit. | 0, the model's. |
| `max_seq` | The sessions' sequence limit. | 0, the model's. |

A bundle path may not contain a comma. Two models may not share a name.
`--listen` and `--max-message-bytes` are refused when given twice;
`--model` repeats, once per model.
The tier and the device are the model's, not a request's: a request
that names `precision` or `device` is refused. To offer one bundle at
two tiers, serve it twice under two names:

```
turbo-kserve --listen 0.0.0.0:8081 \
    --model bundle=/models/bge-small-en-v1.5,device=0,sessions=4 \
    --model name=bge-small-en-v1.5-fastest,bundle=/models/bge-small-en-v1.5,device=0,sessions=4,precision=PRECISION_FASTEST
```

Device indices are the runtime's: every device of each linked backend,
in the order `turbo_version()` lists the backends, the cpu last. With
the cpu backend alone, the CPU is device 0; `turbo_runtime_device_info`
reports each index's kind and name.

### Environment

Each variable is read when its flag is absent; a flag on the command
line replaces it. A variable holds one value, so none can be given
twice; `TURBO_KSERVE_MODELS` lists every model in one.

| Variable | Flag |
|---|---|
| `TURBO_KSERVE_LISTEN` | `--listen` |
| `TURBO_KSERVE_MODELS` | `--model`, one or more values separated by `;` |
| `TURBO_KSERVE_BUNDLE_DIR` | `--bundle-dir` |
| `TURBO_KSERVE_MAX_MESSAGE_BYTES` | `--max-message-bytes` |
| `TURBO_KSERVE_ADMIN` | `--admin` (`1`, `true`, `yes`, `0`, `false`, `no`) |
| `TURBO_KSERVE_VOCABULARY_DIR` | `--vocabulary-dir` |
| `TURBO_KSERVE_SINK_ENDPOINT` | `--sink-endpoint` |
| `TURBO_KSERVE_SINK_QUEUE_ITEMS` | `--sink-queue-items` |
| `TURBO_KSERVE_SINK_QUEUE_BYTES` | `--sink-queue-bytes` |
| `TURBO_KSERVE_CATALOG_DIR` | `--catalog-dir` |
| `TURBO_KSERVE_CATALOG_INDEX` | `--catalog-index` |
| `TURBO_KSERVE_TRAINING_DIR` | `--training-dir` |

```
TURBO_KSERVE_LISTEN=0.0.0.0:8081 \
TURBO_KSERVE_MODELS='bundle=/models/a,device=0,sessions=4;bundle=/models/b,device=0,sessions=2' \
turbo-kserve
```

The library's own variables apply as well: `TURBO_AUTOTUNE_CACHE`, the
tuner's disk cache (docs/autotune.md), and each backend's (its page
under `docs/`).

## The service

The six RPCs served, and what each answers, are in kserve.md:

| RPC | Answer |
|---|---|
| `ServerLive` | The process is up. |
| `ServerReady` | Every configured model is loaded. |
| `ModelReady` | That model is loaded. |
| `ServerMetadata` | `turboembed`, and the library's version with its linked backends. |
| `ModelMetadata` | The inputs, the output, and the model's facts on its device as properties: the bundle's settings, the sessions' limits and tier, the device, and the capability cell with its conformance floor. |
| `ModelInfer` | The vectors, and the run's summary (bytes moved, allocations, where each stage ran) as response parameters. |

Every other method of the Open Inference Protocol is `UNIMPLEMENTED`.
The server also serves `grpc.health.v1.Health` (`Check` and `Watch`),
reporting `SERVING` when `ServerReady` is true and `NOT_SERVING` before
load and during drain.
`turbo-kserve health [--addr ADDR:PORT] [--service NAME] [--timeout-ms N]`
makes one `Check` against a running server (the address
`TURBO_KSERVE_LISTEN`, on loopback when it is `0.0.0.0` or `::`, when
`--addr` is absent) and exits 0 on `SERVING`, 1 otherwise: the release
image's health check. The image, its `/bundles` volume, its variables
and the Kubernetes probes for it are in docs/release.md, Images.

A `ModelInfer` carries either `texts` (BYTES, `[batch]`) or token rows
`ids` and `mask` (INT32, `[batch, seq]`), with `types` optional, and
answers one `vectors` tensor (FP32, `[batch, dim]`) in
`raw_output_contents[0]`, little-endian, row-major, as the library gives
it. Its request parameters are the embedding options by their header
names: `truncate`, `max_tokens`, `prompt_role`, `normalize`, `pooling`,
`output_dim`; one left out is what the bundle says. A batch is one run:
to embed many texts at once, send them in one request, up to the
model's `session_info.max_batch`.

### Texts longer than the model takes

A served model has two lengths, both in its metadata: the bundle's
`model_info.max_seq`, the length the model was evaluated at, and the
sessions' `session_info.max_seq`, what this device runs. They differ
where the artifact is compiled for a fixed shape (an NPU IR or a Hailo
HEF at 128 tokens, say, for a bundle evaluated at 256 or 512), or where
`max_seq` is configured below the model's.

With no options, a text is cut as the bundle says (`TRUNCATE_MODEL`, at
`model_info.max_seq`), and a row that then does not fit the session is
refused with `OUT_OF_RANGE` (`TURBO_E_CAPACITY`, `turbo-code` 771) and a
message naming the row and the limit:

```
TURBO_E_CAPACITY: texts[0]: 203 tokens is over the session's max_seq 128
```

The server never cuts a text on its own: that is the library's rule,
and it keeps the vectors the same for the same request on every device.
A client that wants long texts cut to what the device runs asks for it
in the request:

```
"parameters": {"max_tokens": {"int64_param": 128},
               "truncate": {"string_param": "TRUNCATE_RIGHT"}}
```

`max_tokens` is the budget per row, specials included, at most
`session_info.max_seq`; `truncate` says which end to cut (`TRUNCATE_LEFT`
keeps the end), and `TRUNCATE_NONE` refuses instead. Rows sent as
tokens are never cut: a row longer than `session_info.max_seq` is the
same `OUT_OF_RANGE`.

Errors are gRPC statuses with the library's status name and message,
and trailing metadata `turbo-code` and `turbo-field` (kserve.md,
Errors).

## The proto

`server/proto/open_inference_grpc.proto` is the Open Inference
Protocol's own file (`inference`), vendored (Apache-2.0); its first lines
say where from, and `server/proto/health.proto` is the standard gRPC health
protocol (`grpc.health.v1`). First-party TurboEmbed definitions live under
versioned `turbo.<name>.v1` packages:

- `turbo.result.v1` (`server/proto/turbo_analysis.proto`, `server/proto/turbo_typed_annotations.proto`): the analysis result layer's messages, typed document annotations, and `AnalysisService` (docs/grpc-analysis.md, docs/grpc-inspect.md).
- `turbo.search.v1` (`server/proto/turbo_search.proto`): hybrid search, index lifecycle, aliases, and `TurboSearchService`.
- `turbo.vocabulary.v1` (`server/proto/turbo_vocabulary.proto`): the vocabulary store and `TurboVocabularyService` (docs/grpc-vocabulary.md).
- `turbo.training.v1` (`server/proto/turbo_training.proto`): the model catalog (`TurboCatalogService`, docs/grpc-catalog.md) and static model training (`TurboTrainingService`, docs/grpc-training.md).
- `turbo.sink.v1` (`server/proto/turbo_sink.proto`): the document sink and `TurboSinkService` (docs/grpc-sink.md).

Generate a client from any of them in any language with gRPC, or use the
published release artifacts (`scripts/package-proto-stubs.sh`,
docs/release.md):

- **Proto archive**: `turboembed-proto-v<version>.tar.gz` (plus `.sha256`),
  containing `proto/*.proto`, `proto/buf.yaml`, `licenses/LICENSE`, and
  `licenses/NOTICE`.
- **Java Maven stub artifact**: `ai.pipestream:turboembed-grpc:<version>`
  (`turboembed-grpc-<version>.jar`, `turboembed-grpc-<version>-sources.jar`,
  `turboembed-grpc-<version>.pom`, and `.sha256`), generated by
  `clients/java/pom.xml` with pinned `com.google.protobuf:protoc:3.25.5`
  and `io.grpc:protoc-gen-grpc-java:1.68.2` (`clients/java/src/test/java/ai/pipestream/turboembed/ClientSmoke.java`).
- **Python wheel stub package**: `turboembed_proto-<version>-py3-none-any.whl`
  (plus `.sha256`), generated by `clients/python/build_wheel.py` with pinned
  `grpcio-tools==1.70.0` and `protobuf==5.29.3` (`clients/python/pyproject.toml`,
  `clients/python/smoke_test.py`).

The server build compiles `server/proto/*.proto` with a Rust protobuf
compiler, so building the server needs no external `protoc`.

### Proto compatibility policy

Every pull request that touches `server/proto/` runs `buf lint` and
`buf breaking` (`server/proto/buf.yaml`) in CI against `main`.

- **What counts as breaking**: removing or renaming a package, service, RPC,
  message, field, oneof, or enum value; changing a field's tag number, wire
  type, label (`repeated` vs scalar), or oneof membership; or changing an
  RPC's request/response message type or streaming mode (`FILE` and
  `WIRE_JSON` breaking rules). When a field or enum value is retired within a
  major package version, both its numeric tag (`reserved 6;`) and its name
  (`reserved "old_field";`) must be reserved. Adding new optional/repeated
  fields with unused tag numbers, new enum values, new messages, or new RPCs
  is non-breaking within `v1`.
- **How `v2` is introduced**: a change that cannot be expressed additively in
  `turbo.<name>.v1` is introduced as a new sibling package `turbo.<name>.v2`
  in its own `.proto` file, registered on the same `tonic` server beside the
  `v1` service rather than modifying `turbo.<name>.v1` in place.
- **How long `v1` stays served**: once `turbo.<name>.v2` is introduced,
  `turbo-kserve` continues serving `turbo.<name>.v1` on the same listener for
  at least one full minor release cycle (and at least 6 months after the first
  release containing `v2`) before `v1` may be removed in a subsequent major
  release.

## A client call

With [grpcurl](https://github.com/fullstorydev/grpcurl), against a
server started as above with `testdata/tiny-bert-bundle`:

```
grpcurl -plaintext -import-path server/proto -proto open_inference_grpc.proto \
    localhost:8081 inference.GRPCInferenceService/ServerReady

grpcurl -plaintext -import-path server/proto -proto open_inference_grpc.proto \
    -d '{"name": "tiny-bert-bundle"}' \
    localhost:8081 inference.GRPCInferenceService/ModelMetadata

grpcurl -plaintext -import-path server/proto -proto open_inference_grpc.proto \
    -d '{"model_name": "tiny-bert-bundle",
         "parameters": {"prompt_role": {"string_param": "PROMPT_QUERY"}},
         "inputs": [{"name": "texts", "datatype": "BYTES", "shape": [2],
                     "contents": {"bytes_contents": ["aGVsbG8=", "d29ybGQ="]}}]}' \
    localhost:8081 inference.GRPCInferenceService/ModelInfer
```

`bytes_contents` are the texts, base64 in JSON as grpcurl wants them.
The reply's `raw_output_contents[0]` is `batch × dim` little-endian
floats, base64 in grpcurl's JSON; its `outputs[0].shape` gives the two
extents, and its `parameters` the run's summary.

From Rust, the crate's generated client does the same:

```rust
use turbo_kserve::proto::grpc_inference_service_client::GrpcInferenceServiceClient;
use turbo_kserve::proto::model_infer_request::InferInputTensor;
use turbo_kserve::proto::{InferTensorContents, ModelInferRequest};

let mut c = GrpcInferenceServiceClient::connect("http://localhost:8081").await?;
let r = ModelInferRequest {
    model_name: "tiny-bert-bundle".into(),
    inputs: vec![InferInputTensor {
        name: "texts".into(),
        datatype: "BYTES".into(),
        shape: vec![2],
        contents: Some(InferTensorContents {
            bytes_contents: vec![b"hello".to_vec(), b"world".to_vec()],
            ..Default::default()
        }),
        ..Default::default()
    }],
    ..Default::default()
};
let resp = c.model_infer(r).await?.into_inner();
let dim = resp.outputs[0].shape[1] as usize;
let floats: Vec<f32> =
    resp.raw_output_contents[0].chunks(4).map(|b| f32::from_le_bytes(b.try_into().unwrap())).collect();
let rows: Vec<&[f32]> = floats.chunks(dim).collect();
```

## Tests

`cargo test -p turbo-kserve` starts the server in-process on the CPU
with `testdata/tiny-bert-bundle` and calls every RPC over gRPC with a
real client: readiness before and after load, the metadata, texts and
token rows in both encodings against the bundle's reference vectors,
every parameter and its refusals, the batch and sequence limits, a tier
per model, every session held, a message over the limit, the analysis
result round trips, the capability listing across load and unload, and
the named-limit refusals, plus the binary itself. Nothing is mocked. On
a machine with another backend, serve a bundle on that device and run
the same calls; the vectors are checked against the reference the same
way (docs/conformance.md).
