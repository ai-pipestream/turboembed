# The gRPC server

`turbo-kserve` serves turboembed bundles over gRPC with the Open
Inference Protocol (KServe v2, `inference.GRPCInferenceService`), so a
KServe deployment, or any client of that protocol, embeds text through
the library without linking it. This page says how to run it. What every
RPC, tensor, parameter and status means, call by call against the C
interface, is in [kserve.md](kserve.md).

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
and the library's message. Ctrl-C (SIGINT) stops it.

### Flags

| Flag | Meaning | Absent |
|---|---|---|
| `--listen ADDR:PORT` | Where to answer gRPC, without TLS. | Required. |
| `--model SETTINGS` | One served model; repeat for more. | At least one is required. |
| `--max-message-bytes N` | The largest request message read; a larger one is refused with `RESOURCE_EXHAUSTED` before it is read. | 64 MiB. |

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
line replaces it.

| Variable | Flag |
|---|---|
| `TURBO_KSERVE_LISTEN` | `--listen` |
| `TURBO_KSERVE_MODELS` | `--model`, one or more values separated by `;` |
| `TURBO_KSERVE_MAX_MESSAGE_BYTES` | `--max-message-bytes` |

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

Every other method of the protocol, and `grpc.health.v1.Health`, is
`UNIMPLEMENTED`. Probe liveness with `ServerLive` and readiness with
`ServerReady`.

A `ModelInfer` carries either `texts` (BYTES, `[batch]`) or token rows
`ids` and `mask` (INT32, `[batch, seq]`), with `types` optional, and
answers one `vectors` tensor (FP32, `[batch, dim]`) in
`raw_output_contents[0]`, little-endian, row-major, as the library gives
it. Its request parameters are the embedding options by their header
names: `truncate`, `max_tokens`, `prompt_role`, `normalize`, `pooling`,
`output_dim`; one left out is what the bundle says. A batch is one run:
to embed many texts at once, send them in one request, up to the
model's `session_info.max_batch`.

Errors are gRPC statuses with the library's status name and message,
and trailing metadata `turbo-code` and `turbo-field` (kserve.md,
Errors).

## The proto

`server/proto/open_inference_grpc.proto` is the Open Inference
Protocol's own file, vendored (Apache-2.0); its first lines say where
from. Generate a client from it in any language with gRPC. The build
compiles it with a Rust protobuf compiler, so building the server needs
no `protoc`.

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
per model, every session held, a message over the limit, and the binary
itself. Nothing is mocked. On a machine with another backend, serve a
bundle on that device and run the same calls; the vectors are checked
against the reference the same way (docs/conformance.md).
