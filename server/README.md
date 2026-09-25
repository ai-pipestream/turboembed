# turbo-kserve

A gRPC server for the Open Inference Protocol (KServe v2,
`inference.GRPCInferenceService`) that serves turboembed bundles through
the library's C interface. What every RPC, tensor, parameter and status
means is in [docs/kserve.md](../docs/kserve.md); this file only says how
to run it.

## Run

```
cargo run --release -p turbo-kserve -- \
    --listen 0.0.0.0:8081 \
    --model bundle=testdata/tiny-bert-bundle,device=0,sessions=2
```

`--features cuda` links the CUDA backend (docs/cuda.md).

The server answers gRPC as soon as it listens: `ServerLive` is true while
the bundles load, and `ServerReady` turns true once every model and all its
sessions are made. A model that fails to load stops the server with the
failing call, its status, the field it names and the library's message.

## Configuration

`--listen ADDR:PORT` is required. Each served model is one `--model`,
repeated for more, with comma-separated settings:

| Setting | Meaning | Absent |
|---|---|---|
| `bundle` | The bundle directory. Its last path component is the model name. | Required. |
| `device` | A runtime device index, or `select` for the device `turbo_runtime_select` picks for embedding (never a CPU). | Required. |
| `sessions` | How many sessions the model has; one request holds one, and a request that finds none idle is refused with `UNAVAILABLE` at once. | Required, at least 1. |
| `precision` | `PRECISION_MODEL`, `PRECISION_FASTEST` or `PRECISION_EXACT`. | `PRECISION_MODEL`. |
| `max_batch` | The sessions' batch limit. | 0, the model's. |
| `max_seq` | The sessions' sequence limit. | 0, the model's. |

A bundle path may not contain a comma.

## Try it

With [grpcurl](https://github.com/fullstorydev/grpcurl) and the vendored
proto:

```
grpcurl -plaintext -import-path server/proto -proto open_inference_grpc.proto \
    -d '{"model_name": "tiny-bert-bundle",
         "inputs": [{"name": "texts", "datatype": "BYTES", "shape": [1],
                     "contents": {"bytes_contents": ["aGVsbG8="]}}]}' \
    localhost:8081 inference.GRPCInferenceService/ModelInfer
```

The vectors come back as little-endian FP32 in `raw_output_contents[0]`.

## Proto

`proto/open_inference_grpc.proto` is vendored from the Open Inference
Protocol specification (Apache-2.0); its first lines say where from. The
build compiles it with a Rust protobuf compiler, so no `protoc` is needed.
