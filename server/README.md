# turbo-kserve

A gRPC server for the Open Inference Protocol (KServe v2,
`inference.GRPCInferenceService`) that serves turboembed bundles through
the library's C interface.

- [docs/grpc.md](../docs/grpc.md): building, starting and calling it.
- [docs/kserve.md](../docs/kserve.md): what every RPC, tensor, parameter
  and status means, against the header.

```
cargo run --release -p turbo-kserve -- \
    --listen 0.0.0.0:8081 \
    --model bundle=testdata/tiny-bert-bundle,device=0,sessions=2
```

`proto/open_inference_grpc.proto` is vendored from the Open Inference
Protocol specification (Apache-2.0); its first lines say where from. The
build compiles it with a Rust protobuf compiler, so no `protoc` is needed.
