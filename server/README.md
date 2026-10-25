# turbo-kserve

A gRPC server for the Open Inference Protocol (KServe v2,
`inference.GRPCInferenceService`) that serves turboembed bundles through
the library's C interface, and `turbo.result.AnalysisService`, the shared
analysis result layer beside it.

- [docs/grpc.md](../docs/grpc.md): building, starting and calling it.
- [docs/kserve.md](../docs/kserve.md): what every RPC, tensor, parameter
  and status of the Open Inference Protocol means, against the header.
- [docs/grpc-analysis.md](../docs/grpc-analysis.md): the analysis result
  messages, the capability call, and load and unload.
- [docs/grpc-catalog.md](../docs/grpc-catalog.md): the model catalog,
  licence acceptances and verified installs from the zoo.

```
cargo run --release -p turbo-kserve -- \
    --listen 0.0.0.0:8081 \
    --model bundle=testdata/tiny-bert-bundle,device=0,sessions=2
```

`proto/open_inference_grpc.proto` is vendored from the Open Inference
Protocol specification (Apache-2.0); its first lines say where from.
`proto/turbo_analysis.proto` and `proto/turbo_training.proto` are this
crate's own. The build compiles all three with a Rust protobuf compiler,
so no `protoc` is needed.
