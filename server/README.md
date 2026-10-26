# turbo-kserve

A gRPC server for the Open Inference Protocol (KServe v2,
`inference.GRPCInferenceService`) that serves turboembed bundles through
the library's C interface, `turbo.result.AnalysisService`, the shared
analysis result layer beside it, and `turbo.vocabulary.TurboVocabularyService`,
the dictionary and corpus vocabulary store.

- [docs/grpc.md](../docs/grpc.md): building, starting and calling it.
- [docs/kserve.md](../docs/kserve.md): what every RPC, tensor, parameter
  and status of the Open Inference Protocol means, against the header.
- [docs/grpc-analysis.md](../docs/grpc-analysis.md): the analysis result
  messages, the capability call, and load and unload.
- [docs/grpc-vocabulary.md](../docs/grpc-vocabulary.md): dictionary
  uploads, corpus vocabulary learning and the store that keeps both.
- [docs/grpc-sink.md](../docs/grpc-sink.md): forwarding analyzed
  documents to a downstream sink, and serving one.
- [docs/grpc-catalog.md](../docs/grpc-catalog.md): the model catalog,
  licence acceptances and verified installs from the zoo.
- [docs/grpc-training.md](../docs/grpc-training.md): static models
  distilled from a served encoder, sealed, and served at once.

```
cargo run --release -p turbo-kserve -- \
    --listen 0.0.0.0:8081 \
    --model bundle=testdata/tiny-bert-bundle,device=0,sessions=2
```

`proto/open_inference_grpc.proto` is vendored from the Open Inference
Protocol specification (Apache-2.0); its first lines say where from.
`proto/turbo_analysis.proto`, `proto/turbo_vocabulary.proto`,
`proto/turbo_training.proto` and `proto/turbo_sink.proto` are this
crate's own. The build compiles them
with a Rust protobuf compiler, so no `protoc` is needed.
