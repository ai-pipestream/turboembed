//! Compiles the `turbo-kserve` client protobufs with `protox` when the `grpc`
//! feature is active, requiring no external `protoc` binary.

const PROTOS: [&str; 13] = [
    "../../server/proto/open_inference_grpc.proto",
    "../../server/proto/turbo_typed_annotations.proto",
    "../../server/proto/turbo_search_projection.proto",
    "../../server/proto/turbo_analyze_query.proto",
    "../../server/proto/turbo_rerank_highlight.proto",
    "../../server/proto/turbo_pii_policy.proto",
    "../../server/proto/turbo_analysis.proto",
    "../../server/proto/turbo_search.proto",
    "../../server/proto/turbo_vocabulary.proto",
    "../../server/proto/turbo_training.proto",
    "../../server/proto/turbo_sink.proto",
    "../../server/proto/turbo_audit.proto",
    "../../server/proto/health.proto",
];

fn main() -> Result<(), Box<dyn std::error::Error>> {
    for proto in PROTOS {
        println!("cargo:rerun-if-changed={proto}");
    }
    if std::env::var_os("CARGO_FEATURE_GRPC").is_none() {
        return Ok(());
    }
    let build_server = std::env::var_os("CARGO_FEATURE_SERVER_STUBS").is_some();
    let fds = protox::compile(PROTOS, ["../../server/proto"])?;
    tonic_prost_build::configure()
        .build_client(true)
        .build_server(build_server)
        .bytes(".inference.ModelInferResponse.raw_output_contents")
        .bytes(".turbo.result.v1.TokenVectorsResult.data")
        .bytes(".turbo.result.v1.AnalyzeVectors.data")
        .bytes(".turbo.result.v1.VectorCodesResult.data")
        .bytes(".turbo.result.v1.ChunkLevelsResult.data")
        .bytes(".turbo.result.v1.DocumentGraphRequest.graph")
        .bytes(".turbo.result.v1.DocumentGraphResponse.graph")
        .compile_fds(fds)?;
    Ok(())
}
