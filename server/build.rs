//! Generates the gRPC services and messages from the vendored
//! proto/open_inference_grpc.proto and the crate's own
//! proto/turbo_analysis.proto, proto/turbo_training.proto and
//! proto/turbo_sink.proto with a protobuf compiler written in Rust, so
//! building needs no protoc.

const PROTOS: [&str; 4] = [
    "proto/open_inference_grpc.proto",
    "proto/turbo_analysis.proto",
    "proto/turbo_training.proto",
    "proto/turbo_sink.proto",
];

fn main() -> Result<(), Box<dyn std::error::Error>> {
    for proto in PROTOS {
        println!("cargo:rerun-if-changed={proto}");
    }
    let fds = protox::compile(PROTOS, ["proto"])?;
    tonic_prost_build::configure()
        .build_client(true)
        .build_server(true)
        // The vectors go to the encoder from the library's memory in place,
        // a token-vector block, a chunk-levels vector block and a
        // document graph each cross in one piece, and a stream's documents
        // are slices of their batch's one copy.
        .bytes(".inference.ModelInferResponse.raw_output_contents")
        .bytes(".turbo.result.TokenVectorsResult.data")
        .bytes(".turbo.result.AnalyzeVectors.data")
        .bytes(".turbo.result.ChunkLevelsResult.data")
        .bytes(".turbo.result.DocumentGraphRequest.graph")
        .bytes(".turbo.result.DocumentGraphResponse.graph")
        .compile_fds(fds)?;
    Ok(())
}
