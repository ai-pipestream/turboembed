//! Generates the gRPC services and messages from the vendored
//! proto/open_inference_grpc.proto and the crate's own
//! proto/turbo_analysis.proto with a protobuf compiler written in
//! Rust, so building needs no protoc.

fn main() -> Result<(), Box<dyn std::error::Error>> {
    for proto in ["proto/open_inference_grpc.proto", "proto/turbo_analysis.proto"] {
        println!("cargo:rerun-if-changed={proto}");
    }
    let fds = protox::compile(["proto/open_inference_grpc.proto", "proto/turbo_analysis.proto"], ["proto"])?;
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
