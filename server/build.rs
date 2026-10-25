//! Generates the gRPC services and messages from the vendored
//! proto/open_inference_grpc.proto and the crate's own
//! proto/turbo_analysis.proto with a protobuf compiler written in
//! Rust, so building needs no protoc.

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let protos = ["proto/open_inference_grpc.proto", "proto/turbo_analysis.proto", "proto/turbo_search.proto"];
    for proto in protos {
        println!("cargo:rerun-if-changed={proto}");
    }
    let fds = protox::compile(protos, ["proto"])?;
    tonic_prost_build::configure()
        .build_client(true)
        .build_server(true)
        // The vectors go to the encoder from the library's memory in place,
        // and a token-vector block crosses in one piece.
        .bytes(".inference.ModelInferResponse.raw_output_contents")
        .bytes(".turbo.result.TokenVectorsResult.data")
        .compile_fds(fds)?;
    Ok(())
}
