//! Generates the gRPC service and messages from the vendored
//! proto/open_inference_grpc.proto with a protobuf compiler written in
//! Rust, so building needs no protoc.

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let proto = "proto/open_inference_grpc.proto";
    println!("cargo:rerun-if-changed={proto}");
    let fds = protox::compile([proto], ["proto"])?;
    tonic_prost_build::configure()
        .build_client(true)
        .build_server(true)
        // The vectors go to the encoder from the library's memory in place.
        .bytes(".inference.ModelInferResponse.raw_output_contents")
        .compile_fds(fds)?;
    Ok(())
}
