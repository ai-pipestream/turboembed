//! Generates the gRPC client from the server's vendored
//! ../server/proto/open_inference_grpc.proto with a protobuf compiler
//! written in Rust, so building needs no protoc.

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let proto = "../server/proto/open_inference_grpc.proto";
    println!("cargo:rerun-if-changed={proto}");
    let fds = protox::compile([proto], ["../server/proto"])?;
    tonic_prost_build::configure().build_client(true).build_server(false).compile_fds(fds)?;
    Ok(())
}
