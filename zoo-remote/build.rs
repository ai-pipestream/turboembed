//! Generates the zoo server's client from the server's own
//! proto/turbo_training.proto with a protobuf compiler written in Rust, so
//! building needs no protoc.

const PROTO: &str = "../server/proto/turbo_training.proto";

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("cargo:rerun-if-changed={PROTO}");
    let fds = protox::compile([PROTO], ["../server/proto"])?;
    tonic_prost_build::configure().build_client(true).build_server(false).compile_fds(fds)?;
    Ok(())
}
