//! Generates the gRPC clients from the server's vendored
//! ../server/proto/open_inference_grpc.proto and turbo_analysis.proto (for
//! its Capabilities call) with a protobuf compiler written in Rust, so
//! building needs no protoc.

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let protos = ["../server/proto/open_inference_grpc.proto", "../server/proto/turbo_analysis.proto"];
    for proto in protos {
        println!("cargo:rerun-if-changed={proto}");
    }
    let fds = protox::compile(protos, ["../server/proto"])?;
    tonic_prost_build::configure().build_client(true).build_server(false).compile_fds(fds)?;
    Ok(())
}
