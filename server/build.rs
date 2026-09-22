// The gRPC half of the Open Inference Protocol v2 is generated from the
// protocol's own proto file (proto/open_inference_grpc.proto, from
// kserve/open-inference-protocol) at build time. `protoc` must be on the
// path or named by PROTOC.
fn main() {
    println!("cargo:rerun-if-changed=proto/open_inference_grpc.proto");
    tonic_prost_build::configure()
        .build_client(true)
        .build_server(true)
        .compile_protos(&["proto/open_inference_grpc.proto"], &["proto"])
        .unwrap_or_else(|e| panic!("open_inference_grpc.proto did not compile (is protoc installed?): {e}"));
}
