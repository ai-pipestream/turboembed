// The gRPC half of the Open Inference Protocol v2 is generated from the
// protocol's own proto file (proto/open_inference_grpc.proto, from
// kserve/open-inference-protocol) at build time, and Inferstream's
// extension service (proto/turbo_inferstream.proto) after it, referring
// to the first module for the messages it imports. The extension's file
// descriptor set (which includes the import) feeds the reflection
// service. `protoc` must be on the path or named by PROTOC.
fn main() {
    println!("cargo:rerun-if-changed=proto/open_inference_grpc.proto");
    println!("cargo:rerun-if-changed=proto/turbo_inferstream.proto");
    let out = std::path::PathBuf::from(std::env::var("OUT_DIR").expect("OUT_DIR"));
    tonic_prost_build::configure()
        .build_client(true)
        .build_server(true)
        .compile_protos(&["proto/open_inference_grpc.proto"], &["proto"])
        .unwrap_or_else(|e| panic!("open_inference_grpc.proto did not compile (is protoc installed?): {e}"));
    // Its own directory: the pass also emits a stub for the imported
    // package, which must not replace the first pass's module.
    let ext = out.join("ext");
    std::fs::create_dir_all(&ext).expect("create the extension output directory");
    tonic_prost_build::configure()
        .build_client(true)
        .build_server(true)
        .out_dir(&ext)
        .file_descriptor_set_path(out.join("inferstream_descriptor.bin"))
        .extern_path(".inference", "crate::grpc::inference")
        .compile_protos(&["proto/turbo_inferstream.proto"], &["proto"])
        .unwrap_or_else(|e| panic!("turbo_inferstream.proto did not compile: {e}"));
}
