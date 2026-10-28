//! Generates the gRPC services and messages from the vendored
//! proto/open_inference_grpc.proto, proto/health.proto and the crate's own
//! proto/turbo_typed_annotations.proto, proto/turbo_analysis.proto,
//! proto/turbo_search.proto, proto/turbo_vocabulary.proto, proto/turbo_training.proto and
//! proto/turbo_sink.proto with a protobuf compiler written in Rust, so
//! building needs no protoc.

const PROTOS: [&str; 8] = [
    "proto/open_inference_grpc.proto",
    "proto/turbo_typed_annotations.proto",
    "proto/turbo_analysis.proto",
    "proto/turbo_search.proto",
    "proto/turbo_vocabulary.proto",
    "proto/turbo_training.proto",
    "proto/turbo_sink.proto",
    "proto/health.proto",
];

fn main() -> Result<(), Box<dyn std::error::Error>> {
    for proto in PROTOS {
        println!("cargo:rerun-if-changed={proto}");
    }
    let fds = protox::compile(PROTOS, ["proto"])?;
    // Every gRPC path the protos define, so metrics label only those and a
    // client's junk paths cannot grow the label set (#1300).
    let mut methods: Vec<String> = Vec::new();
    for file in &fds.file {
        let package = file.package.as_deref().unwrap_or_default();
        for service in &file.service {
            let service_name = service.name.as_deref().unwrap_or_default();
            let full = if package.is_empty() { service_name.to_owned() } else { format!("{package}.{service_name}") };
            for method in &service.method {
                methods.push(format!("/{full}/{}", method.name.as_deref().unwrap_or_default()));
            }
        }
    }
    methods.sort();
    let list: String = methods.iter().map(|m| format!("    {m:?},\n")).collect();
    let out = std::path::PathBuf::from(std::env::var("OUT_DIR")?).join("grpc_methods.rs");
    std::fs::write(out, format!("pub(crate) const GRPC_METHODS: [&str; {}] = [\n{list}];\n", methods.len()))?;
    tonic_prost_build::configure()
        .build_client(true)
        .build_server(true)
        // The vectors go to the encoder from the library's memory in place,
        // a token-vector block, a chunk-levels vector block and a
        // document graph each cross in one piece, and a stream's documents
        // are slices of their batch's one copy.
        .bytes(".inference.ModelInferResponse.raw_output_contents")
        .bytes(".turbo.result.v1.TokenVectorsResult.data")
        .bytes(".turbo.result.v1.AnalyzeVectors.data")
        .bytes(".turbo.result.v1.ChunkLevelsResult.data")
        .bytes(".turbo.result.v1.DocumentGraphRequest.graph")
        .bytes(".turbo.result.v1.DocumentGraphResponse.graph")
        .compile_fds(fds)?;
    Ok(())
}
