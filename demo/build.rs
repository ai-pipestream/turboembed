//! Generates the gRPC clients from the server's vendored
//! ../server/proto/open_inference_grpc.proto and turbo_analysis.proto with
//! a protobuf compiler written in Rust, so building needs no protoc.
//!
//! The analysis messages also derive serde, so the page's JSON for an
//! AnalysisService call is the proto message itself (docs/demo.md): field
//! names as in the .proto, a oneof as its field's name, enums as their
//! numbers, a document's text as a string and vector data as numbers
//! (src/wire.rs).

const SERDE: &str = "#[derive(serde::Serialize, serde::Deserialize)]";

/// The bytes fields that are not raw bytes to the page.
const FIELDS: [(&str, &str); 4] = [
    (".turbo.result.AnalyzeDocument.text", "utf8"),
    (".turbo.result.AnalyzeVectors.data", "f32le"),
    (".turbo.result.TokenVectorsResult.data", "f32le"),
    (".turbo.result.ChunkLevelsResult.data", "f32le"),
];

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let protos = ["../server/proto/open_inference_grpc.proto", "../server/proto/turbo_analysis.proto"];
    for proto in protos {
        println!("cargo:rerun-if-changed={proto}");
    }
    let fds = protox::compile(protos, ["../server/proto"])?;
    let mut build = tonic_prost_build::configure()
        .build_client(true)
        .build_server(false)
        .message_attribute(".turbo.result", SERDE)
        .message_attribute(".turbo.result", "#[serde(default)]")
        .enum_attribute(".turbo.result", SERDE)
        .enum_attribute(".turbo.result", "#[serde(rename_all = \"snake_case\")]");
    for (field, with) in FIELDS {
        build = build.field_attribute(field, format!("#[serde(with = \"crate::wire::{with}\")]"));
    }
    build.compile_fds(fds)?;
    Ok(())
}
