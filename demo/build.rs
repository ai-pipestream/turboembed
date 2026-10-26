//! Generates the gRPC clients from the server's vendored .proto files
//! (../server/proto) with a protobuf compiler written in Rust, so building
//! needs no protoc.
//!
//! The messages of the analysis, search, vocabulary and training packages
//! also derive serde, so the page's JSON for a call is the proto message
//! itself (docs/demo.md): field names as in the .proto, a oneof as its
//! field's name, enums as their numbers. A document's text is a string,
//! vector data numbers, and the document graphs and file contents base64
//! strings, as the protobuf JSON mapping writes bytes (src/wire.rs).

const SERDE: &str = "#[derive(serde::Serialize, serde::Deserialize)]";

/// The packages whose messages the page reads and writes.
const PACKAGES: [&str; 4] = [".turbo.result.v1", ".turbo.search.v1", ".turbo.vocabulary.v1", ".turbo.training.v1"];

/// The bytes fields that are not arrays of byte values to the page. A
/// field of a oneof is named by its message and its own name.
const FIELDS: [(&str, &str); 17] = [
    (".turbo.result.v1.AnalyzeDocument.text", "utf8"),
    (".turbo.result.v1.AnalyzeVectors.data", "f32le"),
    (".turbo.result.v1.TokenVectorsResult.data", "f32le"),
    (".turbo.result.v1.ChunkLevelsResult.data", "f32le"),
    (".turbo.result.v1.DocumentGraphRequest.graph", "base64"),
    (".turbo.result.v1.DocumentGraphResponse.graph", "base64"),
    (".turbo.result.v1.AnalyzePipelineRequest.source_bytes", "base64"),
    (".turbo.result.v1.AnalyzePipelineRequest.initial_graph", "base64"),
    (".turbo.result.v1.AnalysisLayerBatch.graph", "base64"),
    (".turbo.result.v1.DocumentWindowRequest.graph", "base64"),
    (".turbo.result.v1.ProjectChunksRequest.graph", "base64"),
    (".turbo.result.v1.FormatDocumentRequest.graph", "base64"),
    (".turbo.result.v1.FormatDocumentResponse.content", "base64"),
    (".turbo.result.v1.InspectDocumentGraphRequest.graph", "base64"),
    (".turbo.search.v1.IndexDocumentRequest.graph", "base64"),
    (".turbo.vocabulary.v1.ImportDictionaryRequest.frame.chunk", "base64"),
    (".turbo.vocabulary.v1.VocabularyArtifactChunk.frame.data", "base64"),
];

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let protos = [
        "../server/proto/open_inference_grpc.proto",
        "../server/proto/turbo_analysis.proto",
        "../server/proto/turbo_typed_annotations.proto",
        "../server/proto/turbo_search.proto",
        "../server/proto/turbo_vocabulary.proto",
        "../server/proto/turbo_training.proto",
    ];
    for proto in protos {
        println!("cargo:rerun-if-changed={proto}");
    }
    let fds = protox::compile(protos, ["../server/proto"])?;
    let mut build = tonic_prost_build::configure().build_client(true).build_server(false);
    for package in PACKAGES {
        build = build
            .message_attribute(package, SERDE)
            .message_attribute(package, "#[serde(default)]")
            .enum_attribute(package, SERDE)
            .enum_attribute(package, "#[serde(rename_all = \"snake_case\")]");
    }
    for (field, with) in FIELDS {
        build = build.field_attribute(field, format!("#[serde(with = \"crate::wire::{with}\")]"));
    }
    build.compile_fds(fds)?;
    Ok(())
}
