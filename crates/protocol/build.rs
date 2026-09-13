use std::path::PathBuf;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Source of truth is the repo-root proto/ directory (shared with the
    // Swift Apple server). Do not compile a per-crate fork.
    let proto_root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../proto");
    println!("cargo:rerun-if-changed={}", proto_root.display());
    tonic_build::configure()
        .build_server(true)
        .build_client(true)
        // PACKED_BYTES / OIP raw tensors: Bytes so a freelist slab can
        // outlive encode and return on Drop (SOLIDIFY 6).
        .bytes([
            ".inferstream.v1.EmbedResponse.packed_embeddings",
            ".inferstream.v1.EmbedChunk.packed_row",
            ".inference.ModelInferResponse.raw_output_contents",
        ])
        .compile_protos(
            &[
                proto_root.join("open_inference_grpc.proto"),
                proto_root.join("inferstream_extension.proto"),
            ],
            &[&proto_root],
        )?;
    Ok(())
}
