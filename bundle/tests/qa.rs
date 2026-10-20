#[path = "../../core/tests/qa/common.rs"]
mod common;
use std::path::Path;

#[test]
fn sealed_qa_reference_matches_real_encoder_and_span_head() {
    let f = common::Fixture::new(Path::new(env!("CARGO_MANIFEST_DIR")).parent().unwrap());
    turbo_bundle::seal::verify(&f.dir).unwrap();
}

#[test]
fn reference_logits_must_match_even_when_file_hashes_are_updated() {
    let mut f = common::Fixture::new(Path::new(env!("CARGO_MANIFEST_DIR")).parent().unwrap());
    let name = "reference/reference.safetensors";
    let mut bytes = std::fs::read(f.dir.join(name)).unwrap();
    let header_len = u64::from_le_bytes(bytes[..8].try_into().unwrap()) as usize;
    let header: serde_json::Value = serde_json::from_slice(&bytes[8..8 + header_len]).unwrap();
    let offset = header["logits"]["data_offsets"][0].as_u64().unwrap() as usize + 8 + header_len;
    bytes[offset..offset + 4].copy_from_slice(&1000f32.to_le_bytes());
    std::fs::write(f.dir.join(name), bytes).unwrap();
    f.refresh(name);
    assert!(turbo_bundle::seal::verify(&f.dir).unwrap_err().contains("logits differ"));
}
