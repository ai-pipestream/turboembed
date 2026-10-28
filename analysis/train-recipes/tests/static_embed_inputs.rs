use std::fs;
use std::process::Command;
use std::sync::atomic::{AtomicUsize, Ordering};

use serde_json::json;
use turbo_train_recipes::static_embed::{read_tensor, write_safetensors};

const NAME: &str = "embeddings.word_embeddings.weight";

fn file(shape: serde_json::Value, offsets: serde_json::Value, data: &[u8]) -> Vec<u8> {
    let mut header = serde_json::to_vec(&json!({NAME: {"dtype": "F32", "shape": shape, "data_offsets": offsets}})).unwrap();
    while !header.len().is_multiple_of(8) {
        header.push(b' ');
    }
    let mut bytes = (header.len() as u64).to_le_bytes().to_vec();
    bytes.extend(header);
    bytes.extend(data);
    bytes
}

#[test]
fn short_offsets_return_an_error() {
    assert!(read_tensor(&file(json!([1, 1]), json!([]), &[0; 4]), NAME).is_err());
}

#[test]
fn huge_header_returns_an_error() {
    assert!(read_tensor(&u64::MAX.to_le_bytes(), NAME).is_err());
}

#[test]
fn huge_offsets_return_an_error() {
    assert!(read_tensor(&file(json!([1, 1]), json!([u64::MAX, u64::MAX]), &[]), NAME).is_err());
}

#[test]
fn invalid_offset_types_return_an_error() {
    assert!(read_tensor(&file(json!([0, 1]), json!(["wrong", 0]), &[]), NAME).is_err());
}

#[test]
fn shape_must_match_data() {
    assert!(read_tensor(&file(json!([2, 2]), json!([0, 4]), &[0; 4]), NAME).is_err());
}

#[test]
fn empty_and_nonfinite_teacher_matrices_are_refused() {
    for (shape, values) in [(vec![0, 2], vec![]), (vec![1, 1], vec![f32::NAN])] {
        assert!(read_tensor(&write_safetensors(&[(NAME, &shape, values)]), NAME).is_err());
    }
}

fn scratch_preserved(extension: &str) {
    static NEXT: AtomicUsize = AtomicUsize::new(0);
    let root = std::env::temp_dir().join(format!("static-input-{}-{}", std::process::id(), NEXT.fetch_add(1, Ordering::Relaxed)));
    fs::create_dir(&root).unwrap();
    let out = root.join("model.bundle");
    let scratch = out.with_extension(extension);
    fs::create_dir(&scratch).unwrap();
    fs::write(scratch.join("manifest.json"), b"existing bundle").unwrap();
    fs::write(scratch.join("model.safetensors"), b"keep these bytes").unwrap();
    let teacher = root.join("teacher.safetensors");
    fs::write(&teacher, write_safetensors(&[(NAME, &[2, 2], vec![1., 0., 0., 1.])])).unwrap();
    let result = Command::new(env!("CARGO_BIN_EXE_turbo-train-recipes"))
        .args(["static-embed", "--teacher"]).arg(&teacher)
        .arg("--tokenizer").arg(root.join("missing-tokenizer.json"))
        .arg("--recipes").arg(root.join("missing-recipes"))
        .arg("--out").arg(&out)
        .args(["--commit", "0123456789012345678901234567890123456789", "--init-only"])
        .output().unwrap();
    assert!(!result.status.success());
    assert_eq!(fs::read(scratch.join("manifest.json")).unwrap(), b"existing bundle");
    assert_eq!(fs::read(scratch.join("model.safetensors")).unwrap(), b"keep these bytes");
    let stderr = String::from_utf8_lossy(&result.stderr);
    assert!(stderr.contains("already exists"), "{stderr}");
    fs::remove_dir_all(root).unwrap();
}

#[test]
fn cli_preserves_existing_init_bundle() {
    scratch_preserved("init");
}

#[test]
fn cli_preserves_existing_stage_files() {
    scratch_preserved("stage");
}
