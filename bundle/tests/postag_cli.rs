//! Retired archive commands refuse input before attempting filesystem access.
use std::process::Command;

fn refuses_before_read(command: &str) {
    let root = std::env::temp_dir().join(format!("turbo-{command}-retired-{}", std::process::id()));
    assert!(!root.exists());
    let missing = root.join("missing.bin");
    let out = root.join("bundle");
    let args = if command == "postag-convert" {
        vec![command, missing.to_str().unwrap(), "Apache-2.0", out.to_str().unwrap()]
    } else {
        vec![command, out.to_str().unwrap(), missing.to_str().unwrap()]
    };
    let result = Command::new(env!("CARGO_BIN_EXE_turbo-bundle")).args(args).output().unwrap();
    assert!(!result.status.success());
    let stderr = String::from_utf8(result.stderr).unwrap();
    assert!(stderr.contains("retired"), "{stderr}");
    assert!(!root.exists());
}

#[test]
fn archive_convert_refuses_missing_paths_before_file_access() {
    refuses_before_read("postag-convert");
}

#[test]
fn archive_verify_refuses_missing_paths_before_file_access() {
    refuses_before_read("postag-verify");
}
