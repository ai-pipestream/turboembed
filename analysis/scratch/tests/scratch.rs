//! `Scratch` clears its directory at the start and removes it when
//! dropped, also when the test that holds it panics.

use std::fs;

use turbo_scratch::Scratch;

#[test]
fn the_directory_is_cleared_at_the_start_and_gone_at_the_end() {
    let at = std::env::temp_dir().join(format!("turbo-scratch-test-{}-plain", std::process::id()));
    fs::create_dir_all(at.join("stale")).unwrap();
    let dir = Scratch::at(&at);
    assert!(!at.exists(), "what was there before is cleared, and the directory is not made");
    fs::create_dir_all(dir.join("weights")).unwrap();
    fs::write(dir.join("weights/model.bin"), b"bytes").unwrap();
    assert_eq!(dir.path(), at);
    drop(dir);
    assert!(!at.exists());
}

#[test]
fn a_file_written_at_the_path_is_removed_as_a_directory_is() {
    let at = std::env::temp_dir().join(format!("turbo-scratch-test-{}-file.json", std::process::id()));
    let file = Scratch::at(&at);
    fs::write(&file, b"{}").unwrap();
    drop(file);
    assert!(!at.exists());
}

#[test]
fn a_panic_removes_the_directory_too() {
    let at = std::env::temp_dir().join(format!("turbo-scratch-test-{}-panic", std::process::id()));
    let staged = at.clone();
    let ended = std::panic::catch_unwind(move || {
        let dir = Scratch::at(&staged);
        fs::create_dir_all(&dir).unwrap();
        fs::write(dir.join("file"), b"bytes").unwrap();
        panic!("the test failed");
    });
    assert!(ended.is_err());
    assert!(!at.exists());
}

/// A directory kept until the exit is there for the whole process and
/// gone after it: this test runs itself again as that process.
#[test]
fn a_directory_kept_until_exit_is_removed_as_the_process_ends() {
    const CHILD: &str = "TURBO_SCRATCH_TEST_KEEPS";
    if let Some(at) = std::env::var_os(CHILD) {
        let dir = Scratch::at(std::path::PathBuf::from(at)).until_exit();
        fs::create_dir_all(dir.join("weights")).unwrap();
        fs::write(dir.join("weights/model.bin"), b"bytes").unwrap();
        println!("kept {}", dir.is_dir());
        return;
    }
    let at = std::env::temp_dir().join(format!("turbo-scratch-test-{}-exit", std::process::id()));
    let out = std::process::Command::new(std::env::current_exe().unwrap())
        .args(["--exact", "a_directory_kept_until_exit_is_removed_as_the_process_ends", "--nocapture"])
        .env(CHILD, &at)
        .output()
        .unwrap();
    assert!(out.status.success(), "{}", String::from_utf8_lossy(&out.stderr));
    assert!(String::from_utf8_lossy(&out.stdout).contains("kept true"), "the directory was there while it ran");
    assert!(!at.exists(), "and is gone once the process has ended");
}
