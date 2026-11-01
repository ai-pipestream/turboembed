#[path = "encrypted/common.rs"]
mod common;
use common::*;
use std::{
    fs,
    io::{Read, Write},
    sync::Arc,
};
use turbo_eval::artifact::encrypted::*;
use turbo_eval::artifact::{EvaluationArtifactStore, TemporaryFileStore};

#[test]
#[ignore = "subprocess helper for crash recovery"]
fn crash_writer_child() {
    let root = std::path::PathBuf::from(std::env::var_os("ARTIFACT_CRASH_ROOT").expect("subprocess root"));
    let policy = Arc::new(Policy::default());
    let context = SecurityContext {
        principal: PrincipalId([1; 16]),
        encoding: EncodingId([3; 16]),
        keys: Arc::new(Keys::default()),
        authorization: policy.clone(),
        retention: policy.clone(),
        audit: policy,
    };
    let store =
        EncryptedStore::new(TemporaryFileStore::new(root.join("objects")), context, root.join("scratch"), 1000000);
    let mut writer = store.begin(&key(1)).unwrap();
    writer.write_all(&vec![42; 100000]).unwrap();
    writer.flush().unwrap();
    // Immediate process exit deliberately skips Rust destructors, as a crash
    // would. The parent owns this test root and inspects the abandoned stage.
    std::process::exit(0);
}

#[test]
fn abandoned_ciphertext_cannot_be_read_and_does_not_block_a_new_writer() {
    let fixture = Fixture::new();
    let status = std::process::Command::new(std::env::current_exe().unwrap())
        .args(["--ignored", "--exact", "crash_writer_child"])
        .env("ARTIFACT_CRASH_ROOT", &fixture.root)
        .output()
        .unwrap();
    assert!(status.status.success());
    let store = fixture.store(1);
    assert!(store.list().unwrap().is_empty());
    assert!(store.open(&key(1)).is_err());
    let abandoned = fs::read_dir(fixture.root.join("objects")).unwrap().next().unwrap().unwrap().path();
    let old = fs::read(&abandoned).unwrap();
    assert!(old.starts_with(b"TEAEAD01"));
    assert!(!old.windows(64).any(|w| w == [42; 64]));
    let mut writer = store.begin(&key(1)).unwrap();
    writer.write_all(b"new complete artifact").unwrap();
    writer.finalize().unwrap();
    assert_eq!(fs::read(&abandoned).unwrap(), old);
    let mut read = Vec::new();
    store.open(&key(1)).unwrap().read_to_end(&mut read).unwrap();
    assert_eq!(read, b"new complete artifact");
    assert!(fixture.physical(1, &key(1)).exists());
    // Recovery may remove only the proven abandoned name. It must not sweep
    // unknown stages or committed artifacts. The parent observed child exit.
    fs::remove_file(abandoned).unwrap();
    assert_eq!(store.list().unwrap(), vec![key(1)]);
    assert_eq!(fs::read_dir(fixture.root.join("objects")).unwrap().count(), 1);
}
