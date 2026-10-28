use std::{
    fs,
    io::{Read, Write},
    sync::{Arc, Barrier},
};
use turbo_eval::artifact::{ArtifactKey, EvaluationArtifactStore, StorageError, TemporaryFileStore};

fn store(name: &str) -> (std::path::PathBuf, TemporaryFileStore) {
    let dir = std::env::temp_dir().join(format!("eval-storage-{}-{name}", std::process::id()));
    (dir.clone(), TemporaryFileStore::new(dir))
}
fn key() -> ArtifactKey {
    ArtifactKey::new("sample.binary").unwrap()
}

#[test]
fn concurrent_publication_has_one_winner_and_preserves_its_bytes() {
    let (dir, store) = store("race");
    let barrier = Arc::new(Barrier::new(2));
    let threads: Vec<_> = [b"first".as_slice(), b"second".as_slice()]
        .into_iter()
        .map(|bytes| {
            let store = store.clone();
            let barrier = barrier.clone();
            std::thread::spawn(move || {
                let mut pending = store.begin(&key()).unwrap();
                pending.write_all(bytes).unwrap();
                barrier.wait();
                (bytes, pending.finalize())
            })
        })
        .collect();
    let results: Vec<_> = threads.into_iter().map(|t| t.join().unwrap()).collect();
    assert_eq!(results.iter().filter(|(_, result)| result.is_ok()).count(), 1);
    let winner = results.iter().find(|(_, result)| result.is_ok()).unwrap().0;
    assert!(results.iter().any(|(_, r)| matches!(r, Err(StorageError::AlreadyExists(_)))));
    let mut bytes = Vec::new();
    store.open(&key()).unwrap().read_to_end(&mut bytes).unwrap();
    assert_eq!(bytes, winner);
    assert_eq!(fs::read_dir(&dir).unwrap().count(), 1);
    store.delete(&key()).unwrap();
    assert!(store.list().unwrap().is_empty());
    assert!(matches!(store.open(&key()), Err(StorageError::NotFound(_))));
    fs::remove_dir(dir).unwrap();
}

#[test]
fn abort_and_drop_remove_only_their_owned_staging_files() {
    let (dir, store) = store("abort");
    let mut a = store.begin(&key()).unwrap();
    let mut b = store.begin(&key()).unwrap();
    a.write_all(b"aborted").unwrap();
    b.write_all(b"dropped").unwrap();
    fs::write(dir.join("unrelated.partial"), b"preserve").unwrap();
    a.abort().unwrap();
    drop(b);
    assert_eq!(fs::read_dir(&dir).unwrap().count(), 1);
    assert_eq!(fs::read(dir.join("unrelated.partial")).unwrap(), b"preserve");
    fs::remove_dir_all(dir).unwrap();
}

#[test]
fn keys_are_opaque_names_and_cannot_escape_the_store() {
    for name in ["", ".", "..", "../x", "x/y", "x\\y", "/tmp/x", "x.", ".hidden"] {
        assert!(matches!(ArtifactKey::new(name), Err(StorageError::InvalidKey)), "{name}");
    }
    assert!(ArtifactKey::new("eval.scifact.v2").is_ok());
}

#[cfg(unix)]
#[test]
fn existing_and_racing_symlinks_are_never_replaced_or_followed_on_publish() {
    use std::os::unix::fs::symlink;
    let (dir, store) = store("symlink");
    fs::create_dir_all(&dir).unwrap();
    for target_exists in [false, true] {
        let target = dir.join("target");
        if target_exists {
            fs::write(&target, b"original").unwrap();
        }
        symlink(&target, dir.join(key().as_str())).unwrap();
        assert!(matches!(store.begin(&key()), Err(StorageError::AlreadyExists(_))));
        fs::remove_file(dir.join(key().as_str())).unwrap();
        let mut pending = store.begin(&key()).unwrap();
        pending.write_all(b"replacement").unwrap();
        symlink(&target, dir.join(key().as_str())).unwrap();
        assert!(matches!(pending.finalize(), Err(StorageError::AlreadyExists(_))));
        assert!(fs::symlink_metadata(dir.join(key().as_str())).unwrap().file_type().is_symlink());
        if target_exists {
            assert_eq!(fs::read(&target).unwrap(), b"original");
        } else {
            assert!(!target.exists());
        }
        assert!(!store.list().unwrap().contains(&key()));
        fs::remove_file(dir.join(key().as_str())).unwrap();
    }
    fs::remove_dir_all(dir).unwrap();
}

#[cfg(unix)]
#[test]
fn local_staging_and_published_files_are_private_regular_files() {
    use std::os::unix::fs::PermissionsExt;
    let (dir, store) = store("permissions");
    let mut pending = store.begin(&key()).unwrap();
    for entry in fs::read_dir(&dir).unwrap() {
        assert_eq!(entry.unwrap().metadata().unwrap().permissions().mode() & 0o777, 0o600);
    }
    pending.write_all(b"data").unwrap();
    let receipt = pending.finalize().unwrap();
    assert!(receipt.cleanup_error.is_none());
    assert_eq!(fs::metadata(dir.join(key().as_str())).unwrap().permissions().mode() & 0o777, 0o600);
    fs::create_dir(dir.join("directory.cuts")).unwrap();
    assert_eq!(store.list().unwrap(), vec![key()]);
    fs::remove_dir_all(dir).unwrap();
}

#[test]
fn applications_choose_isolated_directories_for_the_same_logical_key() {
    let (a_dir, a) = store("application-a");
    let (b_dir, b) = store("application-b");
    for (dir, store, value) in [(&a_dir, &a, b"app-a"), (&b_dir, &b, b"app-b")] {
        let mut pending = store.begin(&key()).unwrap();
        let staging: Vec<_> = fs::read_dir(dir).unwrap().map(|e| e.unwrap().path()).collect();
        assert_eq!(staging.len(), 1);
        assert_eq!(staging[0].parent(), Some(dir.as_path()));
        assert!(staging[0].file_name().unwrap().to_str().unwrap().ends_with(".partial"));
        assert!(store.list().unwrap().is_empty());
        pending.write_all(value).unwrap();
        pending.finalize().unwrap();
        assert_eq!(store.list().unwrap(), vec![key()]);
        let mut opened = Vec::new();
        store.open(&key()).unwrap().read_to_end(&mut opened).unwrap();
        assert_eq!(opened, value);
    }
    assert_eq!(fs::read(a_dir.join(key().as_str())).unwrap(), b"app-a");
    assert_eq!(fs::read(b_dir.join(key().as_str())).unwrap(), b"app-b");
    a.delete(&key()).unwrap();
    assert!(a.list().unwrap().is_empty());
    assert!(matches!(a.open(&key()), Err(StorageError::NotFound(_))));
    assert_eq!(b.list().unwrap(), vec![key()]);
    assert_eq!(fs::read(b_dir.join(key().as_str())).unwrap(), b"app-b");
    fs::remove_dir_all(a_dir).unwrap();
    fs::remove_dir_all(b_dir).unwrap();
}
