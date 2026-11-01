#[path = "encrypted/common.rs"]
mod common;
use common::*;
use std::{
    fs,
    io::{Read, Write},
    sync::{Arc, Barrier, atomic::Ordering},
};
use turbo_eval::artifact::encrypted::*;
use turbo_eval::artifact::{EvaluationArtifactStore, StorageError, TemporaryFileStore};

fn save(store: &impl EvaluationArtifactStore, id: u8, bytes: &[u8]) {
    let mut writer = store.begin(&key(id)).unwrap();
    writer.write_all(bytes).unwrap();
    writer.finalize().unwrap();
}
fn read(store: &impl EvaluationArtifactStore, id: u8) -> Vec<u8> {
    let mut bytes = Vec::new();
    store.open(&key(id)).unwrap().read_to_end(&mut bytes).unwrap();
    bytes
}

#[test]
fn real_files_are_encrypted_and_isolated_by_principal_and_encoding() {
    let fixture = Fixture::new();
    let a = fixture.store(1);
    let b = fixture.store(2);
    let text = b"private document text kept out of every stored file";
    save(&a, 1, text);
    assert_eq!(read(&a, 1), text);
    assert!(b.open(&key(1)).is_err());
    assert!(b.list().unwrap().is_empty());
    assert!(b.delete(&key(1)).is_err());
    save(&b, 1, b"another user's content");
    assert_eq!(a.list().unwrap(), vec![key(1)]);
    assert_eq!(b.list().unwrap(), vec![key(1)]);
    let bytes = fs::read(fixture.physical(1, &key(1))).unwrap();
    assert!(!bytes.windows(text.len()).any(|w| w == text));
    fs::copy(fixture.physical(1, &key(1)), fixture.physical(2, &key(1))).unwrap();
    assert!(b.open(&key(1)).is_err());
    let mut context = fixture.context(1);
    context.encoding = EncodingId([4; 16]);
    let wrong_encoding = EncryptedStore::new(
        TemporaryFileStore::new(fixture.root.join("objects")),
        context,
        fixture.root.join("scratch"),
        10000,
    );
    assert!(wrong_encoding.open(&key(1)).is_err());
    fs::copy(fixture.physical(1, &key(1)), fixture.physical(1, &key(2))).unwrap();
    assert!(a.open(&key(2)).is_err());
    assert_eq!(fs::read_dir(fixture.root.join("scratch")).unwrap().count(), 0);
}

#[test]
fn authorization_audit_retention_and_publish_rechecks_fail_closed() {
    let fixture = Fixture::new();
    let store = fixture.store(1);
    save(&store, 1, b"retained");
    fixture.policy.held.store(true, Ordering::SeqCst);
    assert!(matches!(store.delete(&key(1)), Err(StorageError::Security(Error::Retained))));
    assert_eq!(read(&store, 1), b"retained");
    let mut pending = store.begin(&key(2)).unwrap();
    pending.write_all(b"revoked while writing").unwrap();
    fixture.policy.denied.store(true, Ordering::SeqCst);
    assert!(pending.finalize().is_err());
    assert!(store.begin(&key(3)).is_err());
    assert!(store.open(&key(1)).is_err());
    assert!(store.list().is_err());
    assert!(store.delete(&key(1)).is_err());
    assert!(!fixture.physical(1, &key(2)).exists());
    fixture.policy.denied.store(false, Ordering::SeqCst);
    fixture.policy.audit_down.store(true, Ordering::SeqCst);
    assert!(store.open(&key(1)).is_err());
    assert!(store.delete(&key(1)).is_err());
    fixture.policy.audit_down.store(false, Ordering::SeqCst);
    fixture.policy.held.store(false, Ordering::SeqCst);
    store.delete(&key(1)).unwrap();
    assert!(store.open(&key(1)).is_err());
    let events = fixture.policy.events.lock().unwrap();
    assert!(events.contains(&(Operation::Delete, Outcome::Failed)));
    assert!(events.contains(&(Operation::Delete, Outcome::Succeeded)));
}

#[test]
fn rotation_and_revocation_apply_to_new_opens_and_publication() {
    let fixture = Fixture::new();
    let store = fixture.store(1);
    save(&store, 1, b"old key");
    fixture.keys.rotated.store(true, Ordering::SeqCst);
    save(&store, 2, b"new key");
    assert_eq!(read(&store, 1), b"old key");
    assert_eq!(read(&store, 2), b"new key");
    let mut pending = store.begin(&key(3)).unwrap();
    pending.write_all(b"must not publish").unwrap();
    fixture.keys.revoked.store(true, Ordering::SeqCst);
    assert!(matches!(store.open(&key(1)), Err(StorageError::Security(Error::RevokedKey))));
    assert!(store.begin(&key(4)).is_err());
    assert!(pending.finalize().is_err());
    assert_eq!(fs::read_dir(fixture.root.join("objects")).unwrap().count(), 2);
}

#[test]
fn competing_writers_abort_drop_and_cleanup_do_not_replace_publication() {
    let fixture = Fixture::new();
    let barrier = Arc::new(Barrier::new(2));
    let paths = fixture.root.clone();
    let handles: Vec<_> = (0..2)
        .map(|n| {
            let paths = paths.clone();
            let barrier = barrier.clone();
            std::thread::spawn(move || {
                let policy = Arc::new(Policy::default());
                let context = SecurityContext {
                    principal: PrincipalId([1; 16]),
                    encoding: EncodingId([3; 16]),
                    keys: Arc::new(Keys::default()),
                    authorization: policy.clone(),
                    retention: policy.clone(),
                    audit: policy,
                };
                let store = EncryptedStore::new(
                    TemporaryFileStore::new(paths.join("objects")),
                    context,
                    paths.join("scratch"),
                    10000,
                );
                let mut writer = store.begin(&key(1)).unwrap();
                writer.write_all(&[n; 100]).unwrap();
                barrier.wait();
                writer.finalize().is_ok()
            })
        })
        .collect();
    assert_eq!(handles.into_iter().filter_map(|h| h.join().unwrap().then_some(())).count(), 1);
    let store = fixture.store(1);
    let winner = read(&store, 1);
    for id in [2, 3] {
        let mut pending = store.begin(&key(id)).unwrap();
        pending.write_all(&vec![7; 100_000]).unwrap();
        if id == 2 {
            pending.abort().unwrap();
        } else {
            drop(pending);
        }
        assert!(store.open(&key(id)).is_err());
    }
    assert_eq!(read(&store, 1), winner);
    assert_eq!(fs::read_dir(fixture.root.join("objects")).unwrap().count(), 1);
}

#[test]
fn full_verification_precedes_reader_and_snapshot_survives_source_replacement() {
    let fixture = Fixture::new();
    let store = fixture.store(1);
    save(&store, 1, &vec![29; 150_000]);
    let original = fs::read(fixture.physical(1, &key(1))).unwrap();
    for end in [0, 56, 65536 + 77, original.len() - 1] {
        fs::write(fixture.physical(1, &key(1)), &original[..end]).unwrap();
        assert!(store.open(&key(1)).is_err());
        assert_eq!(fs::read_dir(fixture.root.join("scratch")).unwrap().count(), 0);
    }
    fs::write(fixture.physical(1, &key(1)), original).unwrap();
    let mut reader = store.open(&key(1)).unwrap();
    fs::write(fixture.physical(1, &key(1)), b"changed").unwrap();
    let mut bytes = Vec::new();
    reader.read_to_end(&mut bytes).unwrap();
    assert_eq!(bytes, vec![29; 150_000]);
    drop(reader);
    assert_eq!(fs::read_dir(fixture.root.join("scratch")).unwrap().count(), 0);
}

#[test]
fn encrypted_size_limit_poisoning_never_leaves_a_completed_artifact() {
    let fixture = Fixture::new();
    let store = EncryptedStore::new(
        TemporaryFileStore::new(fixture.root.join("objects")),
        fixture.context(1),
        fixture.root.join("scratch"),
        100,
    );
    let mut writer = store.begin(&key(1)).unwrap();
    assert!(writer.write_all(b"larger than the ciphertext budget").is_err());
    assert!(writer.write_all(b"x").is_err());
    assert!(writer.finalize().is_err());
    assert!(store.open(&key(1)).is_err());
    assert_eq!(fs::read_dir(fixture.root.join("objects")).unwrap().count(), 0);
}

#[test]
fn minimum_budget_and_opaque_identifiers_are_enforced() {
    let fixture = Fixture::new();
    let store = EncryptedStore::new(
        TemporaryFileStore::new(fixture.root.join("objects")),
        fixture.context(1),
        fixture.root.join("scratch"),
        92,
    );
    assert!(store.begin(&key(1)).is_err());
    let store = fixture.store(1);
    let a = new_artifact_key().unwrap();
    let b = new_artifact_key().unwrap();
    assert_ne!(a, b);
    store.begin(&a).unwrap().finalize().unwrap();
    let sensitive = turbo_eval::artifact::ArtifactKey::new("document-title").unwrap();
    assert!(store.begin(&sensitive).is_err());
    assert!(store.open(&sensitive).is_err());
    assert!(store.delete(&sensitive).is_err());
}

#[test]
#[ignore = "explicit bounded-memory adapter measurement"]
fn bounded_adapter_measurement() {
    let mib: u64 = std::env::var("ARTIFACT_MEASURE_MIB").unwrap().parse().unwrap();
    let fixture = Fixture::new();
    let store = fixture.store(1);
    let mut writer = store.begin(&key(1)).unwrap();
    let block = [42; 65536];
    for _ in 0..mib * 16 {
        writer.write_all(&block).unwrap();
    }
    writer.finalize().unwrap();
    let mut reader = store.open(&key(1)).unwrap();
    let mut buffer = [0; 32768];
    let mut count = 0_u64;
    loop {
        let n = reader.read(&mut buffer).unwrap();
        if n == 0 {
            break;
        }
        assert!(buffer[..n].iter().all(|&b| b == 42));
        count += n as u64;
    }
    assert_eq!(count, mib * 1024 * 1024);
    drop(reader);
    assert_eq!(fs::read_dir(fixture.root.join("scratch")).unwrap().count(), 0);
}

#[test]
fn live_staging_contains_only_ciphertext_and_errors_and_audit_exclude_text() {
    let fixture = Fixture::new();
    let store = fixture.store(1);
    let marker = b"document source must never appear in a staging file";
    let mut writer = store.begin(&key(1)).unwrap();
    for _ in 0..4000 {
        writer.write_all(marker).unwrap();
    }
    writer.flush().unwrap();
    let files: Vec<_> = fs::read_dir(fixture.root.join("objects")).unwrap().collect();
    assert_eq!(files.len(), 1);
    for file in files {
        let bytes = fs::read(file.unwrap().path()).unwrap();
        assert!(bytes.starts_with(b"TEAEAD01"));
        assert!(!bytes.windows(marker.len()).any(|w| w == marker));
    }
    fixture.policy.denied.store(true, Ordering::SeqCst);
    let error = writer.finalize().unwrap_err();
    let diagnostic = format!("{error:?} {error}");
    assert!(!diagnostic.contains(std::str::from_utf8(marker).unwrap()));
    assert!(!format!("{:?}", fixture.policy.events.lock().unwrap()).contains("document source"));
    assert_eq!(fs::read_dir(fixture.root.join("objects")).unwrap().count(), 0);
}

#[test]
fn flush_emits_partial_ciphertext_and_budget_counts_every_partial_frame() {
    let fixture = Fixture::new();
    let store = EncryptedStore::new(
        TemporaryFileStore::new(fixture.root.join("objects")),
        fixture.context(1),
        fixture.root.join("scratch"),
        116,
    );
    let mut writer = store.begin(&key(1)).unwrap();
    writer.write_all(b"a").unwrap();
    writer.flush().unwrap();
    let stage = fs::read_dir(fixture.root.join("objects")).unwrap().next().unwrap().unwrap().path();
    // Header plus kind/length plus one plaintext byte encrypted with its tag.
    assert_eq!(fs::metadata(stage).unwrap().len(), 56 + 5 + 1 + 16);
    // A second flushed frame needs another 21 bytes of overhead; it cannot fit.
    assert!(writer.write_all(b"b").is_err());
    assert!(writer.finalize().is_err());
    assert_eq!(fs::read_dir(fixture.root.join("objects")).unwrap().count(), 0);
}
