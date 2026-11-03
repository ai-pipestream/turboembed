#[path = "encrypted/common.rs"]
mod common;
#[path = "encrypted/remote.rs"]
mod remote;
use common::*;
use remote::Remote;
use std::{
    fs,
    io::{Read, Write},
    sync::{Arc, Barrier, atomic::Ordering},
};
use turbo_eval::artifact::encrypted::EncryptedStore;
use turbo_eval::artifact::{ArtifactKey, EvaluationArtifactStore, StorageError, StorageOperation};

#[test]
#[ignore = "requires an explicitly selected existing RustFS service and isolated test bucket"]
fn real_rustfs_ciphertext_policy_and_failure_paths() {
    let fixture = Fixture::new();
    let provider = Remote::configured(fixture.root.join("remote-stage"));
    let physical = |principal: u8, id: u8| {
        ArtifactKey::new(format!("{}.{}", format!("{principal:02x}").repeat(16), key(id).as_str())).unwrap()
    };
    // Cleanup guard contains only this run's exact opaque object names.
    struct Cleanup(Remote, Vec<ArtifactKey>);
    impl Drop for Cleanup {
        fn drop(&mut self) {
            for key in &self.1 {
                let _ = self.0.erase(key);
            }
        }
    }
    let _cleanup = Cleanup(provider.clone(), (1..=5).map(|id| physical(1, id)).chain([physical(2, 1)]).collect());
    let store =
        EncryptedStore::new(provider.clone(), fixture.context(1), fixture.root.join("scratch"), 16 * 1024 * 1024);
    let other =
        EncryptedStore::new(provider.clone(), fixture.context(2), fixture.root.join("scratch"), 16 * 1024 * 1024);
    assert!(matches!(store.list(), Err(StorageError::Unsupported(StorageOperation::List))));
    let text = b"private artifact content ".repeat(40_000);
    let mut writer = store.begin(&key(1)).unwrap();
    writer.write_all(&text).unwrap();
    writer.finalize().unwrap();
    let mut bytes = Vec::new();
    store.open(&key(1)).unwrap().read_to_end(&mut bytes).unwrap();
    assert_eq!(bytes, text);
    let mut ciphertext = Vec::new();
    provider.open(&physical(1, 1)).unwrap().read_to_end(&mut ciphertext).unwrap();
    assert!(!ciphertext.windows(128).any(|w| w == &text[..128]));
    assert_eq!(provider.get_status(&physical(1, 1), true), 403);
    assert!(other.open(&key(1)).is_err());
    let mut collision = store.begin(&key(1)).unwrap();
    collision.write_all(b"replacement must fail").unwrap();
    assert!(matches!(collision.finalize(), Err(StorageError::AlreadyExists(_))));
    let mut aborted = store.begin(&key(2)).unwrap();
    aborted.write_all(&text).unwrap();
    aborted.abort().unwrap();
    assert_eq!(provider.get_status(&physical(1, 2), false), 404);
    fixture.policy.held.store(true, Ordering::SeqCst);
    assert!(store.delete(&key(1)).is_err());
    assert_eq!(provider.get_status(&physical(1, 1), false), 200);
    fixture.keys.revoked.store(true, Ordering::SeqCst);
    assert!(store.open(&key(1)).is_err());
    fixture.keys.revoked.store(false, Ordering::SeqCst);
    fixture.policy.held.store(false, Ordering::SeqCst);
    let barrier = Arc::new(Barrier::new(2));
    let racers: Vec<_> = (0..2)
        .map(|n| {
            let context = fixture.context(1);
            let root = fixture.root.clone();
            let barrier = barrier.clone();
            std::thread::spawn(move || {
                // Independent local staging makes the conditional PUT, not a local
                // name collision, choose the winner at the real remote service.
                let remote = Remote::configured(root.join(format!("racer-{n}")));
                let store = EncryptedStore::new(remote, context, root.join("scratch"), 10000);
                let mut writer = store.begin(&key(3)).unwrap();
                writer.write_all(&[n; 100]).unwrap();
                barrier.wait();
                match writer.finalize() {
                    Ok(_) => true,
                    Err(StorageError::AlreadyExists(_)) => false,
                    Err(_) => panic!("unexpected remote publication error"),
                }
            })
        })
        .collect();
    assert_eq!(racers.into_iter().filter_map(|h| h.join().unwrap().then_some(())).count(), 1);
    let mut winner = Vec::new();
    store.open(&key(3)).unwrap().read_to_end(&mut winner).unwrap();
    assert!(winner == vec![0; 100] || winner == vec![1; 100]);
    let mutation = fixture.root.join("mutation");
    fs::write(&mutation, &ciphertext).unwrap();
    assert_eq!(provider.put_file(&physical(2, 1), &mut fs::File::open(&mutation).unwrap(), false).unwrap(), 200);
    assert!(other.open(&key(1)).is_err());
    // Deleting the other scope's object cannot delete the original scope.
    other.delete(&key(1)).unwrap();
    assert_eq!(provider.get_status(&physical(1, 1), false), 200);
    for variant in 0..7 {
        let mut changed = ciphertext.clone();
        match variant {
            0 => {
                changed[70] ^= 1;
            }
            1 => {
                changed.pop();
            }
            2 => changed.truncate(56),
            3 => {
                // Exchange the first two full frames.
                let second = 56 + 65536 + 21;
                let third = second + 65536 + 21;
                changed[56..third].rotate_left(second - 56);
            }
            4 => {
                // Duplicate a frame without changing its authenticated index.
                let frame = changed[56..56 + 65536 + 21].to_vec();
                changed.splice(56..56, frame);
            }
            5 => {
                changed.drain(56..56 + 65536 + 21);
            }
            _ => {
                changed[8] ^= 127;
            } // Unknown key identifier fails closed.
        }
        fs::write(&mutation, changed).unwrap();
        assert_eq!(provider.put_file(&physical(1, 1), &mut fs::File::open(&mutation).unwrap(), false).unwrap(), 200);
        assert!(store.open(&key(1)).is_err());
        assert_eq!(fs::read_dir(fixture.root.join("scratch")).unwrap().count(), 0);
    }
    store.delete(&key(1)).unwrap();
    assert_eq!(provider.get_status(&physical(1, 1), false), 404);
    // Exercise the common fixture's local helpers as part of this same scope.
    assert!(fixture.store(1).list().unwrap().is_empty());
    assert!(!fixture.physical(1, &key(1)).exists());
}
