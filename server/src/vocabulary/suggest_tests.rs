//! The spellchecker cache against the vocabulary store (#2341): a
//! vocabulary's file is read and hashed once, not on every query.

use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicUsize, Ordering};

use super::super::Store;
use super::*;

static SERIAL: AtomicUsize = AtomicUsize::new(0);

struct Dir(PathBuf);
impl Drop for Dir {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

fn store() -> (Dir, Store) {
    let id = format!("{}-{}", std::process::id(), SERIAL.fetch_add(1, Ordering::Relaxed));
    let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("../target/vocabulary-suggest-cache-tests").join(id);
    let store = Store::open(&dir).unwrap();
    (Dir(dir), store)
}

fn tsv(terms: &[&str]) -> Vec<u8> {
    let mut out = String::from("# turbo-vocabulary 1\nterm\tkind\ttf\tdf\n");
    for t in terms {
        out.push_str(&format!("{t}\tterm\t50\t10\n"));
    }
    out.into_bytes()
}

fn put(store: &Store, id: &str, bytes: &[u8]) {
    let desc = VocabularyArtifactDescriptor {
        artifact_id: id.into(),
        byte_size: bytes.len() as u64,
        sha256: turbo_maxent::sha256(bytes).to_hex(),
        ..Default::default()
    };
    assert!(store.put_vocabulary(desc, bytes).unwrap().1, "stored");
}

fn corrected(checker: &QuerySpellchecker, query: &str) -> String {
    checker.check_query(query, &[]).corrected_query
}

#[test]
fn an_unchanged_vocabulary_is_read_once() {
    let (_dir, store) = store();
    put(&store, "v", &tsv(&["password", "reset"]));
    let cache = SuggestCache::default();

    let first = cache.spellchecker(&store, "v", 0).unwrap();
    assert_eq!(store.body_reads(), 1, "the first query loads the vocabulary");
    assert_eq!(corrected(&first, "pasword reset"), "password reset");

    for _ in 0..25 {
        let again = cache.spellchecker(&store, "v", 0).unwrap();
        assert!(Arc::ptr_eq(&first, &again));
    }
    // 0 is the default distance, 2: the same checker, still no read.
    assert!(Arc::ptr_eq(&first, &cache.spellchecker(&store, "v", 2).unwrap()));
    assert_eq!(store.body_reads(), 1, "a cached vocabulary is not read again");

    // Another distance is another checker, built from one more read.
    let narrow = cache.spellchecker(&store, "v", 1).unwrap();
    assert!(!Arc::ptr_eq(&first, &narrow));
    assert_eq!(store.body_reads(), 2);
}

#[test]
fn a_replaced_vocabulary_is_read_again() {
    let (_dir, store) = store();
    put(&store, "v", &tsv(&["password"]));
    let cache = SuggestCache::default();
    let old = cache.spellchecker(&store, "v", 0).unwrap();
    assert_eq!(corrected(&old, "pasword"), "password");
    assert_eq!(corrected(&old, "setings"), "setings");

    // The store replaces the artifact under the same id; the cache is not
    // told. The descriptor's digest changed, so the next query reloads.
    assert!(store.delete("v").unwrap());
    put(&store, "v", &tsv(&["settings"]));
    let new = cache.spellchecker(&store, "v", 0).unwrap();
    assert!(!Arc::ptr_eq(&old, &new));
    assert_eq!(store.body_reads(), 2);
    assert_eq!(corrected(&new, "setings"), "settings");
    assert_eq!(corrected(&new, "pasword"), "pasword");

    cache.spellchecker(&store, "v", 0).unwrap();
    assert_eq!(store.body_reads(), 2, "the replacement is cached in turn");
}

#[test]
fn a_deleted_vocabulary_is_not_served_from_the_cache() {
    let (_dir, store) = store();
    put(&store, "v", &tsv(&["password"]));
    let cache = SuggestCache::default();
    cache.spellchecker(&store, "v", 0).unwrap();
    assert!(store.delete("v").unwrap());
    let Err(status) = cache.spellchecker(&store, "v", 0) else { panic!("a deleted vocabulary answered") };
    assert_eq!(status.code(), tonic::Code::NotFound);
    assert_eq!(store.body_reads(), 1, "nothing is read for an artifact the store no longer lists");
}

#[test]
fn an_unknown_distance_is_refused_before_any_read() {
    let (_dir, store) = store();
    put(&store, "v", &tsv(&["password"]));
    assert!(SuggestCache::default().spellchecker(&store, "v", 3).is_err());
    assert_eq!(store.body_reads(), 0);
}
