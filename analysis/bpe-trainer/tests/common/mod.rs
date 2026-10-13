//! What the tests share: where the corpora are, and a directory of its
//! own for a test to write a bundle in.

#![allow(dead_code)]

use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicUsize, Ordering};

pub fn repo() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../..")
}

pub fn testdata() -> PathBuf {
    repo().join("testdata")
}

/// One JSON string per line.
pub fn jsonl_strings(path: &Path) -> Vec<String> {
    let file = std::fs::read_to_string(path).unwrap();
    file.lines().map(|l| serde_json::from_str(l).unwrap()).collect()
}

/// testdata/tokenizer-texts.jsonl: one `{"text": ...}` per line.
pub fn tokenizer_texts() -> Vec<String> {
    let file = std::fs::read_to_string(testdata().join("tokenizer-texts.jsonl")).unwrap();
    file.lines()
        .map(|l| serde_json::from_str::<serde_json::Value>(l).unwrap()["text"].as_str().unwrap().to_owned())
        .collect()
}

/// The directory of the JVM goldens' BPE inputs.
pub fn bpe_inputs_dir() -> PathBuf {
    repo().join("analysis/goldens/data/inputs/bpe")
}

/// The BPE inputs of the JVM goldens, by name, in name order.
pub fn bpe_inputs() -> Vec<(String, String)> {
    let mut names: Vec<PathBuf> = std::fs::read_dir(bpe_inputs_dir()).unwrap().map(|e| e.unwrap().path()).collect();
    names.sort();
    names
        .into_iter()
        .map(|p| (p.file_stem().unwrap().to_string_lossy().into_owned(), std::fs::read_to_string(&p).unwrap()))
        .collect()
}

/// A directory of its own, removed on drop. Tests run side by side.
pub struct TempDir(pub PathBuf);

impl TempDir {
    pub fn new(name: &str) -> TempDir {
        static NEXT: AtomicUsize = AtomicUsize::new(0);
        let n = NEXT.fetch_add(1, Ordering::Relaxed);
        let dir = std::env::temp_dir().join(format!("turbo-bpe-trainer-{}-{name}-{n}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        TempDir(dir)
    }
}

impl Drop for TempDir {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}
