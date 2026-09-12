//! Load the SHA-pinned / committed text corpus for chunking and parity.
//!
//! Default CI uses only committed micro fixtures (no network). `make
//! fetch-corpus` materializes the full Tiny Shakespeare file; STS pairs
//! are committed and hash-verified.

use std::path::{Path, PathBuf};

use serde::Deserialize;

use crate::chunker::{embed_units, Chunk};

const EXCERPT: &str =
    include_str!("../../../testdata/corpus/fixtures/tiny-shakespeare.excerpt.txt");
const STS_MICRO: &str = include_str!("../../../testdata/corpus/fixtures/sts-micro.jsonl");

/// One STS-style pair (original sentences, scores on a 0–5 scale).
#[derive(Debug, Clone, PartialEq, Deserialize)]
pub struct StsPair {
    pub id: String,
    pub score: f32,
    pub text_a: String,
    pub text_b: String,
}

/// Fixed prompts always embedded for cross-arch parity (independent of
/// whether the soak file has been fetched).
pub const PARITY_PROMPTS: &[(&str, &str)] = &[
    ("parity:short", "hello world"),
    ("parity:medium", "The quick brown fox jumps over the lazy dog."),
    ("parity:query", "query: the quick brown fox"),
    (
        "parity:unicode",
        "Unicode check: café, naïve, 東京, emoji 🙂.",
    ),
    (
        "parity:long",
        "Cross-architecture embeddings should stay nearly identical when the same catalog alias, tokenizer, and pooling are used.",
    ),
    (
        "parity:inferstream",
        "Inference should not hop through Python or a JVM.",
    ),
];

/// Locate the workspace testdata/corpus directory.
pub fn corpus_dir(root: &Path) -> PathBuf {
    root.join("testdata/corpus")
}

pub fn parse_sts_jsonl(text: &str) -> Result<Vec<StsPair>, String> {
    let mut out = Vec::new();
    for (i, line) in text.lines().enumerate() {
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        let pair: StsPair =
            serde_json::from_str(line).map_err(|e| format!("sts jsonl line {}: {e}", i + 1))?;
        out.push(pair);
    }
    Ok(out)
}

pub fn load_sts_micro() -> Vec<StsPair> {
    parse_sts_jsonl(STS_MICRO).expect("committed sts-micro.jsonl must parse")
}

/// Full committed STS list when present; otherwise the micro fixture.
pub fn load_sts_pairs(root: &Path) -> Result<Vec<StsPair>, String> {
    let path = corpus_dir(root).join("sts-pairs.jsonl");
    if path.is_file() {
        let text =
            std::fs::read_to_string(&path).map_err(|e| format!("{}: {e}", path.display()))?;
        parse_sts_jsonl(&text)
    } else {
        Ok(load_sts_micro())
    }
}

pub fn shakespeare_excerpt() -> &'static str {
    EXCERPT
}

/// Full Tiny Shakespeare when `make fetch-corpus` has run; otherwise None.
pub fn load_shakespeare(root: &Path) -> Result<Option<String>, String> {
    let path = corpus_dir(root).join("tiny-shakespeare.txt");
    if !path.is_file() {
        return Ok(None);
    }
    std::fs::read_to_string(&path)
        .map(Some)
        .map_err(|e| format!("{}: {e}", path.display()))
}

/// `(id, text)` pairs sent to Embed for parity.
///
/// Always includes the built-in prompts, every unique STS line from the
/// committed list (micro if the full file is absent), and sentence units
/// from the Shakespeare excerpt. When the full play is on disk, adds the
/// first `soak_limit` sentence units so soak stays bounded.
pub fn parity_texts(root: &Path, soak_limit: usize) -> Result<Vec<(String, String)>, String> {
    let mut out = Vec::new();
    let mut seen = std::collections::HashSet::new();
    let mut push = |id: String, text: String| {
        if text.trim().is_empty() {
            return;
        }
        if seen.insert(id.clone()) {
            out.push((id, text));
        }
    };

    for (id, text) in PARITY_PROMPTS {
        push((*id).to_string(), (*text).to_string());
    }

    for pair in load_sts_pairs(root)? {
        push(format!("{}:a", pair.id), pair.text_a);
        push(format!("{}:b", pair.id), pair.text_b);
    }

    for chunk in embed_units("tiny-shakespeare-excerpt", EXCERPT) {
        push(chunk.id, chunk.text);
    }

    if let Some(full) = load_shakespeare(root)? {
        if soak_limit > 0 {
            let extra: Vec<Chunk> = embed_units("tiny-shakespeare", &full)
                .into_iter()
                .take(soak_limit)
                .collect();
            for chunk in extra {
                push(chunk.id, chunk.text);
            }
        }
    }

    Ok(out)
}

/// All embed units of the soak file (excerpt fallback).
pub fn soak_units(root: &Path) -> Result<Vec<Chunk>, String> {
    if let Some(full) = load_shakespeare(root)? {
        return Ok(embed_units("tiny-shakespeare", &full));
    }
    Ok(embed_units("tiny-shakespeare-excerpt", EXCERPT))
}

#[cfg(test)]
mod tests {
    use super::*;
    use inferstream_fetch::workspace_root;

    #[test]
    fn micro_sts_has_twelve_pairs() {
        let pairs = load_sts_micro();
        assert_eq!(pairs.len(), 12);
        assert_eq!(pairs[0].id, "sts-0001");
        assert_eq!(pairs[0].score, 5.0);
    }

    #[test]
    fn committed_sts_has_ninety_six_pairs() {
        let root = workspace_root();
        let pairs = load_sts_pairs(&root).unwrap();
        assert_eq!(pairs.len(), 96, "committed sts-pairs.jsonl");
        assert!(pairs.iter().any(|p| p.id == "sts-0096"));
    }

    #[test]
    fn parity_texts_are_deterministic_without_soak_file() {
        let tmp = tempfile::tempdir().unwrap();
        // No testdata/corpus under tmp → micro STS + excerpt + prompts.
        let a = parity_texts(tmp.path(), 32).unwrap();
        let b = parity_texts(tmp.path(), 32).unwrap();
        assert_eq!(a, b);
        assert!(a.iter().any(|(id, _)| id == "parity:short"));
        assert!(a.iter().any(|(id, _)| id == "sts-0001:a"));
        assert!(a
            .iter()
            .any(|(id, _)| id.starts_with("tiny-shakespeare-excerpt:")));
        assert!(
            a.iter()
                .all(|(id, _)| !id.starts_with("tiny-shakespeare:p")),
            "full play must not appear when the file is missing"
        );
    }

    #[test]
    fn soak_units_excerpt_fallback() {
        let tmp = tempfile::tempdir().unwrap();
        let units = soak_units(tmp.path()).unwrap();
        assert!(!units.is_empty());
        assert!(units[0].id.starts_with("tiny-shakespeare-excerpt:"));
    }
}
