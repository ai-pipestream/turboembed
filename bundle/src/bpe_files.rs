//! A byte-level BPE `tokenizer.json` from a repository that ships only
//! `vocab.json` and `merges.txt`, as GPT-2 and RoBERTa checkpoints often
//! do. The file is what `tokenizers`' RoBERTa converter writes for the
//! core's byte-level BPE: no normalizer, GPT-2's ByteLevel pre-tokenizer
//! with the manifest's `add_prefix_space`, the vocabulary and merges as
//! given, and the manifest's special tokens as added tokens. The verify
//! step then holds it to the reference's ids, which the upstream
//! tokenizer made from the same two files.

use serde::Deserialize;
use serde_json::{Map, Value, json};

use crate::Result;

/// Where the two files are upstream, and where the bundle carries the
/// tokenizer made from them.
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct BpeFiles {
    pub vocab: String,
    pub merges: String,
    pub to: String,
}

/// The `tokenizer.json` text for `vocab` (a JSON object of piece to id)
/// and `merges` (`merges.txt`: an optional `#version` line, then a pair of
/// pieces a line), with the special tokens and `bpe` block of `manifest`.
pub fn tokenizer_json(vocab: &[u8], merges: &str, manifest: &Value) -> Result<Vec<u8>> {
    let vocab: Map<String, Value> = serde_json::from_slice(vocab).map_err(|e| format!("vocab: {e}"))?;
    let mut pairs = Vec::new();
    for (n, line) in merges.lines().enumerate() {
        if n == 0 && line.starts_with("#version") || line.is_empty() {
            continue;
        }
        let mut parts = line.split(' ');
        match (parts.next(), parts.next(), parts.next()) {
            (Some(l), Some(r), None) if !l.is_empty() && !r.is_empty() => pairs.push(Value::from(format!("{l} {r}"))),
            _ => return Err(format!("merges line {}: {line:?} is not two pieces", n + 1)),
        }
    }
    let t = &manifest["tokenizer"];
    let add_prefix_space =
        t["bpe"]["add_prefix_space"].as_bool().ok_or("manifest.tokenizer.bpe.add_prefix_space: missing")?;
    let mut added = Vec::new();
    for s in t["special_tokens"].as_array().ok_or("manifest.tokenizer.special_tokens: missing")? {
        let flag = |k: &str| s[k].as_bool().unwrap_or(false);
        added.push(json!({
            "id": s["id"], "content": s["content"], "single_word": flag("single_word"), "lstrip": flag("lstrip"),
            "rstrip": flag("rstrip"), "normalized": flag("normalized"), "special": true,
        }));
    }
    added.sort_by_key(|a| a["id"].as_u64());
    let byte_level =
        json!({ "type": "ByteLevel", "add_prefix_space": add_prefix_space, "trim_offsets": true, "use_regex": true });
    let file = json!({
        "version": "1.0",
        "truncation": null,
        "padding": null,
        "added_tokens": added,
        "normalizer": null,
        "pre_tokenizer": byte_level,
        "post_processor": null,
        "decoder": byte_level,
        "model": {
            "type": "BPE", "dropout": null, "unk_token": null, "continuing_subword_prefix": "",
            "end_of_word_suffix": "", "fuse_unk": false, "byte_fallback": false, "ignore_merges": false,
            "vocab": vocab, "merges": pairs,
        },
    });
    serde_json::to_vec(&file).map_err(|e| e.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn merges_skip_the_version_line_and_reject_a_bad_pair() {
        let manifest = json!({"tokenizer": {"bpe": {"add_prefix_space": false}, "special_tokens": [
            {"role": "SPECIAL_MASK", "content": "<mask>", "id": 3, "lstrip": true},
            {"role": "SPECIAL_BOS", "content": "<s>", "id": 0}]}});
        let vocab = br#"{"<s>":0,"a":1,"b":2,"<mask>":3,"ab":4}"#;
        let out: Value =
            serde_json::from_slice(&tokenizer_json(vocab, "#version: 0.2\na b\n", &manifest).unwrap()).unwrap();
        assert_eq!(out["model"]["merges"], json!(["a b"]));
        assert_eq!(out["model"]["vocab"]["ab"], 4);
        assert_eq!(out["added_tokens"][0]["content"], "<s>");
        assert_eq!(out["added_tokens"][1]["lstrip"], true);
        assert_eq!(out["pre_tokenizer"]["add_prefix_space"], false);
        assert!(tokenizer_json(vocab, "a b c\n", &manifest).is_err());
        assert!(tokenizer_json(vocab, "#version: 0.2\nab\n", &manifest).is_err());
    }
}
