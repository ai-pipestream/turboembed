//! Malformed bundle records must fail before tokenizer reference checks.
mod common;

use common::*;
use turbo::{bundle::Bundle, tokenizer::Tokenizer};

#[test]
fn duplicate_added_token_cannot_hide_an_omitted_special() {
    let mut fixture = Fixture::standard("duplicate-added-token");
    let path = fixture.dir.join("tokenizer.json");
    let mut json: serde_json::Value = serde_json::from_slice(&std::fs::read(&path).unwrap()).unwrap();
    let added = json["added_tokens"].as_array_mut().unwrap();
    assert!(added.len() > 1);
    added[1] = added[0].clone();
    std::fs::write(&path, serde_json::to_vec(&json).unwrap()).unwrap();
    fixture.list("tokenizer.json");
    fixture.write();
    let bundle = Bundle::open(&fixture.dir).unwrap();
    let result = Tokenizer::unchecked(&bundle);
    assert!(result.is_err(), "duplicate record replaced a required special token without rejection");
    assert!(result.err().unwrap().message.contains("duplicate"));
}

#[test]
fn added_token_special_flag_must_match_manifest_decode_behavior() {
    let mut fixture = Fixture::standard("ordinary-added-token");
    let path = fixture.dir.join("tokenizer.json");
    let mut json: serde_json::Value = serde_json::from_slice(&std::fs::read(&path).unwrap()).unwrap();
    let added = &mut json["added_tokens"][0];
    let id = added["id"].as_u64().unwrap() as u32;
    let content = added["content"].as_str().unwrap().to_owned();
    added["special"] = serde_json::json!(false);
    std::fs::write(&path, serde_json::to_vec(&json).unwrap()).unwrap();
    // Upstream keeps this token even when special tokens are skipped.
    let upstream = tokenizers::Tokenizer::from_file(&path).unwrap();
    assert_eq!(upstream.decode(&[id], true).unwrap(), content);
    fixture.list("tokenizer.json");
    fixture.write();
    let bundle = Bundle::open(&fixture.dir).unwrap();
    let result = Tokenizer::unchecked(&bundle);
    assert!(result.is_err(), "accepted an ordinary token that the manifest treats as special");
}
