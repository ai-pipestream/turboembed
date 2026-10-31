//! `tokenizer.added_tokens` from the bundle's tokenizer.json: every added
//! token of the file that is not one of the recipe's `special_tokens`, with
//! its id and flags as the file gives them (ModernBERT's 111 role-less
//! tokens beside its five role tokens). The loader checks the file's list
//! against `special_tokens` plus `added_tokens` field by field
//! (docs/bundle.md), so what is written here is what it reads back.

use std::path::Path;

use serde_json::{Map, Value};

use crate::Result;

const FLAGS: [&str; 5] = ["special", "normalized", "lstrip", "rstrip", "single_word"];

/// Fill `manifest.tokenizer.added_tokens` when the recipe leaves it out and
/// the file has added tokens besides the role tokens. A recipe that writes
/// the list keeps it, and the loader checks it.
pub fn fill(m: &mut Value, bundle: &Path) -> Result<()> {
    let t = &m["tokenizer"];
    if t.get("added_tokens").is_some() || t.get("sentencepiece").is_some() {
        return Ok(());
    }
    let Some(file) = t["file"].as_str() else { return Ok(()) };
    let path = bundle.join(file);
    let bytes = std::fs::read(&path).map_err(|e| format!("{}: {e}", path.display()))?;
    let json: Value = serde_json::from_slice(&bytes).map_err(|e| format!("{}: {e}", path.display()))?;
    let added = role_less(&json, t)?;
    if !added.is_empty() {
        m["tokenizer"]["added_tokens"] = Value::Array(added);
    }
    Ok(())
}

/// The file's added tokens whose content no special token has, in the
/// file's order, each flag written only when true (the manifest's
/// canonical form).
fn role_less(file: &Value, tokenizer: &Value) -> Result<Vec<Value>> {
    let roles: Vec<&str> =
        tokenizer["special_tokens"].as_array().into_iter().flatten().filter_map(|s| s["content"].as_str()).collect();
    let mut out = Vec::new();
    for (i, a) in file["added_tokens"].as_array().into_iter().flatten().enumerate() {
        let content = a["content"].as_str().ok_or(format!("tokenizer.json added_tokens[{i}]: no content"))?;
        if roles.contains(&content) {
            continue;
        }
        let id = a["id"].as_u64().ok_or(format!("tokenizer.json added_tokens[{i}]: no id"))?;
        let mut t = Map::new();
        t.insert("content".into(), content.into());
        t.insert("id".into(), id.into());
        for f in FLAGS {
            match &a[f] {
                Value::Bool(true) => {
                    t.insert(f.into(), true.into());
                }
                Value::Bool(false) | Value::Null => {}
                v => return Err(format!("tokenizer.json added_tokens[{i}].{f}: {v} is not a boolean")),
            }
        }
        out.push(Value::Object(t));
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::*;

    #[test]
    fn the_role_less_tokens_are_listed_with_their_flags() {
        let file = json!({ "added_tokens": [
            { "id": 0, "content": "|||IP_ADDRESS|||", "single_word": false, "lstrip": false, "rstrip": false,
              "normalized": true, "special": false },
            { "id": 1, "content": "<|padding|>", "single_word": false, "lstrip": false, "rstrip": false,
              "normalized": false, "special": true },
            { "id": 50284, "content": "[MASK]", "single_word": false, "lstrip": true, "rstrip": false,
              "normalized": false, "special": true }
        ]});
        let tokenizer = json!({ "special_tokens": [{ "role": "SPECIAL_MASK", "content": "[MASK]", "id": 50284 }] });
        assert_eq!(
            role_less(&file, &tokenizer).unwrap(),
            vec![
                json!({ "content": "|||IP_ADDRESS|||", "id": 0, "normalized": true }),
                json!({ "content": "<|padding|>", "id": 1, "special": true }),
            ]
        );
        let bad = json!({ "added_tokens": [{ "id": 3, "content": "x", "special": "yes" }] });
        assert!(role_less(&bad, &tokenizer).unwrap_err().contains("added_tokens[0].special"));
    }

    #[test]
    fn a_listed_or_empty_list_is_left_as_it_is() {
        let d = std::env::temp_dir().join(format!("turbo-added-tokens-{}", std::process::id()));
        std::fs::create_dir_all(&d).unwrap();
        std::fs::write(d.join("tokenizer.json"), r#"{"added_tokens":[{"id":0,"content":"[PAD]"}]}"#).unwrap();
        let roles = json!([{ "role": "SPECIAL_PAD", "content": "[PAD]", "id": 0 }]);
        let mut m = json!({ "tokenizer": { "file": "tokenizer.json", "special_tokens": roles } });
        fill(&mut m, &d).unwrap();
        assert!(m["tokenizer"].get("added_tokens").is_none(), "no role-less token, no list");
        m["tokenizer"]["added_tokens"] = json!([]);
        fill(&mut m, &d).unwrap();
        assert_eq!(m["tokenizer"]["added_tokens"], json!([]), "a recipe's own list is kept");
        std::fs::remove_dir_all(d).unwrap();
    }
}
