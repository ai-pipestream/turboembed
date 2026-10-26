use super::*;

#[test]
fn normalized_special_token_word_boundaries_match_upstream() {
    let mut json: serde_json::Value =
        serde_json::from_slice(&std::fs::read(upstream_tokenizer_json()).unwrap()).unwrap();
    let pad = json["added_tokens"].as_array_mut().unwrap().iter_mut().find(|a| a["content"] == "[PAD]").unwrap();
    pad["normalized"] = json!(true);
    pad["single_word"] = json!(true);
    let path = std::env::temp_dir().join(format!("turbo-unicode-word-{}.json", std::process::id()));
    std::fs::write(&path, serde_json::to_vec(&json).unwrap()).unwrap();
    let mut m = manifest();
    let pad =
        m["tokenizer"]["special_tokens"].as_array_mut().unwrap().iter_mut().find(|a| a["content"] == "[PAD]").unwrap();
    pad["normalized"] = json!(true);
    pad["single_word"] = json!(true);
    let f = Fixture::with_tokenizer("unicode-word", m, &path);
    let core = core_tokenizer(&f);
    let mut up = upstream_at(&path);
    up.with_truncation(None).unwrap();
    let mut mismatches = Vec::new();
    for c in ['\u{7fd}', '\u{8ca}', '\u{1e4f0}', '\u{1e944}', '\u{301}', '\u{558}', 'a', '_'] {
        for text in [format!("{c}[PAD]"), format!("[PAD]{c}")] {
            let (ours, theirs) = both_spans(&core, &up, &text);
            check_well_formed(&text, &ours);
            if ours != theirs {
                mismatches.push(format!("text={text:?}: ours={ours:?}, upstream={theirs:?}"));
            }
        }
    }
    std::fs::remove_file(path).unwrap();
    assert!(mismatches.is_empty(), "{}", mismatches.join("\n"));
}
