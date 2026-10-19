use super::*;
use serde_json::json;

fn fixture() -> (Value, manifest::Unigram) {
    let m = manifest::Unigram {
        precompiled_charsmap: false,
        collapse_spaces: false,
        metaspace: "▁".into(),
        add_prefix_space: false,
        space_punctuation: false,
        collapse_whitespace: false,
        strip: false,
        whole_text: false,
    };
    let json = json!({"model":{"type":"Unigram","unk_id":0,"byte_fallback":false,
        "vocab":[["<unk>",0.0],["a",-1.0],["b",-1.0],["ab",-1.1],["ba",-1.7]]},
        "normalizer":null,"pre_tokenizer":{"type":"Metaspace","replacement":"▁","prepend_scheme":"never","split":true}});
    (json, m)
}

#[test]
fn malformed_normalizer_is_not_silently_ignored() {
    let (mut j, m) = fixture();
    for bad in [
        json!({}),
        json!(42),
        json!({"type":"Sequence","normalizers":false}),
        json!({"type":"Sequence","normalizers":[null]}),
    ] {
        j["normalizer"] = bad;
        assert!(Unigram::parse("test", &j, &m, 0).is_err());
    }
}

#[test]
fn malformed_boolean_settings_are_rejected() {
    let (j, m) = fixture();
    for (object, key) in [("model", "byte_fallback"), ("pre_tokenizer", "split")] {
        let mut j = j.clone();
        j[object][key] = json!("false");
        assert!(Unigram::parse("test", &j, &m, 0).is_err());
    }
    let mut j = j;
    j["pre_tokenizer"].as_object_mut().unwrap().remove("prepend_scheme");
    j["pre_tokenizer"]["add_prefix_space"] = json!("true");
    let mut m = m;
    m.add_prefix_space = true;
    assert!(Unigram::parse("test", &j, &m, 0).is_err());
}

#[test]
fn direct_parse_rejects_invalid_metaspace_without_panicking() {
    let (mut j, mut m) = fixture();
    for bad in ["", "xx"] {
        m.metaspace = bad.into();
        j["pre_tokenizer"]["replacement"] = json!(bad);
        assert!(Unigram::parse("test", &j, &m, 0).is_err());
    }
}

#[test]
fn lattice_preserves_best_path_for_extreme_finite_scores() {
    let (j, m) = fixture();
    let mut normal = Vec::new();
    Unigram::parse("test", &j, &m, 0).unwrap().encode("aba", 0, "aba", &mut normal);
    assert_eq!(normal, vec![3, 1]);
    let mut extreme = j;
    for item in extreme["model"]["vocab"].as_array_mut().unwrap() {
        item[1] = json!(item[1].as_f64().unwrap() * 1e308);
    }
    let mut actual = Vec::new();
    Unigram::parse("test", &extreme, &m, 0).unwrap().encode("aba", 0, "aba", &mut actual);
    assert_eq!(actual, normal);
}

#[test]
fn base64_rejects_nonterminal_padding_and_nonzero_unused_bits() {
    for bad in ["YQ==Yg==", "YR==", "YWJ="] {
        assert!(base64(bad).is_none(), "{bad}");
    }
}

#[test]
fn precompiled_map_rejects_invalid_replacement_bytes() {
    // Four-byte trie root, followed by malformed UTF-8 or an unterminated string.
    for bad in ["BAAAAAAAAAD/AA==", "BAAAAAAAAABh"] {
        assert!(Charsmap::parse("test", bad).is_err());
    }
}
