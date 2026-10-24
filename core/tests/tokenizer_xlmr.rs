//! The core's Unigram tokenizer on a file in xlm-roberta-base's older
//! form against upstream `tokenizers` (#1020): a model section with no
//! type, the precompiled map alone as the normalizer, and WhitespaceSplit
//! then Metaspace as the pre-tokenizer. The file is BGE-M3's (the same
//! XLM-R vocabulary and map) rewritten to that form.

mod common;

use std::path::{Path, PathBuf};

use common::*;
use serde_json::{Value, json};

const MASK: i64 = 28232;

/// BGE-M3's tokenizer.json in xlm-roberta-base's form, written once per
/// process.
fn tokenizer_json() -> PathBuf {
    let path = std::env::temp_dir().join(format!("turbo-xlmr-tokenizer-{}.json", std::process::id()));
    if !path.exists() {
        let raw = std::fs::read(testdata().join("bge-m3/tokenizer.json")).unwrap();
        let mut j: Value = serde_json::from_slice(&raw).unwrap();
        let model = j["model"].as_object_mut().unwrap();
        model.remove("type");
        model.remove("byte_fallback");
        let precompiled = j["normalizer"]["normalizers"][0].clone();
        assert_eq!(precompiled["type"], "Precompiled");
        j["normalizer"] = precompiled;
        j["pre_tokenizer"] = json!({
            "type": "Sequence",
            "pretokenizers": [
                { "type": "WhitespaceSplit" },
                { "type": "Metaspace", "replacement": "\u{2581}", "add_prefix_space": true }
            ]
        });
        let tmp = path.with_extension(format!("{:?}.tmp", std::thread::current().id()));
        std::fs::write(&tmp, serde_json::to_vec(&j).unwrap()).unwrap();
        std::fs::rename(&tmp, &path).unwrap();
    }
    path
}

/// The MiniLM fixture's manifest with that tokenizer and XLM-R's family.
fn manifest_xlmr() -> Value {
    let mut m = manifest();
    m["model"]["id"] = json!("FacebookAI/xlm-roberta-base");
    m["tokenizer"] = json!({
        "file": "tokenizer.json",
        "unigram": {
            "precompiled_charsmap": true,
            "collapse_spaces": false,
            "metaspace": "\u{2581}",
            "add_prefix_space": true,
            "whitespace_split": true
        },
        "special_tokens": [
            { "role": "SPECIAL_BOS",  "content": "<s>",    "id": 0 },
            { "role": "SPECIAL_PAD",  "content": "<pad>",  "id": 1 },
            { "role": "SPECIAL_EOS",  "content": "</s>",   "id": 2 },
            { "role": "SPECIAL_UNK",  "content": "<unk>",  "id": 3 },
            { "role": "SPECIAL_MASK", "content": "<mask>", "id": MASK, "lstrip": true }
        ],
        "template": ["<s>", "$TEXT", "</s>"],
        "truncation": "TRUNCATE_RIGHT"
    });
    let a = &mut m["architecture"];
    a["family"] = json!("FAMILY_ROBERTA");
    a["position_offset"] = json!(2);
    a["max_positions"] = json!(514);
    a["layer_norm_eps"] = json!(1e-5);
    a["token_types"] = json!(1);
    a["vocab_size"] = json!(MASK + 1);
    m
}

fn lines(path: &Path) -> Vec<String> {
    std::fs::read_to_string(path).unwrap().lines().map(|l| serde_json::from_str::<String>(l).unwrap()).collect()
}

/// The parity list, the SaT texts, and whitespace of every kind: runs,
/// either end, alone, and the characters the map leaves as they are.
fn texts() -> Vec<String> {
    let mut t = parity_texts();
    t.extend(lines(&Path::new(env!("CARGO_MANIFEST_DIR")).join("../analysis/sat/tests/data/texts.jsonl")));
    t.extend(
        [
            "",
            " ",
            "\t\n\r ",
            "  two  and   three    spaces   ",
            " leading space",
            "trailing space ",
            "line\u{2028}separator and paragraph\u{2029}separator and next\u{85}line",
            "non\u{a0}breaking\u{a0}spaces and\u{2003}em\u{2009}thin\u{3000}ideographic",
            "tabs\tand\nnewlines\r\nand\u{b}vertical\u{c}form feeds",
            "the metaspace \u{2581} itself, \u{2581}twice\u{2581}\u{2581} and at the end\u{2581}",
            "before   <mask>   after",
            "before\t\n <mask>after<mask><mask> and <mask>",
            "<s> and </s> and <pad> and <unk> in the text",
            "    fn indented() {\n        code();\n    }\n",
            "zero-width\u{200d}joiner\u{200b}space\u{2060}word joiner",
        ]
        .map(str::to_owned),
    );
    t
}

fn fixture(name: &str) -> Fixture {
    Fixture::with_tokenizer(&format!("xlmr-{name}"), manifest_xlmr(), &tokenizer_json())
}

fn whole() -> turbo::tokenizer::Encode {
    turbo::tokenizer::Encode {
        add_special_tokens: true,
        truncation: turbo::manifest::Truncation::None,
        max_tokens: u32::MAX,
        prompt: turbo::manifest::PromptRole::None,
    }
}

/// `span` with the whitespace at either end left out, unless that is all.
fn trimmed(text: &str, span: [u32; 2]) -> [u32; 2] {
    let Some(piece) = text.get(span[0] as usize..span[1] as usize) else { return span };
    let inner = piece.trim();
    if inner.is_empty() {
        return span;
    }
    let start = span[0] + (inner.as_ptr() as usize - piece.as_ptr() as usize) as u32;
    [start, start + inner.len() as u32]
}

#[test]
fn ids_match_upstream() {
    let f = fixture("ids");
    let tok = f.open().expect("tokenizer loads");
    let up = upstream_at(&tokenizer_json());
    for text in texts() {
        assert_eq!(tok.row(&text, None).unwrap(), upstream_ids(&up, &text), "text {text:?}");
    }
}

/// Every span is upstream's offset trimmed of whitespace, except where
/// upstream's ends inside a grapheme cluster the map replaced whole: the
/// core's then covers the cluster.
#[test]
fn spans_match_upstream_offsets() {
    use unicode_segmentation::UnicodeSegmentation;
    let f = fixture("spans");
    f.write();
    let tok = turbo::tokenizer::Tokenizer::load(&turbo::bundle::Bundle::open(&f.dir).unwrap()).unwrap();
    let mut up = upstream_at(&tokenizer_json());
    up.with_truncation(None).unwrap();
    for text in texts() {
        let (mut ids, mut spans) = (Vec::new(), Vec::new());
        tok.encode_spans(&text, whole(), &mut ids, &mut spans).unwrap();
        let e = up.encode(text.as_str(), true).unwrap();
        assert_eq!(ids.iter().map(|&i| i as u32).collect::<Vec<_>>(), e.get_ids(), "{text:?}");
        let bounds: Vec<usize> = text.grapheme_indices(true).map(|(i, _)| i).chain([text.len()]).collect();
        for (&ours, &(a, b)) in spans.iter().zip(e.get_offsets()) {
            let theirs = trimmed(&text, [a as u32, b as u32]);
            if ours != theirs {
                assert!(ours[0] <= theirs[0] && ours[1] >= theirs[1], "{text:?}: ours {ours:?} theirs {theirs:?}");
                assert!(!bounds.contains(&(theirs[1] as usize)), "{text:?}: upstream's {theirs:?} ends on a cluster");
            }
        }
    }
}

/// The file and the manifest must say the same of the WhitespaceSplit.
#[test]
fn the_manifest_must_agree_with_the_file() {
    let refused = |name: &str, edit: &dyn Fn(&mut Value)| {
        let mut f = fixture(name);
        edit(&mut f.manifest);
        f.open().err().unwrap_or_else(|| panic!("{name}: the bundle loaded"))
    };
    let e = refused("no-split", &|m| m["tokenizer"]["unigram"]["whitespace_split"] = json!(false));
    assert!(e.is(turbo::status::BUNDLE_INVALID, "WhitespaceSplit"), "{e:?}");
    let e = refused("whole", &|m| m["tokenizer"]["unigram"]["whole_text"] = json!(true));
    assert!(e.is(turbo::status::BUNDLE_INVALID, "whole_text"), "{e:?}");
    // BGE-M3's own file has no WhitespaceSplit.
    let mut m = manifest_xlmr();
    m["tokenizer"]["unigram"]["collapse_spaces"] = json!(true);
    let f = Fixture::with_tokenizer("xlmr-m3-file", m, &testdata().join("bge-m3/tokenizer.json"));
    let Err(e) = f.open() else { panic!("the bundle loaded") };
    assert!(e.is(turbo::status::BUNDLE_INVALID, "WhitespaceSplit"), "{e:?}");
}
