//! Added tokens with no role, in the shapes ModernBERT's tokenizer file
//! has them (runs of spaces, `[unusedN]`, `|||IP_ADDRESS|||`,
//! `<|padding|>`, `[MASK]` taking the whitespace before it), against
//! upstream `tokenizers` on byte-level BPE with NFC and on WordPiece.

use super::*;
use std::path::{Path, PathBuf};
use std::str::FromStr;
use turbo_unicode::hf::aligned;
use turbo_unicode::normalize::Form;

pub fn tokenizer_json() -> PathBuf {
    testdata().join("tiny-bpe-added/tokenizer.json")
}

/// testdata/tiny-bpe-added/tokenizer.json as upstream writes it: the
/// tiny BPE's model with its first five pieces renamed to ModernBERT's
/// `|||IP_ADDRESS|||`, `<|padding|>`, `<|endoftext|>` and runs of four
/// and two spaces, NFC, and ModernBERT's added tokens in its order, so
/// upstream gives the pieces their ids and the others ids past the
/// vocabulary; then `[CLS] $A [SEP]` around the text.
fn tiny_bpe_added() -> String {
    use tokenizers::processors::template::TemplateProcessing;
    use tokenizers::{AddedToken, Tokenizer};
    let mut json: serde_json::Value = serde_json::from_slice(&std::fs::read(bpe::tokenizer_json()).unwrap()).unwrap();
    let vocab = json["model"]["vocab"].as_object_mut().unwrap();
    for (old, new) in [
        ("<s>", "|||IP_ADDRESS|||"),
        ("<pad>", "<|padding|>"),
        ("</s>", "<|endoftext|>"),
        ("<unk>", "    "),
        ("<mask>", "  "),
    ] {
        let id = vocab.remove(old).unwrap();
        vocab.insert(new.into(), id);
    }
    json["added_tokens"] = json!([]);
    json["normalizer"] = json!({ "type": "NFC" });
    json["post_processor"] = serde_json::Value::Null;
    let mut tok = Tokenizer::from_str(&json.to_string()).unwrap();
    let plain = |c: &str| AddedToken::from(c, false);
    let special = |c: &str| AddedToken::from(c, true);
    let mut added = vec![plain("|||IP_ADDRESS|||"), special("<|padding|>"), special("<|endoftext|>")];
    added.extend(["    ", "  ", "   ", "|||EMAIL_ADDRESS|||"].map(plain));
    added.extend(["[UNK]", "[CLS]", "[SEP]", "[PAD]"].map(special));
    added.push(special("[MASK]").lstrip(true));
    added.extend((0..12).map(|i| plain(&format!("[unused{i}]"))));
    tok.add_tokens(added).unwrap();
    let id = |c: &str| tok.token_to_id(c).unwrap();
    let template = TemplateProcessing::builder()
        .try_single("[CLS] $A [SEP]")
        .unwrap()
        .try_pair("[CLS] $A [SEP] $B [SEP]")
        .unwrap()
        .special_tokens(vec![("[CLS]", id("[CLS]")), ("[SEP]", id("[SEP]"))])
        .build()
        .unwrap();
    tok.with_post_processor(Some(template));
    tok.to_string(false).unwrap()
}

/// Writes testdata/tiny-bpe-added/tokenizer.json.
#[test]
#[ignore = "writes the test data; run by hand"]
fn make_tiny_bpe_added() {
    std::fs::create_dir_all(tokenizer_json().parent().unwrap()).unwrap();
    std::fs::write(tokenizer_json(), tiny_bpe_added()).unwrap();
}

#[test]
fn the_file_is_what_upstream_writes() {
    assert_eq!(std::fs::read_to_string(tokenizer_json()).unwrap(), tiny_bpe_added());
}

/// The manifest for the file at `path`: `base` with its special tokens
/// (role, content) taken from the file under their ids and flags, and
/// every other added token of the file in `added_tokens`.
fn manifest_for(mut base: serde_json::Value, roles: &[(&str, &str)], path: &Path) -> serde_json::Value {
    let file: serde_json::Value = serde_json::from_slice(&std::fs::read(path).unwrap()).unwrap();
    let flags = ["special", "normalized", "lstrip", "rstrip", "single_word"];
    let (mut specials, mut added) = (Vec::new(), Vec::new());
    for a in file["added_tokens"].as_array().unwrap() {
        let mut t = json!({ "content": a["content"], "id": a["id"] });
        match roles.iter().find(|r| a["content"] == r.1) {
            Some(r) => {
                t["role"] = json!(r.0);
                for f in &flags[1..] {
                    t[f] = a[f].clone();
                }
                specials.push(t);
            }
            None => {
                for f in flags {
                    t[f] = a[f].clone();
                }
                added.push(t);
            }
        }
    }
    let size = file["added_tokens"].as_array().unwrap().iter().map(|a| a["id"].as_u64().unwrap() + 1).max().unwrap();
    base["architecture"]["vocab_size"] = json!(size.max(base["architecture"]["vocab_size"].as_u64().unwrap()));
    base["tokenizer"]["special_tokens"] = json!(specials);
    base["tokenizer"]["added_tokens"] = json!(added);
    base
}

const ROLES: [(&str, &str); 5] = [
    ("SPECIAL_UNK", "[UNK]"),
    ("SPECIAL_BOS", "[CLS]"),
    ("SPECIAL_EOS", "[SEP]"),
    ("SPECIAL_PAD", "[PAD]"),
    ("SPECIAL_MASK", "[MASK]"),
];

fn manifest_bpe(path: &Path, nfc: bool) -> serde_json::Value {
    let mut m = bpe::manifest_bpe();
    m["tokenizer"]["bpe"] = json!({ "add_prefix_space": false, "nfc": nfc });
    m["tokenizer"]["template"] = json!(["[CLS]", "$TEXT", "[SEP]"]);
    manifest_for(m, &ROLES, path)
}

/// Texts that reach each added token and each flag.
fn added_texts() -> Vec<String> {
    let mut t: Vec<String> = [
        "[MASK]",
        "a [MASK] b",
        "a  \t [MASK]b [MASK][MASK]  [MASK]",
        "x<|padding|>y <|endoftext|> <|padding| <|padding|>",
        "|||IP_ADDRESS||| at |||EMAIL_ADDRESS|||, call |||PHONE_NUMBER|||",
        "a  b   c    d     e      f       g",
        "  leading",
        "trailing  ",
        "\t  \n   x \u{3000}  y\u{a0}  z",
        "[unused0][unused1][unused10][unused11][unused2]x[unused1",
        "[UNUSED1] [unused12] [unused1 0] [unused11]]",
        "e\u{301}[unused1] [unused1]\u{301} A\u{30a}[unused3]\u{212b}  \u{212b}",
        "caf\u{e9}  [MASK] nai\u{308}ve   |||IP_ADDRESS|||",
        "<|padding|>[CLS]  [SEP][UNK][PAD]",
        "\u{e9}<|padding|>\u{301}  [unused4]",
    ]
    .map(str::to_owned)
    .to_vec();
    t.push(format!("x{}y", " ".repeat(30)));
    t.push(format!("{}[MASK]", " ".repeat(9)));
    t
}

/// The core against upstream on `path`, on bpe::texts() and these: the
/// ids, the spans (upstream's, trimmed, where NFC leaves the text as it
/// is; a character NFC composes spans all its sources, docs/tokenizer.md)
/// and decode.
fn check_bpe(name: &str, m: serde_json::Value, path: &Path) {
    let nfc = m["tokenizer"]["bpe"]["nfc"] == true;
    let f = Fixture::with_tokenizer(&format!("added-{name}"), m, path);
    let tok = core_tokenizer(&f);
    let c = f.open().unwrap();
    let up = upstream_at(path);
    let mut whole = upstream_at(path);
    whole.with_truncation(None).unwrap();
    for text in bpe::texts().into_iter().chain(added_texts()) {
        assert_eq!(c.row(&text, None).unwrap(), upstream_ids(&up, &text), "{text:?}");
        let (ours, theirs) = both_spans(&tok, &whole, &text);
        let changed = nfc && aligned(Form::Nfc, &text).map(|(c, _)| c).ne(text.chars());
        if changed {
            assert_eq!(ours.iter().map(|t| t.0).collect::<Vec<_>>(), theirs.iter().map(|t| t.0).collect::<Vec<_>>());
        } else {
            assert_eq!(ours, theirs, "spans of {text:?}");
        }
        check_well_formed(&text, &ours);
        decode_matches_upstream(&tok, &whole, &text);
    }
}

#[test]
fn bpe_with_nfc_matches_upstream() {
    check_bpe("nfc", manifest_bpe(&tokenizer_json(), true), &tokenizer_json());
}

/// Each shape of ModernBERT's added tokens gives the id upstream gives
/// it, where upstream gives it.
#[test]
fn each_shape_is_its_id() {
    let path = tokenizer_json();
    let f = Fixture::with_tokenizer("added-shapes", manifest_bpe(&path, true), &path);
    let tok = f.open().unwrap();
    let up = upstream_at(&path);
    let id = |c: &str| up.token_to_id(c).unwrap() as i32;
    for (text, token) in [
        ("a  b", "  "),
        ("a   b", "   "),
        ("a      b", "    "),
        ("x [unused5] y", "[unused5]"),
        ("x [unused10] y", "[unused10]"),
        ("mail |||IP_ADDRESS|||.", "|||IP_ADDRESS|||"),
        ("a<|padding|>b", "<|padding|>"),
        ("fill the   [MASK] in", "[MASK]"),
    ] {
        let ours = tok.row(text, None).unwrap();
        assert_eq!(ours, upstream_ids(&up, text), "{text:?}");
        assert!(ours.contains(&id(token)), "{text:?} holds {token:?}: {ours:?}");
    }
    // [MASK] takes the spaces before it: none is left to encode.
    let ours = tok.row("fill the   [MASK]", None).unwrap();
    assert_eq!(ours[ours.len() - 2], id("[MASK]"));
    assert_eq!(ours.len(), tok.row("fill the[MASK]", None).unwrap().len());
}

/// The same file without NFC, and with the flags ModernBERT's file does
/// not set: single_word and rstrip on raw and normalized tokens, and a
/// normalized token NFC changes.
#[test]
fn bpe_with_each_flag_matches_upstream() {
    let base = std::fs::read_to_string(tokenizer_json()).unwrap();
    for variant in ["no-nfc", "flags"] {
        let mut json: serde_json::Value = serde_json::from_str(&base).unwrap();
        let nfc = variant != "no-nfc";
        if !nfc {
            json["normalizer"] = serde_json::Value::Null;
        } else {
            for a in json["added_tokens"].as_array_mut().unwrap() {
                match a["content"].as_str().unwrap() {
                    "<|padding|>" => (a["single_word"], a["rstrip"]) = (json!(true), json!(true)),
                    "[unused2]" | "<|endoftext|>" => a["single_word"] = json!(true),
                    "|||IP_ADDRESS|||" => (a["lstrip"], a["rstrip"]) = (json!(true), json!(true)),
                    _ => {}
                }
            }
        }
        let mut up = tokenizers::Tokenizer::from_str(&json.to_string()).unwrap();
        if nfc {
            up.add_tokens([tokenizers::AddedToken::from("e\u{301}x", false)]).unwrap();
        }
        let path = std::env::temp_dir().join(format!("turbo-added-{variant}-{}.json", std::process::id()));
        up.save(&path, false).unwrap();
        check_bpe(variant, manifest_bpe(&path, nfc), &path);
        let tok = Fixture::with_tokenizer(&format!("added-{variant}-more"), manifest_bpe(&path, nfc), &path);
        let tok = tok.open().unwrap();
        let up = upstream_at(&path);
        for text in [
            "a<|padding|>  b <|padding|>  c <|padding|>x",
            "x[unused2] [unused2]x [unused2]. <|endoftext|>",
            "ip  |||IP_ADDRESS|||   ip",
            "\u{e9}x ex e\u{301}x",
        ] {
            assert_eq!(tok.row(text, None).unwrap(), upstream_ids(&up, text), "{variant}: {text:?}");
        }
        std::fs::remove_file(path).unwrap();
    }
}

/// MiniLM's WordPiece with added tokens of each shape: normalized ones
/// matched in the lowercased text, a run of spaces BertNormalizer makes
/// from a tab, a raw one that takes the whitespace after it, and a
/// normalized one that must be a whole word.
#[test]
fn wordpiece_matches_upstream() {
    use tokenizers::AddedToken;
    let mut up = upstream_at(&upstream_tokenizer_json());
    up.add_tokens([
        AddedToken::from("[unused1]", false),
        AddedToken::from("  ", false),
        AddedToken::from("|||IP_ADDRESS|||", false),
        AddedToken::from("<|padding|>", true).rstrip(true),
        AddedToken::from("Foo Bar", false).single_word(true),
    ])
    .unwrap();
    let path = std::env::temp_dir().join(format!("turbo-added-wordpiece-{}.json", std::process::id()));
    up.save(&path, false).unwrap();
    let roles = [
        ("SPECIAL_PAD", "[PAD]"),
        ("SPECIAL_UNK", "[UNK]"),
        ("SPECIAL_BOS", "[CLS]"),
        ("SPECIAL_EOS", "[SEP]"),
        ("SPECIAL_MASK", "[MASK]"),
    ];
    let m = manifest_for(manifest(), &roles, &path);
    let f = Fixture::with_tokenizer("added-wordpiece", m, &path);
    let c = f.open().expect("tokenizer loads");
    let tok = core_tokenizer(&f);
    let up = upstream_at(&path);
    let mut whole = upstream_at(&path);
    whole.with_truncation(None).unwrap();
    let mut all = texts();
    all.extend(added_texts());
    all.extend(
        ["x\t y", "FOO BAR foo bar", "foo barx foo bar.", "<|padding|>   x<|padding|>y", "[UNUSED1][unused1]"]
            .map(str::to_owned),
    );
    for text in all {
        assert_eq!(c.row(&text, None).unwrap(), upstream_ids(&up, &text), "{text:?}");
        let (ours, theirs) = both_spans(&tok, &whole, &text);
        assert_eq!(ours, theirs, "spans of {text:?}");
        decode_matches_upstream(&tok, &whole, &text);
    }
    std::fs::remove_file(path).unwrap();
}

/// A manifest whose added tokens are not the file's is refused.
#[test]
fn the_manifest_must_list_the_file_added_tokens() {
    let path = tokenizer_json();
    type Edit = fn(&mut serde_json::Value);
    let cases: [(&str, Edit); 4] = [
        ("is not a special or added token", |m| {
            m["tokenizer"]["added_tokens"].as_array_mut().unwrap().pop();
        }),
        ("matched differently", |m| m["tokenizer"]["added_tokens"][0]["rstrip"] = json!(true)),
        ("upstream gives it", |m| m["tokenizer"]["added_tokens"][5]["id"] = json!(1300)),
        ("listed twice", |m| m["tokenizer"]["added_tokens"][1]["content"] = json!("[MASK]")),
    ];
    for (want, edit) in cases {
        let mut f = Fixture::with_tokenizer("added-bad", manifest_bpe(&path, true), &path);
        edit(&mut f.manifest);
        let e = f.open().unwrap_err();
        assert!(e.message.contains(want), "{want}: {e:?}");
    }
}

/// ModernBERT's own file (answerdotai/ModernBERT-base), from the
/// directory TURBO_BPE_FILES names: modernbert-base/tokenizer.json.
#[test]
#[ignore = "needs TURBO_BPE_FILES"]
fn matches_upstream_on_modernbert() {
    let dir = PathBuf::from(std::env::var("TURBO_BPE_FILES").expect("TURBO_BPE_FILES"));
    let path = dir.join("modernbert-base/tokenizer.json");
    let mut m = manifest_bpe(&path, true);
    m["architecture"]["vocab_size"] = json!(50368);
    check_bpe("modernbert", m, &path);
}
