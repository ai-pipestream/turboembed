//! Byte-level BPE with Llama-3's and Qwen2's Split pattern and Qwen2's
//! NFC normalizer (#906): ids, spans and decode against upstream
//! `tokenizers` on testdata/tiny-bpe-split, which upstream's trainer made.

mod common;

use std::path::{Path, PathBuf};

use common::*;
use serde_json::{Value, json};
use turbo::tokenizer::{Encode, NOWHERE, Tokenizer};
use unicode_normalization_alignments::UnicodeNormalization;

fn tokenizer_json() -> PathBuf {
    testdata().join("tiny-bpe-split/tokenizer.json")
}

fn corpus() -> Vec<String> {
    let file = std::fs::read_to_string(testdata().join("tiny-bpe/corpus.jsonl")).unwrap();
    file.lines().map(|l| serde_json::from_str(l).unwrap()).collect()
}

/// Writes testdata/tiny-bpe-split/tokenizer.json: upstream's BPE trainer
/// on testdata/tiny-bpe's corpus with Llama-3's pre-tokenizer, GPT-2's
/// byte-level alphabet, 1200 pieces, and RoBERTa's special tokens,
/// post-processor and decoder.
#[test]
#[ignore = "writes the test data; run by hand"]
fn make_tiny_bpe_split() {
    use tokenizers::models::TrainerWrapper;
    use tokenizers::models::bpe::{BPE, BpeTrainerBuilder};
    use tokenizers::pre_tokenizers::PreTokenizerWrapper;
    use tokenizers::pre_tokenizers::byte_level::ByteLevel;
    use tokenizers::pre_tokenizers::sequence::Sequence;
    use tokenizers::pre_tokenizers::split::{Split, SplitPattern};
    use tokenizers::processors::roberta::RobertaProcessing;
    use tokenizers::{AddedToken, SplitDelimiterBehavior, Tokenizer};
    let mut tok = Tokenizer::new(BPE::default());
    let split = Split::new(SplitPattern::Regex(turbo::bpe::LLAMA3.into()), SplitDelimiterBehavior::Isolated, false);
    tok.with_pre_tokenizer(Some(Sequence::new(vec![
        PreTokenizerWrapper::Split(split.unwrap()),
        PreTokenizerWrapper::ByteLevel(ByteLevel::new(false, true, false)),
    ])));
    tok.with_decoder(Some(ByteLevel::default()));
    tok.with_post_processor(Some(
        RobertaProcessing::new(("</s>".into(), 2), ("<s>".into(), 0)).trim_offsets(true).add_prefix_space(false),
    ));
    let mut specials = ["<s>", "<pad>", "</s>", "<unk>"].map(|s| AddedToken::from(s, true)).to_vec();
    specials.push(AddedToken::from("<mask>", true).lstrip(true));
    let mut trainer: TrainerWrapper = BpeTrainerBuilder::new()
        .show_progress(false)
        .vocab_size(1200)
        .min_frequency(1)
        .special_tokens(specials)
        .initial_alphabet(ByteLevel::alphabet().into_iter().collect())
        .build()
        .into();
    tok.train(&mut trainer, corpus().into_iter()).unwrap();
    std::fs::create_dir_all(tokenizer_json().parent().unwrap()).unwrap();
    tok.save(tokenizer_json(), false).unwrap();
}

/// The MiniLM fixture's manifest with a byte-level BPE of `pattern`, NFC
/// when `nfc`, and RoBERTa's special tokens and template.
fn manifest_split(pattern: &str, nfc: bool) -> Value {
    let mut m = manifest();
    m["tokenizer"] = json!({
        "file": "tokenizer.json",
        "bpe": { "add_prefix_space": false, "pattern": pattern, "nfc": nfc },
        "special_tokens": [
            { "role": "SPECIAL_BOS",  "content": "<s>",    "id": 0 },
            { "role": "SPECIAL_PAD",  "content": "<pad>",  "id": 1 },
            { "role": "SPECIAL_EOS",  "content": "</s>",   "id": 2 },
            { "role": "SPECIAL_UNK",  "content": "<unk>",  "id": 3 },
            { "role": "SPECIAL_MASK", "content": "<mask>", "id": 4, "lstrip": true }
        ],
        "template": ["<s>", "$TEXT", "</s>"],
        "truncation": "TRUNCATE_RIGHT"
    });
    m
}

/// The file as Qwen2 has it: its pattern and an NFC normalizer.
fn as_qwen2(json: &mut Value) {
    json["pre_tokenizer"]["pretokenizers"][0]["pattern"]["Regex"] = json!(turbo::bpe::QWEN2);
    json["normalizer"] = json!({ "type": "NFC" });
}

/// The corpus, the corners of the pattern, text NFC changes, and random
/// strings of the corpus's characters.
fn texts() -> Vec<String> {
    let mut t = corpus();
    let pool: Vec<char> = t.concat().chars().collect();
    let mut x: u64 = 0x5851_f42d_4c95_7f2d;
    for _ in 0..300 {
        let mut s = String::new();
        for _ in 0..(x >> 58) + 1 {
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            s.push(pool[(x % pool.len() as u64) as usize]);
        }
        t.push(s);
    }
    t.extend(
        [
            "it's IT'S they're WE'VE I'm you'll he'd 'tis '\u{17f} ''s",
            "12345 1234567 ① ②③④⑤ ١٢٣٤ x1y22z333 3,141.59",
            "  two  spaces\t\ttabs \n\n lines\r\n crlf \n  \n  trailing   ",
            "!!! ... ?!\n\n\n --- ,abc .def 'ghi \tjkl \u{3000}mno",
            "   ",
            "Cafe\u{301} cafe\u{301}s e\u{302}\u{301} \u{212b}ngstro\u{308}m \u{1100}\u{1161}\u{11a8} \u{344}",
            "a lone mark \u{301} and \u{e9}\u{301} composed then marked",
            "<mask> <s>x</s> <pad><unk>",
            "before   <mask>   after",
        ]
        .map(str::to_owned),
    );
    t
}

fn tokenizer(name: &str, m: Value, json: &Value) -> (Fixture, Tokenizer, tokenizers::Tokenizer) {
    let path = std::env::temp_dir().join(format!("turbo-bpe-split-{name}-{}.json", std::process::id()));
    std::fs::write(&path, serde_json::to_vec(json).unwrap()).unwrap();
    let f = Fixture::with_tokenizer(&format!("bpe-split-{name}"), m, &path);
    f.write();
    let tok = Tokenizer::load(&turbo::bundle::Bundle::open(&f.dir).unwrap()).unwrap();
    let mut up = upstream_at(&path);
    up.with_truncation(None).unwrap();
    std::fs::remove_file(path).unwrap();
    (f, tok, up)
}

fn whole_text() -> Encode {
    Encode {
        add_special_tokens: true,
        truncation: turbo::manifest::Truncation::None,
        max_tokens: u32::MAX,
        prompt: turbo::manifest::PromptRole::None,
    }
}

/// `span` with the whitespace at either end left out, unless that is all.
fn trimmed(text: &str, s: [u32; 2]) -> [u32; 2] {
    let Some(piece) = text.get(s[0] as usize..s[1] as usize) else { return s };
    let inner = piece.trim();
    if inner.is_empty() {
        return s;
    }
    let start = s[0] + (inner.as_ptr() as usize - piece.as_ptr() as usize) as u32;
    [start, start + inner.len() as u32]
}

/// The core against upstream on `text`: the same ids and decode, and
/// spans that are upstream's offsets trimmed of whitespace, but where
/// upstream's are short of the text:
/// - a token of whitespace alone is that whitespace, not the empty
///   offset at its end that upstream's trim leaves;
/// - where NFC changes the text, a character comes from every character
///   it holds a part of, where upstream's offset keeps the first alone
///   and loses marks NFC reorders; there the core's span must be that
///   text, which in NFC holds the token's bytes.
fn check(tok: &Tokenizer, up: &tokenizers::Tokenizer, text: &str, nfc: bool) {
    let (mut ids, mut spans) = (Vec::new(), Vec::new());
    tok.encode_spans(text, whole_text(), &mut ids, &mut spans).unwrap();
    let e = up.encode(text, true).unwrap();
    let theirs: Vec<i32> = e.get_ids().iter().map(|&i| i as i32).collect();
    assert_eq!(ids, theirs, "{text:?}");
    let changed = nfc && text.nfc().map(|(c, _)| c).ne(text.chars());
    let mut last = 0;
    for (i, (&a, &(x, y))) in spans.iter().zip(e.get_offsets()).enumerate() {
        let b = trimmed(text, [x as u32, y as u32]);
        assert!(text.is_char_boundary(a[0] as usize) && text.is_char_boundary(a[1] as usize), "{text:?}: {a:?}");
        if a != NOWHERE {
            assert!(a[0] >= last, "{text:?}: token {i} starts before {last}");
            last = a[0];
        }
        let blank = text[a[0] as usize..a[1] as usize].trim().is_empty();
        let piece = token_bytes(&e.get_tokens()[i]);
        let piece = piece.trim_ascii();
        let from: String = text[a[0] as usize..a[1] as usize].nfc().map(|(c, _)| c).collect();
        let holds = !piece.is_empty() && from.as_bytes().windows(piece.len()).any(|w| w == piece);
        assert!(
            a == b || (blank && b[0] == b[1] && b[1] == a[1]) || (changed && holds),
            "{text:?}: token {i} ({}): ours {a:?} {:?}, upstream {b:?}",
            ids[i],
            &text[a[0] as usize..a[1] as usize],
        );
    }
    let raw: Vec<u32> = ids.iter().map(|&i| i as u32).collect();
    for skip in [true, false] {
        assert_eq!(tok.decode(&ids, skip).unwrap(), up.decode(&raw, skip).unwrap(), "{text:?} skip {skip}");
    }
}

/// The bytes of a byte-level token: GPT-2's alphabet keeps the printable
/// bytes as themselves and writes each other byte, in order, from U+0100.
fn token_bytes(token: &str) -> Vec<u8> {
    let kept = |b: u8| matches!(b, b'!'..=b'~' | 0xa1..=0xac | 0xae..=0xff);
    let moved: Vec<u8> = (0..=255u8).filter(|&b| !kept(b)).collect();
    token.chars().map(|c| if c < '\u{100}' { c as u8 } else { moved[c as usize - 0x100] }).collect()
}

fn base() -> Value {
    serde_json::from_slice(&std::fs::read(tokenizer_json()).unwrap()).unwrap()
}

#[test]
fn llama3_matches_upstream() {
    let (_f, tok, up) = tokenizer("llama3", manifest_split("BPE_PATTERN_LLAMA3", false), &base());
    for text in texts() {
        check(&tok, &up, &text, false);
    }
}

#[test]
fn qwen2_with_nfc_matches_upstream() {
    let mut json = base();
    as_qwen2(&mut json);
    let (_f, tok, up) = tokenizer("qwen2", manifest_split("BPE_PATTERN_QWEN2", true), &json);
    for text in texts() {
        check(&tok, &up, &text, true);
    }
    // The span of a composed character covers what it was composed from.
    let text = "x e\u{302}\u{301}";
    let (mut ids, mut spans) = (Vec::new(), Vec::new());
    tok.encode_spans(text, whole_text(), &mut ids, &mut spans).unwrap();
    assert_eq!(spans.iter().rfind(|s| **s != NOWHERE), Some(&[2, 7]), "{spans:?}");
    // The acute composes with the a past the cedilla, so the a's span
    // reaches over the acute too.
    let text = "a\u{327}\u{301}";
    ids.clear();
    spans.clear();
    tok.encode_spans(text, whole_text(), &mut ids, &mut spans).unwrap();
    assert_eq!(spans.iter().find(|s| **s != NOWHERE), Some(&[0, 5]), "{spans:?}");
    check(&tok, &up, text, true);
}

/// Llama-3's file with Qwen2's pattern, with merges ignored for a word
/// that is a piece, as Llama-3's own file has it.
#[test]
fn ignore_merges_and_the_other_pattern_match_upstream() {
    let mut json = base();
    json["pre_tokenizer"]["pretokenizers"][0]["pattern"]["Regex"] = json!(turbo::bpe::QWEN2);
    json["model"]["ignore_merges"] = json!(true);
    let mut m = manifest_split("BPE_PATTERN_QWEN2", false);
    m["tokenizer"]["bpe"]["ignore_merges"] = json!(true);
    let (_f, tok, up) = tokenizer("ignore-merges", m, &json);
    for text in texts() {
        check(&tok, &up, &text, false);
    }
}

/// A file that does not run what the manifest says is refused.
#[test]
fn the_manifest_must_agree_with_the_file() {
    type Edit = fn(&mut Value, &mut Value);
    let cases: [(&str, Edit); 7] = [
        ("Split pattern", |j, _| j["pre_tokenizer"]["pretokenizers"][0]["pattern"]["Regex"] = json!(r"\s+")),
        ("Split pattern", |_, m| m["tokenizer"]["bpe"]["pattern"] = json!("BPE_PATTERN_QWEN2")),
        ("GPT-2's pattern", |_, m| m["tokenizer"]["bpe"]["pattern"] = json!("BPE_PATTERN_GPT2")),
        ("use_regex", |j, _| j["pre_tokenizer"]["pretokenizers"][1]["use_regex"] = json!(true)),
        ("normalizer", |j, _| j["normalizer"] = json!({ "type": "NFC" })),
        ("NFC", |_, m| m["tokenizer"]["bpe"]["nfc"] = json!(true)),
        ("matches no token after NFC", |j, m| {
            j["normalizer"] = json!({ "type": "NFC" });
            j["added_tokens"][4]["normalized"] = json!(true);
            m["tokenizer"]["bpe"]["nfc"] = json!(true);
            m["tokenizer"]["special_tokens"][4]["normalized"] = json!(true);
        }),
    ];
    for (want, edit) in cases {
        let (mut json, mut m) = (base(), manifest_split("BPE_PATTERN_LLAMA3", false));
        edit(&mut json, &mut m);
        let mut f = Fixture::with_tokenizer("bpe-split-bad", m, &tokenizer_json());
        std::fs::write(f.dir.join("tokenizer.json"), serde_json::to_vec(&json).unwrap()).unwrap();
        f.list("tokenizer.json");
        let e = f.open().unwrap_err();
        assert!(e.message.contains(want), "{want}: {e:?}");
    }
}

/// The published Llama-3 and Qwen2 files, from the directory
/// TURBO_BPE_FILES names: llama3/tokenizer.json and qwen2/tokenizer.json.
/// Their reserved added tokens are not in their model's vocabulary, which
/// the manifest's special tokens cannot describe yet, so each file's
/// added tokens and post-processor are replaced by one unknown token put
/// at the end of the vocabulary: what is checked is the pattern, NFC and
/// the merges on the published vocabulary.
const UNK: &str = "<turbo unk>";

#[test]
#[ignore = "needs TURBO_BPE_FILES"]
fn matches_upstream_on_llama3_and_qwen2() {
    let dir = PathBuf::from(std::env::var("TURBO_BPE_FILES").expect("TURBO_BPE_FILES"));
    for (name, pattern, nfc) in [("llama3", "BPE_PATTERN_LLAMA3", false), ("qwen2", "BPE_PATTERN_QWEN2", true)] {
        let mut json: Value = serde_json::from_slice(&std::fs::read(dir.join(name).join("tokenizer.json")).unwrap())
            .unwrap_or_else(|e| panic!("{name}: {e}"));
        let vocab = json["model"]["vocab"].as_object_mut().unwrap();
        // A byte-level piece never holds a raw space (it is Ġ there), so
        // this name cannot be one of the vocabulary's own pieces, as
        // "<unk>" is in Qwen2's.
        let unk = vocab.len();
        assert!(vocab.insert(UNK.into(), json!(unk)).is_none(), "{name}: {UNK} is in the vocabulary");
        json["added_tokens"] = json!([{ "id": unk, "content": UNK, "single_word": false, "lstrip": false,
            "rstrip": false, "normalized": false, "special": true }]);
        json["post_processor"] = Value::Null;
        let mut m = manifest_split(pattern, nfc);
        m["tokenizer"]["bpe"]["ignore_merges"] = json!(json["model"]["ignore_merges"] == true);
        m["tokenizer"]["special_tokens"] = json!([{ "role": "SPECIAL_UNK", "content": UNK, "id": unk }]);
        m["tokenizer"]["template"] = json!(["$TEXT"]);
        m["architecture"]["vocab_size"] = json!(unk + 1);
        let (_f, tok, up) = tokenizer(name, m, &json);
        let texts = texts();
        for text in &texts {
            check(&tok, &up, text, nfc);
        }
        println!("{name}: {} texts match upstream", texts.len());
    }
}

#[test]
fn the_fixture_is_what_the_generator_writes() {
    // The file names the pattern the core matches, so a change to either
    // is caught here first.
    let json = base();
    assert_eq!(json["pre_tokenizer"]["pretokenizers"][0]["pattern"]["Regex"], json!(turbo::bpe::LLAMA3));
    assert!(Path::new(&tokenizer_json()).is_file());
}

/// Encoding speed of the Split patterns and NFC beside GPT-2's on
/// testdata/tiny-bpe, on the corpus repeated, the best of 30 runs each:
/// run with --release.
#[test]
#[ignore = "measures; run by hand with --release"]
fn speed_beside_gpt2() {
    let text = corpus().join(" ").repeat(40);
    let mut gpt2 = manifest_split("BPE_PATTERN_GPT2", false);
    gpt2["tokenizer"]["bpe"].as_object_mut().unwrap().remove("pattern");
    let gpt2_json: Value =
        serde_json::from_slice(&std::fs::read(testdata().join("tiny-bpe/tokenizer.json")).unwrap()).unwrap();
    let mut qwen2 = base();
    as_qwen2(&mut qwen2);
    let mut decomposed = String::new();
    for c in text.chars() {
        decomposed.push(c);
        if c == 'e' {
            decomposed.push('\u{301}');
        }
    }
    let composed: String = text.as_str().nfc().map(|(c, _)| c).collect();
    for (name, m, json, text) in [
        ("gpt2", gpt2, gpt2_json, &text),
        ("qwen2-nfc-composed", manifest_split("BPE_PATTERN_QWEN2", true), qwen2.clone(), &composed),
        ("llama3", manifest_split("BPE_PATTERN_LLAMA3", false), base(), &text),
        ("qwen2-nfc", manifest_split("BPE_PATTERN_QWEN2", true), qwen2.clone(), &text),
        ("qwen2-nfc-decomposed", manifest_split("BPE_PATTERN_QWEN2", true), qwen2, &decomposed),
    ] {
        let (_f, tok, _) = tokenizer(name, m, &json);
        let mut ids = Vec::new();
        // The best of 30, as the machine's other work only slows a run.
        let mut s = f64::MAX;
        for _ in 0..30 {
            ids.clear();
            let start = std::time::Instant::now();
            tok.encode_into(text, whole_text(), &mut ids).unwrap();
            s = s.min(start.elapsed().as_secs_f64());
        }
        println!("{name}: {:.1} MB/s, {} tokens", text.len() as f64 / s / 1e6, ids.len());
    }
}
