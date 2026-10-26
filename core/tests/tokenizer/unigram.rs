use super::*;
use std::path::PathBuf;

const MASK: i64 = 28232;

fn tokenizer_json() -> PathBuf {
    testdata().join("bge-m3/tokenizer.json")
}

/// The MiniLM fixture's manifest with BGE-M3's tokenizer and family.
pub fn manifest_m3() -> serde_json::Value {
    let mut m = manifest();
    m["model"]["id"] = json!("BAAI/bge-m3");
    m["tokenizer"] = json!({
        "file": "tokenizer.json",
        "unigram": {
            "precompiled_charsmap": true,
            "collapse_spaces": true,
            "metaspace": "\u{2581}",
            "add_prefix_space": true
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

fn fixture(name: &str) -> Fixture {
    Fixture::with_tokenizer(&format!("m3-{name}"), manifest_m3(), &tokenizer_json())
}

fn up() -> tokenizers::Tokenizer {
    upstream_at(&tokenizer_json())
}

/// The parity list and the reference cases, then the corners of the
/// SentencePiece pipeline: the character map's compatibility forms,
/// graphemes over and under six bytes, scripts with no spaces, runs
/// of whitespace, the metaspace itself in the text, special tokens in
/// the text with and without the whitespace `<mask>` takes.
pub fn texts() -> Vec<String> {
    let mut t = super::texts();
    t.extend(
        [
            "Ｆｕｌｌｗｉｄｔｈ ｔｅｘｔ and ﬁ ﬂ ligatures, ① ② ③, ½ ¾, ™ ℃, Ⅻ",
            "É (composed) vs É (decomposed), ñ vs ñ, 한 vs 한",
            "families 👨‍👩‍👧‍👦 flags 🇫🇷🇯🇵 skin 👍🏽 keycaps 1️⃣ ZWJ 🏳️‍🌈",
            "日本語の文章にはスペースがありません。中文也是这样。",
            "ภาษาไทยไม่มีการเว้นวรรคระหว่างคำ และภาษาลาวก็เช่นกัน",
            "العربية تكتب من اليمين إلى اليسار، والعبرية עברית كذلك",
            "Русский текст, українська мова, български език, ελληνικά",
            "हिन्दी और বাংলা और தமிழ் और తెలుగు और ಕನ್ನಡ और മലയാളം",
            "Tiếng Việt có nhiều dấu: ắ ằ ẳ ẵ ặ ớ ờ ở ỡ ợ",
            "Türkçe İstanbul ı I ğ Ğ ş Ş; Deutsch ß ẞ Straße; Français œ Œ æ Æ",
            "  two  and   three    spaces   ",
            " leading space",
            "trailing space ",
            "non\u{a0}breaking\u{a0}spaces and\u{2003}em\u{2009}thin\u{3000}ideographic",
            "tabs\tand\nnewlines\r\nand\u{b}vertical\u{c}form feeds",
            "the metaspace \u{2581} itself, \u{2581}twice\u{2581}\u{2581} and at the end\u{2581}",
            "<s> and </s> and <pad> and <unk> in the text",
            "before <mask> after",
            "before   <mask>   after",
            "before\t\n <mask>after<mask><mask> and <mask>",
            "<mask>",
            " <mask>",
            "<mask> ",
            "<s><mask></s>",
            "text<s>text</s>text",
            "control\u{0}chars\u{1}and\u{7f}format\u{200b}chars\u{fffd}here\u{feff}",
            "private use \u{e000}\u{f8ff} and a lone combining mark \u{301}",
            "𝔘𝔫𝔦𝔠𝔬𝔡𝔢 𝕞𝕒𝕥𝕙 𝓼𝓬𝓻𝓲𝓹𝓽 and 𠀀 𪜀 𫝀 outside the BMP",
            "zero-width\u{200d}joiner\u{200c}non-joiner\u{2060}word joiner",
            "a\u{300}\u{301}\u{302}\u{303}\u{304}\u{305} many marks on one base",
            "https://example.org/path?q=1&r=2#frag user@example.com 3.14159 1,000,000",
            "mixed日本語and한국어andEnglishand中文in one word",
            "Ⅳ ⅳ ㍿ ㌀ ㈱ ⑴ ⒈ ㎡ ㎏ ㏒ ㍻ compatibility characters",
            "",
            " ",
            "\u{2581}",
            "x",
        ]
        .map(str::to_owned),
    );
    t
}

#[test]
fn ids_match_upstream() {
    let f = fixture("ids-match-upstream");
    let tok = f.open().expect("tokenizer loads");
    let up = up();
    for text in texts() {
        let ours = tok.row(&text, None).unwrap();
        assert_eq!(ours, upstream_ids(&up, &text), "text {text:?}");
    }
}

#[test]
fn count_is_untruncated_with_specials() {
    let f = fixture("count");
    let tok = f.open().unwrap();
    let mut up = up();
    up.with_truncation(None).unwrap();
    for text in texts() {
        let mut n = 0u32;
        let mut err = new_error();
        let rc = unsafe { turbo_tokenizer_count(tok.t, common::text(&text), TURBO_PROMPT_NONE, &mut n, &mut err) };
        assert_eq!(rc, 0);
        assert_eq!(n as usize, up.encode(text.as_str(), true).unwrap().get_ids().len(), "text {text:?}");
    }
}

#[test]
fn truncation_cuts_inside_the_specials() {
    let f = fixture("truncation");
    let tok = f.open().unwrap();
    let text = long_text();
    let full = {
        let mut up = up();
        up.with_truncation(None).unwrap();
        upstream_ids(&up, &text)
    };
    assert_eq!((full[0], full[full.len() - 1]), (0, 2));
    let body = &full[1..full.len() - 1];
    let model = tok.row(&text, None).unwrap();
    assert_eq!(model.len(), MAX_SEQ);
    assert_eq!(&model[1..MAX_SEQ - 1], &body[..MAX_SEQ - 2]);
    let right = tok.row(&text, Some(&options(0, TURBO_TRUNCATE_RIGHT, 32, 0))).unwrap();
    assert_eq!(right, [&[0], &body[..30], &[2]].concat());
    let left = tok.row(&text, Some(&options(0, TURBO_TRUNCATE_LEFT, 32, 0))).unwrap();
    assert_eq!(left, [&[0], &body[body.len() - 30..], &[2]].concat());
    let bare = tok.row(&text, Some(&options(1, TURBO_TRUNCATE_RIGHT, 32, 0))).unwrap();
    assert_eq!(bare, &body[..32]);
}

#[test]
fn rows_are_padded_with_the_pad_id() {
    let f = fixture("padding");
    let tok = f.open().unwrap();
    let texts = ["short", "", "a somewhat longer row of text"];
    let e = tok.encode(&texts, None, 16).unwrap();
    for (i, text) in texts.iter().enumerate() {
        let n = e.lengths[i] as usize;
        let at = i * 16;
        assert_eq!(e.row(i), upstream_ids(&up(), text));
        assert!(e.ids[at + n..at + 16].iter().all(|&x| x == 1), "padded with pad id 1");
        assert!(e.mask[at..at + n].iter().all(|&x| x == 1));
        assert!(e.mask[at + n..at + 16].iter().all(|&x| x == 0));
    }
}

#[test]
fn info_names_the_kind_and_the_ids() {
    let f = fixture("info");
    let tok = f.open().unwrap();
    let info = tok.info();
    assert_eq!(info.vocab_size, MASK as u32 + 1);
    assert_eq!((info.pad_id, info.bos_id, info.eos_id, info.unk_id), (1, 0, 2, 3));
    let s = |b: &[std::ffi::c_char]| {
        b.iter().take_while(|&&c| c != 0).map(|&c| char::from(u8::from_ne_bytes(c.to_ne_bytes()))).collect::<String>()
    };
    assert_eq!(s(&info.kind), "unigram");
}

#[test]
fn the_manifest_must_agree_with_the_file() {
    let refused = |name: &str, edit: &dyn Fn(&mut serde_json::Value)| {
        let mut f = fixture(name);
        edit(&mut f.manifest);
        match f.open() {
            Ok(_) => panic!("{name}: the bundle loaded"),
            Err(e) => e,
        }
    };
    let e = refused("no-charsmap", &|m| m["tokenizer"]["unigram"]["precompiled_charsmap"] = json!(false));
    assert!(e.is(turbo::status::BUNDLE_INVALID, "normalizer is not what manifest.json says"), "{e:?}");
    let e = refused("no-collapse", &|m| m["tokenizer"]["unigram"]["collapse_spaces"] = json!(false));
    assert!(e.is(turbo::status::BUNDLE_INVALID, "normalizer is not what manifest.json says"), "{e:?}");
    let e = refused("no-prefix", &|m| m["tokenizer"]["unigram"]["add_prefix_space"] = json!(false));
    assert!(e.is(turbo::status::BUNDLE_INVALID, "add_prefix_space"), "{e:?}");
    let e = refused("other-metaspace", &|m| m["tokenizer"]["unigram"]["metaspace"] = json!("_"));
    assert!(e.is(turbo::status::BUNDLE_INVALID, "replacement"), "{e:?}");
    let e = refused("mask-lstrip", &|m| m["tokenizer"]["special_tokens"][4]["lstrip"] = json!(false));
    assert!(e.is(turbo::status::BUNDLE_INVALID, "<mask>"), "{e:?}");
    let e = refused("unk-id", &|m| m["tokenizer"]["special_tokens"][3]["id"] = json!(4));
    assert!(e.is(turbo::status::BUNDLE_INVALID, "model.unk_id"), "{e:?}");
    let e = refused("as-wordpiece", &|m| {
        m["tokenizer"]["unigram"] = json!(null);
        m["tokenizer"]["normalizer"] = manifest()["tokenizer"]["normalizer"].clone();
        m["tokenizer"]["wordpiece"] = manifest()["tokenizer"]["wordpiece"].clone();
    });
    assert!(e.is(turbo::status::BUNDLE_INVALID, "not WordPiece"), "{e:?}");
}

/// The BGE-M3 tokenizer with the normalizer and pre-tokenizer of
/// Model2Vec's multilingual models: the map and the collapse of spaces
/// in a nested sequence, a space around each ASCII punctuation
/// character, runs of whitespace to one space, the strip, and the
/// metaspace with no cut at it.
#[test]
fn the_multilingual_static_normalizer_matches_upstream() {
    let mut json: serde_json::Value = serde_json::from_slice(&std::fs::read(tokenizer_json()).unwrap()).unwrap();
    let first = json["normalizer"].clone();
    let mut steps = vec![first];
    for c in "!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~".chars() {
        steps.push(json!({ "type": "Replace", "pattern": { "String": c.to_string() }, "content": format!(" {c} ") }));
    }
    steps.push(json!({ "type": "Replace", "pattern": { "Regex": "\\s+" }, "content": " " }));
    steps.push(json!({ "type": "Strip", "strip_left": true, "strip_right": true }));
    json["normalizer"] = json!({ "type": "Sequence", "normalizers": steps });
    json["pre_tokenizer"]["split"] = json!(false);
    let path = std::env::temp_dir().join(format!("turbo-m3-static-{}.json", std::process::id()));
    std::fs::write(&path, serde_json::to_vec(&json).unwrap()).unwrap();
    let mut m = manifest_m3();
    let u = &mut m["tokenizer"]["unigram"];
    for flag in ["space_punctuation", "collapse_whitespace", "strip", "whole_text"] {
        u[flag] = json!(true);
    }
    let f = Fixture::with_tokenizer("m3-static", m, &path);
    let tok = f.open().expect("tokenizer loads");
    let up = upstream_at(&path);
    let mut all = texts();
    all.extend(["a,b.c!d", "  ¿qué?  ", "x\u{3000}\u{3000}y", "...", "\t \n"].map(str::to_owned));
    let core = core_tokenizer(&f);
    let mut whole = upstream_at(&path);
    whole.with_truncation(None).unwrap();
    for text in all {
        assert_eq!(tok.row(&text, None).unwrap(), upstream_ids(&up, &text), "text {text:?}");
        let (ours, theirs) = both_spans(&core, &whole, &text);
        spans_near_upstream(&text, &ours, &theirs);
    }
    std::fs::remove_file(path).unwrap();
}
