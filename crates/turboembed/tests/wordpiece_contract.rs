//! Compare the native tokenizer with an independent model tokenizer implementation.
use std::ffi::{c_char, c_void, CString};
use std::sync::atomic::{AtomicU64, Ordering};

use serde_json::{json, Value};
use tokenizers::{PaddingParams, PaddingStrategy, Tokenizer, TruncationParams};

unsafe extern "C" {
    fn wordpiece_vocab_load(path: *const c_char, out: *mut *mut c_void) -> i32;
    fn wordpiece_vocab_destroy(v: *mut c_void);
    fn wordpiece_tokenize(
        v: *const c_void,
        text: *const u8,
        len: usize,
        ids: *mut i32,
        cap: usize,
        width: u32,
        count: *mut usize,
    ) -> i32;
    fn wordpiece_encode_sentence(
        v: *const c_void,
        text: *const u8,
        len: usize,
        ids: *mut i32,
        mask: *mut i32,
        types: *mut i32,
        pos: *mut i32,
        seq: u32,
        stride: u32,
        width: u32,
    ) -> i32;
    fn wordpiece_pack_pair(
        v: *const c_void,
        a: *const u8,
        an: usize,
        b: *const u8,
        bn: usize,
        ids: *mut i32,
        mask: *mut i32,
        types: *mut i32,
        pos: *mut i32,
        seq: u32,
        stride: u32,
        width: u32,
        truncation: u32,
        max_length: u32,
    ) -> i32;
}

struct Native(*mut c_void);
impl Drop for Native {
    fn drop(&mut self) {
        unsafe { wordpiece_vocab_destroy(self.0) }
    }
}
impl Native {
    fn load_bytes(bytes: &[u8]) -> Result<Self, i32> {
        assert!(turboembed::abi_version() > 0); // Link the native archive.
        static NEXT: AtomicU64 = AtomicU64::new(0);
        let path = std::env::temp_dir().join(format!(
            "turboembed-tokenizer-{}-{}.json",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        ));
        std::fs::write(&path, bytes).unwrap();
        let cpath = CString::new(path.to_str().unwrap()).unwrap();
        let mut raw = std::ptr::null_mut();
        let status = unsafe { wordpiece_vocab_load(cpath.as_ptr(), &mut raw) };
        std::fs::remove_file(path).unwrap();
        if status == 0 {
            assert!(!raw.is_null());
            Ok(Self(raw))
        } else {
            assert!(raw.is_null());
            Err(status)
        }
    }
    fn load(config: &Value) -> Self {
        Self::load_bytes(&serde_json::to_vec(config).unwrap()).expect("native tokenizer load")
    }
    fn tokens(&self, text: &[u8], cap: usize) -> Result<Vec<u32>, i32> {
        let mut ids = vec![-777; cap + 1];
        let mut count = usize::MAX;
        let status = unsafe {
            wordpiece_tokenize(
                self.0,
                text.as_ptr(),
                text.len(),
                ids.as_mut_ptr(),
                cap,
                4,
                &mut count,
            )
        };
        assert_eq!(ids[cap], -777, "output canary");
        if status != 0 {
            return Err(status);
        }
        assert!(count <= cap);
        Ok(ids[..count].iter().map(|&v| v as u32).collect())
    }
}

fn config() -> Value {
    let mut vocab = serde_json::Map::new();
    for token in [
        "[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]", "hello", "world", "中", "文", "𠀀", "😀",
        "€", "##€", "!", "?", "[", "]", "—", "ᄀ", "##ᅡ", "##ᆨ",
    ] {
        vocab.insert(token.into(), json!(vocab.len()));
    }
    for ch in "abcdefghijklmnopqrstuvwxyzßøæðıłσςαωжяéᾀ".chars() {
        for token in [ch.to_string(), format!("##{ch}")] {
            if !vocab.contains_key(&token) {
                vocab.insert(token, json!(vocab.len()));
            }
        }
    }
    json!({
        "version":"1.0", "truncation":null, "padding":null,
        "added_tokens": (["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"].iter().enumerate().map(|(id, token)| json!({"id":id,"content":token,"single_word":false,"lstrip":false,"rstrip":false,"normalized":false,"special":true})).collect::<Vec<_>>()),
        "normalizer":{"type":"BertNormalizer","clean_text":true,"handle_chinese_chars":true,"strip_accents":null,"lowercase":true},
        "pre_tokenizer":{"type":"BertPreTokenizer"},
        "post_processor":{"type":"BertProcessing","sep":["[SEP]",3],"cls":["[CLS]",2]},
        "decoder":{"type":"WordPiece","prefix":"##","cleanup":true},
        "model":{"type":"WordPiece","unk_token":"[UNK]","continuing_subword_prefix":"##","max_input_chars_per_word":100,"vocab":vocab}
    })
}
fn reference(config: &Value) -> Tokenizer {
    Tokenizer::from_bytes(serde_json::to_vec(config).unwrap()).unwrap()
}

#[test]
fn unicode_and_literal_special_tokens_match_reference() {
    let config = config();
    let native = Native::load(&config);
    let reference = reference(&config);
    for text in [
        "",
        "Hello WORLD!",
        "Straße Ø Æ Ð Ł ı",
        "Café Cafe\u{301} İ",
        "ΑΩ Σς ЖЯ",
        "각",
        "hello\0world",
        "hello\u{200d}world\u{e000}",
        "hello\u{fffd}world",
        "hello\u{0378}world",
        "a€b — 中中文𠀀",
        "😀hello",
        "[MASK]hello[SEP]world",
        "[CLS][UNK][PAD]",
        "[mask]",
        "a\u{1ab0}b",
        "hello\u{2028}world",
    ] {
        let expected = reference.encode(text, false).unwrap();
        assert_eq!(
            native.tokens(text.as_bytes(), 512).unwrap(),
            expected.get_ids(),
            "{text:?}"
        );
    }
}

#[test]
fn long_words_and_output_prefixes_match_reference() {
    let config = config();
    let native = Native::load(&config);
    let reference = reference(&config);
    for text in [
        "hello world".into(),
        "ab".repeat(40),
        "a".repeat(100),
        "a".repeat(101),
        "a".repeat(4096),
        format!("{}!hello", "a".repeat(500)),
        "hello[MASK]world".into(),
        "hello🦀".into(),
    ] {
        let expected = reference.encode(text.as_str(), false).unwrap();
        for cap in [0, 1, 2, 7, 64, 100, 512] {
            assert_eq!(
                native.tokens(text.as_bytes(), cap).unwrap(),
                expected.get_ids()[..cap.min(expected.len())],
                "length={}, cap={cap}",
                text.len()
            );
        }
    }
}

#[test]
fn padded_sentence_and_pair_rows_match_reference() {
    let config = config();
    let native = Native::load(&config);
    for seq in [3usize, 4, 7, 8, 32] {
        let mut reference = reference(&config);
        reference
            .with_truncation(Some(TruncationParams {
                max_length: seq,
                ..Default::default()
            }))
            .unwrap();
        reference.with_padding(Some(PaddingParams {
            strategy: PaddingStrategy::Fixed(seq),
            pad_id: 0,
            pad_token: "[PAD]".into(),
            ..Default::default()
        }));
        for (a, b) in [
            ("hello", "world"),
            ("hello world", "hello world"),
            ("ababa", "ab"),
            ("ab", "ababa"),
            ("", "world"),
            ("", ""),
        ] {
            for pair in [false, true] {
                let expected = if pair {
                    reference.encode((a, b), true).unwrap()
                } else {
                    reference.encode(a, true).unwrap()
                };
                let mut ids = vec![-1; seq];
                let mut mask = ids.clone();
                let mut types = ids.clone();
                let mut pos = ids.clone();
                let status = unsafe {
                    if pair {
                        wordpiece_pack_pair(
                            native.0,
                            a.as_ptr(),
                            a.len(),
                            b.as_ptr(),
                            b.len(),
                            ids.as_mut_ptr(),
                            mask.as_mut_ptr(),
                            types.as_mut_ptr(),
                            pos.as_mut_ptr(),
                            seq as u32,
                            seq as u32,
                            4,
                            0,
                            0,
                        )
                    } else {
                        wordpiece_encode_sentence(
                            native.0,
                            a.as_ptr(),
                            a.len(),
                            ids.as_mut_ptr(),
                            mask.as_mut_ptr(),
                            types.as_mut_ptr(),
                            pos.as_mut_ptr(),
                            seq as u32,
                            seq as u32,
                            4,
                        )
                    }
                };
                assert_eq!(status, 0, "seq={seq}, pair={pair}, a={a:?}, b={b:?}");
                assert_eq!(
                    ids.iter().map(|&v| v as u32).collect::<Vec<_>>(),
                    expected.get_ids(),
                    "seq={seq}, pair={pair}, a={a:?}, b={b:?}"
                );
                assert_eq!(
                    mask.iter().map(|&v| v as u32).collect::<Vec<_>>(),
                    expected.get_attention_mask()
                );
                assert_eq!(
                    types.iter().map(|&v| v as u32).collect::<Vec<_>>(),
                    expected.get_type_ids()
                );
            }
        }
    }
}

#[test]
fn incompatible_or_malformed_tokenizers_are_rejected() {
    for (path, value) in [
        ("/model/type", json!("BPE")),
        ("/normalizer/lowercase", json!(false)),
        ("/normalizer/strip_accents", json!(false)),
        ("/pre_tokenizer/type", json!("Whitespace")),
        ("/model/max_input_chars_per_word", json!(256)),
        ("/model/continuing_subword_prefix", json!("@@")),
        ("/added_tokens/0/normalized", json!(true)),
        ("/added_tokens/0/lstrip", json!(true)),
        ("/post_processor/cls/1", json!(999)),
        ("/model/vocab/hello", json!(-1)),
        ("/model/vocab/hello", json!(4294967297u64)),
        ("/model/vocab/hello", json!(3)),
    ] {
        let mut cfg = config();
        *cfg.pointer_mut(path).unwrap() = value;
        assert!(
            Native::load_bytes(&serde_json::to_vec(&cfg).unwrap()).is_err(),
            "accepted incompatible {path}"
        );
    }
    for bytes in [
        b"{\"model\":{\"type\":\"WordPiece\",\"vocab\":{\"a\":1}}}".as_slice(),
        b"{\"vocab\":{\"a\":1}}",
        b"{\"vocab\":{\"\\uD800\":1}}",
        b"{\"vocab\":{\"a\":99999999999999999999999999999}}",
    ] {
        assert!(Native::load_bytes(bytes).is_err());
    }
}

#[test]
fn malformed_utf8_is_rejected_even_after_output_capacity() {
    let native = Native::load(&config());
    for bytes in [
        b"\xc0\xaf".as_slice(),
        b"\xe2\x82",
        b"\xed\xa0\x80",
        b"\xf4\x90\x80\x80",
        b"hello world \xff",
    ] {
        for cap in [0, 1, 64] {
            assert_eq!(native.tokens(bytes, cap), Err(1));
        }
    }
}

#[test]
fn removing_added_token_rules_changes_literal_special_handling() {
    let mut config = config();
    config["added_tokens"] = json!([]);
    let native = Native::load(&config);
    let reference = reference(&config);
    let text = "hello[MASK]world [SEP]";
    assert_eq!(
        native.tokens(text.as_bytes(), 128).unwrap(),
        reference.encode(text, false).unwrap().get_ids()
    );
}

#[test]
#[ignore = "requires TURBOEMBED_TOKENIZER_JSON pointing to the pinned MiniLM tokenizer"]
fn actual_minilm_unicode_corpus_matches_reference() {
    let path =
        std::env::var("TURBOEMBED_TOKENIZER_JSON").expect("TURBOEMBED_TOKENIZER_JSON is required");
    let bytes = std::fs::read(path).unwrap();
    let config: Value = serde_json::from_slice(&bytes).unwrap();
    let native = Native::load(&config);
    let mut reference = reference(&config);
    reference.with_padding(None);
    reference.with_truncation(None).unwrap();
    for text in [
        "hello world",
        "Straße Ø Æ Ð Ł ı",
        "Café Cafe\u{301} İ",
        "ΑΩ Σς ЖЯ",
        "각",
        "hello\0world",
        "hello\u{200d}world\u{e000}",
        "a€b — 中中文𠀀",
        "😀hello",
        "[MASK]hello[SEP]world",
    ] {
        assert_eq!(
            native.tokens(text.as_bytes(), 512).unwrap(),
            reference.encode(text, false).unwrap().get_ids(),
            "{text:?}"
        );
    }
    // Every Unicode scalar between known ASCII tokens exposes category/cleaning
    // differences without relying on an unknown standalone character's vector.
    let mut text = String::new();
    for start in (0..=0x10ffff).step_by(256) {
        text.clear();
        for cp in start..=(start + 255).min(0x10ffff) {
            if let Some(ch) = char::from_u32(cp) {
                text.push('a');
                text.push(ch);
                text.push('b');
                text.push(' ');
            }
        }
        let expected = reference.encode(text.as_str(), false).unwrap();
        assert_eq!(
            native.tokens(text.as_bytes(), 4096).unwrap(),
            expected.get_ids(),
            "Unicode block U+{start:04X}"
        );
    }
}

#[test]
fn count_only_ignores_output_capacity() {
    let native = Native::load(&config());
    let text = b"hello world hello world";
    for cap in [0, 1, usize::MAX] {
        let mut count = 0;
        let status = unsafe {
            wordpiece_tokenize(
                native.0,
                text.as_ptr(),
                text.len(),
                std::ptr::null_mut(),
                cap,
                4,
                &mut count,
            )
        };
        assert_eq!(status, 0);
        assert_eq!(count, 4);
    }
}

#[test]
fn explicit_text_vocabulary_rejects_invalid_utf8() {
    let path = std::env::temp_dir().join(format!(
        "turboembed-invalid-vocab-{}.txt",
        std::process::id()
    ));
    std::fs::write(&path, b"[PAD]\n[UNK]\n[CLS]\n[SEP]\ninvalid\xff\n").unwrap();
    let path_c = CString::new(path.to_str().unwrap()).unwrap();
    let mut raw = std::ptr::null_mut();
    let status = unsafe { wordpiece_vocab_load(path_c.as_ptr(), &mut raw) };
    std::fs::remove_file(path).unwrap();
    assert_eq!(status, 1);
    assert!(raw.is_null());
}

#[test]
fn i64_rows_honor_stride_and_preserve_output_canaries() {
    let native = Native::load(&config());
    let mut ids = [-777i64; 11];
    let mut mask = ids;
    let mut types = ids;
    let mut pos = ids;
    let text = b"hello world";
    let status = unsafe {
        wordpiece_encode_sentence(
            native.0,
            text.as_ptr(),
            text.len(),
            ids.as_mut_ptr().cast(),
            mask.as_mut_ptr().cast(),
            types.as_mut_ptr().cast(),
            pos.as_mut_ptr().cast(),
            7,
            10,
            8,
        )
    };
    assert_eq!(status, 0);
    assert_eq!(&ids[..7], &[2, 5, 6, 3, 0, 0, 0]);
    assert_eq!(&mask[..7], &[1, 1, 1, 1, 0, 0, 0]);
    assert_eq!(&types[..7], &[0; 7]);
    assert_eq!(&pos[..7], &[0, 1, 2, 3, 4, 5, 6]);
    for row in [ids, mask, types, pos] {
        assert_eq!(&row[7..10], &[0; 3]);
        assert_eq!(row[10], -777);
    }
}

#[test]
fn duplicate_json_keys_and_missing_specials_are_rejected() {
    let cfg = config();
    let mut missing = cfg.clone();
    missing["model"]["vocab"]
        .as_object_mut()
        .unwrap()
        .remove("[UNK]");
    assert!(Native::load_bytes(&serde_json::to_vec(&missing).unwrap()).is_err());
    let text = serde_json::to_string(&cfg).unwrap();
    let duplicate = text.replacen(
        "\"type\":\"WordPiece\"",
        "\"type\":\"BPE\",\"type\":\"WordPiece\"",
        1,
    );
    assert_ne!(duplicate, text);
    assert!(Native::load_bytes(duplicate.as_bytes()).is_err());
}
