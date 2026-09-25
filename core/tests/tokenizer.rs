//! The core's tokenizer against upstream `tokenizers` on the upstream
//! all-MiniLM-L6-v2 tokenizer.json, through the C interface.

mod common;

use common::*;
use serde_json::json;
use turbo::*;

/// Texts that reach each part of the pipeline: the parity list, the
/// reference cases, and the corners of normalization and WordPiece.
fn texts() -> Vec<String> {
    let mut t = parity_texts();
    for c in cases().as_array().unwrap() {
        t.push(c["text"].as_str().unwrap().to_owned());
    }
    t.extend(
        [
            "[CLS] special tokens [SEP] in the text[MASK]and [PAD][UNK]",
            "[cls] is not special, [SEP is not either",
            "control\u{0}chars\u{1}and\u{7f}format\u{200b}chars\u{fffd}here\u{feff}",
            "private use \u{e000}\u{f8ff} and a lone combining mark \u{301}",
            // Emoji assigned after the Unicode version upstream's category
            // tables were built from, so both must treat them alike.
            "newer emoji 🥺 🫠 🪿 and older 🙂",
            "İstanbul ǅ ß ΣΑΣ Ǆ ﬃ",
            "CJK extensions 𠀀 𪜀 𫝀 𫠠 𬺰 丽 and compatibility 豈",
            "hiragana ひらがな katakana カタカナ hangul 한국어",
            &"x".repeat(100),
            &"x".repeat(101),
            &format!("{} tail", "é".repeat(101)),
            "unbreakable-word: qzxqzxqzxqzxqzxqzx",
            "punctuation…“quotes”—dashes–«guillemets»¡¿",
            "   ",
            "\t\n\r",
        ]
        .map(str::to_owned),
    );
    t
}

#[test]
fn ids_match_upstream() {
    let f = Fixture::standard("ids-match-upstream");
    let tok = f.open().expect("tokenizer loads");
    let up = upstream();
    for text in texts() {
        let ours = tok.row(&text, None).unwrap();
        assert_eq!(ours, upstream_ids(&up, &text), "text {text:?}");
    }
}

#[test]
fn count_is_untruncated_with_specials() {
    let f = Fixture::standard("count");
    let tok = f.open().unwrap();
    let mut up = upstream();
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
fn rows_are_padded_and_masked() {
    let f = Fixture::standard("padding");
    let tok = f.open().unwrap();
    let texts = ["short", "", "a somewhat longer row of text"];
    let e = tok.encode(&texts, None, 16).unwrap();
    for (i, text) in texts.iter().enumerate() {
        let n = e.lengths[i] as usize;
        let at = i * 16;
        assert_eq!(e.row(i), upstream_ids(&upstream(), text));
        assert!(e.ids[at + n..at + 16].iter().all(|&x| x == 0), "padded with pad id 0");
        assert!(e.mask[at..at + n].iter().all(|&x| x == 1));
        assert!(e.mask[at + n..at + 16].iter().all(|&x| x == 0));
        assert!(e.types[at..at + 16].iter().all(|&x| x == 0));
    }
}

#[test]
fn truncation_follows_the_option() {
    let f = Fixture::standard("truncation");
    let tok = f.open().unwrap();
    let text = long_text();
    let full = {
        let mut up = upstream();
        up.with_truncation(None).unwrap();
        upstream_ids(&up, &text)
    };
    let body = &full[1..full.len() - 1];

    // The bundle says TRUNCATE_RIGHT at max_seq.
    let model = tok.row(&text, None).unwrap();
    assert_eq!(model.len(), MAX_SEQ);
    assert_eq!(&model[1..MAX_SEQ - 1], &body[..MAX_SEQ - 2]);

    let right = tok.row(&text, Some(&options(0, TURBO_TRUNCATE_RIGHT, 32, 0))).unwrap();
    assert_eq!(right, [&[101], &body[..30], &[102]].concat());

    let left = tok.row(&text, Some(&options(0, TURBO_TRUNCATE_LEFT, 32, 0))).unwrap();
    assert_eq!(left, [&[101], &body[body.len() - 30..], &[102]].concat());

    let bare = tok.row(&text, Some(&options(1, TURBO_TRUNCATE_RIGHT, 32, 0))).unwrap();
    assert_eq!(bare, &body[..32]);

    let none = tok.row(&text, Some(&options(0, TURBO_TRUNCATE_NONE, 32, 0))).unwrap_err();
    assert!(none.is(turbo::status::CAPACITY, "TRUNCATE_NONE"), "{none:?}");

    // Asked for more than the bundle's max_seq: the tokenizer cuts where it
    // is told, it does not clamp to the model.
    let wide = tok.row(&text, Some(&options(0, TURBO_TRUNCATE_RIGHT, 512, 0))).unwrap();
    assert_eq!(wide, full);
}

#[test]
fn a_row_wider_than_the_stride_is_capacity() {
    let f = Fixture::standard("stride");
    let tok = f.open().unwrap();
    let e = tok.encode(&["fits", "this one does not fit in eight"], None, 8).unwrap_err();
    assert!(e.is(turbo::status::CAPACITY, "texts[1]"), "{e:?}");
}

#[test]
fn prompt_roles_prepend_the_bundle_prefixes() {
    let mut m = manifest();
    m["embed"]["prefix_query"] = json!("query: ");
    m["embed"]["prefix_document"] = json!("passage: ");
    let f = Fixture::new("prompts", m);
    let tok = f.open().expect("reference ids were made with the prefixes");
    let up = upstream();
    let q = tok.row("reset a password", Some(&options(0, 0, 0, TURBO_PROMPT_QUERY))).unwrap();
    assert_eq!(q, upstream_ids(&up, "query: reset a password"));
    let d = tok.row("reset a password", Some(&options(0, 0, 0, TURBO_PROMPT_DOCUMENT))).unwrap();
    assert_eq!(d, upstream_ids(&up, "passage: reset a password"));
    let n = tok.row("reset a password", Some(&options(0, 0, 0, TURBO_PROMPT_NONE))).unwrap();
    assert_eq!(n, upstream_ids(&up, "reset a password"));
}

#[test]
fn info_describes_the_bundle() {
    let f = Fixture::standard("info");
    let tok = f.open().unwrap();
    let info = tok.info();
    assert_eq!(info.vocab_size, 30522);
    assert_eq!(info.max_seq, MAX_SEQ as u32);
    assert_eq!(info.specials_per_sequence, 2);
    assert_eq!((info.pad_id, info.bos_id, info.eos_id, info.unk_id), (0, 101, 102, 100));
    let s = |b: &[std::ffi::c_char]| {
        b.iter().take_while(|&&c| c != 0).map(|&c| char::from(u8::from_ne_bytes(c.to_ne_bytes()))).collect::<String>()
    };
    assert_eq!(s(&info.kind), "wordpiece");
    let bytes = std::fs::read(upstream_tokenizer_json()).unwrap();
    assert_eq!(s(&info.sha256), sha256_hex(&bytes));
    let manifest = std::fs::read(f.dir.join("manifest.json")).unwrap();
    assert_eq!(s(&info.manifest_sha256), sha256_hex(&manifest));
}

#[test]
fn a_zeroed_options_struct_is_what_the_bundle_says() {
    let f = Fixture::standard("zeroed-options");
    let tok = f.open().unwrap();
    let text = "the bundle's template adds [CLS] and [SEP]";
    let zeroed = options(0, 0, 0, 0);
    assert_eq!(tok.row(text, Some(&zeroed)).unwrap(), tok.row(text, None).unwrap());
    assert_eq!(tok.row(text, None).unwrap(), upstream_ids(&upstream(), text));
}

#[test]
fn count_takes_the_prompt_role() {
    let mut m = manifest();
    m["embed"]["prefix_query"] = json!("query: ");
    m["embed"]["prefix_document"] = json!("passage: ");
    let f = Fixture::new("count-prompts", m);
    let tok = f.open().unwrap();
    let text = "reset a password";
    let count = |role: u32| {
        let mut n = 0u32;
        let mut err = new_error();
        let rc = unsafe { turbo_tokenizer_count(tok.t, common::text(text), role, &mut n, &mut err) };
        (rc, n as usize)
    };
    for role in [TURBO_PROMPT_NONE, TURBO_PROMPT_QUERY, TURBO_PROMPT_DOCUMENT] {
        let row = tok.row(text, Some(&options(0, 0, 0, role))).unwrap();
        assert_eq!(count(role), (0, row.len()), "role {role}");
    }
    assert!(count(TURBO_PROMPT_QUERY).1 > count(TURBO_PROMPT_NONE).1, "the prefix is counted");
    assert_eq!(count(9).0, turbo::status::INVALID_ENUM);
}

#[test]
fn bad_options_are_refused() {
    let f = Fixture::standard("bad-options");
    let tok = f.open().unwrap();
    let e = tok.row("x", Some(&options(2, 0, 0, 0))).unwrap_err();
    assert_eq!((e.code, e.field), (turbo::status::INVALID_ARGUMENT, 1), "{e:?}");
    let e = tok.row("x", Some(&options(0, 9, 0, 0))).unwrap_err();
    assert_eq!(e.code, turbo::status::INVALID_ENUM, "{e:?}");
    let e = tok.row("x", Some(&options(0, 0, 0, 9))).unwrap_err();
    assert_eq!(e.code, turbo::status::INVALID_ENUM, "{e:?}");
    let e = tok.row("x", Some(&options(0, 0, 1, 0))).unwrap_err();
    assert_eq!((e.code, e.field), (turbo::status::INVALID_ARGUMENT, 3), "{e:?}");
    let mut short = options(0, 0, 0, 0);
    short.struct_size -= 4;
    let e = tok.row("x", Some(&short)).unwrap_err();
    assert_eq!(e.code, turbo::status::INVALID_STRUCT_SIZE, "{e:?}");
}

#[test]
fn bad_utf8_and_bad_handles_are_refused() {
    let f = Fixture::standard("bad-input");
    let tok = f.open().unwrap();
    let bad = [0x66u8, 0xff, 0x6f];
    let view = turbo_text { ptr: bad.as_ptr() as *const _, len: 3 };
    let mut ids = [0i32; 8];
    let mut mask = [0i32; 8];
    let mut err = new_error();
    let rc = unsafe {
        turbo_tokenizer_encode(
            tok.t,
            &view,
            1,
            std::ptr::null(),
            ids.as_mut_ptr(),
            mask.as_mut_ptr(),
            std::ptr::null_mut(),
            8,
            std::ptr::null_mut(),
            &mut err,
        )
    };
    assert_eq!(rc, turbo::status::INVALID_UTF8);

    let mut n = 0;
    let rc = unsafe {
        turbo_tokenizer_count(std::ptr::null_mut(), common::text("x"), TURBO_PROMPT_NONE, &mut n, std::ptr::null_mut())
    };
    assert_eq!(rc, turbo::status::INVALID_HANDLE, "a NULL turbo_error is allowed");

    let mut small = new_error();
    small.struct_size = 8;
    let rc = unsafe { turbo_tokenizer_count(tok.t, common::text("x"), TURBO_PROMPT_NONE, &mut n, &mut small) };
    assert_eq!(rc, turbo::status::INVALID_STRUCT_SIZE);
}
