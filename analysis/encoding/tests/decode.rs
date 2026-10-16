//! Decoding with a byte view below the text: every token of a decoded
//! document maps back to the bytes that encode it, every replacement
//! character maps to the bytes it replaced, and the edge cases (byte
//! order marks, empty input, a sequence cut off at the end) hold exact
//! offsets.
//!
//! The encoded samples were produced with the encoding_rs crate's
//! encoders, an implementation of the same standard independent of these
//! decoders.

use turbo_analysis::text::{Edit, Side};
use turbo_analysis::{Document, Span, ViewId};
use turbo_encoding::{Decoded, Encoding, decode, decode_without_bom_handling};

const V0: ViewId = ViewId::SOURCE;

fn doc(bytes: &[u8], e: Encoding) -> (Document, Decoded) {
    let d = decode(bytes, e).unwrap();
    (d.clone().into_document(bytes).unwrap(), d)
}

/// Words (runs of alphanumerics) and single other non-space characters.
fn tokens(text: &str) -> Vec<Span> {
    let mut out = Vec::new();
    let mut word: Option<usize> = None;
    for (i, c) in text.char_indices() {
        if c.is_alphanumeric() {
            word.get_or_insert(i);
            continue;
        }
        if let Some(start) = word.take() {
            out.push(Span::new(start as u32, i as u32));
        }
        if !c.is_whitespace() {
            out.push(Span::new(i as u32, (i + c.len_utf8()) as u32));
        }
    }
    if let Some(start) = word {
        out.push(Span::new(start as u32, text.len() as u32));
    }
    out
}

/// Every token maps to a byte range that decodes on its own to exactly
/// the token, and that byte range maps back to exactly the token.
fn check_tokens(bytes: &[u8], e: Encoding, expected: &str) {
    let (doc, d) = doc(bytes, e);
    assert_eq!(d.text, expected, "{e}");
    assert_eq!(d.errors, 0, "{e}");
    let toks = tokens(&d.text);
    assert!(toks.len() >= 3, "{e}: {toks:?}");
    let mut prev_end = 0;
    for t in toks {
        let b = doc.to_source_bytes(V0, t);
        assert!(b.start >= prev_end, "{e}: token byte ranges overlap");
        prev_end = b.end;
        let alone = decode_without_bom_handling(&bytes[b.range()], e).unwrap();
        assert_eq!(alone.text, &d.text[t.range()], "{e}: token {t:?} from bytes {b:?}");
        assert_eq!(doc.from_source_bytes(V0, b), t, "{e}");
        assert_eq!(doc.offset_to_source_bytes(V0, t.start, Side::Start), b.start);
        assert_eq!(doc.offset_to_source_bytes(V0, t.end, Side::End), b.end);
        assert_eq!(doc.offset_from_source_bytes(V0, b.start, Side::Start), t.start);
        assert_eq!(doc.offset_from_source_bytes(V0, b.end, Side::End), t.end);
    }
}

#[test]
fn windows_1252_tokens_map_to_their_bytes() {
    let bytes = b"Caf\xE9 cr\xE8me \x96 \x93na\xEFve\x94 r\xE9sum\xE9, 5\x80.";
    check_tokens(bytes, Encoding::Windows1252, "Café crème – “naïve” résumé, 5€.");
    // "Café" is four bytes in the file and five in the text.
    let (doc, _) = doc(bytes, Encoding::Windows1252);
    assert_eq!(doc.to_source_bytes(V0, Span::new(0, 5)), Span::new(0, 4));
    // The en dash is one byte for three.
    assert_eq!(&doc.source()[13..16], "–");
    assert_eq!(doc.to_source_bytes(V0, Span::new(13, 16)), Span::new(11, 12));
}

#[test]
fn shift_jis_tokens_map_to_their_bytes() {
    let bytes = b"\x82\xB1\x82\xEA\x82\xCD\x95\xB6\x8E\x9A\x83\x52\x81\x5B\x83\x68\x82\xCC\x83\x65\x83\x58\x83\x67\x82\xC5\x82\xB7\x81\x42\xB6\xC0\xB6\xC5\x20\x41\x42\x43\x20\x31\x32\x33";
    check_tokens(bytes, Encoding::ShiftJis, "これは文字コードのテストです。ｶﾀｶﾅ ABC 123");
}

#[test]
fn euc_kr_tokens_map_to_their_bytes() {
    let bytes = b"\xC0\xCC\xB0\xCD\xC0\xBA\x20\xB9\xAE\xC0\xDA\x20\xC0\xCE\xC4\xDA\xB5\xF9\x20\xC5\xD7\xBD\xBA\xC6\xAE\xC0\xD4\xB4\xCF\xB4\xD9\x2E\x20\x41\x42\x43";
    check_tokens(bytes, Encoding::EucKr, "이것은 문자 인코딩 테스트입니다. ABC");
}

#[test]
fn utf16_tokens_map_to_their_bytes() {
    let le = b"\x47\x00\x72\x00\xFC\x00\xDF\x00\x65\x00\x2C\x00\x20\x00\x16\x4E\x4C\x75\x21\x00\x20\x00\x3D\xD8\x00\xDE\x20\x00\x6F\x00\x6B\x00";
    check_tokens(le, Encoding::Utf16Le, "Grüße, 世界! 😀 ok");
    let be = b"\x00\x47\x00\x72\x00\xFC\x00\xDF\x00\x65\x00\x2C\x00\x20\x4E\x16\x75\x4C\x00\x21\x00\x20\xD8\x3D\xDE\x00\x00\x20\x00\x6F\x00\x6B";
    check_tokens(be, Encoding::Utf16Be, "Grüße, 世界! 😀 ok");
    // The emoji is a surrogate pair: four bytes, one edit.
    let (doc, _) = doc(le, Encoding::Utf16Le);
    let emoji = doc.source().find('😀').unwrap() as u32;
    assert_eq!(doc.to_source_bytes(V0, Span::new(emoji, emoji + 4)), Span::new(22, 26));
}

#[test]
fn the_other_cjk_encodings_map_tokens_to_their_bytes() {
    check_tokens(
        b"\xA4\xB3\xA4\xEC\xA4\xCF\xCA\xB8\xBB\xFA\xA5\xB3\xA1\xBC\xA5\xC9\xA4\xCE\xA5\xC6\xA5\xB9\xA5\xC8\xA4\xC7\xA4\xB9\xA1\xA3\x8E\xB6\x8E\xC0\x8E\xB6\x8E\xC5",
        Encoding::EucJp,
        "これは文字コードのテストです。ｶﾀｶﾅ",
    );
    check_tokens(
        b"\xD5\xE2\xCA\xC7\xD2\xBB\xB8\xF6\xD7\xD6\xB7\xFB\xB1\xE0\xC2\xEB\xB2\xE2\xCA\xD4\xA1\xA3\x41\x42\x43",
        Encoding::Gbk,
        "这是一个字符编码测试。ABC",
    );
    check_tokens(
        b"\xD5\xE2\xCA\xC7\xA2\xE3\xD2\xBB\xB8\xF6\x94\x39\xFC\x36\xB2\xE2\xCA\xD4",
        Encoding::Gb18030,
        "这是€一个😀测试",
    );
    check_tokens(
        b"\xB3\x6F\xAC\x4F\xA4\x40\xAD\xD3\xA6\x72\xB2\xC5\xBD\x73\xBD\x58\xB4\xFA\xB8\xD5\xA1\x43\x41\x42\x43",
        Encoding::Big5,
        "這是一個字符編碼測試。ABC",
    );
    check_tokens(
        b"\xFC\xD4\xCF\x20\xD4\xC5\xD3\xD4\x20\xCB\xCF\xC4\xC9\xD2\xCF\xD7\xCB\xC9\x2E",
        Encoding::Koi8R,
        "Это тест кодировки.",
    );
}

#[test]
fn iso_2022_jp_escapes_belong_to_no_token() {
    let bytes = b"\x1B\x24\x42\x24\x33\x24\x6C\x24\x4F\x1B\x28\x42\x20\x41\x42\x43\x20\x1B\x24\x42\x25\x46\x25\x39\x25\x48\x1B\x28\x42";
    let (doc, d) = doc(bytes, Encoding::Iso2022Jp);
    assert_eq!(d.text, "これは ABC テスト");
    // Each escape sequence is a deletion of its three bytes.
    let deletions: Vec<Span> = d.alignment.edits().iter().filter(|e| e.child.is_empty()).map(|e| e.parent).collect();
    assert_eq!(deletions, [Span::new(0, 3), Span::new(9, 12), Span::new(17, 20), Span::new(26, 29)]);
    // A token maps to its character bytes, without the escapes around it.
    let tesuto = doc.source().find("テスト").unwrap() as u32;
    assert_eq!(doc.to_source_bytes(V0, Span::new(tesuto, tesuto + 9)), Span::new(20, 26));
    assert_eq!(doc.to_source_bytes(V0, Span::new(10, 13)), Span::new(13, 16));
}

/// Input bytes, encoding, expected text, and the byte span of each
/// expected replacement character.
type ReplacementCase = (&'static [u8], Encoding, &'static str, &'static [(u32, u32)]);

/// The parent spans of the replacement characters a decode wrote, in order.
fn replacements(d: &Decoded) -> Vec<Span> {
    d.alignment.edits().iter().filter(|e| &d.text[e.child.range()] == "\u{FFFD}").map(|e| e.parent).collect()
}

#[test]
fn replacement_characters_map_to_the_bytes_they_replaced() {
    let cases: &[ReplacementCase] = &[
        // UTF-8: one replacement per maximal subpart.
        (b"a\xE2\x82b\xF0\x9F\x98", Encoding::Utf8, "a\u{FFFD}b\u{FFFD}", &[(1, 3), (4, 7)]),
        (b"\xC0\xAF", Encoding::Utf8, "\u{FFFD}\u{FFFD}", &[(0, 1), (1, 2)]),
        (b"\xED\xA0\x80", Encoding::Utf8, "\u{FFFD}\u{FFFD}\u{FFFD}", &[(0, 1), (1, 2), (2, 3)]),
        // A lead byte before ASCII: the ASCII byte is its own character.
        (b"\x82A", Encoding::ShiftJis, "\u{FFFD}A", &[(0, 1)]),
        (b"\x82\xFF", Encoding::ShiftJis, "\u{FFFD}", &[(0, 2)]),
        (b"\xFFA", Encoding::EucKr, "\u{FFFD}A", &[(0, 1)]),
        (b"\xA1\x20", Encoding::Big5, "\u{FFFD} ", &[(0, 1)]),
        (b"\x8F\xA1\x41", Encoding::EucJp, "\u{FFFD}A", &[(0, 2)]),
        // A four-byte gb18030 sequence broken at its fourth byte: only the
        // first byte is the error, the other three are decoded again.
        (b"\x81\x30\x81\x20", Encoding::Gb18030, "\u{FFFD}0\u{FFFD} ", &[(0, 1), (2, 3)]),
        // UTF-16: an unpaired surrogate, and an odd byte at the end.
        (b"\x3D\xD8A\x00", Encoding::Utf16Le, "\u{FFFD}A", &[(0, 2)]),
        (b"\x00\xDCA\x00", Encoding::Utf16Le, "\u{FFFD}A", &[(0, 2)]),
        (b"A\x00B", Encoding::Utf16Le, "A\u{FFFD}", &[(2, 3)]),
        // ISO-2022-JP: an escape cut off at the end is an error for the
        // escape byte alone.
        (b"A\x1B$", Encoding::Iso2022Jp, "A\u{FFFD}$", &[(1, 2)]),
        (b"\x1B$B\x1B(B", Encoding::Iso2022Jp, "\u{FFFD}", &[(3, 6)]),
        // Unmapped single bytes.
        (b"a\x81b", Encoding::Windows1253, "a\u{81}b", &[]),
        (b"a\xAAb", Encoding::Windows1253, "a\u{FFFD}b", &[(1, 2)]),
        // The replacement encoding: one character for the whole input.
        (b"anything", Encoding::Replacement, "\u{FFFD}", &[(0, 8)]),
    ];
    for &(bytes, e, text, spans) in cases {
        let d = decode_without_bom_handling(bytes, e).unwrap();
        assert_eq!(d.text, text, "{e} {bytes:02X?}");
        let expected: Vec<Span> = spans.iter().map(|&(s, end)| Span::new(s, end)).collect();
        assert_eq!(replacements(&d), expected, "{e} {bytes:02X?}");
        assert_eq!(d.errors as usize, expected.len(), "{e} {bytes:02X?}");
        let doc = d.clone().into_document(bytes).unwrap();
        // Each replacement character maps to its bytes and back.
        for (i, _) in d.text.match_indices('\u{FFFD}') {
            let s = Span::new(i as u32, i as u32 + 3);
            let b = doc.to_source_bytes(V0, s);
            assert!(expected.contains(&b), "{e}: {s:?} -> {b:?}");
            assert_eq!(doc.from_source_bytes(V0, b), s);
        }
    }
}

#[test]
fn a_byte_order_mark_wins_and_is_dropped() {
    let cases: &[(&[u8], Encoding, Encoding, u32)] = &[
        (b"\xEF\xBB\xBFabc", Encoding::Windows1252, Encoding::Utf8, 3),
        (b"\xFF\xFEa\x00b\x00c\x00", Encoding::Utf8, Encoding::Utf16Le, 2),
        (b"\xFE\xFF\x00a\x00b\x00c", Encoding::Utf16Le, Encoding::Utf16Be, 2),
    ];
    for &(bytes, asked, used, bom) in cases {
        let (doc, d) = doc(bytes, asked);
        assert_eq!((d.text.as_str(), d.encoding, d.bom), ("abc", used, true));
        assert_eq!(d.alignment.edits()[0], Edit { parent: Span::new(0, bom), child: Span::new(0, 0) });
        // The text starts after the mark, and the mark maps to no text.
        assert_eq!(doc.to_source_bytes(V0, Span::new(0, 1)).start, bom);
        assert_eq!(doc.from_source_bytes(V0, Span::new(0, bom)), Span::new(0, 0));
    }
    // Without byte order mark handling the mark is a character.
    let d = decode_without_bom_handling(b"\xEF\xBB\xBFabc", Encoding::Utf8).unwrap();
    assert_eq!(d.text, "\u{FEFF}abc");
    // A mark alone is an empty text.
    let (doc, d) = doc(b"\xFF\xFE", Encoding::Utf8);
    assert_eq!(d.text, "");
    assert_eq!(doc.bytes(), b"\xFF\xFE");
}

#[test]
fn empty_input_is_an_empty_text_in_every_encoding() {
    for e in Encoding::ALL {
        let (doc, d) = doc(b"", e);
        assert_eq!(d.text, "", "{e}");
        assert!(d.alignment.edits().is_empty(), "{e}");
        assert_eq!(d.errors, 0, "{e}");
        assert_eq!(doc.to_source_bytes(V0, Span::new(0, 0)), Span::new(0, 0));
    }
}

#[test]
fn a_sequence_cut_off_at_the_end_is_one_replacement_for_its_bytes() {
    // Each sample ends in one character that starts at the given byte.
    let cases: &[(&[u8], Encoding, usize)] = &[
        ("caf\u{E9}".as_bytes(), Encoding::Utf8, 3),
        ("a😀".as_bytes(), Encoding::Utf8, 1),
        (b"a\x82\xB1", Encoding::ShiftJis, 1),
        (b"a\xC0\xCC", Encoding::EucKr, 1),
        (b"a\xA4\xB3", Encoding::EucJp, 1),
        (b"a\x8F\xB0\xA1", Encoding::EucJp, 1),
        (b"a\xD5\xE2", Encoding::Gbk, 1),
        (b"a\x94\x39\xFC\x36", Encoding::Gb18030, 1),
        (b"a\xB3\x6F", Encoding::Big5, 1),
        (b"a\x00\x3D\xD8\x00\xDE", Encoding::Utf16Le, 2),
        (b"\x00a\xD8\x3D\xDE\x00", Encoding::Utf16Be, 2),
    ];
    for &(whole, e, last) in cases {
        let full = decode_without_bom_handling(whole, e).unwrap();
        assert_eq!(full.errors, 0, "{e}");
        let head = decode_without_bom_handling(&whole[..last], e).unwrap().text;
        assert_eq!(full.text.chars().count(), head.chars().count() + 1, "{e}");
        // Every cut inside the last character.
        for cut in last + 1..whole.len() {
            let bytes = &whole[..cut];
            let d = decode_without_bom_handling(bytes, e).unwrap();
            assert_eq!(d.text, format!("{head}\u{FFFD}"), "{e} cut at {cut}");
            assert_eq!(replacements(&d), [Span::new(last as u32, cut as u32)], "{e} cut at {cut}");
            let doc = d.into_document(bytes).unwrap();
            let at = head.len() as u32;
            assert_eq!(doc.to_source_bytes(V0, Span::new(at, at + 3)), Span::new(last as u32, cut as u32));
        }
    }
}

/// A deterministic byte generator for the tests below.
fn noise(seed: u64, len: usize) -> Vec<u8> {
    let mut x = seed | 1;
    (0..len)
        .map(|_| {
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            (x >> 24) as u8
        })
        .collect()
}

#[test]
fn every_edit_decodes_on_its_own_to_its_text() {
    // For every encoding without shift states, an edit is self-contained:
    // its bytes alone decode to exactly its text. This holds for errors
    // too, which is what makes a replacement character's byte range the
    // bytes it replaced.
    for e in Encoding::ALL {
        if matches!(e, Encoding::Iso2022Jp | Encoding::Replacement) {
            continue;
        }
        for seed in 0..200u64 {
            let bytes = noise(seed * 7919 + e as u64, 40);
            let d = decode_without_bom_handling(&bytes, e).unwrap();
            for edit in d.alignment.edits() {
                let alone = decode_without_bom_handling(&bytes[edit.parent.range()], e).unwrap();
                assert_eq!(alone.text, &d.text[edit.child.range()], "{e} {bytes:02X?} edit {edit:?}");
            }
            let doc = d.clone().into_document(bytes.clone()).unwrap();
            assert_eq!(doc.source(), d.text);
        }
    }
}

#[test]
fn single_byte_encodings_map_every_byte_to_one_character() {
    let all: Vec<u8> = (0..=255).collect();
    for e in Encoding::ALL {
        if !e.ascii_compatible()
            || matches!(
                e,
                Encoding::Utf8
                    | Encoding::Gbk
                    | Encoding::Gb18030
                    | Encoding::Big5
                    | Encoding::EucJp
                    | Encoding::ShiftJis
                    | Encoding::EucKr
            )
        {
            continue;
        }
        let (doc, d) = doc(&all, e);
        assert_eq!(d.text.chars().count(), 256, "{e}");
        for (k, (i, c)) in d.text.char_indices().enumerate() {
            let s = Span::new(i as u32, (i + c.len_utf8()) as u32);
            assert_eq!(doc.to_source_bytes(V0, s), Span::new(k as u32, k as u32 + 1), "{e} byte {k}");
        }
    }
}

#[test]
fn decoded_text_matches_the_windows_1252_table_and_x_user_defined() {
    let d = decode(b"\x80\x81\x9F\xFF", Encoding::Windows1252).unwrap();
    assert_eq!(d.text, "\u{20AC}\u{81}\u{178}\u{FF}");
    let d = decode(b"a\x80\xFF", Encoding::XUserDefined).unwrap();
    assert_eq!(d.text, "a\u{F780}\u{F7FF}");
}
