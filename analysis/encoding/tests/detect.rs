//! Detection accuracy against chardetng's own tests, and the
//! `turbo:encoding` layer that `open` records.
//!
//! `tests/data/chardetng_cases.tsv` holds the bytes and the expected
//! encoding of every chardetng test that goes through its `check`
//! helpers; the tests that drive the detector by hand are written out
//! below.

use turbo_analysis::text::Side;
use turbo_analysis::{Key, Span, ViewId};
use turbo_encoding::{Encoding, EncodingDetector, EncodingEvidence, OPEN_ID, SourceEncoding, detect, open, settle};

fn by_name(name: &str) -> Encoding {
    Encoding::ALL.into_iter().find(|e| e.name() == name).unwrap_or_else(|| panic!("no encoding {name}"))
}

fn unhex(hex: &str) -> Vec<u8> {
    (0..hex.len()).step_by(2).map(|i| u8::from_str_radix(&hex[i..i + 2], 16).unwrap()).collect()
}

fn cases() -> Vec<(String, Encoding, Vec<u8>)> {
    let data = include_str!("data/chardetng_cases.tsv");
    data.lines()
        .filter(|l| !l.starts_with('#') && !l.is_empty())
        .map(|l| {
            let f: Vec<&str> = l.split('\t').collect();
            (f[0].to_string(), by_name(f[1]), unhex(f[2]))
        })
        .collect()
}

#[test]
fn every_chardetng_case_is_guessed() {
    let cases = cases();
    assert_eq!(cases.len(), 82);
    let mut wrong = Vec::new();
    for (name, expected, bytes) in &cases {
        let mut det = EncodingDetector::new();
        det.feed(bytes, true);
        let got = det.guess(false);
        if got != *expected {
            wrong.push(format!("{name}: {got}, expected {expected}"));
        }
    }
    assert!(wrong.is_empty(), "{} of {} wrong:\n{}", wrong.len(), cases.len(), wrong.join("\n"));
}

#[test]
fn the_guess_does_not_depend_on_how_the_input_is_split() {
    for (name, expected, bytes) in cases() {
        for size in [1, 2, 3, 7] {
            let mut det = EncodingDetector::new();
            let chunks: Vec<&[u8]> = bytes.chunks(size).collect();
            for (i, chunk) in chunks.iter().enumerate() {
                det.feed(chunk, i + 1 == chunks.len());
            }
            if chunks.is_empty() {
                det.feed(b"", true);
            }
            assert_eq!(det.guess(false), expected, "{name} in chunks of {size}");
        }
    }
}

#[test]
fn a_lone_curly_apostrophe_is_windows_1252() {
    let mut det = EncodingDetector::new();
    det.feed(b"I\x92", true);
    assert_eq!(det.guess(false), Encoding::Windows1252);
}

#[test]
fn the_numero_sign_is_found_however_it_is_fed() {
    // chardetng's streaming tests: "n.\xBA1" (n.º1) split in five ways.
    let feeds: [&[&[u8]]; 5] = [
        &[b"n", b".", b"\xBA", b"1"],
        &[b"n.", b"\xBA", b"1"],
        &[b" n", b".", b"\xBA", b"1"],
        &[b"n", b".\xBA", b"1"],
        &[b"rrn.", b"\xBA", b"1"],
    ];
    for feed in feeds {
        let mut det = EncodingDetector::new();
        for (i, part) in feed.iter().enumerate() {
            det.feed(part, i + 1 == feed.len());
        }
        assert_eq!(det.guess(false), Encoding::Windows1252, "{feed:?}");
    }
}

#[test]
fn empty_input_is_windows_1252_with_no_non_ascii() {
    let mut det = EncodingDetector::new();
    assert!(!det.feed(b"", true));
    assert_eq!(det.guess(false), Encoding::Windows1252);
    let d = detect(b"");
    assert_eq!((d.encoding, d.non_ascii), (Encoding::Utf8, false));
}

#[test]
#[should_panic(expected = "already fed its last bytes")]
fn feeding_after_the_last_bytes_panics() {
    let mut det = EncodingDetector::new();
    det.feed(b"a", true);
    det.feed(b"b", true);
}

#[test]
fn utf8_is_guessed_only_when_allowed() {
    let bytes = "Grüße aus Köln".as_bytes();
    let mut det = EncodingDetector::new();
    assert!(det.feed(bytes, true));
    assert_eq!(det.guess(true), Encoding::Utf8);
    assert_eq!(det.guess(false), Encoding::Windows1252);
    // That windows-1252 is a policy, not a finding.
    assert_eq!(det.detection(false).confidence, 0.0);
    let d = detect(bytes);
    assert_eq!((d.encoding, d.confidence, d.non_ascii), (Encoding::Utf8, 1.0, true));
}

#[test]
fn confidence_is_in_range_and_follows_the_margin() {
    for (name, _, bytes) in cases() {
        let d = detect(&bytes);
        assert!((0.0..=1.0).contains(&d.confidence), "{name}: {}", d.confidence);
    }
    // A long Russian text in windows-1251 is far ahead of any other
    // reading; two bytes are barely ahead.
    let long = detect(b"\xDD\xF2\xEE \xF2\xE5\xF1\xF2 \xEA\xEE\xE4\xE8\xF0\xEE\xE2\xEA\xE8 \xF1\xE8\xEC\xE2\xEE\xEB\xEE\xE2. \xDD\xF2\xEE \xE5\xF9\xB8 \xEE\xE4\xED\xE0 \xF1\xF2\xF0\xEE\xEA\xE0.");
    assert_eq!(long.encoding, Encoding::Windows1251);
    let short = detect(b"\xE0\xE1");
    assert!(long.confidence > 0.5, "{}", long.confidence);
    assert!(long.confidence > short.confidence);
    // ASCII with an ISO-2022-JP escape is decided by validity.
    let jis = detect(b"\x1B$B$3$l\x1B(B");
    assert_eq!((jis.encoding, jis.confidence), (Encoding::Iso2022Jp, 1.0));
}

#[test]
fn settle_prefers_a_byte_order_mark_then_a_declaration_then_detection() {
    let s = settle(b"\xFF\xFEa\x00", Some(Encoding::ShiftJis));
    assert_eq!((s.encoding, s.evidence, s.confidence), (Encoding::Utf16Le, EncodingEvidence::Bom, 1.0));
    let s = settle(b"\xE0\xE1", Some(Encoding::ShiftJis));
    assert_eq!((s.encoding, s.evidence), (Encoding::ShiftJis, EncodingEvidence::Declared));
    assert_eq!(settle(b"plain", None).evidence, EncodingEvidence::Ascii);
    assert_eq!(settle("café".as_bytes(), None).evidence, EncodingEvidence::ValidUtf8);
    let s = settle(b"caf\xE9 cr\xE8me", None);
    assert_eq!((s.encoding, s.evidence), (Encoding::Windows1252, EncodingEvidence::Detected));
}

#[test]
fn open_records_the_encoding_and_keeps_the_bytes_below_the_text() {
    let bytes = b"\x93Caf\xE9\x94 au lait".to_vec();
    let doc = open(bytes.clone(), None).unwrap();
    assert_eq!(doc.source(), "“Café” au lait");
    assert_eq!(doc.bytes(), &bytes[..]);
    let layer: &SourceEncoding = doc.get_key(&Key::<SourceEncoding>::DEFAULT).unwrap();
    assert_eq!((layer.encoding, layer.evidence), (Encoding::Windows1252, EncodingEvidence::Detected));
    assert_eq!(doc.provenance(Key::<SourceEncoding>::DEFAULT.id()).unwrap().annotator, OPEN_ID);
    // "Café" is bytes 1..5.
    let cafe = doc.source().find("Café").unwrap() as u32;
    assert_eq!(doc.to_source_bytes(ViewId::SOURCE, Span::new(cafe, cafe + 5)), Span::new(1, 5));
    assert_eq!(doc.offset_to_source_bytes(ViewId::SOURCE, doc.source().len() as u32, Side::End), bytes.len() as u32);
}
