//! Spans that cut a surrogate pair. OpenNLP's `SimpleTokenizer` classifies
//! UTF-16 units: the two halves of U+1F600 are each `OTHER` and differ
//! from each other, so it makes a token of each. The hand written
//! `tests/fixtures/smile.golden`, for the input `Smile 😀`
//! (`tests/fixtures/inputs/test/smile.txt`, 10 UTF-8 bytes, 8 UTF-16
//! units), holds what the generator writes for that: both token rows
//! widened to the character's bytes 6..10, and `turbo:utf16<opennlp:tokens>`
//! with the JVM's UTF-16 spans 6..7 and 7..8, each marked as a cut.

use std::path::PathBuf;

use turbo_analysis::{Document, Key, Provenance, Span, Spans, ViewId};
use turbo_analysis_layers::{Sentences, Tokens, Utf16Spans};
use turbo_goldens::format::ColumnType;
use turbo_goldens::*;

const TEXT: &str = "Smile 😀";

fn fixtures() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures")
}

fn fixture_text() -> String {
    std::fs::read_to_string(fixtures().join("smile.golden")).unwrap()
}

fn golden() -> Golden {
    Golden::read(fixtures().join("smile.golden")).unwrap()
}

/// The fixture with `from` replaced by `to` (which must occur), parsed.
fn parse_with_change(from: &str, to: &str) -> Result<Golden, GoldenError> {
    let text = fixture_text();
    assert!(text.contains(from), "{from:?} is not in the fixture");
    Golden::parse(&text.replacen(from, to, 1))
}

fn tokens(spans: &[(u32, u32)]) -> Tokens {
    let mut s = Spans::new(ViewId::SOURCE);
    for &(a, b) in spans {
        s.push(Span::new(a, b));
    }
    Tokens { spans: s, sentence: None }
}

/// A document with these tokens and, when given, their UTF-16 spans.
fn doc(token_spans: &[(u32, u32)], units: Option<Utf16Spans<Tokens>>) -> Document {
    let mut doc = Document::new(TEXT);
    doc.add(&Key::DEFAULT, tokens(token_spans), Provenance::by("test")).unwrap();
    if let Some(u) = units {
        doc.add(&Key::DEFAULT, u, Provenance::by("test")).unwrap();
    }
    doc
}

/// The JVM's tokens, as the golden has them.
const JVM_TOKENS: [(u32, u32); 3] = [(0, 5), (6, 10), (6, 10)];

fn jvm_units() -> Utf16Spans<Tokens> {
    Utf16Spans::new(vec![0, 6, 7], vec![5, 7, 8], vec![false, true, true])
}

// ---------------------------------------------------------------------
// Reading and writing

#[test]
fn the_utf16_layer_reads_back_with_the_cut_rows() {
    let g = golden();
    assert_eq!(g.input_text(fixtures()).unwrap(), TEXT);
    let t = g.layer("opennlp:tokens").unwrap();
    // Two rows over the same bytes: the halves of one character, widened.
    assert_eq!(t.u32s("start").unwrap(), [0, 6, 6]);
    assert_eq!(t.u32s("end").unwrap(), [5, 10, 10]);
    let u = g.layer("turbo:utf16<opennlp:tokens>").unwrap();
    assert_eq!(u.anchor, GoldenAnchor::Aligned("opennlp:tokens".into()));
    assert_eq!(u.column_names(), ["start16:u32", "end16:u32", "cut:u32"]);
    assert_eq!(u.u32s("start16").unwrap(), [0, 6, 7]);
    assert_eq!(u.u32s("end16").unwrap(), [5, 7, 8]);
    assert_eq!(u.u32s("cut").unwrap(), [0, 1, 1]);
}

#[test]
fn the_utf16_layer_writes_back_byte_for_byte() {
    let g = golden();
    let text = g.to_text().unwrap();
    assert_eq!(text, fixture_text());
    assert_eq!(Golden::parse(&text).unwrap(), g);
}

#[test]
fn the_shapes_come_from_the_layer_type() {
    let columns = [
        ColumnSpec::new("start16", ColumnType::U32),
        ColumnSpec::new("end16", ColumnType::U32),
        ColumnSpec::new("cut", ColumnType::U32),
    ];
    let t = shape_of::<Utf16Spans<Tokens>>();
    assert_eq!(t.id, "turbo:utf16<opennlp:tokens>");
    assert_eq!(t.anchor, GoldenAnchor::Aligned("opennlp:tokens".into()));
    assert_eq!(t.columns, columns);
    let s = shape_of::<Utf16Spans<Sentences>>();
    assert_eq!(s.id, "turbo:utf16<opennlp:sentences>");
    assert_eq!(s.anchor, GoldenAnchor::Aligned("opennlp:sentences".into()));
    assert_eq!(s.columns, columns);
    let known = known_shapes();
    assert!(known.contains(&t) && known.contains(&s));
}

#[test]
fn the_reader_checks_the_utf16_layer_against_its_shape() {
    let e = parse_with_change("\tcut:u32", "\tcut:str").unwrap_err();
    assert!(matches!(e, GoldenError::Columns { line: 13, .. }), "{e:?}");
    assert!(e.to_string().contains("expected [start16:u32, end16:u32, cut:u32]"), "{e}");
    let e = parse_with_change("\taligned:opennlp:tokens\t3", "\ttext\t3").unwrap_err();
    assert!(matches!(e, GoldenError::Anchor { line: 13, .. }), "{e:?}");
    let e =
        parse_with_change("<opennlp:tokens>\taligned:opennlp:tokens\t3", "<opennlp:tokens>\taligned:opennlp:tokens\t2")
            .unwrap_err();
    assert!(matches!(e, GoldenError::AlignedRows { line: 13, expected: 3, found: 2, .. }), "{e:?}");
}

#[test]
fn the_reader_refuses_a_utf16_span_backwards_or_past_the_input() {
    let e = parse_with_change("6\t7\t1\n", "7\t6\t1\n").unwrap_err();
    assert!(matches!(e, GoldenError::Span { line: 15, .. }), "{e:?}");
    assert!(e.to_string().contains("the UTF-16 span 7..6 ends before it starts"), "{e}");
    // The input is 8 UTF-16 units long: 8 is its end, 9 is past it.
    let e = parse_with_change("7\t8\t1\n", "7\t9\t1\n").unwrap_err();
    assert!(matches!(e, GoldenError::Span { line: 16, .. }), "{e:?}");
    assert!(e.to_string().contains("the UTF-16 span 7..9 ends past the input's 8 UTF-16 units"), "{e}");
}

// ---------------------------------------------------------------------
// Comparing

#[test]
fn one_rust_token_over_the_character_is_a_count_mismatch() {
    // The Rust tokenizer works on characters: one token over U+1F600
    // where the JVM made two. The golden's widened rows show that as one
    // row more, the first past the Rust side being the second half.
    let g = golden();
    let d = doc(&[(0, 5), (6, 10)], None);
    let m = matches_golden::<Tokens>(&d, &g, Tolerance::exact()).unwrap_err();
    assert!(matches!(m, Mismatch::Count { rust: 2, golden: 3, .. }), "{m:?}");
    let shown = m.to_string();
    assert!(shown.contains("opennlp:tokens: 2 items in Rust, 3 in the golden"), "{shown}");
    assert!(shown.contains("golden: bytes 6..10 \"😀\""), "{shown}");
    // The golden keeps the JVM's answer: the two halves, each a cut.
    let u = g.layer("turbo:utf16<opennlp:tokens>").unwrap();
    assert_eq!(u.u32s("cut").unwrap()[1..], [1, 1]);
    // With no UTF-16 layer on the Rust side, that comparison says so.
    let m = matches_golden::<Utf16Spans<Tokens>>(&d, &g, Tolerance::exact()).unwrap_err();
    assert!(matches!(m, Mismatch::MissingLayer { .. }), "{m:?}");
}

#[test]
fn a_cut_inside_the_text_shows_as_the_next_span() {
    // In `a😀b` the JVM makes four tokens and Rust three, so the first
    // difference is the row after the character, not the counts.
    let text = "a😀b";
    let golden_text = format!(
        "golden 1\ngenerator\tg\t1\nopennlp\t0123456789abcdef0123456789abcdef01234567\tv\njdk\t21\t21+1\tV\n\
         unicode\t15.0\ninput\ttest/x\t6\t4\t{}\nannotator\tC\tm\n\
         layer\topennlp:tokens\ttext\t4\tstart:u32\tend:u32\n0\t1\n1\t5\n1\t5\n5\t6\nend\topennlp:tokens\n\
         layer\tturbo:utf16<opennlp:tokens>\taligned:opennlp:tokens\t4\tstart16:u32\tend16:u32\tcut:u32\n\
         0\t1\t0\n1\t2\t1\n2\t3\t1\n3\t4\t0\nend\tturbo:utf16<opennlp:tokens>\n",
        hex::to_hex(&sha256(text.as_bytes()))
    );
    let g = Golden::parse(&golden_text).unwrap();
    let mut d = Document::new(text);
    d.add(&Key::DEFAULT, tokens(&[(0, 1), (1, 5), (5, 6)]), Provenance::by("test")).unwrap();
    let m = matches_golden::<Tokens>(&d, &g, Tolerance::exact()).unwrap_err();
    assert!(matches!(m, Mismatch::Item { index: 2, what: "span", .. }), "{m:?}");
    let shown = m.to_string();
    assert!(shown.contains("rust:   bytes 5..6 \"b\""), "{shown}");
    assert!(shown.contains("golden: bytes 1..5 \"😀\""), "{shown}");
}

#[test]
fn rust_utf16_spans_that_mirror_the_jvm_match() {
    let g = golden();
    let d = doc(&JVM_TOKENS, Some(jvm_units()));
    assert_matches_golden::<Tokens>(&d, &g, Tolerance::exact());
    assert_matches_golden::<Utf16Spans<Tokens>>(&d, &g, Tolerance::exact());
}

#[test]
fn a_utf16_span_or_cut_that_differs_is_reported_with_its_token() {
    let g = golden();
    let mut u = jvm_units();
    u.end16[1] = 8;
    let m = matches_golden::<Utf16Spans<Tokens>>(&doc(&JVM_TOKENS, Some(u)), &g, Tolerance::exact()).unwrap_err();
    assert!(matches!(m, Mismatch::Item { index: 1, what: "UTF-16 span", .. }), "{m:?}");
    let shown = m.to_string();
    for part in [
        "turbo:utf16<opennlp:tokens> item 1: the UTF-16 span differs",
        "rust:   bytes 6..10 \"😀\", UTF-16 units 6..8, cut",
        "golden: bytes 6..10 \"😀\", UTF-16 units 6..7, cut",
    ] {
        assert!(shown.contains(part), "{part:?} is not in the report:\n{shown}");
    }

    let mut u = jvm_units();
    u.cut[2] = false;
    let m = matches_golden::<Utf16Spans<Tokens>>(&doc(&JVM_TOKENS, Some(u)), &g, Tolerance::exact()).unwrap_err();
    assert!(matches!(m, Mismatch::Item { index: 2, what: "cut", .. }), "{m:?}");
    let shown = m.to_string();
    assert!(shown.contains("rust:   bytes 6..10 \"😀\", UTF-16 units 7..8, not cut"), "{shown}");
    assert!(shown.contains("golden: bytes 6..10 \"😀\", UTF-16 units 7..8, cut"), "{shown}");

    // A golden with fewer rows: the count, as for any layer.
    let mut d = doc(&JVM_TOKENS[..2], Some(Utf16Spans::new(vec![0, 6], vec![5, 7], vec![false, true])));
    let m = matches_golden::<Utf16Spans<Tokens>>(&d, &g, Tolerance::exact()).unwrap_err();
    assert!(matches!(m, Mismatch::Count { rust: 2, golden: 3, .. }), "{m:?}");
    d = doc(&JVM_TOKENS, Some(jvm_units()));
    assert!(matches_golden::<Utf16Spans<Tokens>>(&d, &g, Tolerance::abs(1.0)).is_ok());
}

#[test]
fn a_golden_cut_other_than_0_or_1_is_refused() {
    let g = parse_with_change("7\t8\t1\n", "7\t8\t2\n").unwrap();
    let d = doc(&JVM_TOKENS, Some(jvm_units()));
    let m = matches_golden::<Utf16Spans<Tokens>>(&d, &g, Tolerance::exact()).unwrap_err();
    assert!(matches!(m, Mismatch::BadGolden { .. }), "{m:?}");
    assert!(m.to_string().contains("row 2 has cut 2, expected 0 or 1"), "{m}");
}

#[test]
fn sentences_have_a_utf16_layer_too() {
    // A sentence that ends between the halves of U+1F600, as a sentence
    // detector reading UTF-16 units could make it: widened to bytes 0..10.
    let text = fixture_text()
        .replace("annotator\topennlp.tools.tokenize.SimpleTokenizer\ttokenizePos\n", "annotator\tC\tm\n")
        .replace(
            "layer\topennlp:tokens",
            "layer\topennlp:sentences\ttext\t1\tstart:u32\tend:u32\n0\t10\nend\topennlp:sentences\n\
             layer\tturbo:utf16<opennlp:sentences>\taligned:opennlp:sentences\t1\tstart16:u32\tend16:u32\tcut:u32\n\
             0\t7\t1\nend\tturbo:utf16<opennlp:sentences>\nlayer\topennlp:tokens",
        );
    let g = Golden::parse(&text).unwrap();
    assert_eq!(g.to_text().unwrap(), text);
    let mut d = doc(&JVM_TOKENS, None);
    let mut spans = Spans::new(ViewId::SOURCE);
    spans.push(Span::new(0, 10));
    d.add(&Key::DEFAULT, Sentences { spans }, Provenance::by("test")).unwrap();
    d.add(&Key::DEFAULT, Utf16Spans::<Sentences>::new(vec![0], vec![7], vec![true]), Provenance::by("test")).unwrap();
    assert_matches_golden::<Sentences>(&d, &g, Tolerance::exact());
    assert_matches_golden::<Utf16Spans<Sentences>>(&d, &g, Tolerance::exact());
}
