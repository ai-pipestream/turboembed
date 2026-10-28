//! Focused tests for the WNUT-17 kept-corpus reader of issue #1533.
//!
//! Every fixture here is original, written for these tests.

use turbo_formats::NameSample;
use turbo_formats::error::FormatError;
use turbo_formats::ner::{WNUT17_TYPES, WnutError, WnutReader};

/// The sample's shape: whether it clears adaptive data, its token texts
/// and its names.
type Shape<'a> = (bool, Vec<String>, Vec<(u32, u32, Option<&'a str>)>);

fn shape<'a>(s: &NameSample<'a>) -> Shape<'a> {
    (
        s.clear_adaptive_data(),
        s.tokens().iter().map(|t| t.as_str().to_owned()).collect(),
        s.names().iter().map(|n| (n.start, n.end, n.kind)).collect(),
    )
}

fn strings(items: &[&str]) -> Vec<String> {
    items.iter().map(|s| (*s).to_owned()).collect()
}

fn read(source: &str) -> Vec<Result<Shape<'_>, String>> {
    WnutReader::new(source).unwrap().map(|r| r.map(|s| shape(&s)).map_err(|e| e.to_string())).collect()
}

#[test]
fn reads_sentences_and_names() {
    let text = "Sarah\tO\nloves\tO\nManchester\tB-location\nUnited\tI-location\n\nunited\tB-group\nwon\tO\n";
    let samples: Vec<_> = read(text).into_iter().map(|r| r.unwrap()).collect();
    assert_eq!(
        samples,
        [
            (false, strings(&["Sarah", "loves", "Manchester", "United"]), vec![(2, 4, Some("location"))]),
            (false, strings(&["united", "won"]), vec![(0, 1, Some("group"))]),
        ]
    );
    let first = WnutReader::new(text).unwrap().next().unwrap().unwrap();
    assert_eq!(first.names().iter().map(|n| (n.start, n.end, n.kind)).collect::<Vec<_>>(), [(2, 4, Some("location"))]);
    assert!(!first.clear_adaptive_data());
}

#[test]
fn tokens_are_the_sources_bytes_with_multi_byte_characters() {
    // "Zoë" holds a two-byte character, the emoji is a supplementary
    // character of four bytes and two UTF-16 code units.
    let text = "Zoë\tB-person\n\u{1F600}\tO\n";
    let sample = WnutReader::new(text).unwrap().next().unwrap().unwrap();
    let tokens = sample.tokens();
    assert_eq!(tokens.iter().map(|t| t.as_str()).collect::<Vec<_>>(), ["Zoë", "\u{1F600}"]);
    for t in tokens {
        assert!(t.is_borrowed());
        assert_eq!(&text[t.span.range()], t.as_str());
    }
    assert_eq!(&text[sample.source().range()], "Zoë\tB-person\n\u{1F600}\tO");
    assert_eq!(shape(&sample), (false, strings(&["Zoë", "\u{1F600}"]), vec![(0, 1, Some("person"))]));
}

#[test]
fn a_row_of_another_field_count_ends_the_read_at_its_line() {
    let text = "a\tO\nb\n\nc\tO\n";
    assert_eq!(
        read(text),
        [
            Err("Expected two tab-separated fields per line in WNUT-17 data, got 1 for line 'b'!".to_owned()),
            Ok((false, strings(&["c"]), Vec::new())),
        ]
    );
    let err = WnutReader::new("a\tb\tc\n").unwrap().next().unwrap().unwrap_err();
    assert!(matches!(err, WnutError::Fields { line: 1, count: 3, .. }));
}

#[test]
fn bad_tags_fail_after_the_sentence_is_read() {
    let text = "a\tX\nb\tO\n\nc\tO\n";
    assert_eq!(read(text), [Err("Invalid tag: X".to_owned()), Ok((false, strings(&["c"]), Vec::new()))]);
    let text = "a\to\n\nb\tO\n";
    assert_eq!(read(text)[0], Err("Invalid tag: o".to_owned()));
    // a name tag with an empty type is an unknown type, as in GermEval
    let text = "a\tB-\n\nb\tO\n";
    assert_eq!(read(text)[0], Err("Unknown type: ".to_owned()));
    // an unknown type is found when its name ends
    let text = "a\tB-widget\nb\tO\n\nc\tO\n";
    assert_eq!(read(text), [Err("Unknown type: widget".to_owned()), Ok((false, strings(&["c"]), Vec::new()))]);
    let err = WnutReader::new("a\tB-widget\n").unwrap().next().unwrap().unwrap_err();
    assert!(matches!(err, WnutError::UnknownType { line: 1, kind } if kind == "widget"));
}

#[test]
fn i_tags_extend_only_a_matching_open_name() {
    let text = "a\tO\nb\tB-person\nc\tI-person\nd\tO\n";
    assert_eq!(read(text), [Ok((false, strings(&["a", "b", "c", "d"]), vec![(1, 3, Some("person"))]))]);
    let err = WnutReader::new("a\tI-person\n").unwrap().next().unwrap().unwrap_err();
    assert!(matches!(err, WnutError::Sample { source: FormatError::UnexpectedNameContinuation { .. }, .. }));
    let err = WnutReader::new("a\tB-person\nb\tI-group\n").unwrap().next().unwrap().unwrap_err();
    assert!(matches!(err, WnutError::Sample { source: FormatError::NameTypeMismatch { .. }, .. }));
}

#[test]
fn every_type_of_the_task_is_kept_as_written() {
    assert_eq!(WNUT17_TYPES, ["corporation", "creative-work", "group", "location", "person", "product"]);
    let text = "a\tB-corporation\nb\tB-creative-work\nc\tB-group\n";
    let sample = WnutReader::new(text).unwrap().next().unwrap().unwrap();
    let kinds: Vec<_> = sample.names().iter().map(|n| n.kind.unwrap()).collect();
    assert_eq!(kinds, ["corporation", "creative-work", "group"]);
}

#[test]
fn empty_text_gives_no_samples_and_a_sentence_at_the_end_is_read() {
    assert!(read("").is_empty());
    assert_eq!(read("a\tO"), [Ok((false, strings(&["a"]), Vec::new()))]);
    assert_eq!(read("\n\na\tO\n\n"), [Ok((false, strings(&["a"]), Vec::new()))]);
}
