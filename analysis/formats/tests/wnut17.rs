//! Focused tests for the WNUT-17 kept-corpus reader of issue #1533.
//!
//! Every fixture here is original, written for these tests.

use turbo_formats::ner::{WNUT17_TYPES, WnutError, WnutReader};

fn read(source: &str) -> Vec<Result<String, String>> {
    WnutReader::new(source).unwrap().map(|r| r.map(|s| s.to_string()).map_err(|e| e.to_string())).collect()
}

#[test]
fn reads_sentences_and_names() {
    let text = "Sarah\tO\nloves\tO\nManchester\tB-location\nUnited\tI-location\n\nunited\tB-group\nwon\tO\n";
    let samples: Vec<String> = read(text).into_iter().map(|r| r.unwrap()).collect();
    assert_eq!(samples, ["Sarah loves <START:location> Manchester United <END>", "<START:group> united <END> won"]);
    let first = WnutReader::new(text).unwrap().next().unwrap().unwrap();
    assert_eq!(first.names().iter().map(|n| n.java_string()).collect::<Vec<_>>(), ["[2..4) location"]);
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
    assert_eq!(sample.to_string(), "<START:person> Zoë <END> \u{1F600}");
}

#[test]
fn a_row_of_another_field_count_ends_the_read_at_its_line() {
    let text = "a\tO\nb\n\nc\tO\n";
    assert_eq!(
        read(text),
        [
            Err("Expected two tab-separated fields per line in WNUT-17 data, got 1 for line 'b'!".to_owned()),
            Ok("c ".to_owned()),
        ]
    );
    let err = WnutReader::new("a\tb\tc\n").unwrap().next().unwrap().unwrap_err();
    assert!(matches!(err, WnutError::Fields { line: 1, count: 3, .. }));
}

#[test]
fn bad_tags_fail_after_the_sentence_is_read() {
    let text = "a\tX\nb\tO\n\nc\tO\n";
    assert_eq!(read(text), [Err("Invalid tag: X".to_owned()), Ok("c ".to_owned())]);
    let text = "a\to\n\nb\tO\n";
    assert_eq!(read(text)[0], Err("Invalid tag: o".to_owned()));
    // a name tag with an empty type is an unknown type, as in GermEval
    let text = "a\tB-\n\nb\tO\n";
    assert_eq!(read(text)[0], Err("Unknown type: ".to_owned()));
    // an unknown type is found when its name ends
    let text = "a\tB-widget\nb\tO\n\nc\tO\n";
    assert_eq!(read(text), [Err("Unknown type: widget".to_owned()), Ok("c ".to_owned())]);
    let err = WnutReader::new("a\tB-widget\n").unwrap().next().unwrap().unwrap_err();
    assert!(matches!(err, WnutError::UnknownType { line: 1, kind } if kind == "widget"));
}

#[test]
fn i_tags_extend_the_open_name_or_are_read_as_o() {
    // an `I-` with no name open is read as `O`; an `I-` of another type
    // still extends the open name, as `bio_names` reads it
    let text = "a\tI-person\nb\tO\nc\tB-person\nd\tI-group\n";
    assert_eq!(read(text), [Ok("a b <START:person> c d <END>".to_owned())]);
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
    assert_eq!(read("a\tO"), [Ok("a ".to_owned())]);
    assert_eq!(read("\n\na\tO\n\n"), [Ok("a ".to_owned())]);
}
