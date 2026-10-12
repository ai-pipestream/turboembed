//! The error record: a golden for a run where the annotator threw on the
//! JVM. `tests/fixtures/newline-first.golden` is hand written for the
//! input `tests/fixtures/inputs/test/newline-first.txt`, a line feed and
//! then `It rained.`: OpenNLP's `SimpleTokenizer` with `keepNewLines`
//! makes `new Span(-1, 0)` for a line break before the first token, and
//! `Span`'s constructor throws `IllegalArgumentException` with the message
//! the file holds. The digests in the header other than the input's are
//! stand-ins.

use std::path::PathBuf;

use turbo_analysis::{Document, Key, Provenance, Span, Spans, ViewId};
use turbo_analysis_layers::Tokens;
use turbo_goldens::*;

const TEXT: &str = "\nIt rained.";

fn fixtures() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures")
}

fn fixture_text() -> String {
    std::fs::read_to_string(fixtures().join("newline-first.golden")).unwrap()
}

fn golden() -> Golden {
    Golden::read(fixtures().join("newline-first.golden")).unwrap()
}

/// The fixture's text with `extra` appended.
fn with_lines(extra: &str) -> String {
    format!("{}{extra}", fixture_text())
}

const TOKENS: &str = "layer\topennlp:tokens\ttext\t0\tstart:u32\tend:u32\nend\topennlp:tokens\n";

#[test]
fn the_error_record_reads_into_the_header() {
    let g = golden();
    assert_eq!(
        g.header.error,
        Some(JvmError {
            class: "java.lang.IllegalArgumentException".into(),
            message: "start index must be zero or greater: -1".into(),
        })
    );
    assert_eq!(g.header.config, [ConfigEntry { key: "keepNewLines".into(), value: "true".into() }]);
    assert!(g.layers.is_empty());
    assert_eq!(g.input_text(fixtures()).unwrap(), TEXT);
}

#[test]
fn a_golden_without_the_record_has_no_error() {
    let g = Golden::read(fixtures().join("zurich.golden")).unwrap();
    assert_eq!(g.header.error, None);
}

#[test]
fn the_error_record_writes_back_byte_for_byte() {
    let g = golden();
    assert_eq!(g.to_text().unwrap(), fixture_text());
}

#[test]
fn the_message_is_escaped_like_any_str_field() {
    let mut g = golden();
    for message in ["", "tab\there", "two\nlines", "cr\r", "back\\slash", "nul\u{0}", "😀 \u{1f}"] {
        g.header.error.as_mut().unwrap().message = message.into();
        let text = g.to_text().unwrap();
        assert_eq!(text.matches('\n').count(), 9, "the record is one line: {text:?}");
        assert_eq!(Golden::parse(&text).unwrap(), g, "{message:?}");
    }
    g.header.error.as_mut().unwrap().message = "a\tb\nc\\".into();
    assert!(g.to_text().unwrap().ends_with("\ta\\tb\\nc\\\\\n"));
}

#[test]
fn the_reader_refuses_a_layer_block_after_the_error_record() {
    let e = Golden::parse(&with_lines(TOKENS)).unwrap_err();
    assert!(matches!(e, GoldenError::AfterError { line: 10, .. }), "{e:?}");
    let shown = e.to_string();
    assert!(
        shown.starts_with(
            "line 10: expected the end of the file after the error record (a run that threw has no layers)"
        ),
        "{shown}"
    );
    assert!(shown.contains("found \"layer\\topennlp:tokens"), "{shown}");
}

#[test]
fn the_error_record_is_the_last_line() {
    // A config record after it, or a second error record, is refused too.
    let e = Golden::parse(&with_lines("config\tz\t1\n")).unwrap_err();
    assert!(matches!(e, GoldenError::AfterError { line: 10, .. }), "{e:?}");
    let e = Golden::parse(&with_lines("error\tC\tm\n")).unwrap_err();
    assert!(matches!(e, GoldenError::AfterError { line: 10, .. }), "{e:?}");
}

#[test]
fn a_malformed_error_record_is_refused_with_its_line() {
    let text = fixture_text();
    let error_line = "error\tjava.lang.IllegalArgumentException\tstart index must be zero or greater: -1\n";
    let swap = |to: &str| text.replace(error_line, to);
    let e = Golden::parse(&swap("error\tjava.lang.IllegalArgumentException\n")).unwrap_err();
    assert!(matches!(e, GoldenError::Fields { line: 9, found: 2, .. }), "{e:?}");
    assert!(e.to_string().contains("expected the fields error <class> <message>"), "{e}");
    let e = Golden::parse(&swap("error\tC\tm\textra\n")).unwrap_err();
    assert!(matches!(e, GoldenError::Fields { line: 9, found: 4, .. }), "{e:?}");
    let e = Golden::parse(&swap("error\t\tm\n")).unwrap_err();
    assert!(matches!(e, GoldenError::Field { line: 9, .. }), "{e:?}");
    assert!(e.to_string().contains("error class: expected a non-empty field"), "{e}");
    let e = Golden::parse(&swap("error\tC\tbad \\q escape\n")).unwrap_err();
    assert!(matches!(e, GoldenError::Escape { line: 9, .. }), "{e:?}");
    assert!(e.to_string().contains("error message"), "{e}");
    // An empty message is a message `getMessage()` returned as null.
    let g = Golden::parse(&swap("error\tC\t\n")).unwrap();
    assert_eq!(g.header.error, Some(JvmError { class: "C".into(), message: String::new() }));
}

#[test]
fn the_writer_refuses_layers_beside_an_error_record() {
    let mut g = Golden::read(fixtures().join("zurich.golden")).unwrap();
    g.header.error = Some(JvmError { class: "java.lang.IllegalStateException".into(), message: "m".into() });
    let e = g.to_text().unwrap_err();
    assert_eq!(e.kind(), std::io::ErrorKind::InvalidInput);
    assert!(e.to_string().contains("a run that threw java.lang.IllegalStateException has layers"), "{e}");
    // A tab in the class would break the record.
    let mut g = golden();
    g.header.error.as_mut().unwrap().class = "a\tb".into();
    assert_eq!(g.to_text().unwrap_err().kind(), std::io::ErrorKind::InvalidInput);
}

/// What the Rust tokenizer makes of the input, which the JVM could not
/// tokenize: the line feed is skipped, then `It`, `rained` and `.`.
fn corrected_doc() -> Document {
    let mut doc = Document::new(TEXT);
    let mut spans = Spans::new(ViewId::SOURCE);
    for (a, b) in [(1, 3), (4, 10), (10, 11)] {
        spans.push(Span::new(a, b));
    }
    doc.add(&Key::DEFAULT, Tokens { spans, sentence: None }, Provenance::by("test")).unwrap();
    doc
}

#[test]
fn a_comparison_reports_the_jvm_error() {
    let g = golden();
    let doc = corrected_doc();
    let m = matches_golden::<Tokens>(&doc, &g, Tolerance::exact()).unwrap_err();
    assert_eq!(
        m,
        Mismatch::JvmError {
            layer: "opennlp:tokens".into(),
            class: "java.lang.IllegalArgumentException".into(),
            message: "start index must be zero or greater: -1".into(),
        }
    );
    assert_eq!(m.layer(), "opennlp:tokens");
    assert_eq!(
        m.to_string(),
        "opennlp:tokens: the annotator threw on the JVM, so the golden has no layers: \
         java.lang.IllegalArgumentException: \"start index must be zero or greater: -1\""
    );
    // With the error asserted, the test checks the corrected output itself.
    let tokens = doc.get::<Tokens>().unwrap();
    let (start, end) = tokens.spans.columns();
    assert_eq!((start, end), (&[1, 4, 10][..], &[3, 10, 11][..]));

    // A document that is not the golden's input is reported as that first.
    let other = Document::new("It rained.");
    let m = matches_golden::<Tokens>(&other, &g, Tolerance::exact()).unwrap_err();
    assert!(matches!(m, Mismatch::Input { .. }), "{m:?}");
}
