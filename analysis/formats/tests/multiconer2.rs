//! Focused tests for the MultiCoNER v2 CoNLL reader of issue #1533.
//!
//! Every fixture here is original, written for these tests in the released
//! corpus's `token _ _ TAG` form.

use turbo_formats::NameSample;
use turbo_formats::error::FormatError;
use turbo_formats::ner::{MULTICONER_TYPES, MulticonerReader};

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
    MulticonerReader::new(source).unwrap().map(|r| r.map(|s| shape(&s)).map_err(|e| e.to_string())).collect()
}

#[test]
fn reads_the_released_conll_form_and_document_headers() {
    let source = concat!(
        "# id f5458a3a-cd23-4df4-8384-4e23fe33a66b\tdomain=en\n",
        "doris _ _ B-Artist\n",
        "day _ _ I-Artist\n",
        "included _ _ O\n",
        "album _ _ B-MusicalWork\n",
        "billy _ _ I-MusicalWork\n",
        "\n",
        "# id second\n",
        "acme _ _ B-ORG\n",
    );
    assert_eq!(
        read(source),
        [
            Ok((
                true,
                strings(&["doris", "day", "included", "album", "billy"]),
                vec![(0, 2, Some("Artist")), (3, 5, Some("MusicalWork"))],
            )),
            Ok((true, strings(&["acme"]), vec![(0, 1, Some("ORG"))])),
        ]
    );
    let samples: Vec<_> = MulticonerReader::new(source).unwrap().map(|s| s.unwrap()).collect();
    assert!(samples[0].clear_adaptive_data() && samples[1].clear_adaptive_data());
    let names: Vec<_> = samples[0].names().iter().map(|n| (n.start, n.end, n.kind)).collect();
    assert_eq!(names, [(0, 2, Some("Artist")), (3, 5, Some("MusicalWork"))]);
    assert_eq!(&source[samples[0].tokens()[3].span.range()], "album");
    for token in samples[0].tokens() {
        assert!(token.is_borrowed());
    }
}

#[test]
fn a_document_header_directly_after_a_token_row_still_clears_the_next_document() {
    // No blank line before the second `# id`: the first sentence ends at
    // the header, and the second document's first sentence must clear.
    let source =
        concat!("# id first\n", "doris _ _ B-Artist\n", "# id second\n", "acme _ _ B-ORG\n", "billy _ _ I-ORG\n",);
    assert_eq!(
        read(source),
        [
            Ok((true, strings(&["doris"]), vec![(0, 1, Some("Artist"))])),
            Ok((true, strings(&["acme", "billy"]), vec![(0, 2, Some("ORG"))])),
        ]
    );
    let samples: Vec<_> = MulticonerReader::new(source).unwrap().map(|s| s.unwrap()).collect();
    assert!(samples[0].clear_adaptive_data() && samples[1].clear_adaptive_data());
}

#[test]
fn every_fine_grained_type_is_kept_as_written() {
    assert_eq!(MULTICONER_TYPES.len(), 36);
    for kind in MULTICONER_TYPES {
        let source = format!("token _ _ B-{kind}\n");
        let sample = MulticonerReader::new(&source).unwrap().next().unwrap().unwrap();
        assert_eq!(sample.names()[0].kind, Some(kind));
    }
}

#[test]
fn iob_boundaries_follow_the_shared_rule() {
    let source =
        concat!("a _ _ O\n", "b _ _ B-Artist\n", "c _ _ I-Artist\n", "d _ _ O\n", "e _ _ B-ORG\n", "f _ _ I-ORG\n",);
    let sample = MulticonerReader::new(source).unwrap().next().unwrap().unwrap();
    let names: Vec<_> = sample.names().iter().map(|n| (n.start, n.end, n.kind)).collect();
    assert_eq!(names, [(1, 3, Some("Artist")), (4, 6, Some("ORG"))]);
}

#[test]
fn tokens_keep_multi_byte_and_supplementary_source_spans() {
    let source = "caf\u{e9} \u{1f600} _ _ B-Facility\n";
    let sample = MulticonerReader::new(source).unwrap().next().unwrap().unwrap();
    assert_eq!(&source[sample.tokens()[0].span.range()], "caf\u{e9} \u{1f600}");
    assert_eq!(sample.names()[0].kind, Some("Facility"));
}

#[test]
fn malformed_rows_are_refused_at_their_line() {
    for source in
        ["token B-Artist\n", "token _ _ B-Artist _ _ O\n", " _ _ B-Artist\n", "token _ _ \n", "token _ _ B Artist\n"]
    {
        let error = MulticonerReader::new(source).unwrap().next().unwrap().unwrap_err();
        assert!(matches!(error, FormatError::MulticonerFields { line: 1, .. }), "{error:?} for {source:?}");
    }
}

#[test]
fn unknown_types_and_invalid_continuations_are_refused() {
    let error = MulticonerReader::new("token _ _ B-NotAType\n").unwrap().next().unwrap().unwrap_err();
    assert!(matches!(error, FormatError::UnknownNameType { .. }));
    let error = MulticonerReader::new("token _ _ I-Artist\n").unwrap().next().unwrap().unwrap_err();
    assert!(matches!(error, FormatError::UnexpectedNameContinuation { .. }));
    let error = MulticonerReader::new("a _ _ B-Artist\nb _ _ I-ORG\n").unwrap().next().unwrap().unwrap_err();
    assert!(matches!(error, FormatError::NameTypeMismatch { .. }));
}

#[test]
fn empty_input_and_a_final_unterminated_sentence_are_handled() {
    assert_eq!(MulticonerReader::new("").unwrap().count(), 0);
    let sample = MulticonerReader::new("token _ _ O").unwrap().next().unwrap().unwrap();
    assert!(sample.names().is_empty());
}

#[test]
fn pinned_english_training_file_streams_with_exact_offsets() {
    let Ok(path) = std::env::var("TURBO_MULTICONER_EN_TRAIN") else { return };
    let source = std::fs::read_to_string(path).unwrap();
    let mut sentences = 0;
    let mut names = 0;
    for sample in MulticonerReader::new(&source).unwrap() {
        let sample = sample.unwrap();
        sentences += 1;
        names += sample.names().len();
        for token in sample.tokens() {
            assert!(token.is_borrowed());
            assert_eq!(&source[token.span.range()], token.as_str());
        }
        for name in sample.names() {
            assert!(name.end as usize <= sample.tokens().len());
        }
    }
    assert!(sentences > 10_000, "only {sentences} sentences");
    assert!(names > 10_000, "only {names} names");
}
