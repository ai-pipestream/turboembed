//! Property tests: flag alignment, kept tokens, and offset invariants.
//!
//! Every `turbo:stopwords` layer is aligned to `opennlp:tokens`: one flag
//! per token. Offsets live on the tokens; the flags must match that length
//! so a round trip through the document keeps every span.

use turbo_analysis::{Analyzer, Bound, Document, Key, Layer, OffsetTable, Provenance, Span, Spans, ViewId};
use turbo_analysis_layers::{Language, Strs, Tokens};
use turbo_stopword::{ListChoice, StopwordAnnotator, StopwordLanguage, StopwordList, Stopwords};

fn space_tokens(text: &str) -> Tokens {
    let mut spans = Spans::new(ViewId::SOURCE);
    let mut start = 0usize;
    for piece in text.split(' ') {
        let end = start + piece.len();
        spans.push(Span::new(start as u32, end as u32));
        start = end + 1;
    }
    Tokens { spans, sentence: None }
}

fn document(text: &str, language: Option<&str>) -> Document {
    let mut doc = Document::new(text);
    doc.add(&Key::DEFAULT, space_tokens(text), Provenance::by("test")).unwrap();
    if let Some(code) = language {
        let mut codes = Strs::default();
        codes.push(code);
        doc.add(&Key::DEFAULT, Language { code: codes, score: vec![1.0] }, Provenance::by("test")).unwrap();
    }
    doc
}

#[test]
fn flag_length_matches_the_token_count() {
    for language in StopwordLanguage::ALL {
        let list = StopwordList::bundled(language);
        let samples = ["", "a", "The cat and the dog", "uno dos tres cuatro cinco seis siete ocho nueve diez"];
        for text in samples {
            let tokens: Vec<&str> = if text.is_empty() { Vec::new() } else { text.split(' ').collect() };
            let mut flags = Vec::new();
            list.flag(tokens.iter().copied(), &mut flags);
            assert_eq!(flags.len(), tokens.len(), "{language:?} {text:?}");
            for (i, &tok) in tokens.iter().enumerate() {
                assert_eq!(flags[i], list.contains(tok), "{language:?} {tok}");
            }
            let kept: Vec<usize> = flags.iter().enumerate().filter(|&(_, &s)| !s).map(|(i, _)| i).collect();
            let via_layer = Stopwords { language, stop: flags.clone() };
            assert_eq!(via_layer.kept().map(|i| i.get()).collect::<Vec<_>>(), kept);
        }
    }
}

#[test]
fn annotator_flags_match_token_count_and_source_spans() {
    let texts = [
        ("The cat and THE dog", "eng"),
        ("Der Hund und die Katze", "deu"),
        ("Le chat et le chien", "fra"),
        ("", "eng"),
    ];
    for &(text, code) in &texts {
        let mut docs = vec![document(text, Some(code))];
        let analyzer = Analyzer::builder()
            .given(Key::<Tokens>::DEFAULT)
            .given(Key::<Language>::DEFAULT)
            .step(Bound::new(StopwordAnnotator::new(ListChoice::Detected { fallback: None })))
            .build()
            .unwrap();
        let results = analyzer.run(&mut docs);
        assert!(results.iter().all(Result::is_ok), "{text:?}: {results:?}");
        let doc = &docs[0];
        let tokens = doc.get::<Tokens>().unwrap();
        let stops = doc.get::<Stopwords>().unwrap();
        assert_eq!(stops.len(), tokens.len(), "{text:?}");
        let source = doc.source();
        for (i, span) in tokens.spans.iter() {
            let piece = &source[span.range()];
            let list = StopwordList::bundled(stops.language);
            assert_eq!(stops.stop[i.get()], list.contains(piece), "{text:?} {piece}");
            assert_eq!(&source[span.start as usize..span.end as usize], piece);
        }
    }
}

#[test]
fn utf16_and_code_point_tables_round_trip_on_flagged_text() {
    let text = "The café and THE dog";
    let table = OffsetTable::build(text);
    for (byte, _) in text.char_indices() {
        let u16 = table.to_utf16(text, byte as u32);
        let cp = table.to_code_point(text, byte as u32);
        assert_eq!(table.from_utf16(text, u16).unwrap(), byte as u32);
        assert_eq!(table.from_code_point(text, cp).unwrap(), byte as u32);
    }
    let mut docs = vec![document(text, Some("eng"))];
    let analyzer = Analyzer::builder()
        .given(Key::<Tokens>::DEFAULT)
        .given(Key::<Language>::DEFAULT)
        .step(Bound::new(StopwordAnnotator::new(ListChoice::Fixed(StopwordLanguage::English))))
        .build()
        .unwrap();
    assert!(analyzer.run(&mut docs).iter().all(Result::is_ok));
    let tokens = docs[0].get::<Tokens>().unwrap();
    for (_, span) in tokens.spans.iter() {
        for edge in [span.start, span.end] {
            let u16 = table.to_utf16(text, edge);
            let cp = table.to_code_point(text, edge);
            assert_eq!(table.from_utf16(text, u16).unwrap(), edge);
            assert_eq!(table.from_code_point(text, cp).unwrap(), edge);
        }
    }
}

fn escape_field(s: &str) -> String {
    s.replace('\\', "\\\\").replace('\t', "\\t").replace('\n', "\\n").replace('\r', "\\r")
}

fn unescape_field(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    let mut chars = s.chars();
    while let Some(c) = chars.next() {
        if c != '\\' {
            out.push(c);
            continue;
        }
        match chars.next() {
            Some('\\') => out.push('\\'),
            Some('t') => out.push('\t'),
            Some('n') => out.push('\n'),
            Some('r') => out.push('\r'),
            Some(other) => {
                out.push('\\');
                out.push(other);
            }
            None => out.push('\\'),
        }
    }
    out
}

#[test]
fn unescape_round_trips_tsv_fields() {
    for s in ["", "a", "a b", "a\tb", "a\\b", "a\nb", "a\rb", "café", "\\t\\n\\\\"] {
        assert_eq!(unescape_field(&escape_field(s)), s, "{s:?}");
    }
}

#[test]
fn layer_invariants_align_flags_to_tokens_and_keep_non_empty_spans() {
    let texts = [
        ("The cat and the dog", "eng"),
        ("Der Hund und die Katze", "deu"),
        ("hot dog New York", "eng"),
    ];
    for &(text, code) in &texts {
        let mut docs = vec![document(text, Some(code))];
        let analyzer = Analyzer::builder()
            .given(Key::<Tokens>::DEFAULT)
            .given(Key::<Language>::DEFAULT)
            .step(Bound::new(StopwordAnnotator::new(ListChoice::Detected { fallback: None })))
            .build()
            .unwrap();
        assert!(analyzer.run(&mut docs).iter().all(Result::is_ok), "{text:?}");
        let doc = &docs[0];
        let tokens = doc.get::<Tokens>().unwrap();
        let stops = doc.get::<Stopwords>().unwrap();
        // stopword_positions.len() == token_positions.len()
        assert_eq!(stops.len(), tokens.spans.len(), "{text:?}");
        assert_eq!(stops.stop.len(), tokens.spans.len(), "{text:?}");
        stops.check().unwrap_or_else(|e| panic!("Stopwords::check: {e}"));
        let source = doc.source();
        let token_view = tokens.spans.view();
        for (i, span) in tokens.spans.iter() {
            let piece = &source[span.range()];
            // Non-empty source tokens stay non-empty slices.
            if span.start != span.end {
                assert!(!piece.is_empty(), "{text:?} token {i:?} empty slice");
                assert!(!source[span.range()].is_empty(), "{text:?}");
            }
            // Filtered (kept) tokens stay on the same token view; the annotator
            // does not invent a derived view (StopwordFilter is not ported).
            assert_eq!(tokens.spans.view(), token_view);
            let _ = stops.stop[i.get()];
        }
        let kept: Vec<_> = stops.kept().collect();
        for ix in kept {
            let span = tokens.spans.get(ix);
            assert!(!source[span.range()].is_empty() || span.start == span.end, "{text:?}");
        }
    }
}
