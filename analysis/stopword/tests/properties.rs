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

#[test]
fn layer_check_passes_on_annotator_output() {
    let texts = [("The cat and the dog", "eng"), ("Der Hund und die Katze", "deu"), ("hot dog New York", "eng")];
    for &(text, code) in &texts {
        let mut docs = vec![document(text, Some(code))];
        let analyzer = Analyzer::builder()
            .given(Key::<Tokens>::DEFAULT)
            .given(Key::<Language>::DEFAULT)
            .step(Bound::new(StopwordAnnotator::new(ListChoice::Detected { fallback: None })))
            .build()
            .unwrap();
        assert!(analyzer.run(&mut docs).iter().all(Result::is_ok), "{text:?}");
        let stops = docs[0].get::<Stopwords>().unwrap();
        let tokens = docs[0].get::<Tokens>().unwrap();
        assert_eq!(stops.len(), tokens.spans.len(), "{text:?}");
        stops.check().unwrap_or_else(|e| panic!("Stopwords::check: {e}"));
    }
}

#[test]
fn lookup_matches_documented_case_and_script_surprises() {
    let en = StopwordList::bundled(StopwordLanguage::English);
    // U+0130 LATIN CAPITAL LETTER I WITH DOT ABOVE lowercases to ASCII i.
    assert!(en.contains("İT"), "İT should be the stopword it");
    assert!(en.contains("it"));
    // LONG S and DOTLESS I do not land in the list word's hash bucket.
    assert!(!en.contains("ſo"), "long-s o must not match so");
    assert!(!en.contains("ıt"), "dotless-i + t must not match it");
    // Decomposed vs composed: accents are not folded.
    let es = StopwordList::bundled(StopwordLanguage::Spanish);
    assert!(es.contains("también"), "también is a Spanish stopword");
    assert!(!es.contains("tambien"), "no accent fold");
    assert!(!es.contains("también"), "decomposed accent is a different word");
    // German: DASS is not daß; DAẞ (capital sharp S) is. Also in lists.rs.
    let de = StopwordList::bundled(StopwordLanguage::German);
    assert!(de.contains("daß"));
    assert!(!de.contains("DASS"), "DASS is not daß");
    assert!(de.contains("DAẞ"), "capital sharp S lowercases to daß");
}

#[test]
fn every_bundled_word_has_no_long_s_dotless_i_or_cyrillic_ve_spelling() {
    // Documents that compareToIgnoreCase would equate these, but the HashSet
    // bucket means none of the bundled words are found under those spellings.
    for language in StopwordLanguage::ALL {
        let list = StopwordList::bundled(language);
        for &word in list.words() {
            let long_s = word.replace(['s', 'S'], "ſ");
            if long_s != word {
                assert!(!list.contains(&long_s), "{language:?} long-s spelling of {word}");
            }
            let dotless = word.replace(['i', 'I'], "\u{0131}");
            if dotless != word {
                assert!(!list.contains(&dotless), "{language:?} dotless-i spelling of {word}");
            }
            let ve = word.replace(['в', 'В'], "\u{1C80}");
            if ve != word {
                assert!(!list.contains(&ve), "{language:?} cyrillic-ve spelling of {word}");
            }
        }
    }
}
