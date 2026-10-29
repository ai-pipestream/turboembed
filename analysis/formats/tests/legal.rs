use turbo_analysis::Span;
use turbo_formats::legal::{BouvierReader, CaselawReader, LegalError};
use turbo_formats::vocabulary::Vocabulary;
use turbo_formats::{DocumentSample, Field};
use turbo_glossary::{DelimitedReader, FirstRecord};
use turbo_stem::porter::PorterStemmer;
use turbo_stopword::{StopwordLanguage, StopwordList};

const CASELAW: &str = include_str!("data/caselaw-case.json");
const BOUVIER: &str = include_str!("data/bouvier.html");

#[test]
fn caselaw_preserves_decoded_text_and_raw_json_spans() {
    let cases: Vec<_> = CaselawReader::new(CASELAW).unwrap().map(Result::unwrap).collect();
    assert_eq!(cases.len(), 2);
    let case = &cases[0];
    assert_eq!(case.category.as_str(), "Goeb v. Tharaldson");
    assert_eq!(&CASELAW[case.category.span.range()], "Goeb v. Tharaldson");
    assert_eq!(&CASELAW[case.source.range()], &CASELAW[..CASELAW.find("\n{").unwrap()]);
    let texts: Vec<_> = case.tokens.iter().map(|token| token.as_str()).collect();
    assert_eq!(texts, ["OPINION", "Frye", "v.", "United", "States,", "293", "F.", "1013.", "Second", "opinion", "😀"]);
    for token in &case.tokens {
        let raw = &CASELAW[token.span.range()];
        if token.as_str() == "😀" {
            assert_eq!(raw, "😀");
        } else if token.as_str() != "OPINION" {
            assert_eq!(raw, token.as_str());
        }
    }
    assert_eq!(cases[1].category.as_str(), "Second Case");
    assert_eq!(cases[1].tokens[0].as_str(), "Another");
    assert_eq!(cases[1].tokens[1].as_str(), "case.");
}

#[test]
fn caselaw_rejects_bad_json_and_missing_fields() {
    let error = CaselawReader::new("{\"name_abbreviation\":\"\\uD800\"}").unwrap_err();
    assert!(matches!(error, LegalError::Json { .. }));
    let mut cases = CaselawReader::new("{\"casebody\":{\"opinions\":[]}}").unwrap();
    assert_eq!(cases.next().unwrap().unwrap_err(), LegalError::MissingField("name_abbreviation"));
}

#[test]
fn bouvier_joins_continuations_and_preserves_html_spans() {
    let entries: Vec<_> = BouvierReader::new(BOUVIER).unwrap().collect();
    assert_eq!(entries.len(), 2);
    assert_eq!(entries[0].category.as_str(), "PACE");
    assert_eq!(&BOUVIER[entries[0].category.span.range()], "PACE");
    let first: Vec<_> = entries[0].tokens.iter().map(|token| token.as_str()).collect();
    assert_eq!(first, [".", "A", "measure", "of", "length."]);
    let second: Vec<_> = entries[1].tokens.iter().map(|token| token.as_str()).collect();
    assert_eq!(
        second,
        [",", "civil", "law.", "An", "agreement", "&", "promise.", "2.", "A", "second", "paragraph", "§755."]
    );
    let amp = entries[1].tokens.iter().find(|token| token.as_str() == "&").unwrap();
    assert_eq!(&BOUVIER[amp.span.range()], "&amp;");
    assert!(BOUVIER[entries[1].source.range()].starts_with("<P><B>PACT</B>"));
    assert!(BOUVIER[entries[1].source.range()].ends_with("A second paragraph &sect;755.</P>"));
}

#[test]
fn bouvier_rejects_unclosed_paragraphs() {
    let error = BouvierReader::new("<P><B>PACE</B>. A measure").unwrap_err();
    assert!(matches!(error, LegalError::Markup { .. }));
}

fn document(tokens: &[&str]) -> DocumentSample<'static> {
    DocumentSample {
        category: Field::made("legal".to_owned(), Span::new(0, 0)),
        tokens: tokens.iter().map(|text| Field::made((*text).to_owned(), Span::new(0, 0))).collect(),
        source: Span::new(0, 0),
    }
}

#[test]
fn vocabulary_counts_stems_and_loads_as_a_glossary() {
    let documents =
        vec![document(&["The", "judges", "judged", "and", "judging"]), document(&["Judges", "and", "the", "law"])];
    let stopwords = StopwordList::bundled(StopwordLanguage::English);
    let vocabulary = Vocabulary::build(documents.clone(), stopwords, &PorterStemmer);
    assert_eq!(vocabulary, Vocabulary::build(documents, stopwords, &PorterStemmer));
    let entries = vocabulary.entries();
    assert_eq!(entries.len(), 2);
    assert_eq!((entries[0].term.as_str(), entries[0].frequency, entries[0].document_frequency), ("judg", 4, 2));
    assert_eq!((entries[1].term.as_str(), entries[1].frequency, entries[1].document_frequency), ("law", 1, 1));
    let tsv = vocabulary.write_tsv();
    assert_eq!(tsv, "id\tterm\tfrequency\tdocument_frequency\njudg\tjudg\t4\t2\nlaw\tlaw\t1\t1\n");
    let loaded = DelimitedReader::tsv(FirstRecord::Header).read(tsv.as_bytes()).unwrap();
    assert_eq!(loaded.glossary.iter().count(), 2);
}

#[test]
fn vocabulary_quotes_terms_for_the_glossary_reader() {
    let vocabulary = Vocabulary::build(
        vec![document(&["quote\"term"])],
        StopwordList::bundled(StopwordLanguage::English),
        &PorterStemmer,
    );
    let tsv = vocabulary.write_tsv();
    assert!(tsv.contains("\"quote\"\"term\""));
    let loaded = DelimitedReader::tsv(FirstRecord::Header).read(tsv.as_bytes()).unwrap();
    let (_, entry) = loaded.glossary.iter().next().unwrap();
    assert_eq!(entry.term, "quote\"term");
}
