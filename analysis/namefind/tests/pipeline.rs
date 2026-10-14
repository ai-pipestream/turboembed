//! The finders and the merge as analysis steps: a dictionary finder and the
//! factory's patterns each write a named instance of `opennlp:entities`,
//! and the merge writes the default one.

use turbo_analysis::{Analyzer, Bound, Document, Ix, Key, Provenance, Span, Spans, ViewId};
use turbo_analysis_layers::{Entities, Sentences, Tokens};
use turbo_maxent::Dictionary;
use turbo_namefind::{BuiltIn, DictionaryNameFinder, MergeEntities, RegexNameFinder};

/// Tokens at the spaces of `text`, a new sentence after each " . ".
fn tokens(text: &str) -> Tokens {
    let mut spans = Spans::new(ViewId::SOURCE);
    let mut sentence = Vec::new();
    let mut s = 0;
    let mut start = 0;
    for piece in text.split(' ') {
        let end = start + piece.len();
        spans.push(Span::new(start as u32, end as u32));
        sentence.push(Ix::<Sentences>::new(s));
        if piece == "." {
            s += 1;
        }
        start = end + 1;
    }
    Tokens { spans, sentence: Some(sentence) }
}

fn names(d: &Document, key: &Key<Entities>) -> Vec<(String, String)> {
    let e = d.get_key(key).unwrap();
    e.spans
        .iter()
        .map(|(i, s)| (d.source()[s.range()].to_string(), e.kinds.label(e.kind[i.get()]).to_string()))
        .collect()
}

#[test]
fn finders_then_merge() {
    let xml = "<dictionary case_sensitive='false'><entry><token>New</token><token>York</token></entry>\
               <entry><token>WWW.example.com</token></entry></dictionary>";
    let dictionary = DictionaryNameFinder::new(Dictionary::read(xml.as_bytes()).unwrap(), "place");
    let regex = RegexNameFinder::defaults(&[BuiltIn::Email, BuiltIn::Url, BuiltIn::UsaPhoneNum]);
    let analyzer = Analyzer::builder()
        .given(Key::<Tokens>::DEFAULT)
        .step(Bound::new(dictionary).write(0, "dictionary"))
        .step(Bound::new(regex).write(0, "regex"))
        .step(Bound::new(MergeEntities::<2>).read(0, "dictionary").read(1, "regex"))
        .build()
        .unwrap();

    let text = "Mail ann@example.com in new york . Call (555) 123-4567 or see www.example.com today";
    let mut doc = Document::new(text);
    let mut sentences = Spans::new(ViewId::SOURCE);
    sentences.push(Span::new(0, 34));
    sentences.push(Span::new(35, text.len() as u32));
    doc.add(&Key::DEFAULT, Sentences { spans: sentences }, Provenance::by("test")).unwrap();
    doc.add(&Key::DEFAULT, tokens(text), Provenance::by("test")).unwrap();
    let mut docs = vec![doc];
    let results = analyzer.run(&mut docs);
    assert!(results.iter().all(Result::is_ok), "{results:?}");
    let d = &docs[0];

    let pair = |a: &str, b: &str| (a.to_string(), b.to_string());
    assert_eq!(names(d, &Key::named("dictionary")), [pair("new york", "place"), pair("www.example.com", "place")]);
    let regex = names(d, &Key::named("regex"));
    assert_eq!(regex.len(), 3);
    assert!(regex.contains(&pair("(555) 123-4567", "PHONE_NUM")));
    // The merge sorts by token, and of the two names of www.example.com
    // keeps the URL, since "URL" sorts before "place".
    assert_eq!(
        names(d, &Key::DEFAULT),
        [
            pair("ann@example.com", "EMAIL"),
            pair("new york", "place"),
            pair("(555) 123-4567", "PHONE_NUM"),
            pair("www.example.com", "URL"),
        ]
    );
    let merged = d.get::<Entities>().unwrap();
    assert_eq!(merged.tokens[2].start.get(), 7);
    assert_eq!(merged.tokens[2].end.get(), 9);
    assert!(merged.score.iter().all(|&s| s == 0.0));
}

#[test]
fn filter_entities_step_resolves_nested_entities() {
    use std::sync::Arc;
    use turbo_analysis::IxRange;
    use turbo_analysis_layers::{FilterRule, Tag, TagSet};
    use turbo_namefind::FilterEntities;

    let text = "Bank of New York in Boston";
    let mut doc = Document::new(text);
    let mut sentences = Spans::new(ViewId::SOURCE);
    sentences.push(Span::new(0, text.len() as u32));
    doc.add(&Key::DEFAULT, Sentences { spans: sentences }, Provenance::by("test")).unwrap();
    doc.add(&Key::DEFAULT, tokens(text), Provenance::by("test")).unwrap();

    let kinds = TagSet::new("ner", &["org", "loc"]);
    let mut spans = Spans::new(ViewId::SOURCE);
    spans.push(Span::new(0, 16)); // "Bank of New York"
    spans.push(Span::new(8, 16)); // "New York"
    spans.push(Span::new(20, 26)); // "Boston"
    let raw = Entities {
        spans,
        tokens: vec![IxRange::new(0, 4), IxRange::new(2, 4), IxRange::new(5, 6)],
        kind: vec![Tag(0), Tag(1), Tag(1)],
        kinds: Arc::clone(&kinds),
        score: vec![0.7, 0.95, 0.8],
    };
    doc.add(&Key::named("raw"), raw, Provenance::by("test")).unwrap();

    let analyzer = Analyzer::builder()
        .given(Key::<Tokens>::DEFAULT)
        .given(Key::<Entities>::named("raw"))
        .step(Bound::new(FilterEntities::new(FilterRule::LongestFirst)).read(0, "raw").write(0, "longest"))
        .step(Bound::new(FilterEntities::new(FilterRule::HighestScore)).read(0, "raw").write(0, "highest"))
        .build()
        .unwrap();

    let mut docs = vec![doc];
    let results = analyzer.run(&mut docs);
    assert!(results.iter().all(Result::is_ok), "{results:?}");
    let d = &docs[0];
    let pair = |a: &str, b: &str| (a.to_string(), b.to_string());
    assert_eq!(names(d, &Key::named("longest")), [pair("Bank of New York", "org"), pair("Boston", "loc")]);
    assert_eq!(names(d, &Key::named("highest")), [pair("New York", "loc"), pair("Boston", "loc")]);
}
