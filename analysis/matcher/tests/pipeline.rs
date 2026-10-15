//! The ruler as an analysis step: matches written as `turbo:matches` under
//! a named instance, their spans in the tokens' view and back in source
//! bytes.

use turbo_analysis::{Analyzer, Bound, Document, Key, Provenance, Span, Spans, Tag, TagSet, View, ViewBuilder, ViewId};
use turbo_analysis_layers::{Lemmas, PosTags, Tokens};
use turbo_matcher::{Attr, Matcher, Overlap, PhraseMatcher, PhraseOn, Ruler, RulerAnnotator, Tok};
use turbo_matcher_layers::Matches;

/// Tokens at the spaces of `text`.
fn tokens(text: &str, view: ViewId) -> Tokens {
    let mut spans = Spans::new(view);
    let mut start = 0;
    for piece in text.split(' ') {
        let end = start + piece.len();
        spans.push(Span::new(start as u32, end as u32));
        start = end + 1;
    }
    Tokens { spans, sentence: None }
}

fn ruler(overlap: Overlap) -> Ruler {
    let mut b = Matcher::builder();
    b.add("PROPN_RUN", vec![vec![Tok::pos("NNP").one_or_more()]], None).unwrap();
    let mut p = PhraseMatcher::builder(PhraseOn::Attr(Attr::Lower));
    p.add("GPE", &["new", "york"]).unwrap();
    p.add("GPE", &["new", "york", "city"]).unwrap();
    Ruler::new(Some(b.build().unwrap()), vec![p.build().unwrap()], overlap).unwrap()
}

fn tagged(text: &str, tags: &[&str]) -> Document {
    let mut doc = Document::new(text);
    doc.add(&Key::DEFAULT, tokens(text, ViewId::SOURCE), Provenance::by("test")).unwrap();
    let set = TagSet::new("penn", &["NNP", "VBD", "PRP", "CC", "JJ", "NN", "."]);
    let tags: Vec<Tag> = tags.iter().map(|t| set.tag(t).unwrap()).collect();
    doc.add(&Key::DEFAULT, PosTags { tagset: set, tags, prob: None }, Provenance::by("test")).unwrap();
    doc
}

fn run(doc: Document, overlap: Overlap) -> (Document, Vec<(String, String)>) {
    let analyzer = Analyzer::builder()
        .given(Key::<Tokens>::DEFAULT)
        .given(Key::<PosTags>::DEFAULT)
        .step(Bound::new(RulerAnnotator::new(ruler(overlap))).write(0, "places"))
        .build()
        .unwrap();
    let mut docs = vec![doc];
    let results = analyzer.run(&mut docs);
    assert!(results.iter().all(Result::is_ok), "{results:?}");
    let doc = docs.pop().unwrap();
    let m = doc.get_key(&Key::<Matches>::named("places")).unwrap();
    let rows = (0..m.spans.len())
        .map(|i| {
            let s = m.spans.get(turbo_analysis::Ix::new(i as u32));
            (m.label_of(i).to_string(), doc.text(m.spans.view())[s.range()].to_string())
        })
        .collect();
    (doc, rows)
}

fn rows(r: &[(&str, &str)]) -> Vec<(String, String)> {
    r.iter().map(|&(a, b)| (a.into(), b.into())).collect()
}

const TEXT: &str = "I visited New York City and new york .";
const TAGS: [&str; 9] = ["PRP", "VBD", "NNP", "NNP", "NNP", "CC", "JJ", "NN", "."];

#[test]
fn every_match_by_default() {
    let (_, got) = run(tagged(TEXT, &TAGS), Overlap::All);
    assert_eq!(
        got,
        rows(&[
            ("PROPN_RUN", "New"),
            ("PROPN_RUN", "New York"),
            ("GPE", "New York"),
            ("PROPN_RUN", "New York City"),
            ("GPE", "New York City"),
            ("PROPN_RUN", "York"),
            ("PROPN_RUN", "York City"),
            ("PROPN_RUN", "City"),
            ("GPE", "new york"),
        ])
    );
}

#[test]
fn longest_without_overlaps() {
    // Where one run has two labels, the first label added wins.
    let (_, got) = run(tagged(TEXT, &TAGS), Overlap::Longest);
    assert_eq!(got, rows(&[("PROPN_RUN", "New York City"), ("GPE", "new york")]));
    let (_, got) = run(tagged(TEXT, &TAGS), Overlap::First);
    assert_eq!(got, rows(&[("PROPN_RUN", "New York City"), ("GPE", "new york")]));
}

#[test]
fn matches_map_back_to_source_bytes_through_a_normalized_view() {
    // The source has a ligature and a run of spaces; the tokens are on a
    // view that expands and folds them.
    let source = "the \u{FB01}nal  report";
    let mut doc = Document::new(source);
    let mut vb = ViewBuilder::new(source);
    vb.keep(4);
    vb.replace(7, "fi");
    vb.keep(10);
    vb.replace(12, " ");
    let (text, alignment) = vb.finish();
    assert_eq!(text, "the final report");
    let view = View::derived(doc.get::<View>().unwrap(), text.clone(), alignment);
    doc.add(&Key::named("clean"), view, Provenance::by("test")).unwrap();
    let id = doc.get_key(&Key::<View>::named("clean")).unwrap().id();
    doc.add(&Key::DEFAULT, tokens(&text, id), Provenance::by("test")).unwrap();

    let mut p = PhraseMatcher::builder(PhraseOn::Attr(Attr::Text));
    p.add("TERM", &["final", "report"]).unwrap();
    let ruler = Ruler::new(None, vec![p.build().unwrap()], Overlap::All).unwrap();
    let analyzer = Analyzer::builder()
        .given(Key::<Tokens>::DEFAULT)
        .given(Key::<View>::named("clean"))
        .step(Bound::new(RulerAnnotator::new(ruler)).read(0, "clean"))
        .build()
        .unwrap();
    let mut docs = vec![doc];
    let results = analyzer.run(&mut docs);
    assert!(results.iter().all(Result::is_ok), "{results:?}");
    let m = docs[0].get::<Matches>().unwrap();
    let s = m.spans.get(turbo_analysis::Ix::new(0));
    assert_eq!(m.spans.view(), id);
    assert_eq!(&text[s.range()], "final report");
    let back = docs[0].to_source(id, s);
    assert_eq!(&source[back.range()], "\u{FB01}nal  report");
}

#[test]
fn a_rule_on_a_missing_layer_fails_the_step() {
    let mut b = Matcher::builder();
    b.add("L", vec![vec![Tok::lemma("visit")]], None).unwrap();
    let ruler = Ruler::new(Some(b.build().unwrap()), vec![], Overlap::All).unwrap();
    let analyzer =
        Analyzer::builder().given(Key::<Tokens>::DEFAULT).step(Bound::new(RulerAnnotator::new(ruler))).build().unwrap();
    let mut doc = Document::new("I visited");
    doc.add(&Key::DEFAULT, tokens("I visited", ViewId::SOURCE), Provenance::by("test")).unwrap();
    let mut docs = vec![doc];
    let error = format!("{:?}", analyzer.run(&mut docs));
    assert!(error.contains("Lemma"), "{error}");
}

#[test]
fn lemmas_when_present() {
    let mut b = Matcher::builder();
    b.add("VISIT", vec![vec![Tok::lemma("visit")]], None).unwrap();
    let ruler = Ruler::new(Some(b.build().unwrap()), vec![], Overlap::All).unwrap();
    let analyzer = Analyzer::builder()
        .given(Key::<Tokens>::DEFAULT)
        .given(Key::<Lemmas>::DEFAULT)
        .step(Bound::new(RulerAnnotator::new(ruler)))
        .build()
        .unwrap();
    let text = "I visited and visit";
    let mut doc = Document::new(text);
    doc.add(&Key::DEFAULT, tokens(text, ViewId::SOURCE), Provenance::by("test")).unwrap();
    let mut lemma = turbo_analysis_layers::Strs::default();
    for l in ["I", "visit", "and", "visit"] {
        lemma.push(l);
    }
    doc.add(&Key::DEFAULT, Lemmas { lemma }, Provenance::by("test")).unwrap();
    let mut docs = vec![doc];
    assert!(analyzer.run(&mut docs).iter().all(Result::is_ok));
    let m = docs[0].get::<Matches>().unwrap();
    let ranges: Vec<(usize, usize)> = m.tokens.iter().map(|r| (r.start.get(), r.end.get())).collect();
    assert_eq!(ranges, vec![(1, 2), (3, 4)]);
}
