use super::support::*;
use turbo_analysis::{Analyzer, Annotator, Document, Ix, Key, Provenance, Span, Spans, Tag, TagSet, ViewId};
use turbo_analysis_layers::{
    Dependencies, MwtLanguage, PosTags, RuleMwtExpander, Sentences, Tokens, WordPosTags, Words,
};
use turbo_depparse::*;
fn spans<L>(ranges: &[(u32, u32)]) -> Spans<L> {
    let mut out = Spans::new(ViewId::SOURCE);
    for &(a, b) in ranges {
        out.push(Span::new(a, b));
    }
    out
}
#[test]
fn neural_token_annotation_records_model_and_beam_provenance() {
    let model = FeedforwardDependencyTrainer::train(&samples(), &settings(false)).unwrap();
    let parser = FeedforwardDependencyParser::new(model.clone(), 4).unwrap();
    let identity = parser.identity();
    let annotator = DependencyAnnotator::<Tokens>::new(parser);
    let other = DependencyAnnotator::<Tokens>::new(FeedforwardDependencyParser::new(model, 1).unwrap());
    assert_ne!(annotator.config_hash(), other.config_hash());
    let mut doc = Document::new("I see birds");
    doc.add(&Key::DEFAULT, Sentences { spans: spans(&[(0, 11)]) }, Provenance::by("test")).unwrap();
    doc.add(&Key::DEFAULT, Tokens { spans: spans(&[(0, 1), (2, 5), (6, 11)]), sentence: None }, Provenance::by("test"))
        .unwrap();
    doc.add(
        &Key::DEFAULT,
        PosTags {
            tags: vec![Tag(0), Tag(1), Tag(2)],
            tagset: TagSet::new("upos", &["PRON", "VERB", "NOUN"]),
            prob: None,
        },
        Provenance::by("test"),
    )
    .unwrap();
    let analyzer = Analyzer::builder()
        .given(Key::<Sentences>::DEFAULT)
        .given(Key::<Tokens>::DEFAULT)
        .given(Key::<PosTags>::DEFAULT)
        .add(annotator)
        .build()
        .unwrap();
    let mut docs = vec![doc];
    analyzer.run(&mut docs)[0].as_ref().unwrap();
    docs[0].get::<Dependencies>().unwrap().check_forest().unwrap();
    let p = docs[0].provenance(Key::<Dependencies>::DEFAULT.id()).unwrap();
    assert_eq!(p.model.as_ref(), Some(&identity));
    assert!(p.reads.contains(&Key::<PosTags>::DEFAULT.id()));
}
#[test]
fn neural_word_annotation_supports_multiword_expansion() {
    let mut doc = Document::new("Voy al mar");
    doc.add(&Key::DEFAULT, Sentences { spans: spans(&[(0, 10)]) }, Provenance::by("test")).unwrap();
    doc.add(&Key::DEFAULT, Tokens { spans: spans(&[(0, 3), (4, 6), (7, 10)]), sentence: None }, Provenance::by("test"))
        .unwrap();
    let words = RuleMwtExpander::new(MwtLanguage::Spanish)
        .expand(doc.get::<turbo_analysis::View>().unwrap().text(), doc.get::<Tokens>().unwrap());
    doc.add(&Key::DEFAULT, words, Provenance::by("test")).unwrap();
    doc.add(
        &Key::DEFAULT,
        WordPosTags { tags: vec![Tag(0); 4], tagset: TagSet::new("upos", &["X"]), prob: None },
        Provenance::by("test"),
    )
    .unwrap();
    let parser = FeedforwardDependencyParser::new(minimal(FeatureLayout::ChenManning18), 4).unwrap();
    let analyzer = Analyzer::builder()
        .given(Key::<Sentences>::DEFAULT)
        .given(Key::<Tokens>::DEFAULT)
        .given(Key::<Words>::DEFAULT)
        .given(Key::<WordPosTags>::DEFAULT)
        .add(DependencyAnnotator::<Words>::new(parser))
        .build()
        .unwrap();
    let mut docs = vec![doc];
    analyzer.run(&mut docs)[0].as_ref().unwrap();
    let layer = docs[0].get::<Dependencies<Words>>().unwrap();
    assert_eq!(layer.head.len(), 4);
    layer.check_forest().unwrap();
    assert!(layer.head.iter().flatten().all(|h| h.get() < 4));
    let _: Option<Ix<Words>> = layer.head[0];
}
