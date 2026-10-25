use std::collections::{BTreeMap, HashMap};
use std::sync::Arc;
use turbo_analysis::export::{DocumentGraph, export, import};
use turbo_analysis::{
    Analyzer, Annotator, Bound, Document, Ix, Key, Layer, Provenance, Span, Spans, View, ViewBuilder, ViewId,
};
use turbo_analysis_layers::{ExpansionRow, LexicalExpansions, SAME_SYNSET, Tokens, WORDNET_RELATIONS, WordNetSenses};
use turbo_wordnet::{
    InMemoryWordNetLexicon, LemmaKey, MorphyExceptions, Synset, WordNetAnnotator, WordNetOptions, WordNetPos,
    WordNetRelation,
};

fn lexicon() -> Arc<InMemoryWordNetLexicon> {
    let mut sets = HashMap::new();
    let mut order = HashMap::new();
    for (id, lemma, synonym) in [("s-dog", "dog", "canine"), ("s-cat", "cat", "feline")] {
        sets.insert(
            id.to_string(),
            Synset::new(id, WordNetPos::Noun, vec![lemma.into(), synonym.into()], "", BTreeMap::new()).unwrap(),
        );
        order.insert(LemmaKey::of(lemma, WordNetPos::Noun), vec![id.to_string()]);
    }
    Arc::new(InMemoryWordNetLexicon::new(sets, order).unwrap())
}
fn add_tokens(doc: &mut Document, key: Key<Tokens>, view: ViewId, pairs: &[(u32, u32)]) {
    let mut spans = Spans::new(view);
    for &(a, b) in pairs {
        spans.push(Span::new(a, b));
    }
    doc.add(&key, Tokens { spans, sentence: None }, Provenance::by("test:tokens")).unwrap();
}
#[test]
fn named_tokens_and_unicode_derived_view_survive_wordnet_graph_roundtrip() {
    let mut doc = Document::new("é dog😀");
    let source = doc.get::<View>().unwrap().clone();
    let mut builder = ViewBuilder::new(source.text());
    builder.keep(source.text().len());
    let (text, alignment) = builder.finish();
    doc.add(&Key::named("derived"), View::derived(&source, text, alignment), Provenance::by("test:view")).unwrap();
    add_tokens(&mut doc, Key::DEFAULT, ViewId::SOURCE, &[(0, 2), (3, 6), (6, 10)]);
    add_tokens(&mut doc, Key::named("gold"), ViewId(1), &[(3, 6)]);
    let annotator = WordNetAnnotator::new(lexicon(), "test:wordnet", WordNetOptions::default()).unwrap();
    let analyzer = Analyzer::builder()
        .given(Key::<View>::named("derived"))
        .given(Key::<Tokens>::named("gold"))
        .step(Bound::new(annotator).read(0, "derived").read(1, "gold"))
        .build()
        .unwrap();
    assert_eq!(analyzer.run(std::slice::from_mut(&mut doc)), [Ok(())]);
    let senses = doc.get::<WordNetSenses>().unwrap();
    assert_eq!(senses.spans.view(), ViewId(1));
    assert_eq!(senses.token, [Ix::new(0)]);
    assert_eq!(senses.spans.get(Ix::new(0)), Span::new(3, 6));
    let codecs = turbo_analysis_layers::export::codecs();
    let bytes = export(&doc, &codecs).unwrap().to_bytes().unwrap();
    let back = import(&DocumentGraph::from_bytes(&bytes).unwrap(), &codecs).unwrap();
    for d in [&doc, &back] {
        assert_eq!(d.made_on(Key::<WordNetSenses>::DEFAULT.id()), [Key::<Tokens>::named("gold").id()]);
        assert_eq!(d.made_on(Key::<LexicalExpansions>::DEFAULT.id()), [Key::<Tokens>::named("gold").id()]);
        assert_eq!(d.get::<WordNetSenses>().unwrap().synset_id.get(0), "s-dog");
    }
    assert_eq!(export(&back, &codecs).unwrap().to_bytes().unwrap(), bytes);
}
#[test]
fn relation_codes_are_bijective_and_roundtrip_across_crates() {
    assert_eq!(WordNetRelation::ALL.len(), usize::from(WORDNET_RELATIONS));
    assert_eq!(WordNetRelation::from_layer_code(SAME_SYNSET), None);
    let mut doc = Document::new("x");
    add_tokens(&mut doc, Key::DEFAULT, ViewId::SOURCE, &[(0, 1)]);
    let mut layer = LexicalExpansions::new(ViewId::SOURCE, "test:relations".into());
    for (i, &relation) in WordNetRelation::ALL.iter().enumerate() {
        let code = relation.layer_code();
        assert_eq!(usize::from(code), i + 1);
        assert_eq!(WordNetRelation::from_layer_code(code), Some(relation));
        layer.push(
            Span::new(0, 1),
            ExpansionRow {
                token: Ix::new(0),
                term: "x",
                relation: code,
                depth: 1,
                sense_rank: 0,
                weight: 0.5,
                synset_id: relation.lmf_name(),
            },
        );
    }
    for code in WORDNET_RELATIONS + 1..=u8::MAX {
        assert_eq!(WordNetRelation::from_layer_code(code), None);
    }
    layer.check().unwrap();
    doc.add(&Key::DEFAULT, layer, Provenance::by("test:relations")).unwrap();
    let codecs = turbo_analysis_layers::export::codecs();
    let bytes = export(&doc, &codecs).unwrap().to_bytes().unwrap();
    let back = import(&DocumentGraph::from_bytes(&bytes).unwrap(), &codecs).unwrap();
    let got = back.get::<LexicalExpansions>().unwrap();
    for (i, &relation) in WordNetRelation::ALL.iter().enumerate() {
        assert_eq!(WordNetRelation::from_layer_code(got.relation[i]), Some(relation));
        assert_eq!(got.synset_id.get(i), relation.lmf_name());
    }
}
#[test]
fn morphy_exception_resources_change_actual_senses_and_configuration_identity() {
    let make = |lemma: &str| {
        let exceptions = MorphyExceptions::from_map(HashMap::from([(
            WordNetPos::Noun,
            HashMap::from([("mice".into(), vec![lemma.to_string()])]),
        )]));
        WordNetAnnotator::new(lexicon(), "test:same-lexicon", WordNetOptions::default())
            .unwrap()
            .with_morphy(exceptions)
    };
    let (dog, cat) = (make("dog"), make("cat"));
    let mut doc = Document::new("mice");
    add_tokens(&mut doc, Key::DEFAULT, ViewId::SOURCE, &[(0, 4)]);
    let run = |a: &WordNetAnnotator<InMemoryWordNetLexicon>| {
        a.annotate((doc.get::<View>().unwrap(), doc.get::<Tokens>().unwrap(), None, None)).unwrap().0
    };
    assert_eq!(run(&dog).synset_id.get(0), "s-dog");
    assert_eq!(run(&cat).synset_id.get(0), "s-cat");
    assert_ne!(dog.config_hash(), cat.config_hash(), "different exception resources alias the pipeline cache key");
}
