use std::num::NonZeroU32;
use turbo_analysis::Layer;
use turbo_analysis::{
    Analyzer, Document, Evidence, FilterRule, IxRange, Key, Provenance, Span, Spans, Tag, TagSet, ViewId,
};
use turbo_analysis_layers::{Entities, Tokens};
use turbo_namefind::{ConsensusEntityMerger, EntityMergePolicy, consensus};

fn tokens(spans: &[(u32, u32)]) -> Tokens {
    let mut out = Spans::new(ViewId::SOURCE);
    for &(a, b) in spans {
        out.push(Span::new(a, b));
    }
    Tokens { spans: out, sentence: None }
}
fn entity() -> Entities {
    let mut spans = Spans::new(ViewId::SOURCE);
    spans.push(Span::new(0, 3));
    Entities {
        spans,
        tokens: vec![IxRange::new(0, 1)],
        kind: vec![Tag(0)],
        kinds: TagSet::new("test:kind", &["person"]),
        score: vec![0.8],
    }
}
fn document() -> Document {
    let mut doc = Document::new("Ada Bob");
    doc.add(&Key::DEFAULT, tokens(&[(0, 3), (4, 7)]), Provenance::by("test:tokens")).unwrap();
    // No explicit reads: Document::add resolves the default token instance.
    doc.add(&Key::named("one"), entity(), Provenance::by("test:finder")).unwrap();
    doc.add(&Key::named("gold"), tokens(&[(0, 1), (1, 7)]), Provenance::by("test:other")).unwrap();
    assert_eq!(doc.made_on(Key::<Entities>::named("one").id()), [Key::<Tokens>::DEFAULT.id()]);
    doc
}
#[test]
fn merger_refuses_resolved_default_tokens_when_named_tokens_selected() {
    let doc = document();
    let mut merger =
        ConsensusEntityMerger::new(vec![Key::named("one")], Key::named("merged"), EntityMergePolicy::RawUnion);
    merger.tokens = Key::named("gold");
    assert!(merger.build(&doc).is_err(), "accepted an entity bound to default tokens as gold");
}
#[test]
fn pipeline_refuses_wrong_resolved_token_binding() {
    let mut merger =
        ConsensusEntityMerger::new(vec![Key::named("one")], Key::named("merged"), EntityMergePolicy::RawUnion);
    merger.tokens = Key::named("gold");
    let analyzer = Analyzer::builder()
        .given(Key::<Entities>::named("one"))
        .given(Key::<Tokens>::named("gold"))
        .step(merger)
        .build()
        .unwrap();
    let mut docs = [document()];
    let results = analyzer.run(&mut docs);
    assert!(
        results[0].is_err(),
        "pipeline accepted foreign tokens: {results:?}; made_on={:?}",
        docs[0].made_on(Key::<Entities>::named("merged").id())
    );
}
#[test]
fn repeated_input_instance_cannot_satisfy_distinct_recognizer_quorum() {
    let doc = document();
    let merger = ConsensusEntityMerger::new(
        vec![Key::named("one"), Key::named("one")],
        Key::named("merged"),
        EntityMergePolicy::Consensus { min_votes: NonZeroU32::new(2).unwrap(), overlap_rule: FilterRule::HighestScore },
    );
    match merger.build(&doc) {
        Err(_) => {}
        Ok((entities, _)) => assert!(entities.score.is_empty(), "one recognizer received two votes"),
    }
}
#[test]
fn selected_token_layer_is_required_by_public_build() {
    let doc = document();
    let mut merger =
        ConsensusEntityMerger::new(vec![Key::named("one")], Key::named("merged"), EntityMergePolicy::RawUnion);
    merger.tokens = Key::named("missing");
    assert!(merger.build(&doc).is_err(), "public build ignored its required tokens");
}
#[test]
fn default_binding_control_produces_verifiable_evidence() {
    let doc = document();
    let merger = ConsensusEntityMerger::new(vec![Key::named("one")], Key::named("merged"), EntityMergePolicy::RawUnion);
    let (entities, evidence) = merger.build(&doc).unwrap();
    assert_eq!(entities.tokens, [IxRange::new(0, 1)]);
    evidence.verify(&doc).unwrap();
    assert_eq!(evidence.support_of(0)[0].start, 0);
    let _ = Key::<Evidence<Entities>>::named("merged");
}

#[test]
fn direct_consensus_refuses_a_layer_with_nan_score() {
    let mut layer = entity();
    layer.score[0] = f64::NAN;
    assert!(layer.check().is_err());
    assert!(consensus(&[&layer], EntityMergePolicy::RawUnion).is_err(), "direct consensus bypassed layer validation");
}

#[test]
fn named_resolved_binding_and_evidence_survive_graph_transport() {
    use turbo_analysis::export::{DocumentGraph, export, import};
    let mut doc = Document::new("Ada Bob");
    doc.add(&Key::DEFAULT, tokens(&[(0, 3), (4, 7)]), Provenance::by("test:tokens")).unwrap();
    let gold = Key::<Tokens>::named("gold");
    doc.add(&gold, tokens(&[(0, 3), (4, 7)]), Provenance::by("test:gold")).unwrap();
    doc.add(&Key::named("one"), entity(), Provenance { on: vec![gold.id()], ..Provenance::by("test:finder") }).unwrap();
    let mut merger =
        ConsensusEntityMerger::new(vec![Key::named("one")], Key::named("merged"), EntityMergePolicy::RawUnion);
    merger.tokens = gold;
    let pipeline = Analyzer::builder().given(gold).given(Key::<Entities>::named("one")).step(merger).build().unwrap();
    let codecs = turbo_analysis_layers::export::codecs();
    let bytes = export(&doc, &codecs).unwrap().to_bytes().unwrap();
    let mut back = import(&DocumentGraph::from_bytes(&bytes).unwrap(), &codecs).unwrap();
    assert_eq!(pipeline.run(std::slice::from_mut(&mut back)), [Ok(())]);
    assert_eq!(back.made_on(Key::<Entities>::named("merged").id()), [gold.id()]);
    let evidence = back.get_key(&Key::<Evidence<Entities>>::named("merged")).unwrap();
    evidence.verify(&back).unwrap();
    assert_eq!(evidence.producer_of(0).annotator, "test:finder");
    let bytes = export(&back, &codecs).unwrap().to_bytes().unwrap();
    let restored = import(&DocumentGraph::from_bytes(&bytes).unwrap(), &codecs).unwrap();
    restored.get_key(&Key::<Evidence<Entities>>::named("merged")).unwrap().verify(&restored).unwrap();
    assert_eq!(restored.made_on(Key::<Entities>::named("merged").id()), [gold.id()]);
}

#[test]
fn transitive_grouping_retains_distinct_recognizer_votes() {
    let layers: Vec<_> = (0..3)
        .map(|i| {
            let mut e = entity();
            e.spans = Spans::new(ViewId::SOURCE);
            e.spans.push(Span::new(i, i + 2));
            e.tokens[0] = IxRange::new(i, i + 2);
            e.score[0] = 0.9 - f64::from(i) * 0.1;
            e
        })
        .collect();
    let refs: Vec<_> = layers.iter().collect();
    let out = consensus(
        &refs,
        EntityMergePolicy::Consensus { min_votes: NonZeroU32::new(3).unwrap(), overlap_rule: FilterRule::HighestScore },
    )
    .unwrap();
    assert_eq!(out.entities.tokens, [IxRange::new(0, 2)]);
    assert_eq!(out.votes_of(0).len(), 3);
    assert_eq!(out.votes_of(0).iter().map(|v| v.input).collect::<Vec<_>>(), [0, 1, 2]);
}

#[test]
fn two_candidates_from_one_recognizer_do_not_form_a_quorum() {
    let mut e = entity();
    e.tokens[0] = IxRange::new(0, 3);
    e.spans.push(Span::new(1, 4));
    e.tokens.push(IxRange::new(1, 4));
    e.kind.push(Tag(0));
    e.score.push(0.9);
    let out = consensus(
        &[&e],
        EntityMergePolicy::Consensus { min_votes: NonZeroU32::new(2).unwrap(), overlap_rule: FilterRule::HighestScore },
    )
    .unwrap();
    assert!(out.entities.score.is_empty());
}
