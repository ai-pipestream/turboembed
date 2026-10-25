//! `EntitySalienceAnnotator` and the `EntitySalience` layer: the lead
//! subject outranks a late passing modifier, every feature's direction,
//! the heading bonus, PageRank centrality, determinism, provenance and the
//! codec round trip.

mod salience_common;

use salience_common::{M, O, S, Sentence, X, build, doc, ranked};
use turbo_analysis::{Analyzer, Annotator, DocumentGraph, Key, Layer, export, import};
use turbo_analysis_layers::{DocumentEntities, EntitySalience, EntitySalienceScore, Sentences};
use turbo_entity_linker::{EntitySalienceAnnotator, SalienceConfig, codecs};

const STORY: &[Sentence] = &[
    &[("Acme", S), ("Beta", O)],
    &[("Acme", S), ("Gamma", O)],
    &[("Beta", S), ("Acme", O)],
    &[("Acme", S)],
    &[("Delta", M), ("Gamma", X)],
];

#[test]
fn lead_subject_outranks_a_late_passing_modifier() {
    let d = doc(STORY);
    let s = EntitySalienceAnnotator::new().salience(&d).unwrap();
    s.check().unwrap();
    assert_eq!(ranked(&d, &s), ["Acme", "Beta", "Gamma", "Delta"]);
    let (acme, delta) = (s.get(0).unwrap(), s.get(3).unwrap());
    assert_eq!((acme.rank, delta.rank), (0, 3));
    assert_eq!(acme.frequency_score, 1.0);
    assert_eq!(acme.centrality_score, 1.0);
    assert!(acme.position_score > delta.position_score);
    assert!(acme.syntax_score > delta.syntax_score);
    assert_eq!(acme.position_score, 1.0);
    assert!((delta.syntax_score - 0.25).abs() < 1e-6, "one modifier: 0.5 / 2");
}

#[test]
fn heading_bonus_lifts_an_entity_named_in_a_heading() {
    let late: &[Sentence] = &[&[("Sweden", S)], &[("Sweden", S)], &[("Volvo", X)]];
    let off = SalienceConfig { heading_bonus: 0.0, ..SalienceConfig::default() };
    let without = EntitySalienceAnnotator::with_config(off).salience(&build(late, Some(2), true)).unwrap();
    let with = EntitySalienceAnnotator::new().salience(&build(late, Some(2), true)).unwrap();
    let plain = EntitySalienceAnnotator::new().salience(&doc(late)).unwrap();
    let decayed = (-0.3f32 * 2.0).exp();
    assert!((without.get(1).unwrap().position_score - decayed).abs() < 1e-6);
    assert!((plain.get(1).unwrap().position_score - decayed).abs() < 1e-6);
    assert_eq!(with.get(1).unwrap().position_score, 1.0, "0.55 + 0.5, capped at 1");
    assert_eq!(with.get(0).unwrap().position_score, 1.0);
}

#[test]
fn centrality_is_zero_without_edges_and_follows_co_occurrence() {
    let alone: &[Sentence] = &[&[("Acme", S)], &[("Beta", S)]];
    let s = EntitySalienceAnnotator::new().salience(&doc(alone)).unwrap();
    assert!(s.as_slice().iter().all(|r| r.centrality_score == 0.0));
    let star: &[Sentence] = &[&[("Hub", X), ("A", X)], &[("Hub", X), ("B", X)], &[("Hub", X), ("C", X)]];
    let s = EntitySalienceAnnotator::new().salience(&doc(star)).unwrap();
    assert_eq!(s.get(0).unwrap().centrality_score, 1.0);
    assert!(s.as_slice()[1..].iter().all(|r| r.centrality_score < 0.6));
}

#[test]
fn deterministic_ties_and_config_hash() {
    let twins: &[Sentence] = &[&[("Acme", S), ("Beta", S)]];
    let d = doc(twins);
    let a = EntitySalienceAnnotator::new().salience(&d).unwrap();
    assert_eq!(a, EntitySalienceAnnotator::new().salience(&d).unwrap());
    assert_eq!((a.get(0).unwrap().rank, a.get(1).unwrap().rank), (0, 1), "ties go to the lower row");
    let h = EntitySalienceAnnotator::new().config_hash();
    assert_eq!(h, EntitySalienceAnnotator::new().config_hash());
    let other = SalienceConfig { pagerank_iters: 31, ..SalienceConfig::default() };
    assert_ne!(h, EntitySalienceAnnotator::with_config(other).config_hash());
}

#[test]
fn check_rejects_bad_rows() {
    let row = |entity_ix, rank, salience| EntitySalienceScore {
        entity_ix,
        salience,
        rank,
        position_score: 0.5,
        syntax_score: 0.5,
        frequency_score: 0.5,
        centrality_score: 0.5,
    };
    assert!(EntitySalience { scores: vec![row(0, 1, 0.2), row(1, 0, 0.4)] }.check().is_ok());
    assert!(EntitySalience { scores: vec![row(1, 0, 0.2)] }.check().is_err(), "not by entity");
    assert!(EntitySalience { scores: vec![row(0, 0, 0.2), row(1, 0, 0.4)] }.check().is_err(), "rank twice");
    assert!(EntitySalience { scores: vec![row(0, 0, 0.2), row(1, 1, 0.4)] }.check().is_err(), "rank order");
    assert!(EntitySalience { scores: vec![row(0, 0, 1.5)] }.check().is_err(), "outside [0, 1]");
    let top = EntitySalience { scores: vec![row(0, 1, 0.2), row(1, 0, 0.4)] }.top_k(1);
    assert_eq!(top.iter().map(|r| r.entity_ix).collect::<Vec<_>>(), [1]);
}

#[test]
fn analyzer_provenance_and_codec_round_trip() {
    let analyzer = Analyzer::builder()
        .given(Key::<Sentences>::DEFAULT)
        .given(Key::<DocumentEntities>::DEFAULT)
        .add(EntitySalienceAnnotator::new())
        .build()
        .unwrap();
    let mut batch = vec![doc(STORY)];
    for r in analyzer.run(&mut batch) {
        r.unwrap();
    }
    let d = batch.pop().unwrap();
    let s = d.require(&Key::<EntitySalience>::DEFAULT).unwrap().clone();
    let prov = d.provenance(Key::<EntitySalience>::DEFAULT.id()).unwrap();
    assert_eq!(prov.annotator, EntitySalienceAnnotator::ID);
    assert_eq!(prov.config, EntitySalienceAnnotator::new().config_hash());
    assert!(prov.reads.contains(&Key::<DocumentEntities>::DEFAULT.id()));
    let codecs = codecs();
    let bytes = export(&d, &codecs).unwrap().to_bytes().unwrap();
    let back = import(&DocumentGraph::from_bytes(&bytes).unwrap(), &codecs).unwrap();
    assert_eq!(back.require(&Key::<EntitySalience>::DEFAULT).unwrap(), &s);
}

#[test]
fn without_sentences_position_is_the_byte_fraction() {
    let d = build(STORY, None, false);
    let s = EntitySalienceAnnotator::new().salience(&d).unwrap();
    assert_eq!(s.get(0).unwrap().position_score, 1.0);
    assert!(s.get(3).unwrap().position_score < 0.25);
    assert!(s.as_slice().iter().all(|r| r.centrality_score == 0.0), "no sentences, no relations, no edges");
}
