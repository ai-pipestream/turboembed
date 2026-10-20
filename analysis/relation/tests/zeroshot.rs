//! Zero-shot relation extraction with the lexical scorer: the label
//! semantics (thresholds, directed and symmetric labels, domain and range),
//! sentence scoping, named limits, request checks, and the relations'
//! evidence records through an `Analyzer`, nested and overlapping mentions
//! included, and across the document graph.

use std::sync::Arc;

use turbo_analysis::export::{DocumentGraph, export, import};
use turbo_analysis::{
    Analyzer, Annotator, Document, Evidence, EvidenceStep, IxRange, Key, Provenance, Span, Spans, TagSet, View, ViewId,
};
use turbo_analysis_layers::{Entities, Sentences, Tokens};
use turbo_relation::{
    LexicalRelationScorer, RelationError, RelationLabel, Relations, ZeroShotRelationAnnotator, ZeroShotRelationConfig,
};

const TEXT: &str = "Jobs founded Apple in Cupertino. Cook works for Apple.";

fn spans<L>(ranges: &[(u32, u32)]) -> Spans<L> {
    let mut out = Spans::new(ViewId::SOURCE);
    for &(a, b) in ranges {
        out.push(Span::new(a, b));
    }
    out
}

/// Entities at `ranges` with kinds from PERSON, ORG and LOC.
fn entities(ranges: &[(u32, u32, &str)]) -> Entities {
    let kinds = TagSet::new("kinds", &["PERSON", "ORG", "LOC"]);
    Entities {
        spans: spans(&ranges.iter().map(|r| (r.0, r.1)).collect::<Vec<_>>()),
        tokens: (0..ranges.len() as u32).map(|i| IxRange::new(i, i + 1)).collect(),
        kind: ranges.iter().map(|r| kinds.tag(r.2).unwrap()).collect(),
        kinds: Arc::clone(&kinds),
        score: vec![1.0; ranges.len()],
    }
}

/// Jobs, Apple, Cupertino in the first sentence; Cook, Apple in the
/// second.
fn fixture() -> (Document, Entities, Sentences) {
    let e = entities(&[(0, 4, "PERSON"), (13, 18, "ORG"), (22, 31, "LOC"), (33, 37, "PERSON"), (48, 53, "ORG")]);
    (Document::new(TEXT), e, Sentences { spans: spans(&[(0, 32), (33, 54)]) })
}

fn label(name: &str, domain: &str, range: &str) -> RelationLabel {
    RelationLabel { domain: Some(vec![domain.into()]), range: Some(vec![range.into()]), ..RelationLabel::new(name) }
}

fn labels() -> Vec<RelationLabel> {
    vec![label("founded", "PERSON", "ORG"), label("located in", "ORG", "LOC"), label("works for", "PERSON", "ORG")]
}

/// (head, tail, label, score) of every relation.
fn rows(r: &Relations) -> Vec<(usize, usize, String, f64)> {
    (0..r.subject.len())
        .map(|i| (r.subject[i].get(), r.object[i].get(), r.kinds.label(r.kind[i]).to_string(), r.score[i]))
        .collect()
}

fn extract(config: ZeroShotRelationConfig, with_sentences: bool) -> Relations {
    let (doc, e, s) = fixture();
    let a = ZeroShotRelationAnnotator::new(config, LexicalRelationScorer).unwrap();
    a.extract(doc.get::<View>().unwrap(), &e, with_sentences.then_some(&s)).unwrap().0
}

fn row(h: usize, t: usize, l: &str, s: f64) -> (usize, usize, String, f64) {
    (h, t, l.to_string(), s)
}

#[test]
fn labels_score_pairs_their_domain_and_range_admit_within_a_sentence() {
    let got = rows(&extract(ZeroShotRelationConfig::new(labels()), true));
    // "located in" finds one of its two words between Apple and
    // Cupertino: 0.5, at the default threshold.
    assert_eq!(got, [row(0, 1, "founded", 1.0), row(1, 2, "located in", 0.5), row(3, 4, "works for", 1.0)]);
    // A label's own threshold overrides the default.
    let mut l = labels();
    l[1].threshold = Some(0.6);
    assert_eq!(rows(&extract(ZeroShotRelationConfig::new(l), true)).len(), 2);
    // Across sentences Jobs pairs with the second Apple too, the words
    // between them holding both labels.
    let cross = ZeroShotRelationConfig { within_sentence: false, ..ZeroShotRelationConfig::new(labels()) };
    let got = rows(&extract(cross, true));
    assert!(got.contains(&row(0, 4, "founded", 1.0)) && got.contains(&row(0, 4, "works for", 1.0)));
    // Without sentences every pair is a candidate.
    assert_eq!(rows(&extract(ZeroShotRelationConfig::new(labels()), false)).len(), got.len());
}

#[test]
fn a_symmetric_label_is_written_once_per_pair() {
    let doc = Document::new("Ann sibling of Bob.");
    let e = entities(&[(0, 3, "PERSON"), (15, 18, "PERSON")]);
    let view = doc.get::<View>().unwrap();
    let directed = ZeroShotRelationConfig::new(vec![RelationLabel::new("sibling")]);
    let a = ZeroShotRelationAnnotator::new(directed, LexicalRelationScorer).unwrap();
    assert_eq!(rows(&a.extract(view, &e, None).unwrap().0), [row(0, 1, "sibling", 1.0), row(1, 0, "sibling", 1.0)]);
    let symmetric = RelationLabel { symmetric: true, ..RelationLabel::new("sibling") };
    let a =
        ZeroShotRelationAnnotator::new(ZeroShotRelationConfig::new(vec![symmetric]), LexicalRelationScorer).unwrap();
    let (r, edges) = a.extract(view, &e, None).unwrap();
    assert_eq!(rows(&r), [row(0, 1, "sibling", 1.0)]);
    assert_eq!(turbo_analysis::Layer::len(&edges), 1);
}

#[test]
fn requests_over_a_limit_fail_with_the_limit_named() {
    let many: Vec<RelationLabel> = (0..5).map(|i| RelationLabel::new(format!("r{i}"))).collect();
    let config = ZeroShotRelationConfig { max_labels: 4, ..ZeroShotRelationConfig::new(many) };
    let err = ZeroShotRelationAnnotator::new(config, LexicalRelationScorer).err().unwrap();
    assert_eq!(err, RelationError::OverLimit { limit: "max_labels", max: 4, got: 5 });
    let (doc, e, s) = fixture();
    let config = ZeroShotRelationConfig { max_mentions: 4, ..ZeroShotRelationConfig::new(labels()) };
    let a = ZeroShotRelationAnnotator::new(config, LexicalRelationScorer).unwrap();
    let err = a.extract(doc.get::<View>().unwrap(), &e, Some(&s)).err().unwrap();
    assert_eq!(err, RelationError::OverLimit { limit: "max_mentions", max: 4, got: 5 });
    assert!(err.to_string().contains("max_mentions"));
    // Across the view the five mentions make 6 admitted pairs, past 5.
    let config =
        ZeroShotRelationConfig { within_sentence: false, max_pairs: 5, ..ZeroShotRelationConfig::new(labels()) };
    let a = ZeroShotRelationAnnotator::new(config, LexicalRelationScorer).unwrap();
    let err = a.extract(doc.get::<View>().unwrap(), &e, Some(&s)).err().unwrap();
    assert!(matches!(err, RelationError::OverLimit { limit: "max_pairs", max: 5, .. }), "{err:?}");
}

#[test]
fn a_score_equal_to_the_threshold_is_kept() {
    let doc = Document::new("Ann a b c d e f g Bob.");
    let e = entities(&[(0, 3, "PERSON"), (18, 21, "PERSON")]);
    let label = RelationLabel { threshold: Some(0.7), ..RelationLabel::new("a b c d e f g h i j") };
    let a = ZeroShotRelationAnnotator::new(ZeroShotRelationConfig::new(vec![label]), LexicalRelationScorer).unwrap();
    assert_eq!(rows(&a.extract(doc.get::<View>().unwrap(), &e, None).unwrap().0).len(), 2);
}

#[test]
fn a_symmetric_label_with_a_domain_unlike_its_range_is_refused() {
    let l = RelationLabel { symmetric: true, ..label("employs", "PERSON", "ORG") };
    let err = ZeroShotRelationAnnotator::new(ZeroShotRelationConfig::new(vec![l]), LexicalRelationScorer).err();
    assert_eq!(err, Some(RelationError::AsymmetricKinds("employs".into())));
}

#[test]
fn bad_requests_are_refused() {
    let new = |labels: Vec<RelationLabel>, threshold: f64| {
        let c = ZeroShotRelationConfig { threshold, ..ZeroShotRelationConfig::new(labels) };
        ZeroShotRelationAnnotator::new(c, LexicalRelationScorer).err()
    };
    assert_eq!(new(vec![], 0.5), Some(RelationError::NoLabels));
    assert_eq!(new(vec![RelationLabel::new(" ")], 0.5), Some(RelationError::BlankLabel(0)));
    let twice = vec![RelationLabel::new("a"), RelationLabel::new("a")];
    assert_eq!(new(twice, 0.5), Some(RelationError::DuplicateLabel("a".into())));
    assert!(matches!(new(labels(), 1.5), Some(RelationError::BadThreshold { label: None, .. })));
    let mut l = labels();
    l[0].threshold = Some(-0.1);
    assert!(matches!(new(l, 0.5), Some(RelationError::BadThreshold { label: Some(_), .. })));
}

#[test]
fn the_configuration_hash_follows_every_setting() {
    let hash =
        |c: ZeroShotRelationConfig| ZeroShotRelationAnnotator::new(c, LexicalRelationScorer).unwrap().config_hash();
    let base = hash(ZeroShotRelationConfig::new(labels()));
    assert_eq!(base, hash(ZeroShotRelationConfig::new(labels())));
    let mut l = labels();
    l.push(RelationLabel::new("sibling of"));
    let with = hash(ZeroShotRelationConfig::new(l.clone()));
    l[3].symmetric = true;
    assert_ne!(with, hash(ZeroShotRelationConfig::new(l)));
    let mut l = labels();
    l[2].range = Some(vec!["LOC".into()]);
    assert_ne!(base, hash(ZeroShotRelationConfig::new(l)));
    assert_ne!(base, hash(ZeroShotRelationConfig { threshold: 0.4, ..ZeroShotRelationConfig::new(labels()) }));
    assert_ne!(base, hash(ZeroShotRelationConfig { within_sentence: false, ..ZeroShotRelationConfig::new(labels()) }));
    assert_ne!(base, hash(ZeroShotRelationConfig { max_pairs: 9, ..ZeroShotRelationConfig::new(labels()) }));
}

/// "New York City is in New York State." with nested and overlapping
/// places, through an analyzer with the evidence step.
fn nested() -> Document {
    let text = "New York City is in New York State.";
    let mut doc = Document::new(text);
    let words = [(0, 3), (4, 8), (9, 13), (14, 16), (17, 19), (20, 23), (24, 28), (29, 34), (34, 35)];
    doc.add(&Key::DEFAULT, Tokens { spans: spans(&words), sentence: None }, Provenance::by("test")).unwrap();
    let mut e = entities(&[(0, 13, "LOC"), (4, 13, "LOC"), (20, 34, "LOC"), (24, 28, "LOC")]);
    e.tokens = vec![IxRange::new(0, 3), IxRange::new(1, 3), IxRange::new(5, 8), IxRange::new(6, 7)];
    doc.add(&Key::DEFAULT, e, Provenance::by("test")).unwrap();
    let a = ZeroShotRelationAnnotator::new(
        ZeroShotRelationConfig::new(vec![RelationLabel::new("in")]),
        LexicalRelationScorer,
    )
    .unwrap();
    let analyzer =
        Analyzer::builder().given(Key::<Entities>::DEFAULT).add(a).step(EvidenceStep::new(Key::<Relations>::DEFAULT));
    let mut docs = vec![doc];
    for r in analyzer.build().unwrap().run(&mut docs) {
        r.unwrap();
    }
    docs.pop().unwrap()
}

#[test]
fn evidence_records_cover_each_head_and_tail_byte_for_byte() {
    let doc = nested();
    let (e, r) = (doc.get::<Entities>().unwrap(), doc.get::<Relations>().unwrap());
    let ev = doc.get::<Evidence<Relations>>().unwrap();
    // Each of the two places in the city pairs with each of the two in
    // the state, both ways: "in" lies between them, and the lexical
    // scorer reads no order.
    let pairs = [(0, 2), (0, 3), (1, 2), (1, 3), (2, 0), (2, 1), (3, 0), (3, 1)];
    assert_eq!(rows(r).iter().map(|x| (x.0, x.1)).collect::<Vec<_>>(), pairs);
    assert_eq!(ev.records(), 8);
    for i in 0..8 {
        let (h, t) = (r.subject[i], r.object[i]);
        let text = |s: Span| &doc.source()[s.start as usize..s.end as usize];
        let got: Vec<&str> = ev.spans_of(i).iter().map(|&s| text(s)).collect();
        assert_eq!(got, [text(e.spans.get(h)), text(e.spans.get(t))]);
        assert_eq!(ev.raw_of(i), Some(r.score[i]));
        let support: Vec<(u32, u32)> = ev.support_of(i).iter().map(|s| (s.start, s.end)).collect();
        assert_eq!(support, [(h.get() as u32, h.get() as u32 + 1), (t.get() as u32, t.get() as u32 + 1)]);
        let p = ev.producer_of(i);
        assert_eq!((p.annotator.as_str(), &p.model), ("turbo:zero-shot-relations", &None));
    }
    assert_eq!(ev.layers[0].layer, "opennlp:entities");
    assert_eq!(ev.verify(&doc), Ok(()));
}

#[test]
fn relation_evidence_crosses_the_graph() {
    let codecs = turbo_relation::export::codecs();
    let doc = nested();
    let graph = export(&doc, &codecs).unwrap();
    let back = import(&DocumentGraph::from_bytes(&graph.to_bytes().unwrap()).unwrap(), &codecs).unwrap();
    let (a, b) = (doc.get::<Evidence<Relations>>().unwrap(), back.get::<Evidence<Relations>>().unwrap());
    assert_eq!((&a.spans, &a.raw, &a.support), (&b.spans, &b.raw, &b.support));
    assert_eq!(export(&back, &codecs).unwrap().to_bytes().unwrap(), graph.to_bytes().unwrap());
}
