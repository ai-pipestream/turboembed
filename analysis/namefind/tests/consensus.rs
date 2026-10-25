//! Consensus over three recognizers' entities, run as a pipeline step:
//! a dictionary (no model) and two statistical finders, each in its own
//! instance of `opennlp:entities`.

use std::num::NonZeroU32;

use turbo_analysis::{
    Analyzer, Document, Evidence, FilterRule, IxRange, Key, ModelRef, Provenance, Span, Spans, Tag, TagSet, ViewId,
};
use turbo_analysis_layers::{Entities, Tokens};
use turbo_namefind::{ConsensusEntityMerger, EntityMergePolicy, Vote, consensus};

const TEXT: &str = "Ada Lovelace met Charles Babbage in London.";
const TOKENS: [(u32, u32); 8] = [(0, 3), (4, 12), (13, 16), (17, 24), (25, 32), (33, 35), (36, 42), (42, 43)];

fn model(id: &str) -> ModelRef {
    ModelRef { id: id.into(), revision: "r1".into(), sha256: "0".repeat(64) }
}

/// Entities over token ranges, labelled from `person`, `location`, `organization`.
fn entities(items: &[(usize, usize, u16, f64)]) -> Entities {
    let kinds = TagSet::new("ner", &["person", "location", "organization"]);
    let mut spans = Spans::new(ViewId::SOURCE);
    for &(a, b, _, _) in items {
        spans.push(Span::new(TOKENS[a].0, TOKENS[b - 1].1));
    }
    Entities {
        spans,
        tokens: items.iter().map(|&(a, b, _, _)| IxRange::new(a as u32, b as u32)).collect(),
        kind: items.iter().map(|&(_, _, k, _)| Tag(k)).collect(),
        kinds,
        score: items.iter().map(|&(_, _, _, s)| s).collect(),
    }
}

const PERSON: u16 = 0;
const LOCATION: u16 = 1;
const ORGANIZATION: u16 = 2;

fn inputs() -> [Entities; 3] {
    [
        entities(&[(0, 2, PERSON, 0.0), (6, 7, LOCATION, 0.0)]),
        entities(&[(0, 2, PERSON, 0.9), (3, 5, PERSON, 0.8), (6, 7, ORGANIZATION, 0.6)]),
        entities(&[(1, 2, PERSON, 0.7), (3, 5, PERSON, 0.85), (6, 7, LOCATION, 0.95)]),
    ]
}

const NAMES: [&str; 3] = ["dictionary", "maxent", "crf"];

fn document() -> Document {
    let mut doc = Document::new(TEXT);
    let mut spans = Spans::new(ViewId::SOURCE);
    for (a, b) in TOKENS {
        spans.push(Span::new(a, b));
    }
    doc.add(&Key::DEFAULT, Tokens { spans, sentence: None }, Provenance::by("test")).unwrap();
    for (name, layer) in NAMES.into_iter().zip(inputs()) {
        let mut p = Provenance { reads: vec![Key::<Tokens>::DEFAULT.id()], ..Provenance::by(name) };
        if name != "dictionary" {
            p.model = Some(model(name));
        }
        doc.add(&Key::named(name), layer, p).unwrap();
    }
    doc
}

/// (first token, end token, label, score) of each merged entity.
fn rows(e: &Entities) -> Vec<(usize, usize, &str, f64)> {
    (0..e.score.len())
        .map(|i| (e.tokens[i].start.get(), e.tokens[i].end.get(), e.kinds.label(e.kind[i]), e.score[i]))
        .collect()
}

fn merged(policy: EntityMergePolicy) -> Vec<(usize, usize, String, f64)> {
    let layers = inputs();
    let refs: Vec<&Entities> = layers.iter().collect();
    rows(&consensus(&refs, policy).unwrap().entities).into_iter().map(|(a, b, l, s)| (a, b, l.to_owned(), s)).collect()
}

fn row(a: usize, b: usize, label: &str, score: f64) -> (usize, usize, String, f64) {
    (a, b, label.to_owned(), score)
}

#[test]
fn raw_union_keeps_every_distinct_range_once_with_its_votes() {
    assert_eq!(
        merged(EntityMergePolicy::RawUnion),
        [
            row(0, 2, "person", 0.9),
            row(1, 2, "person", 0.7),
            row(3, 5, "person", 0.85),
            row(6, 7, "organization", 0.6),
            row(6, 7, "location", 0.95),
        ]
    );
    let layers = inputs();
    let refs: Vec<&Entities> = layers.iter().collect();
    let c = consensus(&refs, EntityMergePolicy::RawUnion).unwrap();
    assert_eq!(c.votes_of(0), [Vote { input: 0, entity: 0 }, Vote { input: 1, entity: 0 }]);
    assert_eq!(c.winner[0], Vote { input: 1, entity: 0 });
}

#[test]
fn priority_fallback_takes_the_first_recognizer_then_fills_gaps() {
    // The dictionary's two names win their places; the maxent finder's
    // organization overlaps the dictionary's location and is dropped. The
    // score is still the highest any voter gave.
    assert_eq!(
        merged(EntityMergePolicy::PriorityFallback),
        [row(0, 2, "person", 0.9), row(3, 5, "person", 0.85), row(6, 7, "location", 0.95)]
    );
    let layers = inputs();
    let refs: Vec<&Entities> = layers.iter().collect();
    let c = consensus(&refs, EntityMergePolicy::PriorityFallback).unwrap();
    assert_eq!(c.winner, [Vote { input: 0, entity: 0 }, Vote { input: 1, entity: 1 }, Vote { input: 0, entity: 1 }]);
    // Overlapping ranges of one type vote together: the CRF's narrower
    // "Lovelace" counts for "Ada Lovelace".
    assert_eq!(c.votes_of(0).len(), 3);
}

#[test]
fn consensus_keeps_what_enough_recognizers_agree_on() {
    let at_least = |n| EntityMergePolicy::Consensus {
        min_votes: NonZeroU32::new(n).unwrap(),
        overlap_rule: FilterRule::HighestScore,
    };
    assert_eq!(merged(at_least(2)), [row(0, 2, "person", 0.9), row(3, 5, "person", 0.85), row(6, 7, "location", 0.95)]);
    assert_eq!(merged(at_least(3)), [row(0, 2, "person", 0.9)]);
    assert_eq!(merged(at_least(4)), []);
    // With one vote enough, the location and the organization over
    // "London" overlap, and the higher score stays.
    let one = merged(at_least(1));
    assert!(one.contains(&row(6, 7, "location", 0.95)) && !one.iter().any(|r| r.2 == "organization"), "{one:?}");
}

#[test]
fn the_step_writes_entities_and_evidence_that_verifies() {
    let inputs: Vec<Key<Entities>> = NAMES.into_iter().map(Key::named).collect();
    let merger = ConsensusEntityMerger::new(inputs, Key::named("merged"), EntityMergePolicy::PriorityFallback);
    let mut builder = Analyzer::builder().given(Key::<Tokens>::DEFAULT);
    for name in NAMES {
        builder = builder.given(Key::<Entities>::named(name));
    }
    let analyzer = builder.step(merger).build().unwrap();
    let mut docs = vec![document()];
    let results = analyzer.run(&mut docs);
    assert!(results.iter().all(Result::is_ok), "{results:?}");
    let doc = &docs[0];
    let entities = doc.get_key(&Key::<Entities>::named("merged")).unwrap();
    assert_eq!(rows(entities).len(), 3);
    let ev = doc.get_key(&Key::<Evidence<Entities>>::named("merged")).unwrap();
    ev.verify(doc).unwrap();
    assert_eq!(ev.records(), 3);
    // "Ada Lovelace": the dictionary's and maxent's spans, then the CRF's
    // narrower one, each pointing at its voter's entity.
    assert_eq!(ev.spans_of(0), [Span::new(0, 12), Span::new(0, 12), Span::new(4, 12)]);
    let support: Vec<(u32, u32)> = ev.support_of(0).iter().map(|s| (s.layer, s.start)).collect();
    assert_eq!(support, [(0, 0), (1, 0), (2, 0)]);
    assert_eq!(ev.layers.iter().map(|l| l.instance.as_str()).collect::<Vec<_>>(), NAMES);
    assert_eq!(ev.producer_of(0).annotator, "dictionary");
    assert_eq!(ev.producer_of(1).model, Some(model("maxent")));
    // The dictionary ran no model, so the raw score is the best of the
    // two statistical finders.
    assert_eq!(ev.raw_of(0), Some(0.9));
    assert_eq!(ev.raw_of(2), Some(0.95));
}

#[test]
fn inputs_in_different_views_are_refused() {
    let mut layers = inputs();
    layers[2].spans = Spans::new(ViewId(1));
    let refs: Vec<&Entities> = layers.iter().collect();
    assert!(consensus(&refs, EntityMergePolicy::RawUnion).err().unwrap().contains("different views"));
}

#[test]
fn an_entity_that_bridges_two_keeps_the_first_recognizers_both() {
    // The CRF's "Lovelace met Charles" overlaps both of the dictionary's
    // names: it votes for the first, and the second is still kept.
    let first = entities(&[(0, 2, PERSON, 0.9), (3, 5, PERSON, 0.8)]);
    let bridge = entities(&[(1, 4, PERSON, 0.95)]);
    let c = consensus(&[&first, &bridge], EntityMergePolicy::PriorityFallback).unwrap();
    assert_eq!(rows(&c.entities), [(0, 2, "person", 0.95), (3, 5, "person", 0.8)]);
    assert_eq!(c.votes_of(0), [Vote { input: 0, entity: 0 }, Vote { input: 1, entity: 0 }]);
    assert_eq!(c.votes_of(1), [Vote { input: 0, entity: 1 }]);
    // Best score first, the bridge wins and both names vote for it.
    let one = EntityMergePolicy::Consensus { min_votes: NonZeroU32::MIN, overlap_rule: FilterRule::HighestScore };
    let c = consensus(&[&first, &bridge], one).unwrap();
    assert_eq!(rows(&c.entities), [(1, 4, "person", 0.95)]);
    assert_eq!(c.votes_of(0).len(), 3);
}

#[test]
fn every_vote_overlaps_the_entity_it_is_for() {
    // A chain of overlapping ranges from two recognizers: no vote may
    // reach past its own entity to the far end of the chain.
    let a = entities(&[(0, 2, PERSON, 0.5), (3, 5, PERSON, 0.6), (6, 8, PERSON, 0.7)]);
    let b = entities(&[(1, 4, PERSON, 0.9), (4, 7, PERSON, 0.4)]);
    let layers = [&a, &b];
    for policy in [
        EntityMergePolicy::PriorityFallback,
        EntityMergePolicy::Consensus { min_votes: NonZeroU32::MIN, overlap_rule: FilterRule::LongestFirst },
        EntityMergePolicy::RawUnion,
    ] {
        let c = consensus(&layers, policy).unwrap();
        for i in 0..c.winner.len() {
            let w = c.entities.tokens[i];
            for v in c.votes_of(i) {
                let t = layers[v.input as usize].tokens[v.entity as usize];
                assert!(t.start < w.end && w.start < t.end, "{policy:?}: vote {v:?} {t:?} for {w:?}");
            }
        }
        // Each candidate votes once.
        assert_eq!(c.votes.len(), 5, "{policy:?}");
    }
}

#[test]
fn priority_fallback_preserves_nested_entities_from_the_first_recognizer() {
    for inner_kind in [PERSON, ORGANIZATION] {
        let first = entities(&[(0, 3, PERSON, 0.4), (1, 2, inner_kind, 0.9)]);
        let later = entities(&[(6, 7, LOCATION, 0.8)]);
        let merged = consensus(&[&first, &later], EntityMergePolicy::PriorityFallback).unwrap();
        assert_eq!(rows(&merged.entities), [(0, 3, "person", 0.4),
            (1, 2, if inner_kind == PERSON { "person" } else { "organization" }, 0.9),
            (6, 7, "location", 0.8)]);
    }
}

#[test]
fn rejected_priority_candidate_does_not_consume_a_usable_fallback() {
    let first = entities(&[(0, 1, PERSON, 0.7)]);
    let later = entities(&[(0, 3, ORGANIZATION, 0.9), (2, 4, ORGANIZATION, 0.8)]);
    let merged = consensus(&[&first, &later], EntityMergePolicy::PriorityFallback).unwrap();
    assert_eq!(rows(&merged.entities), [(0, 1, "person", 0.7), (2, 4, "organization", 0.9)]);
    assert_eq!(merged.votes_of(1), [Vote { input: 1, entity: 0 }, Vote { input: 1, entity: 1 }]);
}
