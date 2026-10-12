//! Drift and cohesion in pipelines: the boundary steps from the registry,
//! weighed into the costs the chunker reads, and the cohesive chunker.

use turbo_analysis::annotate::Step;
use turbo_analysis::registry::{Registry, Spec};
use turbo_analysis::*;
use turbo_analysis_layers::{Block, Boundaries, Chunks, Cut, Sentences, Structure, Subwords, Vectors};
use turbo_chunker::{
    Budget, Chunker, Cohesion, CohesiveChunker, DriftProvider, StructureBoundariesProvider, WeighProvider,
};

const TEXT: &str = "Cats purr. Cats nap. Rust compiles. Rust checks. Rust ships. Rust runs.";
const BUDGET: Budget = Budget { max: 12, target: 9, size_weight: 1000, overhead: 0 };

fn toy_model() -> ModelRef {
    ModelRef { id: "toy".into(), revision: "1".into(), sha256: "0".into() }
}

/// Sentences (each ends at a period), subwords (each word and period is
/// one, so each sentence is 3) and a vector per sentence: the two about
/// cats point one way, the four about Rust at right angles.
fn document() -> Document {
    let mut doc = Document::new(TEXT);
    let (mut sentences, mut pieces) = (Spans::new(ViewId::SOURCE), Spans::new(ViewId::SOURCE));
    let mut start = 0;
    for (i, c) in TEXT.char_indices() {
        if c == '.' {
            sentences.push(Span::new(start, i as u32 + 1));
            start = i as u32 + 2;
        }
    }
    let mut at = 0;
    for word in TEXT.split(' ') {
        let w = word.trim_end_matches('.');
        pieces.push(Span::new(at, at + w.len() as u32));
        if w.len() < word.len() {
            pieces.push(Span::new(at + w.len() as u32, at + word.len() as u32));
        }
        at += word.len() as u32 + 1;
    }
    let ids = vec![0; pieces.len()];
    let data = vec![1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0];
    doc.add(&Key::DEFAULT, Sentences { spans: sentences }, Provenance::by("t")).unwrap();
    doc.add(&Key::DEFAULT, Subwords { spans: pieces, ids, model: toy_model(), token: None }, Provenance::by("t"))
        .unwrap();
    doc.add(&Key::DEFAULT, Vectors::<Sentences>::new(2, data, Vec::new(), toy_model()), Provenance::by("t")).unwrap();
    doc
}

fn registry() -> Registry<dyn Step> {
    let mut r = Registry::default();
    r.register(StructureBoundariesProvider).register(DriftProvider).register(WeighProvider);
    r
}

fn given(b: analyzer::Builder) -> analyzer::Builder {
    b.given(Key::<Sentences>::DEFAULT).given(Key::<Subwords>::DEFAULT).given(Key::<Vectors<Sentences>>::DEFAULT)
}

fn texts(d: &Document) -> Vec<String> {
    let c = d.get::<Chunks>().unwrap();
    c.spans.iter().map(|(_, s)| d.source()[s.range()].to_owned()).collect()
}

#[test]
fn drift_weighed_with_structure_moves_the_cut_to_the_topic_shift() {
    let structure_only = registry()
        .pipeline(&[("structure-boundaries", Spec::default())], |b| given(b).add(Chunker { budget: BUDGET }))
        .unwrap();
    let mut docs = vec![document()];
    assert!(structure_only.run(&mut docs)[0].is_ok());
    // Every gap costs a unit, so the size alone cuts 9 + 9 tokens.
    assert_eq!(texts(&docs[0]), ["Cats purr. Cats nap. Rust compiles.", "Rust checks. Rust ships. Rust runs."]);

    let steps = [
        ("structure-boundaries", Spec::default().with("instance", "structure")),
        // Two sentences on each side: the shift is after the second.
        ("drift-boundaries", Spec::default().with("window", "2")),
        ("weigh-boundaries", Spec::default().with("drift", "1000")),
    ];
    let weighed = registry().pipeline(&steps, |b| given(b).add(Chunker { budget: BUDGET })).unwrap();
    assert_eq!(weighed.plan().len(), 3);
    let mut docs = vec![document()];
    assert!(weighed.run(&mut docs)[0].is_ok());
    assert_eq!(texts(&docs[0]), ["Cats purr. Cats nap.", "Rust compiles. Rust checks. Rust ships. Rust runs."]);
    assert_eq!(docs[0].get::<Chunks>().unwrap().cut, [Cut::DocumentStart, Cut::TopicShift]);
    let drift = docs[0].get_key(&Key::<Boundaries>::named("drift")).unwrap();
    let summed = docs[0].get::<Boundaries>().unwrap();
    assert!(drift.cost[1] < -Boundaries::UNIT);
    assert_eq!(summed.cost[1], Boundaries::UNIT + drift.cost[1]);
    // The weighed layer says what it was made from.
    let reads = &docs[0].provenance(Key::<Boundaries>::DEFAULT.id()).unwrap().reads;
    assert!(reads.contains(&Key::<Boundaries>::named("drift").id()));
}

#[test]
fn the_providers_refuse_what_they_do_not_know() {
    let r = registry();
    assert!(r.select("drift-boundaries", &Spec::default().with("windwo", "3")).is_err());
    assert!(r.create("drift-boundaries", &Spec::default().with("window", "three")).is_err());
    assert!(r.create("drift-boundaries", &Spec::default().with("score", "percentile")).is_err());
    assert!(r.create("drift-boundaries", &Spec::default().with("score", "distance").with("scale", "5")).is_ok());
    // Drift only ever lowers a cost, and a window holds a sentence.
    assert!(r.create("drift-boundaries", &Spec::default().with("scale", "-5")).is_err());
    assert!(r.create("drift-boundaries", &Spec::default().with("window", "0")).is_err());
    assert!(r.create("drift-boundaries", &Spec::default().with("window", "1")).is_ok());
    assert!(r.create("weigh-boundaries", &Spec::default().with("drift", "1.5")).is_err());
    assert!(r.select("structure-boundaries", &Spec::default().with("instance", "drift")).is_err());
}

#[test]
fn the_cohesive_chunker_cuts_where_the_vectors_part() {
    let mut doc = document();
    let n = 6;
    let flat = Boundaries { cost: vec![Boundaries::UNIT; n], reason: vec![Cut::Sentence; n] };
    doc.add(&Key::DEFAULT, flat, Provenance::by("t")).unwrap();
    let a = Analyzer::builder()
        .given(Key::<Sentences>::DEFAULT)
        .given(Key::<Subwords>::DEFAULT)
        .given(Key::<Vectors<Sentences>>::DEFAULT)
        .given(Key::<Boundaries>::DEFAULT)
        .add(CohesiveChunker { chunker: Chunker { budget: BUDGET }, cohesion: Cohesion::default() })
        .build()
        .unwrap();
    let mut docs = vec![doc];
    assert!(a.run(&mut docs)[0].is_ok());
    assert_eq!(texts(&docs[0]), ["Cats purr. Cats nap.", "Rust compiles. Rust checks. Rust ships. Rust runs."]);
    // The term is in the chunk's cost: none, for chunks that do not spread.
    assert_eq!(docs[0].get::<Chunks>().unwrap().cost, [1000 + 111, 111]);
}

#[test]
fn cohesion_is_summed_within_each_section() {
    // Two sections; in the second, two sentences point one way and two
    // the other. Flat boundaries keep its 12 tokens one chunk; cohesion
    // splits it where the vectors turn.
    let mut doc = document();
    let n = 6;
    let flat = Boundaries { cost: vec![Boundaries::UNIT; n], reason: vec![Cut::Sentence; n] };
    doc.add(&Key::DEFAULT, flat, Provenance::by("t")).unwrap();
    let data = vec![1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 0.0];
    doc.add(&Key::named("turned"), Vectors::<Sentences>::new(2, data, Vec::new(), toy_model()), Provenance::by("t"))
        .unwrap();
    let mut spans = Spans::new(ViewId::SOURCE);
    spans.push(Span::new(0, 20));
    spans.push(Span::new(21, 71));
    let st = Structure { spans, kind: vec![Block::Section; 2], parent: vec![None; 2], depth: vec![1; 2] };
    doc.add(&Key::DEFAULT, st, Provenance::by("parser")).unwrap();
    let level0 = |cohesion: Option<Cohesion>| {
        let b = Analyzer::builder()
            .given(Key::<Sentences>::DEFAULT)
            .given(Key::<Subwords>::DEFAULT)
            .given(Key::<Vectors<Sentences>>::named("turned"))
            .given(Key::<Boundaries>::DEFAULT)
            .given(Key::<Structure>::DEFAULT);
        let a = match cohesion {
            Some(cohesion) => {
                b.step(Bound::new(CohesiveChunker { chunker: Chunker { budget: BUDGET }, cohesion }).read(4, "turned"))
            }
            None => b.add(Chunker { budget: BUDGET }),
        };
        let mut docs = vec![doc.clone()];
        assert!(a.build().unwrap().run(&mut docs)[0].is_ok());
        let c = docs[0].get::<Chunks>().unwrap();
        c.spans
            .iter()
            .filter(|(i, _)| c.level[i.get()] == 0)
            .map(|(_, s)| TEXT[s.range()].to_owned())
            .collect::<Vec<_>>()
    };
    assert_eq!(level0(None), ["Cats purr. Cats nap.", "Rust compiles. Rust checks. Rust ships. Rust runs."]);
    assert_eq!(
        level0(Some(Cohesion::default())),
        ["Cats purr. Cats nap.", "Rust compiles. Rust checks.", "Rust ships. Rust runs."]
    );
}
