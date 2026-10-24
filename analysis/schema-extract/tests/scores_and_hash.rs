//! The document categories carry every label's probability, chosen or not,
//! and the provenance config hash follows every setting that changes the
//! annotator's output while staying stable across equal settings.

mod common;

use turbo_analysis::{Analyzer, Annotator, Document, Key, Provenance, Span, Spans, ViewId};
use turbo_analysis_layers::{Categories, Tokens};
use turbo_schema_extract::{ClassificationTask, ExtractionSchema, Label, Limits, SchemaAnnotator};

fn task(name: &str, labels: &[&str], multi_label: bool, threshold: f32) -> ClassificationTask {
    ClassificationTask {
        name: name.into(),
        labels: labels.iter().map(|l| Label { name: (*l).into(), description: None }).collect(),
        multi_label,
        threshold,
        top_k: None,
    }
}

/// A single-label task and a multi-label task whose threshold no sigmoid
/// reaches, so each keeps only its best label.
fn schema() -> ExtractionSchema {
    ExtractionSchema {
        classifications: vec![
            task("tone", &["calm", "urgent", "angry"], false, 0.5),
            task("topic", &["billing", "shipping", "refund"], true, 1.0),
        ],
        ..Default::default()
    }
}

fn categories(text: &str) -> Categories {
    let mut doc = Document::new(text);
    let mut spans = Spans::new(ViewId::SOURCE);
    let mut at = 0;
    for w in text.split_whitespace() {
        let start = text[at..].find(w).unwrap() + at;
        spans.push(Span::new(start as u32, (start + w.len()) as u32));
        at = start + w.len();
    }
    doc.add(&Key::DEFAULT, Tokens { spans, sentence: None }, Provenance::by("test")).unwrap();
    let annotator = SchemaAnnotator::new(common::offline(Limits::default()), schema(), 0.5).unwrap();
    let analyzer = Analyzer::builder().given(Key::<Tokens>::DEFAULT).add(annotator).build().unwrap();
    let mut docs = vec![doc];
    for r in analyzer.run(&mut docs) {
        r.unwrap();
    }
    docs.pop().unwrap().get::<Categories>().unwrap().clone()
}

#[test]
fn every_label_keeps_its_probability_not_only_the_chosen_ones() {
    let c = categories("The parcel never arrived and I want my money back today.");
    assert_eq!(turbo_analysis::Layer::check(&c), Ok(()));
    let row = c.get(0).expect("one document row");
    assert_eq!(row.scores.len(), 6);
    let name = |i: usize| row.categories.label(turbo_analysis::Tag(i as u16)).to_owned();
    // Each task keeps one label, so four of the six are unchosen; each
    // still has its probability, strictly inside (0, 1).
    assert_eq!(row.selected.len(), 2, "{:?}", row.selected);
    for (i, &p) in row.scores.iter().enumerate() {
        assert!(p > 0.0 && p < 1.0, "{} scored {p}", name(i));
    }
    // The single-label task is one softmax: its probabilities sum to 1.
    let tone: f64 = row.scores[..3].iter().sum();
    assert!((tone - 1.0).abs() < 1e-5, "tone sums to {tone}");
    // Each task's chosen label is its highest-scoring one.
    for range in [0..3, 3..6] {
        let chosen = row.selected.iter().map(|t| t.0 as usize).find(|i| range.contains(i)).unwrap();
        assert!(range.clone().all(|i| row.scores[i] <= row.scores[chosen]), "{} not the best", name(chosen));
    }
    // The best category is the highest-scoring chosen label.
    let best = row.selected.iter().max_by(|a, b| row.scores[a.0 as usize].total_cmp(&row.scores[b.0 as usize]));
    assert_eq!(Some(&row.best_tag), best);
}

fn hash(schema: ExtractionSchema, threshold: f32, limits: Limits) -> u64 {
    SchemaAnnotator::new(common::offline(limits), schema, threshold).unwrap().config_hash()
}

#[test]
fn the_config_hash_follows_limits_and_thresholds_canonically() {
    let base = hash(schema(), 0.5, Limits::default());
    assert_eq!(base, hash(schema(), 0.5, Limits::default()), "equal settings hash equal");
    // A limit refuses requests the default accepts: another configuration.
    let mut fewer = Limits::default();
    fewer.max_words -= 1;
    assert_ne!(base, hash(schema(), 0.5, fewer), "max_words");
    let mut fewer = Limits::default();
    fewer.max_prompt_subwords -= 1;
    assert_ne!(base, hash(schema(), 0.5, fewer), "max_prompt_subwords");
    // A zero threshold is one setting whatever its sign.
    assert_eq!(hash(schema(), 0.0, Limits::default()), hash(schema(), -0.0, Limits::default()));
    assert_ne!(base, hash(schema(), 0.25, Limits::default()), "threshold");
    // Every schema setting of a task counts, labels' descriptions included.
    let mut s = schema();
    s.classifications[1].top_k = Some(1);
    assert_ne!(base, hash(s, 0.5, Limits::default()), "top_k");
    let mut s = schema();
    s.classifications[0].labels[0].description = Some("relaxed".into());
    assert_ne!(base, hash(s, 0.5, Limits::default()), "description");
    // Names are length-prefixed, so moving a byte between two names is a
    // different schema.
    let mut a = schema();
    a.classifications[0].labels[0].name = "ab".into();
    a.classifications[0].labels[1].name = "c".into();
    let mut b = schema();
    b.classifications[0].labels[0].name = "a".into();
    b.classifications[0].labels[1].name = "bc".into();
    assert_ne!(hash(a, 0.5, Limits::default()), hash(b, 0.5, Limits::default()));
}
