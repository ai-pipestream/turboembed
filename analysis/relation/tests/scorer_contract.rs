//! What the zero-shot annotator does with a scorer that breaks the
//! `RelationScorer` contract, and with one that names a model. The scorers
//! here are the lexical scorer with its output reshaped after it scored:
//! one score short, a score past one, a score that is not a number; and
//! the lexical scorer reporting a model, which the annotator's provenance,
//! its configuration hash and the relations' evidence carry.

use std::sync::Arc;

use turbo_analysis::{
    Analyzer, Annotator, Document, Evidence, EvidenceStep, IxRange, Key, ModelRef, Provenance, Span, Spans, Tag,
    TagSet, View, ViewId,
};
use turbo_analysis_layers::{Entities, Tokens};
use turbo_relation::zeroshot::Mention;
use turbo_relation::{
    LexicalRelationScorer, RelationError, RelationLabel, RelationScorer, Relations, ZeroShotRelationAnnotator,
    ZeroShotRelationConfig,
};

/// The lexical scorer with `edit` applied to what it returns, and `model`
/// as its model.
struct Reshaped {
    edit: fn(&mut Vec<f32>),
    model: Option<ModelRef>,
}

impl RelationScorer for Reshaped {
    fn model_ref(&self) -> Option<ModelRef> {
        self.model.clone()
    }
    fn max_labels(&self) -> usize {
        usize::MAX
    }
    fn max_mentions(&self) -> usize {
        usize::MAX
    }
    fn score(
        &self,
        text: &str,
        mentions: &[Mention<'_>],
        pairs: &[(u32, u32)],
        labels: &[&str],
        out: &mut Vec<f32>,
    ) -> Result<(), RelationError> {
        LexicalRelationScorer.score(text, mentions, pairs, labels, out)?;
        (self.edit)(out);
        Ok(())
    }
}

const TEXT: &str = "Ann founded Acme.";

fn entities() -> Entities {
    let mut spans = Spans::new(ViewId::SOURCE);
    spans.push(Span::new(0, 3));
    spans.push(Span::new(12, 16));
    Entities {
        spans,
        tokens: vec![IxRange::new(0, 1), IxRange::new(2, 3)],
        kind: vec![Tag(0), Tag(0)],
        kinds: Arc::clone(&TagSet::new("kinds", &["THING"])),
        score: vec![1.0; 2],
    }
}

fn extract(edit: fn(&mut Vec<f32>)) -> Result<Relations, RelationError> {
    let doc = Document::new(TEXT);
    let config = ZeroShotRelationConfig::new(vec![RelationLabel::new("founded")]);
    let a = ZeroShotRelationAnnotator::new(config, Reshaped { edit, model: None }).unwrap();
    a.extract(doc.get::<View>().unwrap(), &entities(), None).map(|(r, _)| r)
}

#[test]
fn a_scorer_breaking_its_contract_is_refused() {
    assert_eq!(extract(|_| {}).unwrap().subject.len(), 2);
    let bad = |r: Result<Relations, RelationError>| matches!(r, Err(RelationError::BadScorerOutput(_)));
    assert!(bad(extract(|out| {
        out.pop();
    })));
    assert!(bad(extract(|out| out.push(0.5))));
    assert!(bad(extract(|out| out[0] = 1.5)));
    assert!(bad(extract(|out| out[1] = f32::NAN)));
    assert!(bad(extract(|out| out[0] = -0.25)));
}

#[test]
fn a_scorers_model_is_in_the_provenance_hash_and_evidence() {
    let model = ModelRef { id: "org/model".into(), revision: "1".into(), sha256: "ab".into() };
    let config = || ZeroShotRelationConfig::new(vec![RelationLabel::new("founded")]);
    let with = ZeroShotRelationAnnotator::new(config(), Reshaped { edit: |_| {}, model: Some(model.clone()) }).unwrap();
    let without = ZeroShotRelationAnnotator::new(config(), Reshaped { edit: |_| {}, model: None }).unwrap();
    assert_ne!(with.config_hash(), without.config_hash());
    assert_eq!(with.model(), Some(model.clone()));

    let mut doc = Document::new(TEXT);
    let mut tokens = Spans::new(ViewId::SOURCE);
    for (a, b) in [(0, 3), (4, 11), (12, 16)] {
        tokens.push(Span::new(a, b));
    }
    doc.add(&Key::DEFAULT, Tokens { spans: tokens, sentence: None }, Provenance::by("test")).unwrap();
    doc.add(&Key::DEFAULT, entities(), Provenance::by("test")).unwrap();
    let analyzer = Analyzer::builder()
        .given(Key::<Entities>::DEFAULT)
        .add(with)
        .step(EvidenceStep::new(Key::<Relations>::DEFAULT))
        .build()
        .unwrap();
    let mut docs = vec![doc];
    for r in analyzer.run(&mut docs) {
        r.unwrap();
    }
    let p = docs[0].provenance(Key::<Relations>::DEFAULT.id()).unwrap();
    assert_eq!(p.model.as_ref(), Some(&model));
    let ev = docs[0].get::<Evidence<Relations>>().unwrap();
    assert_eq!(ev.producer_of(0).model.as_ref(), Some(&model));
    assert_eq!(ev.verify(&docs[0]), Ok(()));
}
