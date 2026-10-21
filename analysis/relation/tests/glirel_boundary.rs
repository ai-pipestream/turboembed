//! Requests the GLiREL scorer refuses with an error instead of a panic: a
//! pair naming a mention that is not there, no labels, and a threshold
//! that is not a probability. The model is the reduced fixture's seeded
//! random encoder and heads, with a small SentencePiece model from
//! `testdata/sentencepiece` as the tokenizer; nothing here needs the
//! checkpoint.

use turbo::deberta::{Config, Encoder};
use turbo::safetensors::File;
use turbo::sentencepiece::SentencePiece;
use turbo_analysis::{ModelRef, Span};
use turbo_relation::glirel::heads::Heads;
use turbo_relation::glirel::prompt::Specials;
use turbo_relation::{GlirelRelations, Mention, RelationError, RelationScorer};

const DIR: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/fixtures/glirel");

fn model() -> GlirelRelations {
    let config = Config {
        hidden: 32,
        intermediate: 48,
        heads: 2,
        layers: 2,
        vocabulary: 800,
        max_sequence: 32,
        position_buckets: 8,
        max_relative_positions: 32,
        layer_norm_eps: 1e-7,
    };
    let bytes = std::fs::read(format!("{DIR}/reduced.safetensors")).unwrap();
    let canonical = turbo_bundle::deberta::convert(&config, &bytes, "token_rep_layer.bert_layer.model.").unwrap();
    let encoder = Encoder::from_safetensors(config, &canonical).unwrap();
    let heads = Heads::load(&File::parse("reduced", &bytes).unwrap(), 32).unwrap();
    let spm = concat!(env!("CARGO_MANIFEST_DIR"), "/../../testdata/sentencepiece/unigram-identity.model");
    let tokenizer = SentencePiece::parse(&std::fs::read(spm).unwrap()).unwrap();
    let model = ModelRef { id: "reduced".into(), revision: "seeded".into(), sha256: "none".into() };
    GlirelRelations::new(encoder, heads, tokenizer, Specials { cls: 1, sep: 2, rel: 3 }, model, 512).unwrap()
}

fn mention(start: u32, end: u32) -> Mention<'static> {
    Mention { span: Span::new(start, end), kind: "ENT" }
}

#[test]
fn a_pair_outside_the_mentions_is_an_error() {
    let mut out = Vec::new();
    let got = model().score("a b", &[mention(0, 1)], &[(0, 1)], &["label"], &mut out);
    assert!(matches!(got, Err(RelationError::Scorer(_))), "{got:?}");
    assert!(out.is_empty());
}

#[test]
fn no_labels_is_an_error() {
    let m = model();
    let mut out = Vec::new();
    let got = m.score("a b", &[mention(0, 1), mention(2, 3)], &[(0, 1)], &[], &mut out);
    assert_eq!(got, Err(RelationError::NoLabels));
    assert_eq!(m.extract("a b", &[(0, 0), (1, 1)], &[], 0.5).unwrap_err(), RelationError::NoLabels);
    assert!(m.pass("a b", &[]).is_err());
}

#[test]
fn a_threshold_that_is_not_a_probability_is_an_error() {
    let m = model();
    for t in [f32::NAN, f32::INFINITY, -0.1, 1.5] {
        let got = m.extract("a b", &[(0, 0), (1, 1)], &["label"], t);
        assert!(matches!(got, Err(RelationError::BadThreshold { .. })), "{t}: {got:?}");
    }
    for t in [0.0, 0.5, 1.0] {
        assert!(m.extract("a b", &[(0, 0), (1, 1)], &["label"], t).is_ok(), "{t}");
    }
}

/// A pass's rows are read only through its accessors, so they always fit
/// its prompt; entities and pairs outside them are errors.
#[test]
fn a_pass_is_checked_before_its_rows_are_indexed() {
    let m = model();
    let pass = m.pass("a b", &["label"]).unwrap();
    assert_eq!(pass.lstm().len(), pass.prompt().words.len() * 16);
    assert_eq!(pass.types().len(), 16);
    assert_eq!(m.logits(&pass, &[(0, 0), (1, 1)], &[(0, 1), (1, 0)]).unwrap().len(), 2);
    assert!(m.logits(&pass, &[(0, 2)], &[]).is_err(), "an entity past the last word");
    assert!(m.logits(&pass, &[(1, 0)], &[]).is_err(), "an entity ending before it starts");
    assert!(m.logits(&pass, &[(0, 0)], &[(0, 1)]).is_err(), "a pair past the entities");
}
