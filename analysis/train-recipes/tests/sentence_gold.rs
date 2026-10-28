//! Project-authored samples exercise the native gate, not corpus accuracy.
use std::borrow::Cow;
use turbo_analysis::Span;
use turbo_eval::model_gate::{Counts, ModelId};
use turbo_formats::sample::SentenceSample;
use turbo_train_recipes::{
    Recipe, Reference,
    segmentation::{self, HeldOut},
    sentence_gold,
};

fn samples() -> HeldOut<'static> {
    HeldOut::Sentences(
        (0..30)
            .map(|_| {
                SentenceSample::new(
                    Cow::Borrowed("Café opens. It closes."),
                    vec![Span::new(0, 12), Span::new(13, 23)],
                    vec![Span::new(0, 12), Span::new(13, 23)],
                    Span::new(0, 23),
                )
                .unwrap()
            })
            .collect(),
    )
}
fn candidate() -> ModelId {
    ModelId { id: "project-authored-spans".into(), sha256: "1".repeat(64) }
}
fn predictions() -> Vec<Vec<Span>> {
    vec![vec![Span::new(0, 12), Span::new(13, 23)]; 30]
}
fn run(
    held: &HeldOut<'_>,
    predicted: &[Vec<Span>],
) -> turbo_train_recipes::error::Result<turbo_train_recipes::GateRun> {
    sentence_gold::gate(candidate(), held, predicted, "project-authored held-out split", &"2".repeat(64))
}

#[test]
fn canonical_sentence_recipes_have_no_model_reference() {
    assert_eq!(Recipe::Segmentation(segmentation::SENTENCE).reference().unwrap(), Reference::GoldSentences);
    for recipe in turbo_train_recipes::ud::UD_RECIPES {
        if recipe.component == turbo_train_recipes::ud::Component::Sentence {
            assert_eq!(recipe.reference, Reference::GoldSentences);
            assert_eq!(recipe.segmentation().unwrap().reference, Reference::GoldSentences);
        }
    }
    assert!(Reference::GoldSentences.file().is_err());
    assert!(Reference::GoldSentences.check(b"model bytes").is_err());
    assert!(Reference::GoldSentences.check(b"").is_err());
    assert!(
        segmentation::SENTENCE.gate(b"candidate", "legacy", b"reference", &samples(), "set", &"2".repeat(64)).is_err()
    );
}

#[test]
fn both_predictions_and_baseline_are_scored_against_gold() {
    let result = run(&samples(), &predictions()).unwrap().record;
    result.check().unwrap();
    assert_eq!(result.reference, sentence_gold::baseline());
    assert_eq!(result.units.len(), 30);
    for unit in &result.units {
        assert_eq!(unit.candidate, Counts { true_positives: 2, selected: 2, target: 2 });
        assert_eq!(unit.reference, Counts { true_positives: 0, selected: 1, target: 2 });
    }
    assert_eq!(result.result.candidate.score, 100.0);
    assert_eq!(result.result.reference.score, 0.0);
    assert!(result.result.pass);
    assert!(result.result.difference.mean.is_finite());
}

#[test]
fn malformed_predictions_do_not_enter_the_gate() {
    for bad in [
        vec![Span::new(3, 4)],
        vec![Span::new(0, 30)],
        vec![Span::new(3, 3)],
        vec![Span::new(13, 23), Span::new(0, 12)],
        vec![Span::new(0, 12), Span::new(5, 15)],
    ] {
        let mut pred = predictions();
        pred[0] = bad;
        assert!(run(&samples(), &pred).is_err());
    }
    assert!(run(&samples(), &predictions()[..29]).is_err());
    assert!(run(&HeldOut::Sentences(vec![]), &[]).is_err());
    assert!(run(&HeldOut::Tokens { samples: vec![], skipped: 0 }, &[]).is_err());
}

#[test]
fn source_identity_and_gold_labels_affect_the_sample_digest() {
    let first = run(&samples(), &predictions()).unwrap().record;
    let HeldOut::Sentences(mut items) = samples() else { unreachable!() };
    items[0] = SentenceSample::new(
        Cow::Borrowed("Café opens. It closes."),
        vec![Span::new(0, 12), Span::new(13, 23)],
        vec![Span::new(100, 112), Span::new(113, 123)],
        Span::new(100, 123),
    )
    .unwrap();
    let second = run(&HeldOut::Sentences(items), &predictions()).unwrap().record;
    assert_ne!(first.set.samples_sha256, second.set.samples_sha256);
    assert_eq!(first.units, second.units);
}

#[test]
fn invalid_gold_sources_and_metadata_are_rejected() {
    let HeldOut::Sentences(mut items) = samples() else { unreachable!() };
    items[0] = SentenceSample::new(
        Cow::Borrowed("Café opens. It closes."),
        vec![Span::new(0, 12), Span::new(5, 23)],
        vec![Span::new(0, 12), Span::new(5, 23)],
        Span::new(0, 23),
    )
    .unwrap();
    assert!(run(&HeldOut::Sentences(items), &predictions()).is_err());
    let HeldOut::Sentences(mut items) = samples() else { unreachable!() };
    items[0] = SentenceSample::new(
        Cow::Borrowed("Café opens. It closes."),
        vec![Span::new(0, 12), Span::new(13, 23)],
        vec![Span { start: 12, end: 0 }, Span::new(13, 23)],
        Span::new(0, 23),
    )
    .unwrap();
    assert!(run(&HeldOut::Sentences(items), &predictions()).is_err());
    assert!(sentence_gold::gate(candidate(), &samples(), &predictions(), "set", "not-a-digest").is_err());
}

#[test]
fn changing_gold_labels_changes_the_digest_and_the_scores() {
    let first = run(&samples(), &predictions()).unwrap().record;
    let HeldOut::Sentences(mut items) = samples() else { unreachable!() };
    items[0] = SentenceSample::new(
        Cow::Borrowed("Café opens. It closes."),
        vec![Span::new(0, 23)],
        vec![Span::new(0, 23)],
        Span::new(0, 23),
    )
    .unwrap();
    let second = run(&HeldOut::Sentences(items), &predictions()).unwrap().record;
    assert_ne!(first.set.samples_sha256, second.set.samples_sha256);
    assert_eq!(second.units[0].candidate, Counts { true_positives: 0, selected: 2, target: 1 });
    assert_eq!(second.units[0].reference, Counts { true_positives: 1, selected: 1, target: 1 });
}

#[test]
fn a_completely_unlabelled_set_is_not_an_evaluation() {
    let empty = || SentenceSample::new(Cow::Borrowed(""), vec![], vec![], Span::default()).unwrap();
    let held = HeldOut::Sentences((0..30).map(|_| empty()).collect());
    assert!(run(&held, &vec![vec![]; 30]).is_err());
    let HeldOut::Sentences(mut mixed) = samples() else { unreachable!() };
    mixed[0] = empty();
    let mut predictions = predictions();
    predictions[0].clear();
    let result = run(&HeldOut::Sentences(mixed), &predictions).unwrap().record;
    assert_eq!(result.units[0].candidate, Counts::default());
    assert_eq!(result.units.iter().map(|u| u.candidate.target).sum::<u64>(), 58);
}

#[test]
fn invalid_candidate_identity_cannot_make_a_receipt() {
    for candidate in
        [ModelId { id: String::new(), sha256: "1".repeat(64) }, ModelId { id: "native".into(), sha256: "wrong".into() }]
    {
        assert!(sentence_gold::gate(candidate, &samples(), &predictions(), "set", &"2".repeat(64)).is_err());
    }
}

#[test]
fn merging_retirement_keeps_non_sentence_references_disabled() {
    let reference = Recipe::Segmentation(segmentation::TOKENS).reference().unwrap();
    assert!(matches!(reference, Reference::Retired(_)));
    assert!(reference.file().is_err());
    assert!(reference.sha256().is_err());
    assert!(reference.check(b"model").is_err());
    for recipe in turbo_train_recipes::ud::UD_RECIPES {
        if recipe.component != turbo_train_recipes::ud::Component::Sentence {
            assert!(matches!(recipe.reference, Reference::Retired(_)));
            assert!(recipe.reference.file().is_err());
            assert!(recipe.reference.sha256().is_err());
        }
    }
}
