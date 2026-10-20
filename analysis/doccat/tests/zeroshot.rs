//! Zero-shot NLI categorization against a scripted scorer:
//! hand-computed single-label and multi-label probabilities
//! over permuted 3-outcome and 2-outcome heads, premise
//! windowing with all three aggregations, every construction
//! refusal, and both annotators through an `Analyzer`.

use std::collections::BTreeMap;
use std::sync::Mutex;

use turbo_analysis::{Analyzer, Document, Key, Layer, ModelRef, Provenance, Span, Spans, ViewId};
use turbo_analysis_layers::{Categories, Sentences};
use turbo_doccat::{
    NliScorer, WindowAggregation, ZeroShotAnnotator, ZeroShotCategorizer, ZeroShotConfig, ZeroShotError, ZeroShotMode,
    ZeroShotSpanAnnotator,
};

/// Logits keyed by hypothesis; the default row otherwise.
/// Token counts are word counts, so budgets are exact.
struct FakeNli {
    outcomes: Vec<String>,
    by_hypothesis: BTreeMap<String, Vec<f32>>,
    by_premise: Option<fn(&str, &str) -> Vec<f32>>,
    max_pair_tokens: u32,
    count_zero: bool,
    fail: bool,
    seen: Mutex<Vec<(String, String)>>,
}

impl FakeNli {
    fn three(rows: &[(&str, [f32; 3])]) -> FakeNli {
        FakeNli {
            // Deliberately permuted: [contradiction, neutral, entailment].
            outcomes: ["CONTRADICTION", "Neutral", "entailment"].iter().map(|s| s.to_string()).collect(),
            by_hypothesis: rows.iter().map(|(h, r)| (h.to_string(), r.to_vec())).collect(),
            by_premise: None,
            max_pair_tokens: 64,
            count_zero: false,
            fail: false,
            seen: Mutex::new(Vec::new()),
        }
    }

    fn seen(&self) -> Vec<(String, String)> {
        self.seen.lock().unwrap().clone()
    }
}

impl NliScorer for FakeNli {
    fn outcome_labels(&self) -> &[String] {
        &self.outcomes
    }

    fn model_ref(&self) -> ModelRef {
        ModelRef { id: "fake-nli".into(), revision: "1".into(), sha256: "0".into() }
    }

    fn max_pair_tokens(&self) -> u32 {
        self.max_pair_tokens
    }

    fn count_tokens(&self, text: &str) -> usize {
        if self.count_zero { 0 } else { text.split_whitespace().count() }
    }

    fn score_pairs(&self, pairs: &[(&str, &str)]) -> Result<Vec<Vec<f32>>, ZeroShotError> {
        if self.fail {
            return Err(ZeroShotError::Scorer("boom".into()));
        }
        self.seen.lock().unwrap().extend(pairs.iter().map(|(p, h)| (p.to_string(), h.to_string())));
        Ok(pairs
            .iter()
            .map(|(premise, hypothesis)| {
                if let Some(f) = self.by_premise {
                    f(premise, hypothesis)
                } else {
                    self.by_hypothesis.get(*hypothesis).cloned().unwrap_or_else(|| vec![0.0; self.outcomes.len()])
                }
            })
            .collect())
    }
}

fn close(scores: &[f64], expected: &[f64]) {
    assert_eq!(scores.len(), expected.len());
    for (a, e) in scores.iter().zip(expected) {
        assert!((a - e).abs() <= 1e-6 * (1.0 + e.abs()), "{scores:?} vs {expected:?}");
    }
}

fn labels() -> Vec<String> {
    ["sports", "politics"].iter().map(|s| s.to_string()).collect()
}

fn config(mode: ZeroShotMode) -> ZeroShotConfig {
    ZeroShotConfig { mode, ..ZeroShotConfig::default() }
}

#[test]
fn single_and_multi_label_match_hand_values() {
    // Rows are [contradiction, neutral, entailment]: sports
    // entails at 2.0 against contradiction 0.0, politics at
    // 1.0 against 1.0; the loud neutral logit must not leak
    // into either mode.
    let rows = &[("This example is sports.", [0.0, 5.0, 2.0]), ("This example is politics.", [1.0, 5.0, 1.0])];
    let single = ZeroShotCategorizer::new(FakeNli::three(rows), labels(), config(ZeroShotMode::SingleLabel)).unwrap();
    // softmax([2, 1])
    close(&single.categorize("the match report").unwrap(), &[0.7310585786300049, 0.2689414213699951]);
    assert_eq!(single.best("the match report").unwrap().0, "sports");
    assert_eq!(single.hypothesis("sports"), "This example is sports.");

    let multi = ZeroShotCategorizer::new(FakeNli::three(rows), labels(), config(ZeroShotMode::MultiLabel)).unwrap();
    // softmax2(2, 0) and softmax2(1, 1)
    close(&multi.categorize("the match report").unwrap(), &[0.8807970779778823, 0.5]);
}

#[test]
fn two_outcome_heads_resolve_not_entailment() {
    let mut fake = FakeNli::three(&[]);
    fake.outcomes = ["Not_Entailment", "ENTAILMENT"].iter().map(|s| s.to_string()).collect();
    fake.by_hypothesis = [
        ("This example is sports.".to_string(), vec![0.0, 2.0]),
        ("This example is politics.".to_string(), vec![2.0, 0.0]),
    ]
    .into_iter()
    .collect();
    let multi = ZeroShotCategorizer::new(fake, labels(), config(ZeroShotMode::MultiLabel)).unwrap();
    close(&multi.categorize("the match report").unwrap(), &[0.8807970779778823, 0.11920292202211755]);
}

#[test]
fn construction_refusals_are_typed() {
    let fake = || FakeNli::three(&[]);
    assert!(matches!(
        ZeroShotCategorizer::new(fake(), vec![], ZeroShotConfig::default()),
        Err(ZeroShotError::NoLabels)
    ));
    assert!(matches!(
        ZeroShotCategorizer::new(fake(), vec!["ok".into(), "  ".into()], ZeroShotConfig::default()),
        Err(ZeroShotError::BlankLabel(1))
    ));
    assert!(matches!(
        ZeroShotCategorizer::new(fake(), vec!["a".into(), "a".into()], ZeroShotConfig::default()),
        Err(ZeroShotError::DuplicateLabel(_))
    ));
    for template in ["no placeholder", "{} and {}", "{}", "   "] {
        let config = ZeroShotConfig { template: template.into(), ..ZeroShotConfig::default() };
        assert!(
            matches!(ZeroShotCategorizer::new(fake(), labels(), config), Err(ZeroShotError::InvalidTemplate(_))),
            "template {template:?}"
        );
    }
    let mut odd = FakeNli::three(&[]);
    odd.outcomes = vec!["yes".into(), "no".into()];
    assert!(matches!(
        ZeroShotCategorizer::new(odd, labels(), ZeroShotConfig::default()),
        Err(ZeroShotError::UnsupportedOutcomes(_))
    ));
    let mut cramped = FakeNli::three(&[]);
    cramped.max_pair_tokens = 4;
    assert!(matches!(
        ZeroShotCategorizer::new(cramped, vec!["a very long label".into()], ZeroShotConfig::default()),
        Err(ZeroShotError::HypothesisTooLong { .. })
    ));
    let ok = ZeroShotCategorizer::new(fake(), labels(), ZeroShotConfig::default()).unwrap();
    assert!(matches!(ok.categorize("   "), Err(ZeroShotError::EmptyPremise)));
}

fn windowed_scorer() -> FakeNli {
    let mut fake = FakeNli::three(&[]);
    fake.max_pair_tokens = 8;
    // Entailment counts the word "goal" in the premise window;
    // contradiction is a flat 0.
    fake.by_premise =
        Some(|premise, _| vec![0.0, 0.0, premise.split_whitespace().filter(|w| *w == "goal").count() as f32]);
    fake
}

#[test]
fn long_premises_window_without_truncating_the_hypothesis() {
    // Hypothesis "This example is sports." is 4 word-tokens,
    // the budget is 8, so premise windows hold 4 words:
    // [goal a b c] [goal goal d e] [f g] with goal counts
    // 1, 2, 0 over token weights 4, 4, 2.
    let premise = "goal a b c goal goal d e f g";
    for (aggregation, expected_logit) in
        [(WindowAggregation::Max, 2.0), (WindowAggregation::Mean, 1.0), (WindowAggregation::LengthWeightedMean, 1.2)]
    {
        let categorizer = ZeroShotCategorizer::new(
            windowed_scorer(),
            vec!["sports".into()],
            ZeroShotConfig { mode: ZeroShotMode::MultiLabel, aggregation, ..ZeroShotConfig::default() },
        )
        .unwrap();
        assert_eq!(categorizer.premise_windows(premise).unwrap(), ["goal a b c", "goal goal d e", "f g"]);
        let p = 1.0 / (1.0 + f64::exp(-expected_logit));
        close(&categorizer.categorize(premise).unwrap(), &[p]);
        for (_, hypothesis) in categorizer.scorer().seen() {
            assert_eq!(hypothesis, "This example is sports.", "every window carries the full hypothesis");
        }
    }
    // Overlap slides the windows instead of tiling them.
    let sliding = ZeroShotCategorizer::new(
        windowed_scorer(),
        vec!["sports".into()],
        ZeroShotConfig { overlap_words: 2, ..ZeroShotConfig::default() },
    )
    .unwrap();
    assert_eq!(sliding.premise_windows(premise).unwrap(), ["goal a b c", "b c goal goal", "goal goal d e", "d e f g"]);
    let stuck = ZeroShotCategorizer::new(
        windowed_scorer(),
        vec!["sports".into()],
        ZeroShotConfig { overlap_words: 4, ..ZeroShotConfig::default() },
    )
    .unwrap();
    assert!(matches!(stuck.premise_windows(premise), Err(ZeroShotError::OverlapTooLarge { .. })));
}

fn spans<L>(pairs: &[(u32, u32)]) -> Spans<L> {
    let mut s = Spans::new(ViewId::SOURCE);
    for &(a, b) in pairs {
        s.push(Span::new(a, b));
    }
    s
}

#[test]
fn annotators_write_categories() {
    let rows = &[("This example is sports.", [0.0, 0.0, 2.0]), ("This example is politics.", [2.0, 0.0, 0.0])];
    let text = "The derby ended level. The senate voted today.";
    let mut doc = Document::new(text);
    doc.add(&Key::DEFAULT, Sentences { spans: spans(&[(0, 22), (23, 45)]) }, Provenance::by("test")).unwrap();

    let document = ZeroShotAnnotator::new(
        ZeroShotCategorizer::new(FakeNli::three(rows), labels(), ZeroShotConfig::default()).unwrap(),
    );
    let analyzer = Analyzer::builder()
        .given(Key::<turbo_analysis::View>::DEFAULT)
        .given(Key::<Sentences>::DEFAULT)
        .add(document)
        .build()
        .unwrap();
    let mut docs = vec![doc.clone()];
    analyzer.run(&mut docs)[0].as_ref().unwrap();
    let cats = docs[0].get::<Categories>().unwrap();
    assert_eq!(cats.best(), Some(("sports", 0.8807970779778823)));

    let spans_annotator = ZeroShotSpanAnnotator::<_, Sentences>::new(
        ZeroShotCategorizer::new(FakeNli::three(rows), labels(), config(ZeroShotMode::MultiLabel)).unwrap(),
    );
    let analyzer = Analyzer::builder()
        .given(Key::<turbo_analysis::View>::DEFAULT)
        .given(Key::<Sentences>::DEFAULT)
        .add(spans_annotator)
        .build()
        .unwrap();
    let mut docs = vec![doc];
    analyzer.run(&mut docs)[0].as_ref().unwrap();
    let cats = docs[0].get::<Categories<Sentences>>().unwrap();
    assert_eq!(cats.len(), 2);
    assert_eq!(cats.best_category(0), "sports");
    close(cats.row_scores(1), &[0.8807970779778823, 0.11920292202211755]);
}

#[test]
fn length_weighted_mean_with_zero_token_counts_falls_back_to_mean() {
    // A scorer whose tokenizer strips every word of the
    // premise counts 0 tokens per window: the weighted mean
    // must not divide 0 by 0 into a NaN blamed on the scorer.
    let mut fake = FakeNli::three(&[("This example is sports.", [0.0, 0.0, 2.0])]);
    fake.count_zero = true;
    let categorizer = ZeroShotCategorizer::new(
        fake,
        vec!["sports".into()],
        ZeroShotConfig {
            mode: ZeroShotMode::MultiLabel,
            aggregation: WindowAggregation::LengthWeightedMean,
            ..ZeroShotConfig::default()
        },
    )
    .unwrap();
    close(&categorizer.categorize("!!! ??? ...").unwrap(), &[0.8807970779778823]);
}

#[test]
fn span_annotator_errors_carry_its_own_id() {
    let text = "The derby ended level. The senate voted today.";
    let mut doc = Document::new(text);
    doc.add(&Key::DEFAULT, Sentences { spans: spans(&[(0, 22), (23, 45)]) }, Provenance::by("test")).unwrap();
    let failing = || {
        let mut fake = FakeNli::three(&[]);
        fake.fail = true;
        ZeroShotCategorizer::new(fake, labels(), ZeroShotConfig::default()).unwrap()
    };
    let annotator_id = |err: &turbo_analysis::Error| match err {
        turbo_analysis::Error::Annotator { annotator, .. } => *annotator,
        other => panic!("expected an annotator error, got {other:?}"),
    };

    let analyzer = Analyzer::builder()
        .given(Key::<turbo_analysis::View>::DEFAULT)
        .given(Key::<Sentences>::DEFAULT)
        .add(ZeroShotSpanAnnotator::<_, Sentences>::new(failing()))
        .build()
        .unwrap();
    let mut docs = vec![doc.clone()];
    let outcomes = analyzer.run(&mut docs);
    assert_eq!(annotator_id(outcomes[0].as_ref().unwrap_err()), "turbo:zero-shot-spans");

    let analyzer = Analyzer::builder()
        .given(Key::<turbo_analysis::View>::DEFAULT)
        .given(Key::<Sentences>::DEFAULT)
        .add(ZeroShotAnnotator::new(failing()))
        .build()
        .unwrap();
    let mut docs = vec![doc];
    let outcomes = analyzer.run(&mut docs);
    assert_eq!(annotator_id(outcomes[0].as_ref().unwrap_err()), "turbo:zero-shot");
}
