//! End-to-end integration of Nemotron-PII neural output (`resolve_model_with_rules`)
//! with the `#1882` `RedactionPolicy` engine (`#1888`).

#[path = "pii_policy/common.rs"]
mod common;

use common::*;
use std::sync::Arc;
use turbo_analysis::{Document, Layer, Span, View, ViewId};
use turbo_analysis_layers::CustomKinds;
use turbo_kserve::pii_policy::{NoRegistry, RedactStep, RedactionPolicy};
use turbo_kserve::proto::result as pb;
use turbo_pii::nemotron::{ModelCandidate, ModelProvenance, resolve_model_with_rules, resolve_nemotron_kind};
use turbo_pii::{Pack, PiiDetector};

#[test]
fn redaction_policy_evaluates_calibrated_scores_detector_and_rule_checksums() {
    use pb::pii_rewrite_strategy::Strategy;
    use pb::pii_strategy_condition::Test;

    let msg = pb::RedactionPolicy {
        id: "nemotron-pii-policy".to_owned(),
        revision: 1,
        custom_kinds: vec!["first-name".to_owned(), "medical-record-number".to_owned()],
        kind_rules: vec![
            // Custom kind 0 ("first-name"): full [NAME] redaction when calibrated probability >= 0.80
            // from turbo-tagger-pii; otherwise abbreviate to initial.
            rule(
                "first-name",
                custom(0),
                vec![
                    pb::PiiStrategyEntry {
                        when: vec![
                            pb::PiiStrategyCondition {
                                test: Some(Test::Detector(pb::PiiDetectorCondition {
                                    id: "turbo-tagger-pii".to_owned(),
                                    version: String::new(),
                                })),
                            },
                            pb::PiiStrategyCondition {
                                test: Some(Test::ScoreAtLeast(pb::PiiScoreCondition {
                                    kind: pb::PiiScoreKind::Probability as i32,
                                    min: 0.80,
                                    calibrated: true,
                                })),
                            },
                        ],
                        strategy: Some(strategy(Strategy::StaticText(pb::PiiStaticStrategy {
                            replacement: "[NAME]".to_owned(),
                        }))),
                    },
                    entry(Strategy::Abbreviate(pb::PiiAbbreviateStrategy {
                        separator: ".".to_owned(),
                        trailing: ".".to_owned(),
                        preserve_hyphens: true,
                        preserve_case: true,
                    })),
                ],
            ),
            // Custom kind 1 ("medical-record-number"): [MRN] when calibrated probability >= 0.85.
            rule(
                "mrn",
                custom(1),
                vec![pb::PiiStrategyEntry {
                    when: vec![pb::PiiStrategyCondition {
                        test: Some(Test::ScoreAtLeast(pb::PiiScoreCondition {
                            kind: pb::PiiScoreKind::Probability as i32,
                            min: 0.85,
                            calibrated: true,
                        })),
                    }],
                    strategy: Some(strategy(Strategy::StaticText(pb::PiiStaticStrategy {
                        replacement: "[MRN]".to_owned(),
                    }))),
                }],
            ),
            // Built-in Card: keep last 4 digits when Luhn checksum passed (rules win on overlap!).
            rule(
                "card",
                builtin(pb::PiiKindId::Card),
                vec![pb::PiiStrategyEntry {
                    when: vec![pb::PiiStrategyCondition {
                        test: Some(Test::ChecksumPassed(pb::PiiChecksumPassedCondition {
                            algorithms: vec![pb::PiiChecksumAlgorithm::Luhn as i32],
                        })),
                    }],
                    strategy: Some(strategy(Strategy::LastN(pb::PiiLastNStrategy {
                        count: 4,
                        mask_char: "#".to_owned(),
                        keep_format: true,
                    }))),
                }],
            ),
        ],
        ..Default::default()
    };

    let policy = Arc::new(RedactionPolicy::from_proto(&msg, &NoRegistry).unwrap());
    let step = RedactStep::new(policy);

    let text = "Patients Jane and Bob paid with 4111-1111-1111-1111 under MRN 884210.";
    let rules = PiiDetector::pack(Pack::Payment);
    let mut model_kinds = CustomKinds::new();
    let first_name = resolve_nemotron_kind("first_name", &mut model_kinds).unwrap();
    let mrn = resolve_nemotron_kind("medical_record_number", &mut model_kinds).unwrap();
    let card = resolve_nemotron_kind("credit_debit_card", &mut model_kinds).unwrap();

    let span_of = |needle: &str| -> Span {
        let s = text.find(needle).unwrap();
        Span::new(s as u32, (s + needle.len()) as u32)
    };

    let (pii, candidates) = resolve_model_with_rules(
        text,
        ViewId::SOURCE,
        Some(&rules),
        &model_kinds,
        ModelProvenance {
            detector_id: "turbo-tagger-pii",
            detector_version: "0.1.0",
            model_id: "turbo-en-nemotron-pii-modernbert-base-1",
            model_version: "1",
            calibrated: true,
        },
        None,
        None,
        &[
            ModelCandidate { span: span_of("Jane"), kind: first_name, probability: 0.94 },
            ModelCandidate { span: span_of("Bob"), kind: first_name, probability: 0.55 },
            // Overlaps Luhn-valid Card rule: rule wins and preserves Checksum(Luhn, true)!
            ModelCandidate { span: span_of("4111-1111-1111-1111"), kind: card, probability: 0.99 },
            ModelCandidate { span: span_of("884210"), kind: mrn, probability: 0.91 },
        ],
    )
    .unwrap();

    pii.check().unwrap();
    candidates.check().unwrap();
    let doc = Document::new(text);
    let source = doc.get::<View>().unwrap();
    let (rewritten, edits) = step.rewrite(source, &pii).unwrap();
    edits.check().unwrap();
    assert_eq!(rewritten.text(), "Patients [NAME] and B. paid with ####-####-####-1111 under MRN [MRN].");
}
