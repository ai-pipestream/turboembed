//! Message builders and an in-memory resolver for the redaction policy
//! tests (issue #1882 part A).
#![allow(dead_code)]

use std::collections::HashMap;
use std::sync::Arc;

use pb::pii_rewrite_strategy::Strategy;
use turbo_kserve::pii_policy::{PolicyResolver, ResolveError};
use turbo_kserve::proto::result as pb;

/// A 32-byte test key: fixture material, never a real secret.
pub const KEY: [u8; 32] = [7u8; 32];
pub const DICT_SHA: &str = "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef";

/// Resolves `tokens:1` to [`KEY`], `names-dict` to two terms and
/// `vip-map` to one map entry, each at [`DICT_SHA`].
pub struct TestRegistry;

impl PolicyResolver for TestRegistry {
    fn hmac_key(&self, id: &str, version: u32) -> Result<&[u8], ResolveError> {
        if id == "tokens" && version == 1 { Ok(&KEY) } else { Err(ResolveError::NotRegistered) }
    }

    fn terms(&self, artifact_id: &str, sha256: &str) -> Result<Arc<[String]>, ResolveError> {
        if artifact_id == "names-dict" && sha256 == DICT_SHA {
            Ok(vec!["Acme Corp".to_owned(), "Initech".to_owned()].into())
        } else {
            Err(ResolveError::NotRegistered)
        }
    }

    fn map_entries(&self, artifact_id: &str, sha256: &str) -> Result<Arc<[(String, String)]>, ResolveError> {
        if artifact_id == "vip-map" && sha256 == DICT_SHA {
            Ok(vec![("vip@example.com".to_owned(), "VIP".to_owned())].into())
        } else {
            Err(ResolveError::NotRegistered)
        }
    }
}

pub fn builtin(id: pb::PiiKindId) -> Option<pb::PiiKindRef> {
    Some(pb::PiiKindRef { kind: Some(pb::pii_kind_ref::Kind::Builtin(id as i32)) })
}

pub fn custom(index: u32) -> Option<pb::PiiKindRef> {
    Some(pb::PiiKindRef { kind: Some(pb::pii_kind_ref::Kind::Custom(index)) })
}

pub fn strategy(s: Strategy) -> pb::PiiRewriteStrategy {
    pb::PiiRewriteStrategy { strategy: Some(s) }
}

pub fn entry(s: Strategy) -> pb::PiiStrategyEntry {
    pb::PiiStrategyEntry { when: Vec::new(), strategy: Some(strategy(s)) }
}

pub fn key_ref(id: &str, version: u32) -> Option<pb::PiiPolicyKeyRef> {
    Some(pb::PiiPolicyKeyRef { id: id.to_owned(), version })
}

pub fn keyed(id: &str, version: u32) -> Strategy {
    Strategy::KeyedToken(pb::PiiKeyedTokenStrategy { key: key_ref(id, version), length: 0 })
}

pub fn rule(rule_id: &str, kind: Option<pb::PiiKindRef>, strategies: Vec<pb::PiiStrategyEntry>) -> pb::PiiKindRule {
    pb::PiiKindRule {
        rule_id: rule_id.to_owned(),
        kind,
        state: pb::PiiRuleState::Enabled as i32,
        strategies,
        allow: None,
        labels: None,
        release_ceiling: None,
    }
}

pub fn policy(rules: Vec<pb::PiiKindRule>) -> pb::RedactionPolicy {
    pb::RedactionPolicy { id: "test-policy".to_owned(), revision: 1, kind_rules: rules, ..Default::default() }
}

/// One rule for EMAIL with `s` as its only strategy.
pub fn one(s: Strategy) -> pb::RedactionPolicy {
    policy(vec![rule("email", builtin(pb::PiiKindId::Email), vec![entry(s)])])
}

pub fn map_entries(entries: &[(&str, &str)]) -> pb::PiiMapEntries {
    pb::PiiMapEntries {
        entries: entries.iter().map(|(k, v)| ((*k).to_owned(), (*v).to_owned())).collect::<HashMap<_, _>>(),
    }
}

/// A policy using every message of the schema, for the digest tests.
pub fn rich() -> pb::RedactionPolicy {
    use pb::pii_strategy_condition::Test;
    let checksum = pb::PiiStrategyCondition {
        test: Some(Test::ChecksumPassed(pb::PiiChecksumPassedCondition {
            algorithms: vec![pb::PiiChecksumAlgorithm::Luhn as i32],
        })),
    };
    let score = pb::PiiStrategyCondition {
        test: Some(Test::ScoreAtLeast(pb::PiiScoreCondition {
            kind: pb::PiiScoreKind::Probability as i32,
            min: 0.75,
            calibrated: true,
        })),
    };
    let card = pb::PiiKindRule {
        rule_id: "card".to_owned(),
        kind: builtin(pb::PiiKindId::Card),
        state: pb::PiiRuleState::Enabled as i32,
        strategies: vec![
            pb::PiiStrategyEntry {
                when: vec![checksum],
                strategy: Some(strategy(Strategy::LastN(pb::PiiLastNStrategy {
                    count: 4,
                    mask_char: "#".to_owned(),
                    keep_format: true,
                }))),
            },
            pb::PiiStrategyEntry { when: vec![score], strategy: Some(strategy(keyed("tokens", 1))) },
            entry(Strategy::Mask(pb::PiiMaskStrategy::default())),
        ],
        allow: Some(pb::PiiAllowList {
            terms: vec![pb::PiiTermSource {
                source: Some(pb::pii_term_source::Source::Terms(pb::PiiTerms {
                    terms: vec!["4111 1111 1111 1111".to_owned()],
                })),
            }],
            case_sensitive: false,
            patterns: Vec::new(),
        }),
        labels: Some(pb::PiiLabelOverride {
            requirement: pb::PiiLabelRequirement::Boosts as i32,
            window_words: 6,
            extra_labels: vec!["card no".to_owned()],
        }),
        release_ceiling: None,
    };
    let email = rule(
        "email",
        builtin(pb::PiiKindId::Email),
        vec![entry(Strategy::Map(Box::new(pb::PiiMapStrategy {
            table: Some(pb::pii_map_strategy::Table::Entries(map_entries(&[
                ("vip@example.com", "VIP"),
                ("ceo@example.com", "CEO"),
                ("cfo@example.com", "CFO"),
            ]))),
            case_sensitive: false,
            target: pb::PiiMapTarget::NormalizedOrSurface as i32,
            fallback: Some(Box::new(strategy(Strategy::Format(pb::PiiFormatStrategy { template: "[%t]".to_owned() })))),
        })))],
    );
    let employee = rule(
        "employee",
        custom(0),
        vec![entry(Strategy::Abbreviate(pb::PiiAbbreviateStrategy {
            separator: ".".to_owned(),
            trailing: ".".to_owned(),
            preserve_hyphens: true,
            preserve_case: false,
        }))],
    );
    pb::RedactionPolicy {
        id: "golden".to_owned(),
        revision: 7,
        custom_kinds: vec!["employee_id".to_owned(), "project".to_owned()],
        kind_rules: vec![card, email, employee],
        default_strategy: Some(pb::PiiDefaultStrategy {
            strategy: Some(pb::pii_default_strategy::Strategy::Mask(pb::PiiMaskStrategy {
                mask_char: "x".to_owned(),
                keep_format: false,
            })),
        }),
        allow: Some(pb::PiiAllowList {
            terms: Vec::new(),
            case_sensitive: true,
            patterns: vec![pb::PiiAllowPattern {
                syntax: pb::PiiPatternSyntax::Shape as i32,
                pattern: "TEST-dddd".to_owned(),
                case_sensitive: false,
            }],
        }),
        deny: vec![pb::PiiDenyList {
            rule_id: "projects".to_owned(),
            kind: custom(1),
            terms: Some(pb::PiiTermSource {
                source: Some(pb::pii_term_source::Source::Dictionary(pb::PiiDictionaryRef {
                    artifact_id: "names-dict".to_owned(),
                    sha256: DICT_SHA.to_owned(),
                })),
            }),
            case_sensitive: false,
            max_edits: 1,
            capitalized_only: true,
        }],
        custom_identifiers: vec![pb::PiiCustomIdentifier {
            rule_id: "employee-shape".to_owned(),
            kind: custom(0),
            shape: "EMP-dddddd".to_owned(),
            case_sensitive: true,
        }],
        redact_and_flag_unknown: false,
    }
}
