//! What a `RedactionPolicy` message is refused for, redact-and-flag, and
//! the default-strategy rule (issue #1882 part A).

#[path = "pii_policy/common.rs"]
mod common;

use common::*;
use pb::pii_rewrite_strategy::Strategy;
use prost::Message;
use turbo_kserve::pii_policy::{NoRegistry, PolicyError, PolicyFlag, RedactionPolicy};
use turbo_kserve::proto::result as pb;
use turbo_pii::PiiKind;
use turbo_pii::rewrite::{HmacTokenizer, MaskPolicy, RewriteStrategy};

fn refused(p: &pb::RedactionPolicy) -> PolicyError {
    RedactionPolicy::from_proto(p, &TestRegistry).expect_err("the policy is refused")
}

#[test]
fn unspecified_enums_are_refused() {
    let mut p = one(Strategy::Mask(pb::PiiMaskStrategy::default()));
    p.kind_rules[0].state = pb::PiiRuleState::Unspecified as i32;
    assert!(matches!(refused(&p), PolicyError::Unspecified { field: "state", .. }));

    let p = policy(vec![rule("x", builtin(pb::PiiKindId::Unspecified), vec![])]);
    assert!(matches!(refused(&p), PolicyError::Unspecified { field: "kind.builtin", .. }));

    let p = one(Strategy::Truncate(pb::PiiTruncateStrategy::default()));
    assert!(matches!(refused(&p), PolicyError::Unspecified { field: "truncate.side", .. }));

    let p = one(Strategy::Map(Box::new(pb::PiiMapStrategy {
        table: Some(pb::pii_map_strategy::Table::Entries(map_entries(&[("a", "b")]))),
        ..Default::default()
    })));
    assert!(matches!(refused(&p), PolicyError::Unspecified { field: "map.target", .. }));

    let mut p = one(Strategy::Mask(pb::PiiMaskStrategy::default()));
    p.kind_rules[0].labels = Some(pb::PiiLabelOverride::default());
    assert!(matches!(refused(&p), PolicyError::Unspecified { field: "labels.requirement", .. }));

    let mut p = one(Strategy::Mask(pb::PiiMaskStrategy::default()));
    p.kind_rules[0].strategies[0].when = vec![pb::PiiStrategyCondition {
        test: Some(pb::pii_strategy_condition::Test::ScoreAtLeast(pb::PiiScoreCondition::default())),
    }];
    assert!(matches!(refused(&p), PolicyError::Unspecified { field: "score_at_least.kind", .. }));

    p.kind_rules[0].strategies[0].when = vec![pb::PiiStrategyCondition {
        test: Some(pb::pii_strategy_condition::Test::ChecksumPassed(pb::PiiChecksumPassedCondition {
            algorithms: vec![0],
        })),
    }];
    assert!(matches!(refused(&p), PolicyError::Unspecified { field: "checksum_passed.algorithms", .. }));

    let mut p = policy(vec![]);
    p.allow = Some(pb::PiiAllowList {
        patterns: vec![pb::PiiAllowPattern { pattern: "ddd".to_owned(), ..Default::default() }],
        ..Default::default()
    });
    assert!(matches!(refused(&p), PolicyError::Unspecified { field: "syntax", .. }));
}

#[test]
fn an_unknown_kind_id_is_refused() {
    let p = policy(vec![rule(
        "future",
        Some(pb::PiiKindRef { kind: Some(pb::pii_kind_ref::Kind::Builtin(9999)) }),
        vec![],
    )]);
    assert!(matches!(refused(&p), PolicyError::UnknownKind { id: 9999, .. }));
    let p = policy(vec![rule("c", custom(0), vec![])]);
    assert!(matches!(refused(&p), PolicyError::CustomKindOutOfRange { index: 0, table: 0, .. }));
}

/// A strategy arm this server does not know arrives as an unknown field:
/// field 15 of PiiRewriteStrategy, here a mask message's bytes.
fn with_unknown_arm(flag: bool) -> pb::RedactionPolicy {
    let mut arm = Vec::new();
    prost::encoding::message::encode(15, &pb::PiiMaskStrategy::default(), &mut arm);
    let mut entry_bytes = Vec::new();
    prost::encoding::bytes::encode(2, &arm, &mut entry_bytes);
    let entry = pb::PiiStrategyEntry::decode(&*entry_bytes).unwrap();
    assert_eq!(entry.strategy, Some(pb::PiiRewriteStrategy { strategy: None }), "the arm decodes as unknown");
    let mut p = one(Strategy::Mask(pb::PiiMaskStrategy::default()));
    p.redact_and_flag_unknown = flag;
    p.kind_rules[0].strategies.push(entry);
    p
}

#[test]
fn an_unknown_strategy_arm_is_refused() {
    let err = refused(&with_unknown_arm(false));
    assert!(matches!(err, PolicyError::UnknownArm { field: "strategy", .. }), "{err}");
    let p = policy(vec![rule("e", builtin(pb::PiiKindId::Email), vec![pb::PiiStrategyEntry::default()])]);
    assert!(matches!(refused(&p), PolicyError::UnknownArm { field: "strategy", .. }));
    let mut p = one(Strategy::Mask(pb::PiiMaskStrategy::default()));
    p.kind_rules[0].strategies[0].when = vec![pb::PiiStrategyCondition::default()];
    assert!(matches!(refused(&p), PolicyError::UnknownArm { field: "when.test", .. }));
}

#[test]
fn redact_and_flag_masks_what_it_does_not_know_and_never_passes_through() {
    let p = RedactionPolicy::from_proto(&with_unknown_arm(true), &TestRegistry).unwrap();
    let mask = RewriteStrategy::Mask(MaskPolicy::new('*').unwrap());
    assert_eq!(p.rules[0].strategies[1].strategy, mask);
    assert!(p.rules[0].strategies[1].when.is_empty());
    assert_eq!(p.flags, vec![PolicyFlag::UnknownStrategy { rule_id: "email".to_owned(), entry: 1 }]);

    let mut q = policy(vec![
        rule("future", Some(pb::PiiKindRef { kind: Some(pb::pii_kind_ref::Kind::Builtin(77)) }), vec![]),
        rule("email", builtin(pb::PiiKindId::Email), vec![entry(Strategy::Mask(pb::PiiMaskStrategy::default()))]),
    ]);
    q.redact_and_flag_unknown = true;
    let p = RedactionPolicy::from_proto(&q, &TestRegistry).unwrap();
    assert_eq!(p.rules.len(), 1);
    assert_eq!(p.flags, vec![PolicyFlag::UnknownKind { rule_id: "future".to_owned(), id: 77 }]);
    // Unknown conditions take the conservative mask too, unconditionally.
    q.kind_rules[1].strategies[0].when = vec![pb::PiiStrategyCondition::default()];
    q.kind_rules[1].strategies[0].strategy =
        Some(strategy(Strategy::LastN(pb::PiiLastNStrategy { count: 4, ..Default::default() })));
    let p = RedactionPolicy::from_proto(&q, &TestRegistry).unwrap();
    assert_eq!(p.rules[0].strategies[0].strategy, mask);
    assert!(p.rules[0].strategies[0].when.is_empty());
}

#[test]
fn refs_refuse_paths_and_key_material() {
    for id in ["/etc/keys/tokens", "..\\keys", "keys/tokens", ".hidden"] {
        let err = refused(&one(keyed(id, 1)));
        assert!(matches!(err, PolicyError::BadRef { field: "key.id", .. }), "{id}: {err}");
    }
    let hex_key = "00112233445566778899aabbccddeeff";
    let err = refused(&one(keyed(hex_key, 1)));
    assert!(matches!(err, PolicyError::BadRef { reason, .. } if reason.contains("key material")));
    assert!(!err.to_string().contains(hex_key), "the error never repeats key material");
    let err = refused(&one(Strategy::KeyedToken(pb::PiiKeyedTokenStrategy { key: None, length: 0 })));
    assert!(matches!(err, PolicyError::Missing { field: "keyed_token.key", .. }));
    let err = refused(&one(keyed("tokens", 2)));
    assert!(matches!(err, PolicyError::Unresolved { .. }), "an unregistered version is refused");

    let dict = |artifact_id: &str, sha256: &str| {
        one(Strategy::Map(Box::new(pb::PiiMapStrategy {
            table: Some(pb::pii_map_strategy::Table::Dictionary(pb::PiiDictionaryRef {
                artifact_id: artifact_id.to_owned(),
                sha256: sha256.to_owned(),
            })),
            target: pb::PiiMapTarget::Normalized as i32,
            ..Default::default()
        })))
    };
    assert!(matches!(
        refused(&dict("/srv/vip.tsv", DICT_SHA)),
        PolicyError::BadRef { field: "dictionary.artifact_id", .. }
    ));
    assert!(matches!(refused(&dict("vip-map", "abc")), PolicyError::BadRef { field: "dictionary.sha256", .. }));
    assert!(matches!(refused(&dict("vip-map", &DICT_SHA.to_uppercase())), PolicyError::BadRef { .. }));
    // With no registry every reference is refused; an inline policy converts.
    let err = RedactionPolicy::from_proto(&dict("vip-map", DICT_SHA), &NoRegistry).unwrap_err();
    assert!(matches!(err, PolicyError::Unresolved { .. }));
    RedactionPolicy::from_proto(&one(Strategy::Mask(pb::PiiMaskStrategy::default())), &NoRegistry).unwrap();
}

#[test]
fn structural_refusals() {
    let mut p = one(Strategy::Mask(pb::PiiMaskStrategy::default()));
    p.id.clear();
    assert!(matches!(refused(&p), PolicyError::Missing { field: "id", .. }));
    let m = || entry(Strategy::Mask(pb::PiiMaskStrategy::default()));
    let p = policy(vec![
        rule("a", builtin(pb::PiiKindId::Email), vec![m()]),
        rule("a", builtin(pb::PiiKindId::Phone), vec![m()]),
    ]);
    assert!(matches!(refused(&p), PolicyError::DuplicateRuleId { .. }));
    let p = policy(vec![
        rule("a", builtin(pb::PiiKindId::Email), vec![m()]),
        rule("b", builtin(pb::PiiKindId::Email), vec![m()]),
    ]);
    assert!(matches!(refused(&p), PolicyError::DuplicateKind { .. }));
    let p = policy(vec![rule("", builtin(pb::PiiKindId::Email), vec![m()])]);
    assert!(matches!(refused(&p), PolicyError::Missing { field: "rule_id", .. }));

    let inner = strategy(Strategy::Map(Box::new(pb::PiiMapStrategy {
        table: Some(pb::pii_map_strategy::Table::Entries(map_entries(&[("a", "b")]))),
        target: pb::PiiMapTarget::Surface as i32,
        ..Default::default()
    })));
    let p = one(Strategy::Map(Box::new(pb::PiiMapStrategy {
        table: Some(pb::pii_map_strategy::Table::Entries(map_entries(&[("a", "b")]))),
        target: pb::PiiMapTarget::Surface as i32,
        fallback: Some(Box::new(inner)),
        ..Default::default()
    })));
    assert!(matches!(refused(&p), PolicyError::NestedMap { .. }));

    let p = one(Strategy::Mask(pb::PiiMaskStrategy { mask_char: "ab".to_owned(), keep_format: false }));
    assert!(matches!(refused(&p), PolicyError::Invalid { .. }));
    let p = one(Strategy::Format(pb::PiiFormatStrategy { template: "%q".to_owned() }));
    assert!(matches!(refused(&p), PolicyError::Invalid { .. }));

    let mut p = policy(vec![]);
    p.deny = vec![pb::PiiDenyList {
        rule_id: "d".to_owned(),
        kind: builtin(pb::PiiKindId::Email),
        terms: Some(pb::PiiTermSource {
            source: Some(pb::pii_term_source::Source::Terms(pb::PiiTerms { terms: vec!["x".to_owned()] })),
        }),
        ..Default::default()
    }];
    assert!(matches!(refused(&p), PolicyError::BuiltinKind { .. }));
    p.custom_kinds = vec!["email".to_owned()];
    assert!(matches!(refused(&p), PolicyError::CustomKinds(_)), "a custom kind cannot take a built-in name");
}

#[test]
fn a_kind_without_a_strategy_gets_a_mask_or_a_keyed_token() {
    // No default given: a whole-value mask.
    let p = RedactionPolicy::from_proto(&policy(vec![]), &NoRegistry).unwrap();
    let mask = RewriteStrategy::Mask(MaskPolicy::new('*').unwrap());
    assert_eq!(p.default_strategy, mask);
    assert!(!matches!(p.default_strategy, RewriteStrategy::Pseudonym(_) | RewriteStrategy::Sha256(_)));
    // A rule with no strategies, and a map with no fallback, take it too.
    let facts =
        turbo_kserve::pii_policy::MentionFacts { evidence: &[], detector_id: "d", detector_version: "1", score: None };
    let q = policy(vec![rule("email", builtin(pb::PiiKindId::Email), vec![])]);
    let p = RedactionPolicy::from_proto(&q, &NoRegistry).unwrap();
    assert_eq!(p.strategy_for(PiiKind::Email, &facts), (&mask, None));
    // A keyed-token default resolves its key.
    let mut q = policy(vec![]);
    q.default_strategy = Some(pb::PiiDefaultStrategy {
        strategy: Some(pb::pii_default_strategy::Strategy::KeyedToken(pb::PiiKeyedTokenStrategy {
            key: key_ref("tokens", 1),
            length: 0,
        })),
    });
    let p = RedactionPolicy::from_proto(&q, &TestRegistry).unwrap();
    assert_eq!(p.default_strategy, RewriteStrategy::Hmac(HmacTokenizer::new(&KEY).unwrap()));
    // A set default with no arm (or an arm this server does not know) is refused.
    q.default_strategy = Some(pb::PiiDefaultStrategy { strategy: None });
    assert!(matches!(refused(&q), PolicyError::UnknownArm { .. }));
}
