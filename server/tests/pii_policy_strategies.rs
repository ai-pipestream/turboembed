//! Every `PiiRewriteStrategy` message converts to the native strategy it names,
//! and back where the native value says all its message does (issue #1882
//! part A).

#[path = "pii_policy/common.rs"]
mod common;

use common::*;
use pb::pii_date_strategy::Mode;
use pb::pii_rewrite_strategy::Strategy;
use turbo_analysis_layers::PiiStrategyKind;
use turbo_kserve::pii_policy::{RedactionPolicy, strategy_to_proto};
use turbo_kserve::proto::result as pb;
use turbo_pii::rewrite::{
    Abbreviator, FormatRedactor, HmacTokenizer, MapReplacer, MapTarget, MaskPolicy, Pseudonymizer, RewriteStrategy,
    Sha256Tokenizer, StaticReplacer, Truncate,
};

fn native(s: Strategy) -> RewriteStrategy {
    let p = RedactionPolicy::from_proto(&one(s), &TestRegistry).expect("the policy converts");
    assert_eq!(p.rules.len(), 1);
    assert_eq!(p.rules[0].strategies.len(), 1);
    p.rules[0].strategies[0].strategy.clone()
}

fn round_trip(s: Strategy, want: RewriteStrategy) {
    let got = native(s.clone());
    assert_eq!(got, want);
    let back = strategy_to_proto(&got).expect("this strategy has a reverse");
    assert_eq!(back.strategy, Some(s), "native -> proto gives the message back");
}

#[test]
fn mask_round_trips() {
    round_trip(Strategy::Mask(pb::PiiMaskStrategy::default()), RewriteStrategy::Mask(MaskPolicy::new('*').unwrap()));
    round_trip(
        Strategy::Mask(pb::PiiMaskStrategy { mask_char: "#".to_owned(), keep_format: true }),
        RewriteStrategy::Mask(MaskPolicy::new('#').unwrap().keeping_format()),
    );
}

#[test]
fn format_round_trips() {
    let t = "[%t-%v]";
    round_trip(
        Strategy::Format(pb::PiiFormatStrategy { template: t.to_owned() }),
        RewriteStrategy::Format(FormatRedactor::new(t).unwrap()),
    );
}

#[test]
fn truncate_and_last_n_round_trip() {
    round_trip(
        Strategy::Truncate(pb::PiiTruncateStrategy {
            side: pb::PiiKeepSide::Start as i32,
            count: 3,
            mask_char: String::new(),
            keep_format: false,
        }),
        RewriteStrategy::Truncate(Truncate::keep_start(3)),
    );
    round_trip(
        Strategy::LastN(pb::PiiLastNStrategy { count: 4, mask_char: "x".to_owned(), keep_format: true }),
        RewriteStrategy::Truncate(Truncate::last_n(4).with_mask('x').keeping_format()),
    );
    // A truncation keeping the end is a last-N.
    let end = native(Strategy::Truncate(pb::PiiTruncateStrategy {
        side: pb::PiiKeepSide::End as i32,
        count: 2,
        mask_char: String::new(),
        keep_format: false,
    }));
    assert_eq!(end, RewriteStrategy::Truncate(Truncate::keep_end(2)));
    let RewriteStrategy::Truncate(t) = &end else { unreachable!() };
    assert_eq!(t.apply("123456"), "****56");
}

#[test]
fn sha256_round_trips() {
    round_trip(Strategy::Sha256(pb::PiiSha256Strategy::default()), RewriteStrategy::Sha256(Sha256Tokenizer::new()));
    round_trip(
        Strategy::Sha256(pb::PiiSha256Strategy { length: 12, kind_prefix: true, prefix: "h:".to_owned() }),
        RewriteStrategy::Sha256(Sha256Tokenizer::with_length(12).unwrap().with_kind_prefix().with_prefix("h:")),
    );
}

#[test]
fn pseudonym_round_trips() {
    round_trip(
        Strategy::Pseudonym(pb::PiiPseudonymStrategy { prefix: "[".to_owned(), suffix: "]".to_owned() }),
        RewriteStrategy::Pseudonym(Pseudonymizer::new("[", "]")),
    );
}

#[test]
fn static_text_converts() {
    let got = native(Strategy::StaticText(pb::PiiStaticStrategy { replacement: "[gone]".to_owned() }));
    assert_eq!(got, RewriteStrategy::Static(StaticReplacer::new("[gone]")));
    assert_eq!(strategy_to_proto(&got), None, "a static replacer has per-kind texts the message lacks");
}

#[test]
fn abbreviate_converts_and_abbreviates() {
    let got = native(Strategy::Abbreviate(pb::PiiAbbreviateStrategy {
        separator: ".".to_owned(),
        trailing: ".".to_owned(),
        preserve_hyphens: true,
        preserve_case: false,
    }));
    assert_eq!(got, RewriteStrategy::Abbreviate(Abbreviator::dotted().preserving_hyphens()));
    let RewriteStrategy::Abbreviate(a) = &got else { unreachable!() };
    assert_eq!(a.abbreviate("Jean-Luc Picard"), "J.-L.P.");
    let compact = native(Strategy::Abbreviate(pb::PiiAbbreviateStrategy::default()));
    assert_eq!(compact, RewriteStrategy::Abbreviate(Abbreviator::compact()));
}

#[test]
fn keyed_token_resolves_its_key_and_tokenizes() {
    let got = native(keyed("tokens", 1));
    let want = HmacTokenizer::new(&KEY).unwrap();
    assert_eq!(got, RewriteStrategy::Hmac(want.clone()));
    let RewriteStrategy::Hmac(h) = &got else { unreachable!() };
    assert_eq!(h.token("email", "jane@example.com").unwrap(), want.token("email", "jane@example.com").unwrap());
    assert_eq!(strategy_to_proto(&got), None, "a native tokenizer holds key bytes, not a reference");
    let long = native(Strategy::KeyedToken(pb::PiiKeyedTokenStrategy { key: key_ref("tokens", 1), length: 32 }));
    let RewriteStrategy::Hmac(h) = &long else { unreachable!() };
    let token = h.token("email", "x@example.com").unwrap();
    assert_eq!(token.rsplit('-').next().map(str::len), Some(32), "32 hexadecimal digits after the kind");
}

#[test]
fn map_converts_inline_and_registered_tables() {
    let inline = native(Strategy::Map(Box::new(pb::PiiMapStrategy {
        table: Some(pb::pii_map_strategy::Table::Entries(map_entries(&[("vip@example.com", "VIP"), ("b", "B")]))),
        case_sensitive: true,
        target: pb::PiiMapTarget::Surface as i32,
        fallback: Some(Box::new(strategy(Strategy::StaticText(pb::PiiStaticStrategy { replacement: "?".to_owned() })))),
    })));
    let want = MapReplacer::new([("b", "B"), ("vip@example.com", "VIP")], true)
        .unwrap()
        .with_target(MapTarget::Surface)
        .with_fallback(StaticReplacer::new("?"));
    assert_eq!(inline, RewriteStrategy::Map(Box::new(want)));
    let registered = native(Strategy::Map(Box::new(pb::PiiMapStrategy {
        table: Some(pb::pii_map_strategy::Table::Dictionary(pb::PiiDictionaryRef {
            artifact_id: "vip-map".to_owned(),
            sha256: DICT_SHA.to_owned(),
        })),
        case_sensitive: false,
        target: pb::PiiMapTarget::NormalizedOrSurface as i32,
        fallback: None,
    })));
    let RewriteStrategy::Map(m) = &registered else { panic!("a map") };
    assert_eq!(m.get("VIP@example.com"), Some("VIP"));
    // An unset fallback is the policy's default: a whole-value mask here.
    assert_eq!(m.fallback(), &RewriteStrategy::Mask(MaskPolicy::new('*').unwrap()));
    assert_eq!(strategy_to_proto(&registered), None);
}

#[test]
fn conditions_choose_the_first_entry_that_holds() {
    use turbo_kserve::pii_policy::MentionFacts;
    use turbo_pii::{ChecksumAlgorithm, PiiDetectorRef, PiiEvidence, PiiKind, PiiScore, PiiScoreKind};
    let p = RedactionPolicy::from_proto(&rich(), &TestRegistry).unwrap();
    let passed = [PiiEvidence::Checksum { algorithm: ChecksumAlgorithm::Luhn, passed: true }];
    let failed = [PiiEvidence::Checksum { algorithm: ChecksumAlgorithm::Luhn, passed: false }];
    let facts =
        |evidence, score| MentionFacts { evidence, detector_id: "turbo-pii:cursor", detector_version: "1", score };
    let (s, rule) = p.strategy_for(PiiKind::Card, &facts(&passed, None));
    assert!(matches!(s, RewriteStrategy::Truncate(_)));
    assert_eq!(rule, Some("card"));
    let score =
        PiiScore { value: 0.9, kind: PiiScoreKind::Probability, model: PiiDetectorRef::new(0), calibrated: true };
    let (s, _) = p.strategy_for(PiiKind::Card, &facts(&failed, Some(&score)));
    assert!(matches!(s, RewriteStrategy::Hmac(_)), "a failed checksum falls to the scored entry");
    let uncalibrated = PiiScore { calibrated: false, ..score };
    let (s, _) = p.strategy_for(PiiKind::Card, &facts(&failed, Some(&uncalibrated)));
    assert!(matches!(s, RewriteStrategy::Mask(_)), "an uncalibrated score does not meet a calibrated condition");
    let (s, rule) = p.strategy_for(PiiKind::Phone, &facts(&[], None));
    assert_eq!(s, &p.default_strategy);
    assert_eq!(rule, None);
    assert_eq!(p.default_strategy, RewriteStrategy::Mask(MaskPolicy::new('x').unwrap()));
}

/// One message of every `PiiRewriteStrategy` arm. The match below has no
/// wildcard, so a new arm fails to compile here until it is listed.
/// The date arm is listed once per mode, as each records its own value.
fn date_arm(mode: Mode) -> Strategy {
    Strategy::Date(pb::PiiDateStrategy { mode: Some(mode), mask_char: String::new() })
}

fn every_arm() -> Vec<Strategy> {
    #[cfg_attr(not(feature = "pii-reversible"), allow(unused_mut))]
    let mut arms = vec![
        Strategy::Mask(pb::PiiMaskStrategy::default()),
        Strategy::Format(pb::PiiFormatStrategy { template: "[%t]".to_owned() }),
        Strategy::Map(Box::new(pb::PiiMapStrategy {
            table: Some(pb::pii_map_strategy::Table::Entries(map_entries(&[("a", "b")]))),
            target: pb::PiiMapTarget::Surface as i32,
            ..Default::default()
        })),
        Strategy::Truncate(pb::PiiTruncateStrategy {
            side: pb::PiiKeepSide::Start as i32,
            count: 2,
            ..Default::default()
        }),
        Strategy::LastN(pb::PiiLastNStrategy { count: 4, ..Default::default() }),
        Strategy::Abbreviate(pb::PiiAbbreviateStrategy::default()),
        keyed("tokens", 1),
        Strategy::Sha256(pb::PiiSha256Strategy::default()),
        Strategy::StaticText(pb::PiiStaticStrategy { replacement: "x".to_owned() }),
        Strategy::Pseudonym(pb::PiiPseudonymStrategy::default()),
        Strategy::ZipSafeHarbor(pb::PiiZipSafeHarborStrategy::default()),
        Strategy::AgeSafeHarbor(pb::PiiAgeSafeHarborStrategy::default()),
        date_arm(Mode::Year(pb::PiiDateYear {})),
        date_arm(Mode::Shift(pb::PiiDateShift { days: 3, ..Default::default() })),
        date_arm(Mode::Relative(pb::PiiDateRelative { reference: "2026-07-14".to_owned() })),
        date_arm(Mode::SafeHarbor(pb::PiiDateSafeHarbor { reference: "2026-07-14".to_owned() })),
    ];
    // A server without reversible protection does not know these two.
    #[cfg(feature = "pii-reversible")]
    arms.extend([
        Strategy::Aes256Gcm(pb::PiiAesGcmStrategy { key: key_ref("vault", 1), context: String::new() }),
        Strategy::Ff1(pb::PiiFf1Strategy {
            key: key_ref("vault", 1),
            alphabet: pb::PiiFf1Alphabet::Alphanumeric as i32,
            tweak: String::new(),
        }),
    ]);
    for arm in &arms {
        match arm {
            Strategy::Mask(_)
            | Strategy::Format(_)
            | Strategy::Map(_)
            | Strategy::Truncate(_)
            | Strategy::LastN(_)
            | Strategy::Abbreviate(_)
            | Strategy::KeyedToken(_)
            | Strategy::Sha256(_)
            | Strategy::StaticText(_)
            | Strategy::Pseudonym(_)
            | Strategy::ZipSafeHarbor(_)
            | Strategy::AgeSafeHarbor(_)
            | Strategy::Date(_)
            | Strategy::Aes256Gcm(_)
            | Strategy::Ff1(_) => {}
        }
    }
    arms
}

#[test]
fn every_arm_maps_to_exactly_one_pii_strategy_value() {
    use turbo_kserve::pii_policy::edit_strategy;
    let mut hit = std::collections::BTreeSet::new();
    for arm in every_arm() {
        let value = edit_strategy(&arm);
        assert_ne!(value, pb::PiiStrategy::Unspecified);
        let native = native(arm.clone());
        assert_eq!(i32::from(native.kind().id()), value as i32, "{arm:?} edits record {value:?}");
        hit.insert(value as i32);
    }
    let reversible: &[i32] = if cfg!(feature = "pii-reversible") { &[10, 11] } else { &[] };
    let all: std::collections::BTreeSet<i32> =
        (1..=9).chain([12, 13, 14, 15, 16, 17]).chain(reversible.iter().copied()).collect();
    assert_eq!(
        hit, all,
        "every PiiStrategy value but UNSPECIFIED (and, without reversible protection, 10 and 11) has an arm"
    );
    for (value, native) in
        [(pb::PiiStrategy::Aes256Gcm, PiiStrategyKind::Aes256Gcm), (pb::PiiStrategy::Ff1, PiiStrategyKind::Ff1)]
    {
        assert_eq!(value as i32, i32::from(native.id()));
    }
    assert!(pb::PiiStrategy::try_from(18).is_err(), "a new PiiStrategy value needs an arm here");
}
