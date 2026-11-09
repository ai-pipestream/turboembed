use turbo_analysis::{Span, ViewId};

use super::*;
use crate::pii::{ChecksumAlgorithm, PiiMention};

const SHAPE: &[PiiEvidence] = &[PiiEvidence::Shape];
const LUHN: &[PiiEvidence] = &[PiiEvidence::Checksum { algorithm: ChecksumAlgorithm::Luhn, passed: true }];
const FAILED: &[PiiEvidence] = &[PiiEvidence::Checksum { algorithm: ChecksumAlgorithm::Luhn, passed: false }];
const LABEL: &[PiiEvidence] = &[PiiEvidence::Label { span: Span { start: 0, end: 1 }, phrase: 0 }];
const BOTH: &[PiiEvidence] = &[
    PiiEvidence::Checksum { algorithm: ChecksumAlgorithm::Luhn, passed: true },
    PiiEvidence::Label { span: Span { start: 0, end: 1 }, phrase: 0 },
];
const DENY: &[PiiEvidence] = &[PiiEvidence::DenyHit { list: 0, term: 0 }];
const EVIDENCE: [&[PiiEvidence]; 6] = [SHAPE, LUHN, FAILED, LABEL, BOTH, DENY];

/// A layer over `custom` holding one mention per `(start, kind, evidence)`,
/// each four bytes long; the starts must be distinct.
fn layer(custom: CustomKinds, mentions: &[(u32, PiiKind, &'static [PiiEvidence])]) -> Pii {
    let mut sorted = mentions.to_vec();
    sorted.sort_by_key(|&(start, kind, _)| (start, kind.order_key()));
    let mut p = Pii::with_custom_kinds(ViewId::SOURCE, custom);
    let detector = p.detector("test", "1").unwrap();
    for (start, kind, evidence) in sorted {
        p.push(PiiMention {
            span: Span::new(start, start + 4),
            kind,
            detector,
            evidence,
            score: None,
            normalized: "x",
        });
    }
    p.check().unwrap();
    p
}

fn risk(mentions: &[(u32, PiiKind, &'static [PiiEvidence])]) -> PiiRisk {
    let r = PiiRisk::compute(&layer(CustomKinds::new(), mentions), &"a".repeat(2000), &PiiSeverityTable::new());
    r.check().unwrap();
    r
}

#[test]
fn an_empty_layer_is_none_with_zero_score() {
    let r = risk(&[]);
    assert_eq!((r.score, r.level, r.mentions, r.kinds.len()), (0.0, PiiRiskLevel::None, 0, 0));
    assert_eq!(r.density_per_1000_chars, 0.0);
    let empty = PiiRisk::compute(&layer(CustomKinds::new(), &[]), "", &PiiSeverityTable::new());
    assert_eq!((empty.chars, empty.density_per_1000_chars), (0, 0.0));
}

#[test]
fn cut_points_are_the_documented_ones() {
    let cases = [
        (0.0, PiiRiskLevel::None),
        (-1.0, PiiRiskLevel::None),
        (f32::NAN, PiiRiskLevel::None),
        (0.001, PiiRiskLevel::Low),
        (2.999, PiiRiskLevel::Low),
        (3.0, PiiRiskLevel::Medium),
        (5.999, PiiRiskLevel::Medium),
        (6.0, PiiRiskLevel::High),
        (8.999, PiiRiskLevel::High),
        (9.0, PiiRiskLevel::Critical),
        (1e9, PiiRiskLevel::Critical),
    ];
    for (score, level) in cases {
        assert_eq!(PiiRiskLevel::of(score), level, "{score}");
    }
    for l in PiiRiskLevel::ALL {
        assert_eq!(PiiRiskLevel::from_id(u32::from(l.id())), Some(l));
    }
    assert_eq!(PiiRiskLevel::from_id(0), None);
}

#[test]
fn the_score_follows_the_formula_and_two_ssns_outweigh_fifty_low_ids() {
    // One Luhn-checked card: 10 x log2(1.9).
    let card = risk(&[(0, PiiKind::Card, LUHN)]);
    assert!((card.score - (10.0f64 * 1.9f64.log2()) as f32).abs() < 1e-5);
    assert_eq!(card.level, PiiRiskLevel::Critical);
    // One e-mail address by shape: 2 x log2(1.75), low.
    let email = risk(&[(0, PiiKind::Email, SHAPE)]);
    assert!((email.score - (2.0f64 * 1.75f64.log2()) as f32).abs() < 1e-5);
    assert_eq!(email.level, PiiRiskLevel::Low);
    let two_ssns = risk(&[(0, PiiKind::UsSsn, SHAPE), (10, PiiKind::UsSsn, SHAPE)]);
    let fifty_low: Vec<_> = (0..50).map(|i| (i * 10, PiiKind::FrSiren, LUHN)).collect();
    let fifty_low = risk(&fifty_low);
    assert!(two_ssns.score > fifty_low.score, "{} vs {}", two_ssns.score, fifty_low.score);
    assert_eq!(two_ssns.level, PiiRiskLevel::Critical);
    assert_eq!(fifty_low.level, PiiRiskLevel::Low);
    assert_eq!(fifty_low.mentions, 50);
    assert!((fifty_low.density_per_1000_chars - 25.0).abs() < 1e-6);
}

#[test]
fn confidence_weights_are_ordered_by_support() {
    let w = |e: &[PiiEvidence]| confidence_weight(e);
    assert_eq!(w(SHAPE), 0.75);
    assert_eq!(w(FAILED), 0.75, "a failed checksum counts for nothing");
    assert!(w(SHAPE) < w(LABEL) && w(LABEL) < w(LUHN) && w(LUHN) < w(BOTH));
    assert_eq!(w(BOTH), 1.0);
    assert_eq!(w(DENY), 1.0);
}

#[test]
fn kinds_are_listed_by_code_with_their_suggestion() {
    let r = risk(&[(0, PiiKind::Phone, SHAPE), (10, PiiKind::Card, LUHN), (20, PiiKind::Phone, LABEL)]);
    let codes: Vec<_> = r.kinds.iter().map(|k| k.kind).collect();
    assert_eq!(codes, [PiiKind::Card, PiiKind::Phone]);
    assert_eq!(r.kinds[1].mentions, 2);
    assert!((r.kinds[1].weighted_count - 1.6).abs() < 1e-6);
    assert_eq!(r.kinds[0].suggested, PiiSuggestedRewrite::KeepLastFour);
    assert_eq!(r.kinds[1].suggested, PiiSuggestedRewrite::Label);
    let sum: f32 = r.kinds.iter().map(|k| k.contribution).sum();
    assert!((sum - r.score).abs() < 1e-4);
}

/// A small deterministic generator, so the property runs the same cases
/// on every machine.
struct Lcg(u64);

impl Lcg {
    fn next(&mut self, n: u32) -> u32 {
        self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        ((self.0 >> 33) % u64::from(n)) as u32
    }
}

fn random_mention(g: &mut Lcg, start: u32) -> (u32, PiiKind, &'static [PiiEvidence]) {
    let kind = PiiKind::from_id(g.next(PiiKind::TABLE.len() as u32) as u16 + 1).unwrap();
    (start, kind, EVIDENCE[g.next(EVIDENCE.len() as u32) as usize])
}

#[test]
fn adding_a_mention_never_lowers_the_score_or_the_level() {
    let mut g = Lcg(1889);
    for case in 0..400 {
        let n = g.next(40);
        let mut mentions: Vec<_> = (0..n).map(|i| random_mention(&mut g, i * 10)).collect();
        let before = risk(&mentions);
        // The new mention lands anywhere in layer order.
        let at = g.next(n + 1) * 10 + 5;
        let extra = random_mention(&mut g, at);
        mentions.push(extra);
        let after = risk(&mentions);
        assert!(after.score >= before.score, "case {case}: {} < {}", after.score, before.score);
        assert!(after.level >= before.level, "case {case}");
        assert_eq!(after.mentions, before.mentions + 1);
    }
}

#[test]
fn the_same_layer_gives_the_same_risk() {
    let mut g = Lcg(7);
    let mentions: Vec<_> = (0..200).map(|i| random_mention(&mut g, i * 10)).collect();
    let text = "b".repeat(2000);
    let table = PiiSeverityTable::new().with(PiiKind::Email, PiiSeverity::High);
    let a = PiiRisk::compute(&layer(CustomKinds::new(), &mentions), &text, &table);
    let b = PiiRisk::compute(&layer(CustomKinds::new(), &mentions), &text, &table);
    assert_eq!(a, b);
    assert_eq!(a.content_hash(), b.content_hash());
    assert_eq!(a.score.to_bits(), b.score.to_bits());
}

#[test]
fn severity_overrides_change_the_score_and_the_digest() {
    let pii = layer(CustomKinds::new(), &[(0, PiiKind::Email, SHAPE)]);
    let text = "c".repeat(100);
    let base = PiiRisk::compute(&pii, &text, &PiiSeverityTable::new());
    let raised = PiiSeverityTable::new().with(PiiKind::Email, PiiSeverity::Critical);
    let up = PiiRisk::compute(&pii, &text, &raised);
    assert_eq!(up.kinds[0].severity, PiiSeverity::Critical);
    assert!(up.score > base.score);
    assert_eq!(up.level, PiiRiskLevel::High);
    assert_ne!(up.severity_table_digest, base.severity_table_digest);
    let lowered = PiiRisk::compute(&pii, &text, &PiiSeverityTable::new().with(PiiKind::Email, PiiSeverity::Low));
    assert!(lowered.score < base.score);
    // Overrides fold in sorted order, so build order does not matter.
    let ab = PiiSeverityTable::new().with(PiiKind::Card, PiiSeverity::Low).with(PiiKind::Email, PiiSeverity::High);
    let ba = PiiSeverityTable::new().with(PiiKind::Email, PiiSeverity::High).with(PiiKind::Card, PiiSeverity::Low);
    assert_eq!(ab.digest(), ba.digest());
    assert!(PiiSeverityTable::new().is_default() && !ab.is_default());
}

#[test]
fn custom_kinds_take_their_named_override_or_the_custom_default() {
    let custom = CustomKinds::from_names(["employee-id", "badge"]).unwrap();
    let (employee, badge) = (custom.find("employee-id").unwrap(), custom.find("badge").unwrap());
    let pii = layer(custom, &[(0, employee, DENY), (10, badge, DENY)]);
    let table = PiiSeverityTable::new().with_custom("employee-id", PiiSeverity::Critical);
    let r = PiiRisk::compute(&pii, "d", &table);
    r.check().unwrap();
    assert_eq!(r.kinds[0].severity, PiiSeverity::Critical);
    assert_eq!(r.kinds[1].severity, PiiSeverity::Medium);
    assert_eq!(r.kind_name(r.kinds[0].kind), Some("employee-id"));
    let low = PiiRisk::compute(&pii, "d", &PiiSeverityTable::new().with_custom_default(PiiSeverity::Low));
    assert!(low.kinds.iter().all(|k| k.severity == PiiSeverity::Low));
    // A custom kind passed to `with` is ignored: its index means nothing alone.
    assert!(PiiSeverityTable::new().with(employee, PiiSeverity::Critical).is_default());
}

#[test]
fn check_refuses_an_inconsistent_layer() {
    let good = risk(&[(0, PiiKind::Card, LUHN), (10, PiiKind::Email, SHAPE)]);
    let mut bad = good.clone();
    bad.level = PiiRiskLevel::Low;
    assert!(bad.check().is_err());
    let mut bad = good.clone();
    bad.mentions += 1;
    assert!(bad.check().is_err());
    let mut bad = good.clone();
    bad.kinds.swap(0, 1);
    assert!(bad.check().is_err());
    let mut bad = good.clone();
    bad.score = f32::INFINITY;
    assert!(bad.check().is_err());
    let mut bad = good;
    bad.kinds[0].kind = PiiKind::Custom(crate::CustomKindId::new(0));
    assert!(bad.check().is_err());
}

#[test]
fn every_builtin_kind_has_a_severity_and_a_suggestion() {
    for row in PiiKind::TABLE {
        let s = PiiSeverity::default_for(row.kind);
        assert_eq!(PiiSeverity::from_id(u32::from(s.id())), Some(s));
        let r = PiiSuggestedRewrite::default_for(row.kind);
        assert_eq!(PiiSuggestedRewrite::from_id(u32::from(r.id())), Some(r));
    }
    assert_eq!(PiiSeverity::default_for(PiiKind::UsSsn), PiiSeverity::Critical);
    assert_eq!(PiiSeverity::default_for(PiiKind::Email), PiiSeverity::Medium);
    assert_eq!(PiiSeverity::default_for(PiiKind::Vin), PiiSeverity::Medium);
    for k in [PiiKind::UsPassport, PiiKind::UsDriversLicence] {
        assert_eq!(PiiSeverity::default_for(k), PiiSeverity::Critical, "{k}");
        assert_eq!(PiiSuggestedRewrite::default_for(k), PiiSuggestedRewrite::MaskKeepFormat, "{k}");
    }
    for k in [PiiKind::UsEin, PiiKind::TrackingNumber, PiiKind::Url, PiiKind::UsZip, PiiKind::PhoneExtension] {
        assert_eq!(PiiSeverity::default_for(k), PiiSeverity::Low, "{k}");
    }
    assert_eq!(PiiSeverity::default_for(PiiKind::Age), PiiSeverity::Low);
    assert_eq!(PiiSeverity::default_for(PiiKind::StreetAddress), PiiSeverity::Medium);
    for k in [PiiKind::Age, PiiKind::StreetAddress] {
        assert_eq!(PiiSuggestedRewrite::default_for(k), PiiSuggestedRewrite::Label, "{k}");
    }
    assert_eq!(PiiSuggestedRewrite::default_for(PiiKind::Url), PiiSuggestedRewrite::Label);
    assert_eq!(PiiSeverity::from_id(0), None);
}

#[test]
fn check_refuses_a_summary_its_kinds_contradict() {
    let good = risk(&[(0, PiiKind::Card, LUHN), (10, PiiKind::Email, SHAPE), (20, PiiKind::Email, LABEL)]);
    let refused = |f: &dyn Fn(&mut PiiRisk)| {
        let mut r = good.clone();
        f(&mut r);
        r.check().is_err()
    };
    assert!(
        refused(&|r| {
            r.score = 0.0;
            r.level = PiiRiskLevel::None;
        }),
        "a zero score over positive contributions"
    );
    assert!(
        refused(&|r| {
            r.score = 2.0;
            r.level = PiiRiskLevel::Low;
        }),
        "a score that is not the sum of the contributions"
    );
    assert!(refused(&|r| r.kinds[0].weighted_count = 0.0), "no weight for a mention");
    assert!(refused(&|r| r.kinds[1].weighted_count = 2.5), "more weight than two mentions can carry");
    assert!(refused(&|r| r.kinds[1].weighted_count = 1.0), "less weight than two mentions carry");
    assert!(refused(&|r| r.kinds[0].contribution *= 2.0), "a contribution its weight does not give");
    assert!(refused(&|r| r.kinds[0].severity = PiiSeverity::Low), "a severity its contribution does not match");
    assert!(refused(&|r| r.density_per_1000_chars = 0.0), "no density for mentions in a non-empty view");
    assert!(refused(&|r| r.chars = 0), "mentions in an empty view");
    // Within rounding, the computed summary is accepted as is.
    good.check().unwrap();
}

#[test]
fn every_computed_risk_passes_its_own_check() {
    let mut g = Lcg(42);
    for _ in 0..200 {
        let n = g.next(60);
        let mentions: Vec<_> = (0..n).map(|i| random_mention(&mut g, i * 10)).collect();
        risk(&mentions);
    }
}
