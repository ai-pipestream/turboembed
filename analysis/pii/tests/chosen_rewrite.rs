//! A rewrite whose strategy is chosen mention by mention: each edit names
//! the winner's rule and rule id, overlapping mentions keep each member's
//! chosen strategy, and a refusing strategy refuses its cluster.

use turbo_analysis::{Document, Ix, Layer, PolicyDigest, Span, View, ViewId};
use turbo_pii::rewrite::{
    Abbreviator, Choice, MaskPolicy, RewriteError, RewriteStrategy, StaticReplacer, StrategyChooser, Truncate,
    rewrite_chosen,
};
use turbo_pii::{Pii, PiiDetectorRef, PiiEvidence, PiiKind, PiiMention, PiiRule, PiiScore, PiiScoreKind};

const TEXT: &str = "mail ann@x.io or 555-1234 now";

/// Phones scoring at least 0.5 get `[P]` by rule "sure-phones"; every
/// other mention gets the default mask.
struct ByScore {
    sure: RewriteStrategy,
    default: RewriteStrategy,
}

impl StrategyChooser for ByScore {
    fn choose<'r>(&'r self, _: &Pii, m: &PiiMention<'_>) -> Choice<'r> {
        if m.kind == PiiKind::Phone && m.score.is_some_and(|s| s.value >= 0.5) {
            Choice { strategy: &self.sure, rule: PiiRule::Kind(m.kind), rule_id: "sure-phones" }
        } else {
            Choice { strategy: &self.default, rule: PiiRule::Default, rule_id: "" }
        }
    }
}

fn chooser() -> ByScore {
    ByScore { sure: StaticReplacer::new("[P]").into(), default: MaskPolicy::new('*').unwrap().into() }
}

fn layer(mentions: &[(u32, u32, PiiKind, f32)]) -> Pii {
    let mut pii = Pii::new(ViewId::SOURCE);
    pii.detector("test", "1").unwrap();
    for &(start, end, kind, score) in mentions {
        pii.push(PiiMention {
            span: Span::new(start, end),
            kind,
            detector: PiiDetectorRef::new(0),
            evidence: &[PiiEvidence::Shape],
            score: Some(PiiScore {
                kind: PiiScoreKind::Probability,
                value: score,
                model: PiiDetectorRef::new(0),
                calibrated: false,
            }),
            normalized: "v",
        });
    }
    pii
}

fn src() -> View {
    Document::new(TEXT).get::<View>().unwrap().clone()
}

#[test]
fn each_edit_names_the_rule_that_chose_its_strategy() {
    let pii = layer(&[(5, 13, PiiKind::Email, 0.9), (17, 25, PiiKind::Phone, 0.9)]);
    let r = rewrite_chosen(&src(), &pii, &chooser(), 4).unwrap().with_policy_digest(PolicyDigest::from_bytes([3; 32]));
    assert_eq!(r.text(), "mail ******** or [P] now");
    let e = r.edits();
    assert_eq!((e.get(Ix::new(0)).rule, e.get(Ix::new(0)).rule_id), (PiiRule::Default, ""));
    assert_eq!((e.get(Ix::new(1)).rule, e.get(Ix::new(1)).rule_id), (PiiRule::Kind(PiiKind::Phone), "sure-phones"));
    assert_eq!((e.policy_revision, e.policy_digest), (4, Some(PolicyDigest::from_bytes([3; 32]))));
    assert_eq!(e.check(), Ok(()));
}

#[test]
fn the_same_kind_takes_another_strategy_when_its_facts_differ() {
    let pii = layer(&[(17, 25, PiiKind::Phone, 0.1)]);
    let r = rewrite_chosen(&src(), &pii, &chooser(), 0).unwrap();
    assert_eq!(r.text(), "mail ann@x.io or ******** now");
    assert_eq!(r.edits().get(Ix::new(0)).rule_id, "");
}

#[test]
fn same_kind_overlap_keeps_the_stricter_members_mask() {
    // The outer phone wins and may retain its final four digits; the
    // inner phone's lower score selects a full mask for those digits.
    let pii = layer(&[(17, 25, PiiKind::Phone, 0.9), (21, 25, PiiKind::Phone, 0.1)]);
    let by_score =
        ByScore { sure: MaskPolicy::last_n(4, '*').unwrap().into(), default: MaskPolicy::new('*').unwrap().into() };
    let r = rewrite_chosen(&src(), &pii, &by_score, 0).unwrap();
    assert_eq!(r.text(), "mail ann@x.io or ******** now");
    let edit = r.edits().get(Ix::new(0));
    assert_eq!(edit.members, &[Ix::new(0), Ix::new(1)]);
    assert_eq!(edit.rule_id, "sure-phones");
    assert_eq!(r.edits().check(), Ok(()));
}

#[test]
fn same_kind_overlap_protects_truncation_and_initials_too() {
    let pii = layer(&[(17, 25, PiiKind::Phone, 0.9), (21, 25, PiiKind::Phone, 0.1)]);
    for (sure, expected) in [
        (Truncate::keep_end(4).into(), "mail ann@x.io or ******** now"),
        (Abbreviator::compact().into(), "mail ann@x.io or 5* now"),
    ] {
        let by_score = ByScore { sure, default: MaskPolicy::new('*').unwrap().into() };
        assert_eq!(rewrite_chosen(&src(), &pii, &by_score, 0).unwrap().text(), expected);
    }
}

#[test]
fn unicode_overlap_masks_all_member_bytes_and_preserves_mapping() {
    let text = "☎ １２３４-５６７８";
    let source = Document::new(text).get::<View>().unwrap().clone();
    let pii = layer(&[(4, 29, PiiKind::Phone, 0.9), (17, 29, PiiKind::Phone, 0.1)]);
    let by_score =
        ByScore { sure: MaskPolicy::last_n(4, '*').unwrap().into(), default: MaskPolicy::new('*').unwrap().into() };
    let r = rewrite_chosen(&source, &pii, &by_score, 0).unwrap();
    assert_eq!(r.text(), format!("☎ {}", "*".repeat(25)));
    assert_eq!(r.text().len(), text.len());
    for (_, span) in pii.spans.iter() {
        let derived = r.to_derived(span).unwrap();
        assert!(r.to_original(derived).unwrap().contains(span));
    }
    assert_eq!(r.to_derived(Span::new(0, 4)).unwrap(), Span::new(0, 4));
}

#[test]
fn overlapping_mentions_keep_each_kind_s_choice_and_a_refusing_choice_refuses() {
    // Phone 17..25 and an email 21..25 inside it: the email wins the
    // cluster (smaller priority) and masks; the phone's static text never
    // applies, so the whole union is masked.
    let pii = layer(&[(17, 25, PiiKind::Phone, 0.9), (21, 25, PiiKind::Email, 0.9)]);
    let r = rewrite_chosen(&src(), &pii, &chooser(), 0).unwrap();
    assert_eq!(r.edits().spans.len(), 1);
    assert_eq!(r.edits().get(Ix::new(0)).members, &[Ix::new(0), Ix::new(1)][..]);
    // The same cluster where the phone's chosen strategy refuses overlaps.
    let refusing = ByScore {
        sure: StaticReplacer::new("[P]").refusing_overlaps().into(),
        default: MaskPolicy::new('*').unwrap().into(),
    };
    assert!(matches!(rewrite_chosen(&src(), &pii, &refusing, 0), Err(RewriteError::Overlap(..))));
    // Without the phone scoring high enough, nothing refuses.
    let low = layer(&[(17, 25, PiiKind::Phone, 0.1), (21, 25, PiiKind::Email, 0.9)]);
    assert!(rewrite_chosen(&src(), &low, &refusing, 0).is_ok());
}
