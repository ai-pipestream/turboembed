//! The `PiiDateStrategy` arm (issue #1884): each mode converts to the
//! native strategy and back, its fields are checked and in the digest, a
//! date rule turns the opt-in date pack on, and the redact step reduces
//! dates with edits that record the mode's `PiiStrategy` on the wire.

#[path = "pii_policy/common.rs"]
mod common;
#[path = "pii_redact/fixture.rs"]
mod fixture;

use common::*;
use fixture::*;
use pb::pii_date_strategy::Mode;
use pb::pii_rewrite_strategy::Strategy;
use turbo_analysis::ViewId;
use turbo_kserve::offsets::OffsetMode;
use turbo_kserve::pii_policy::{
    NoRegistry, PolicyError, PolicyResolver, RedactionPolicy, edit_strategy, strategy_to_proto,
};
use turbo_kserve::proto::result as pb;
use turbo_kserve::proto::result::typed_annotation::Value;
use turbo_kserve::results::typed::document_to_typed;
use turbo_pii::rewrite::{Date, DateRewrite, DateShift, RewriteStrategy};
use turbo_pii::{Pack, PiiStrategyKind};

fn date(mode: Mode, mask_char: &str) -> Strategy {
    Strategy::Date(pb::PiiDateStrategy { mode: Some(mode), mask_char: mask_char.to_owned() })
}

fn year() -> Mode {
    Mode::Year(pb::PiiDateYear {})
}

fn shift(s: pb::PiiDateShift) -> Mode {
    Mode::Shift(s)
}

fn relative(reference: &str) -> Mode {
    Mode::Relative(pb::PiiDateRelative { reference: reference.to_owned() })
}

/// One rule for dates with `s` as its only strategy.
fn date_policy(s: Strategy) -> pb::RedactionPolicy {
    policy(vec![rule("date", builtin(pb::PiiKindId::Date), vec![entry(s)])])
}

fn native_with(s: Strategy, r: &dyn PolicyResolver) -> Result<RewriteStrategy, PolicyError> {
    let p = RedactionPolicy::from_proto(&date_policy(s), r)?;
    Ok(p.rules[0].strategies[0].strategy.clone())
}

fn native(s: Strategy) -> Result<RewriteStrategy, PolicyError> {
    native_with(s, &NoRegistry)
}

#[test]
fn each_mode_converts_and_round_trips() {
    let d = |y, m, day| Date::new(y, m, day).unwrap();
    let cases = [
        (date(year(), ""), DateRewrite::year(), PiiStrategyKind::DateYear, pb::PiiStrategy::DateYear),
        (
            date(shift(pb::PiiDateShift { years: -1, months: 2, days: -3, ..Default::default() }), "#"),
            DateRewrite::shift(DateShift::fixed(-1, 2, -3)).with_mask('#'),
            PiiStrategyKind::DateShift,
            pb::PiiStrategy::DateShift,
        ),
        (
            date(relative("2026-07-14"), ""),
            DateRewrite::relative(d(2026, 7, 14)),
            PiiStrategyKind::DateRelative,
            pb::PiiStrategy::DateRelative,
        ),
    ];
    for (arm, want, kind, value) in cases {
        let got = native(arm.clone()).unwrap();
        assert_eq!(got, RewriteStrategy::Date(want), "{arm:?}");
        assert_eq!(strategy_to_proto(&got).unwrap().strategy, Some(arm.clone()), "{arm:?} round trips");
        assert_eq!(got.kind(), kind);
        assert_eq!(edit_strategy(&arm), value);
        assert_eq!(i32::from(kind.id()), value as i32);
    }
    assert_eq!(native(date(year(), "*")).unwrap(), native(date(year(), "")).unwrap());
}

#[test]
fn a_keyed_shift_resolves_its_key_and_never_travels_back() {
    let keyed = |max_days, context: &str| {
        date(
            shift(pb::PiiDateShift {
                key: key_ref("tokens", 1),
                max_days,
                context: context.to_owned(),
                ..Default::default()
            }),
            "",
        )
    };
    let got = native_with(keyed(0, "doc-1"), &TestRegistry).unwrap();
    let RewriteStrategy::Date(d) = &got else { panic!("{got:?}") };
    let turbo_pii::rewrite::DateMode::Shift(s) = d.mode() else { panic!("{d:?}") };
    let want = DateShift::default().keyed(&common::KEY, "doc-1", 30).unwrap();
    assert_eq!(s.keyed_days(), want.keyed_days(), "0 means 30 days");
    assert!(strategy_to_proto(&got).is_none(), "key bytes have no message");
    assert!(matches!(native(keyed(0, "doc-1")), Err(PolicyError::Unresolved { field: "date.shift.key", .. })));
    let err = native_with(keyed(3651, "doc-1"), &TestRegistry).unwrap_err();
    assert!(matches!(err, PolicyError::Invalid { .. }), "{err:?}");
    let digest = |s| RedactionPolicy::from_proto(&date_policy(s), &TestRegistry).unwrap().digest;
    assert_ne!(digest(keyed(0, "doc-1")), digest(keyed(0, "doc-2")), "the context is in the digest");
    assert_ne!(digest(keyed(0, "doc-1")), digest(keyed(31, "doc-1")));
}

#[test]
fn bad_fields_are_refused() {
    let missing = native(Strategy::Date(pb::PiiDateStrategy::default())).unwrap_err();
    assert!(matches!(missing, PolicyError::Missing { field: "date.mode", .. }), "{missing:?}");
    let no_ref = native(date(relative(""), "")).unwrap_err();
    assert!(matches!(no_ref, PolicyError::Missing { field: "date.relative.reference", .. }), "{no_ref:?}");
    for bad in ["2026-7-14", "2026-02-30", "July 14, 2026", "20260714"] {
        assert!(matches!(native(date(relative(bad), "")), Err(PolicyError::Invalid { .. })), "{bad}");
    }
    assert!(matches!(native(date(year(), "**")), Err(PolicyError::Invalid { .. })));
    let unkeyed = shift(pb::PiiDateShift { max_days: 5, ..Default::default() });
    assert!(matches!(native(date(unkeyed, "")), Err(PolicyError::Invalid { .. })), "max_days needs a key");
    let unkeyed = shift(pb::PiiDateShift { context: "doc".to_owned(), ..Default::default() });
    assert!(matches!(native(date(unkeyed, "")), Err(PolicyError::Invalid { .. })), "context needs a key");
}

#[test]
fn the_digest_covers_the_mode_and_its_fields() {
    let digest = |s| RedactionPolicy::from_proto(&date_policy(s), &NoRegistry).unwrap().digest;
    let base = digest(date(year(), ""));
    assert_eq!(base, digest(date(year(), "")));
    assert_ne!(base, digest(date(year(), "#")));
    let days = |n| digest(date(shift(pb::PiiDateShift { days: n, ..Default::default() }), ""));
    assert_ne!(days(1), days(-1));
    assert_ne!(days(0), base, "a shift by nothing is still another mode");
    assert_ne!(digest(date(relative("2026-07-14"), "")), digest(date(relative("2026-07-15"), "")));
}

#[test]
fn a_date_rule_turns_the_date_pack_on() {
    let p = RedactionPolicy::from_proto(&date_policy(date(year(), "")), &NoRegistry).unwrap();
    assert_eq!(p.opted_in_packs().collect::<Vec<_>>(), [Pack::Dates]);
    let none = RedactionPolicy::from_proto(&one(Strategy::Mask(pb::PiiMaskStrategy::default())), &NoRegistry).unwrap();
    assert_eq!(none.opted_in_packs().count(), 0, "a policy without a date rule keeps the default packs");
}

#[test]
fn the_redact_step_reduces_dates_and_the_wire_edits_say_so() {
    let text = "Admitted July 14, 2026 and born 1952-07-04.";
    for (mode, want, value) in [
        (year(), "Admitted 2026 and born 1952.", pb::PiiStrategy::DateYear),
        (
            shift(pb::PiiDateShift { days: -20, ..Default::default() }),
            "Admitted June 24, 2026 and born 1952-06-14.",
            pb::PiiStrategy::DateShift,
        ),
        (relative("2026-07-20"), "Admitted 6 days ago and born 74 years ago.", pb::PiiStrategy::DateRelative),
    ] {
        let p = convert(&date_policy(date(mode, "")), &NoRegistry);
        let doc = run_with(text, p.clone(), false);
        let (view, edits) = redacted(&doc);
        assert_eq!(view.text(), want);
        assert_eq!(edits.edits().count(), 2);
        let typed = document_to_typed("redact", &doc, ViewId::SOURCE, OffsetMode::Utf8).unwrap();
        let layer = typed.layers.iter().find(|l| l.id == "turbo:pii-edits").expect("the edits layer");
        assert_eq!(layer.annotations.len(), 2);
        for a in &layer.annotations {
            let Some(Value::PiiEdit(e)) = &a.value else { panic!("not an edit") };
            assert_eq!(e.strategy, value as i32);
            assert_eq!(e.policy_digest, p.digest.as_bytes().to_vec());
        }
    }
}
