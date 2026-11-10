//! #1884: HIPAA Safe Harbor age aggregation (`AgeSafeHarbor`). Ages of 89
//! or under stay, older ages become `90+`, what is not an age is masked
//! whole, and the strategy works as a rewrite and inside a per-kind
//! policy. Values are made up.

mod support;

use support::*;
use turbo_analysis::{Annotator, Span};
use turbo_pii::rewrite::{
    AgeSafeHarbor, MaskPolicy, RedactedView, RewritePolicy, RewriteStrategy, Truncate, ZipSafeHarbor, source_view,
};
use turbo_pii::{Pack, PiiKind, PiiStrategyKind};

#[test]
fn every_age_over_89_becomes_one_category() {
    let age = AgeSafeHarbor::new();
    assert_eq!(AgeSafeHarbor::default(), age);
    for n in 0..=89u32 {
        assert_eq!(age.apply(&n.to_string()), n.to_string(), "{n} is kept");
    }
    for n in 90..=125u32 {
        assert_eq!(age.apply(&n.to_string()), "90+", "{n}");
    }
    // The boundary on both sides, leading zeros, and runs longer than an
    // age the scanner finds.
    assert_eq!(age.apply("89"), "89");
    assert_eq!(age.apply("90"), "90+");
    assert_eq!(age.apply("089"), "089");
    assert_eq!(age.apply("093"), "90+");
    assert_eq!(age.apply("0"), "0");
    assert_eq!(age.apply("000"), "000");
    assert_eq!(age.apply("100"), "90+");
    assert_eq!(age.apply("99999999999999999999999"), "90+", "too long to parse is still over 89");
}

#[test]
fn what_is_not_an_age_is_masked_whole() {
    let age = AgeSafeHarbor::new();
    for not in ["", "9O", "93.5", "-93", "+93", " 93", "93 ", "jane@example.com"] {
        assert_eq!(age.apply(not), "*".repeat(not.chars().count()), "{not:?}");
    }
    // Non-ASCII digits are not an age; one mask per character.
    assert_eq!(age.apply("\u{0669}\u{0663}"), "**");
    assert_eq!(age.clone().with_mask('#').apply("n/a"), "###");
    assert_eq!(age.with_mask('#').apply("93"), "90+", "the mask never touches an age");
}

#[test]
fn the_rewrite_aggregates_detected_ages_and_keeps_the_words() {
    let text = "A 93-year-old woman, aged 34, and a man age: 102 were seen.";
    let pii = layer(&pack(Pack::AllStructured), text);
    let ages: Vec<Span> = pii.mentions().filter(|m| m.kind == PiiKind::Age).map(|m| m.span).collect();
    assert_eq!(ages.len(), 3, "every age is found");
    let r = AgeSafeHarbor::new().rewrite(&source_view(text), &pii).unwrap();
    assert_eq!(r.text(), "A 90+-year-old woman, aged 34, and a man age: 90+ were seen.");
    let sources: Vec<Span> = pii.mentions().map(|m| m.span).collect();
    assert_offset_and_span_maps(text, &sources, &r);
    assert!(r.edits().edits().all(|e| e.strategy == PiiStrategyKind::AgeSafeHarbor));
    assert_eq!(r.edits().edits().count(), 3);
}

#[test]
fn a_policy_sends_age_here_and_other_kinds_elsewhere() {
    let text = "Patient aged 97, call (555) 123-4567.";
    let pii = layer(&wide(), text);
    assert!(pii.mentions().any(|m| m.kind == PiiKind::Age));
    assert!(pii.mentions().any(|m| m.kind == PiiKind::Phone));
    let policy = RewritePolicy::new(MaskPolicy::new('#').unwrap()).with(PiiKind::Age, AgeSafeHarbor::new());
    assert_eq!(policy.strategy(PiiKind::Age).kind(), PiiStrategyKind::AgeSafeHarbor);
    let r = policy.rewrite(&source_view(text), &pii).unwrap();
    assert_eq!(r.text(), "Patient aged 90+, call ##############.");
    let strategies: Vec<_> = r.edits().edits().map(|e| e.strategy).collect();
    assert_eq!(strategies, [PiiStrategyKind::AgeSafeHarbor, PiiStrategyKind::Mask]);
}

#[test]
fn overlapping_mentions_are_masked_whole() {
    let text = "age 34";
    let pii = manual(&[(4, 6, PiiKind::Age, "34"), (0, 6, PiiKind::Phone, "age 34")]);
    let age = AgeSafeHarbor::new();
    assert_eq!(age.rewrite(&source_view(text), &pii).unwrap().text(), "******");
    let policy = RewritePolicy::new(RewriteStrategy::from(age.clone()));
    assert_eq!(policy.rewrite(&source_view(text), &pii).unwrap().text(), "******");
    // An age member of another winner's cluster keeps none of its digits.
    let p = RewritePolicy::new(Truncate::keep_end(2)).with(PiiKind::Age, age.clone());
    let phone_wins = manual(&[(0, 6, PiiKind::Phone, "age 34"), (4, 6, PiiKind::Age, "34")]);
    assert_eq!(p.rewrite(&source_view(text), &phone_wins).unwrap().text(), "******");
    assert!(age.refusing_overlaps().rewrite(&source_view(text), &pii).is_err());
}

#[test]
fn the_config_hash_covers_the_mask_and_overlap_choice() {
    let hash = |s: RewriteStrategy| {
        RedactedView::new(RewritePolicy::new(MaskPolicy::new('*').unwrap()).with(PiiKind::Age, s)).config_hash()
    };
    let base = hash(AgeSafeHarbor::new().into());
    assert_eq!(base, hash(AgeSafeHarbor::new().into()));
    assert_ne!(base, hash(AgeSafeHarbor::new().with_mask('#').into()));
    assert_ne!(base, hash(AgeSafeHarbor::new().refusing_overlaps().into()));
    assert_ne!(base, hash(MaskPolicy::new('*').unwrap().into()), "the strategy kind is part of the hash");
    assert_ne!(base, hash(ZipSafeHarbor::hhs_2000().into()));
}

/// A short all-digit identifier of another kind (a caller's employee id, a
/// phone extension) is not an age: the strategy masks it whole, alone and
/// as the default of a policy, so it never passes through unchanged.
#[test]
fn a_digit_mention_of_another_kind_is_masked_whole() {
    let employee = PiiKind::Custom(turbo_pii::CustomKindId::new(0));
    let text = "id 42 ext 7 aged 93";
    let pii = manual(&[(3, 5, employee, "42"), (10, 11, PiiKind::PhoneExtension, "7"), (17, 19, PiiKind::Age, "93")]);
    let want = "id ** ext * aged 90+";
    assert_eq!(AgeSafeHarbor::new().rewrite(&source_view(text), &pii).unwrap().text(), want);
    let policy = RewritePolicy::new(AgeSafeHarbor::new());
    assert_eq!(policy.rewrite(&source_view(text), &pii).unwrap().text(), want);
}
