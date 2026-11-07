//! Replacing a detector's custom-kind table never changes what a rule
//! already set means: a scanner, deny list, per-kind allow list or label
//! override whose custom kind the new table drops or renames refuses the
//! replacement, and a table that keeps every used kind (an extension) is
//! taken.

mod support;

use support::*;
use turbo_analysis::Layer;
use turbo_pii::{CustomKindId, CustomKinds, DenyList, KindError, PiiDetector, PiiKind, PiiSink, Scanner};

const FIRST: PiiKind = PiiKind::Custom(CustomKindId::new(0));

fn table(names: &[&str]) -> CustomKinds {
    CustomKinds::from_names(names).unwrap()
}

fn none(_text: &str, _sink: &mut PiiSink<'_>) {}

#[test]
fn a_stale_allow_rule_cannot_suppress_a_renamed_kind() {
    // An allow term for "badge" at index 0, then a table naming index 0
    // "secret": the rule would suppress secrets, so the swap is refused.
    let badge =
        PiiDetector::new(vec![]).with_custom_kinds(table(&["badge"])).allow_kind_term(FIRST, "SECRET1").unwrap();
    let swapped = badge.clone().try_with_custom_kinds(table(&["secret"]));
    assert_eq!(swapped.err(), Some(KindError::Renamed(0)));

    // Extending the table keeps "badge" at 0, so the secret deny list at 1
    // finds SECRET1 and the badge allow rule leaves it alone.
    let extended = badge.try_with_custom_kinds(table(&["badge", "secret"])).unwrap();
    let secret = extended.custom_kinds().find("secret").unwrap();
    let det = extended.with_deny_list(DenyList::from_terms(secret, ["SECRET1"]).unwrap()).unwrap();
    let pii = layer(&det, "code SECRET1 here");
    assert_eq!(pii.mentions().map(|m| m.kind).collect::<Vec<_>>(), [secret]);
}

#[test]
fn an_empty_table_after_a_deny_list_is_refused_not_emitted() {
    let det = PiiDetector::new(vec![])
        .with_custom_kinds(table(&["secret"]))
        .with_deny_list(DenyList::from_terms(FIRST, ["SECRET1"]).unwrap())
        .unwrap();
    assert_eq!(det.clone().try_with_custom_kinds(CustomKinds::new()).err(), Some(KindError::UnknownCustom(0)));
    // Unchanged, the detector still writes a layer its own table checks.
    assert!(layer(&det, "SECRET1").check().is_ok());
}

#[test]
fn scanners_and_label_overrides_hold_their_kinds_too() {
    let scanner = PiiDetector::new(vec![Scanner::custom_fn("none", FIRST, none)]);
    // A scanner given before any table takes the first table naming it.
    let named = scanner.try_with_custom_kinds(table(&["badge"])).unwrap();
    assert_eq!(named.clone().try_with_custom_kinds(CustomKinds::new()).err(), Some(KindError::UnknownCustom(0)));
    assert_eq!(named.try_with_custom_kinds(table(&["ticket"])).err(), Some(KindError::Renamed(0)));

    let labels = PiiDetector::new(vec![]).with_custom_kinds(table(&["badge"])).label_required(FIRST);
    assert_eq!(labels.clone().try_with_custom_kinds(table(&["ticket"])).err(), Some(KindError::Renamed(0)));
    assert!(labels.try_with_custom_kinds(table(&["badge", "ticket"])).is_ok());

    // Built-in rules never tie the table.
    let builtin =
        PiiDetector::cursor().label_required(PiiKind::Email).allow_kind_term(PiiKind::Email, "a@b.c").unwrap();
    assert!(builtin.with_custom_kinds(table(&["badge"])).try_with_custom_kinds(CustomKinds::new()).is_ok());
}

#[test]
#[should_panic(expected = "a custom-kind table every detector rule agrees with")]
fn the_panicking_builder_refuses_too() {
    let det = PiiDetector::new(vec![]).with_custom_kinds(table(&["badge"])).label_boosts(FIRST);
    let _ = det.with_custom_kinds(table(&["secret"]));
}
