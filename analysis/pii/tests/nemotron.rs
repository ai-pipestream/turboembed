//! Tests for the 55-label Nemotron-PII mapping, canonical normalization,
//! and strict rule-overlap precedence (`#1888`).

use std::collections::BTreeSet;

use turbo_analysis::{Layer, Span, ViewId};
use turbo_analysis_layers::{
    AllowRule, AllowScope, CustomKinds, PiiCandidate, PiiDisposition, PiiKind, PiiScoreKind, RejectReason,
};
use turbo_pii::nemotron::{
    ModelCandidate, ModelProvenance, NEMOTRON_TYPES, nemotron_custom_kinds, nemotron_wire_name, normalize_model_value,
    resolve_model_with_rules, resolve_nemotron_kind,
};
use turbo_pii::{AllowList, Case, Pack, PiiDetector};

const PROVENANCE: ModelProvenance<'static> = ModelProvenance {
    detector_id: "turbo-tagger-pii",
    detector_version: "0.1.0",
    model_id: "turbo-en-nemotron-pii-modernbert-base-1",
    model_version: "1",
    calibrated: true,
};

#[test]
fn all_55_nemotron_types_are_sorted_unique_and_resolve_cleanly() {
    assert_eq!(NEMOTRON_TYPES.len(), 55);
    for w in NEMOTRON_TYPES.windows(2) {
        assert!(w[0].0 < w[1].0, "NEMOTRON_TYPES must be sorted: {} >= {}", w[0].0, w[1].0);
    }

    let mut custom_kinds = CustomKinds::new();
    let kinds: Vec<PiiKind> =
        NEMOTRON_TYPES.iter().map(|&(raw, _)| resolve_nemotron_kind(raw, &mut custom_kinds).unwrap()).collect();
    assert_eq!(kinds.len(), 55);

    let mut builtins = Vec::new();
    let mut customs = Vec::new();
    let mut seen_kinds = BTreeSet::new();
    for (&(raw, canonical), &kind) in NEMOTRON_TYPES.iter().zip(&kinds) {
        assert_eq!(nemotron_wire_name(raw), Some(canonical));
        assert!(seen_kinds.insert(kind.code()), "duplicate PiiKind mapping for {raw}");
        if !matches!(kind, PiiKind::Custom(_)) {
            builtins.push((raw, kind));
            assert_eq!(PiiKind::parse(canonical), Some(kind));
        } else {
            customs.push((raw, canonical));
            assert_eq!(custom_kinds.name(kind), Some(canonical));
        }
    }
    assert_eq!(builtins.len(), 15, "15 Nemotron types map to built-in PiiKinds");
    assert_eq!(customs.len(), 40, "40 Nemotron types map to CustomKinds");
    assert_eq!(custom_kinds.len(), 40);
    assert_eq!(nemotron_custom_kinds().len(), 40);

    for kind in kinds {
        custom_kinds.check(kind).unwrap();
    }
}

#[test]
fn unknown_snake_case_kind_name_is_refused_and_wire_lookup_returns_none() {
    assert_eq!(nemotron_wire_name("not_a_nemotron_label"), None);
    let mut kinds = CustomKinds::new();
    // Empty or control-character kind names are rejected by CustomKinds.
    assert!(resolve_nemotron_kind("", &mut kinds).is_err());
}

#[test]
fn model_value_normalization_matches_rule_conventions() {
    assert_eq!(normalize_model_value(PiiKind::UsSsn, "123 45 6789"), "123-45-6789");
    assert_eq!(normalize_model_value(PiiKind::UsSsn, "123456789"), "123-45-6789");
    assert_eq!(normalize_model_value(PiiKind::Card, "4111-1111-1111-1111"), "4111111111111111");
    assert_eq!(normalize_model_value(PiiKind::AbaRouting, "021-000-021"), "021000021");
    assert_eq!(normalize_model_value(PiiKind::Email, "Jane.Doe@Example.COM"), "Jane.Doe@example.com");
    assert_eq!(normalize_model_value(PiiKind::Phone, "+1 (415) 555-0199"), "+14155550199");
    assert_eq!(normalize_model_value(PiiKind::Bic, "deutdeffxxx"), "DEUTDEFFXXX");
    assert_eq!(normalize_model_value(PiiKind::Mac, "AA-BB-CC-DD-EE-FF"), "aa:bb:cc:dd:ee:ff");
    assert_eq!(normalize_model_value(PiiKind::StreetAddress, "  742   Evergreen\nTerrace "), "742 Evergreen Terrace");
}

#[test]
fn rules_win_across_exact_nested_enclosing_and_cross_kind_overlaps() {
    let text = "Patient Jane Doe, SSN 123-45-6789, card 4111-1111-1111-1111, ZIP 94107, MRN 884210.";
    let rules = PiiDetector::pack(Pack::AllStructured);
    let mut model_kinds = CustomKinds::new();
    let first_name = resolve_nemotron_kind("first_name", &mut model_kinds).unwrap();
    let last_name = resolve_nemotron_kind("last_name", &mut model_kinds).unwrap();
    let ssn = resolve_nemotron_kind("ssn", &mut model_kinds).unwrap();
    let card = resolve_nemotron_kind("credit_debit_card", &mut model_kinds).unwrap();
    let postcode = resolve_nemotron_kind("postcode", &mut model_kinds).unwrap();
    let mrn = resolve_nemotron_kind("medical_record_number", &mut model_kinds).unwrap();

    let span_of = |needle: &str| -> Span {
        let s = text.find(needle).unwrap();
        Span::new(s as u32, (s + needle.len()) as u32)
    };

    let model_candidates = vec![
        // Non-overlapping neural PII spans (names + MRN): kept with Probability score.
        ModelCandidate { span: span_of("Jane"), kind: first_name, probability: 0.96 },
        ModelCandidate { span: span_of("Doe"), kind: last_name, probability: 0.94 },
        // Exact (span, kind) match with SSN rule: rule wins, neural candidate becomes Repeats.
        ModelCandidate { span: span_of("123-45-6789"), kind: ssn, probability: 0.99 },
        // Wider enclosing neural span over Luhn-valid Card ("card 4111-1111-1111-1111"):
        // must NOT swallow the Card rule mention!
        ModelCandidate { span: span_of("card 4111-1111-1111-1111"), kind: card, probability: 0.98 },
        // Strictly nested neural span inside Card ("4111-1111"): becomes NestedIn(card_mention).
        ModelCandidate { span: span_of("4111-1111"), kind: mrn, probability: 0.61 },
        // Cross-kind overlap on "94107" (neural custom `postcode` vs built-in `UsZip` rule): rule wins.
        ModelCandidate { span: span_of("94107"), kind: postcode, probability: 0.91 },
        // Non-overlapping neural MRN with exact 0.0 probability: Some(0.0) distinct from rule None!
        ModelCandidate { span: span_of("884210"), kind: mrn, probability: 0.0 },
    ];

    let (pii, candidates) = resolve_model_with_rules(
        text,
        ViewId::SOURCE,
        Some(&rules),
        &model_kinds,
        PROVENANCE,
        None,
        None,
        &model_candidates,
    )
    .unwrap();

    pii.check().unwrap();
    candidates.check().unwrap();
    candidates.check_against(&pii).unwrap();

    let kept: Vec<_> = pii
        .mentions()
        .map(|m| {
            let (det_id, _) = pii.detectors.get(m.detector).unwrap();
            (
                &text[m.span.range()],
                pii.kind_name(m.kind).unwrap(),
                det_id,
                m.score.map(|s| (s.value, s.kind, s.calibrated)),
            )
        })
        .collect();

    assert_eq!(
        kept,
        vec![
            ("Jane", "first-name", "turbo-tagger-pii", Some((0.96, PiiScoreKind::Probability, true))),
            ("Doe", "last-name", "turbo-tagger-pii", Some((0.94, PiiScoreKind::Probability, true))),
            ("123-45-6789", "us-ssn", "turbo-pii:us-identity", None),
            ("4111-1111-1111-1111", "card", "turbo-pii:cursor", None),
            ("94107", "us-zip", "turbo-pii:us-identity", None),
            ("884210", "medical-record-number", "turbo-tagger-pii", Some((0.0, PiiScoreKind::Probability, true))),
        ]
    );

    let ssn_mention = pii.mentions().enumerate().find(|(_, m)| m.kind == PiiKind::UsSsn).unwrap().0 as u32;
    let card_mention = pii.mentions().enumerate().find(|(_, m)| m.kind == PiiKind::Card).unwrap().0 as u32;

    let has_ssn_repeat = candidates.candidates().any(|c: PiiCandidate<'_>| {
        c.span == span_of("123-45-6789")
            && c.score.is_some()
            && c.disposition == PiiDisposition::Repeats { mention: turbo_analysis::Ix::new(ssn_mention) }
    });
    assert!(has_ssn_repeat, "neural SSN twin must record PiiDisposition::Repeats pointing to rule SSN");

    let has_card_nested = candidates.candidates().any(|c: PiiCandidate<'_>| {
        c.span == span_of("4111-1111")
            && c.score.is_some()
            && c.disposition == PiiDisposition::NestedIn { mention: turbo_analysis::Ix::new(card_mention) }
    });
    assert!(has_card_nested, "strictly nested neural span must record PiiDisposition::NestedIn pointing to rule Card");
}

#[test]
fn score_threshold_and_allow_list_apply_to_neural_candidates_without_dropping_rules() {
    let text = "Contact Alice Example or Bob Secret at desk@example.com.";
    let rules = PiiDetector::pack(Pack::Contact);
    let mut model_kinds = CustomKinds::new();
    let first_name = resolve_nemotron_kind("first_name", &mut model_kinds).unwrap();
    let last_name = resolve_nemotron_kind("last_name", &mut model_kinds).unwrap();

    let span_of = |needle: &str| -> Span {
        let s = text.find(needle).unwrap();
        Span::new(s as u32, (s + needle.len()) as u32)
    };

    let allow = AllowList::from_terms(["example"]).unwrap().with_case(Case::Insensitive);

    let uncalibrated = ModelProvenance { calibrated: false, ..PROVENANCE };
    let model_candidates = vec![
        ModelCandidate { span: span_of("Alice"), kind: first_name, probability: 0.40 },
        ModelCandidate { span: span_of("Example"), kind: last_name, probability: 0.92 },
        ModelCandidate { span: span_of("Bob"), kind: first_name, probability: 0.88 },
        ModelCandidate { span: span_of("Secret"), kind: last_name, probability: 0.91 },
    ];

    let (pii, candidates) = resolve_model_with_rules(
        text,
        ViewId::SOURCE,
        Some(&rules),
        &model_kinds,
        uncalibrated,
        Some(0.75),
        Some(&allow),
        &model_candidates,
    )
    .unwrap();

    pii.check().unwrap();
    candidates.check().unwrap();
    candidates.check_against(&pii).unwrap();

    let kept_texts: Vec<&str> = pii.mentions().map(|m| &text[m.span.range()]).collect();
    assert_eq!(kept_texts, vec!["Bob", "Secret", "desk@example.com"]);
    assert!(!pii.mentions().next().unwrap().score.unwrap().calibrated);

    assert!(candidates.candidates().any(|c| {
        c.span == span_of("Alice")
            && c.disposition == PiiDisposition::RejectedByRule { reason: RejectReason::Threshold }
    }));
    assert!(candidates.candidates().any(|c| {
        c.span == span_of("Example")
            && c.disposition == PiiDisposition::AllowedByRule { scope: AllowScope::Global, rule: AllowRule::Term(0) }
    }));
}
