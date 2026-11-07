//! The `turbo:pii-candidates` layer (#1903): every candidate a detector
//! considered with the decision taken on it (kept, allowed by a rule,
//! nested, rejected by a rule, repeated), the detector and evidence each
//! carries, the audit report read from it, and the absence of any value in
//! its binary graph body.

mod support;

use support::*;
use turbo_analysis::{Layer, Span, ViewId};
use turbo_pii::rewrite::{HmacTokenizer, PiiAuditReport};
use turbo_pii::{DenyList, Pack, PiiDetector, PiiEvidence, PiiKind};

const KEY: &[u8] = b"0123456789abcdef0123456789abcdef";

#[test]
fn candidates_record_every_decision_with_its_rule_and_no_values() {
    use turbo_analysis::Ix;
    use turbo_pii::{AllowRule, AllowScope, DENY_LIST_DETECTOR, PiiDisposition, RejectReason};
    let codename = custom("codename");
    // "943 476 5919" is a UK NHS number and a phone on one span; the phone
    // is allowed by the kind list's second term, the e-mail by the global
    // pattern, the card by the global first term; "card" nests the phone
    // and the SIN-shaped number has no label.
    let text = "NHS 943 476 5919, mail help@example.com, card 4111 1111 1111 1111, ID 046 454 286, Palantir.";
    let detector = PiiDetector::pack(Pack::AllStructured)
        .with_custom_kinds(test_kinds())
        .with_deny_list(DenyList::from_terms(codename, ["Gotham", "Palantir"]).unwrap())
        .unwrap()
        .allow_terms(["4111111111111111"])
        .unwrap()
        .allow_glob("*@example.com")
        .unwrap()
        .allow_kind_terms(PiiKind::Phone, ["555", "9434765919"])
        .unwrap();
    let (pii, candidates) = detector.layer_with_candidates(text, ViewId::SOURCE);
    assert_eq!(pii.check(), Ok(()));
    assert_eq!(candidates.check(), Ok(()));
    assert_eq!(candidates.check_against(&pii), Ok(()));
    assert_eq!(candidates.custom_kinds, pii.custom_kinds);
    assert_eq!(candidates.detectors, pii.detectors);
    let rows: Vec<_> = candidates.candidates().map(|c| (c.kind, c.disposition)).collect();
    let nhs = Ix::new(0);
    let codename_ix = Ix::new(1);
    assert_eq!(
        rows,
        [
            (PiiKind::UkNhs, PiiDisposition::Kept { mention: nhs }),
            (PiiKind::Phone, PiiDisposition::AllowedByRule { scope: AllowScope::Kind, rule: AllowRule::Term(1) }),
            (PiiKind::Email, PiiDisposition::AllowedByRule { scope: AllowScope::Global, rule: AllowRule::Pattern(0) }),
            (PiiKind::Card, PiiDisposition::AllowedByRule { scope: AllowScope::Global, rule: AllowRule::Term(0) }),
            (PiiKind::CaSin, PiiDisposition::RejectedByRule { reason: RejectReason::LabelRequired }),
            (codename, PiiDisposition::Kept { mention: codename_ix }),
        ]
    );
    assert_eq!(
        pii.mentions().map(|m| (m.kind, m.normalized)).collect::<Vec<_>>(),
        [(PiiKind::UkNhs, "9434765919"), (codename, "Palantir")]
    );
    let deny = candidates.get(Ix::new(5));
    assert_eq!(deny.evidence, &[PiiEvidence::DenyHit { list: 0, term: 1 }]);
    assert_eq!(candidates.detectors.get(deny.detector), Some((DENY_LIST_DETECTOR, turbo_pii::BUILTIN_VERSION)));
    assert_eq!(
        candidates.detectors.get(candidates.detector[0]),
        Some(("turbo-pii:eu-identity", turbo_pii::BUILTIN_VERSION))
    );
    assert_eq!(candidates.suppressed_by_kind(), [(PiiKind::Card, 1), (PiiKind::Email, 1), (PiiKind::Phone, 1)]);
    let tokenizer = HmacTokenizer::new(KEY).unwrap();
    let report = PiiAuditReport::with_candidates(&pii, &candidates, &tokenizer, 3);
    assert_eq!((report.total, report.total_suppressed), (2, 3));
    assert_eq!(report.suppressed(PiiKind::Phone), 1);

    // The candidates layer carries no values: the binary graph of a
    // document holding only the candidates has none of the suppressed
    // values in it.
    let mut doc = turbo_analysis::Document::new(text);
    doc.add(&turbo_analysis::Key::DEFAULT, candidates, turbo_analysis::Provenance::by("test")).unwrap();
    let graph = turbo_analysis::export::export(&doc, &turbo_analysis_layers::export::codecs()).unwrap();
    let body = graph.layers.iter().find(|l| l.layer == turbo_pii::PiiCandidates::ID).unwrap().body.clone().unwrap();
    for value in ["help@example.com", "4111", "9434765919", "046454286", "Palantir"] {
        assert!(!body.windows(value.len()).any(|w| w == value.as_bytes()), "{value} in the candidates layer");
    }

    // Nested and repeated candidates point at the mention they lost to.
    let text = "TICKET alice@example.com";
    let ticket = custom("ticket");
    let nested = PiiDetector::pack(Pack::Contact)
        .with_custom_kinds(test_kinds())
        .with_custom_fn("ticket", ticket, |text, sink| {
            let span = Span::new(0, text.len() as u32);
            sink.add_text(span, custom("ticket"), &[PiiEvidence::Shape]);
            sink.add_text(span, custom("ticket"), &[PiiEvidence::Rule { rule: 4 }]);
        })
        .unwrap();
    let (pii, candidates) = nested.layer_with_candidates(text, ViewId::SOURCE);
    assert_eq!(pii.mentions().map(|m| m.kind).collect::<Vec<_>>(), [ticket]);
    assert_eq!(
        candidates.disposition,
        [
            PiiDisposition::Kept { mention: Ix::new(0) },
            PiiDisposition::Repeats { mention: Ix::new(0) },
            PiiDisposition::NestedIn { mention: Ix::new(0) },
        ]
    );
    assert_eq!(candidates.get(Ix::new(1)).evidence, &[PiiEvidence::Rule { rule: 4 }]);
}
