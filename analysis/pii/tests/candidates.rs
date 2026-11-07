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

    // The candidates layer carries no values: its binary graph record has
    // none of the suppressed values in it (the kept ones sit in the pii
    // record, which the candidates index).
    let mut doc = turbo_analysis::Document::new(text);
    let by = turbo_analysis::Provenance::by("test");
    doc.add(&turbo_analysis::Key::DEFAULT, pii, by.clone()).unwrap();
    doc.add(&turbo_analysis::Key::DEFAULT, candidates, by).unwrap();
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

/// Replaces the one occurrence of `from` in the body of `layer` by `to`.
fn patch(graph: &mut turbo_analysis::export::DocumentGraph, layer: &str, from: &[u8], to: &[u8]) {
    let body = graph.layers.iter_mut().find(|l| l.layer == layer).unwrap().body.as_mut().unwrap();
    let at: Vec<usize> = (0..body.len() - from.len() + 1).filter(|&i| body[i..].starts_with(from)).collect();
    assert_eq!(at.len(), 1, "one place to patch");
    body[at[0]..at[0] + to.len()].copy_from_slice(to);
}

#[test]
fn imported_label_spans_and_mention_references_are_held_to_the_document() {
    use turbo_analysis::export::{export, import};
    use turbo_analysis::{Document, Key, Provenance};
    use turbo_analysis_layers::export::codecs;
    use turbo_pii::{Pii, PiiCandidates, PiiMention};
    use turbo_pii::{PiiCandidate, PiiDisposition};
    let text = "card 4111 1111 1111 1111.";
    let label = PiiEvidence::Label { span: Span::new(0, 4), phrase: 0 };
    let evidence = [PiiEvidence::Shape, label];
    let mut pii = Pii::new(ViewId::SOURCE);
    let detector = pii.detector("test", "1").unwrap();
    let span = Span::new(5, 24);
    pii.push(PiiMention { span, kind: PiiKind::Card, detector, evidence: &evidence, score: None, normalized: "x" });
    let mut candidates = PiiCandidates::new(ViewId::SOURCE);
    let detector = candidates.detectors.add("test", "1").unwrap();
    let disposition = PiiDisposition::Kept { mention: turbo_analysis::Ix::new(0) };
    candidates.push(PiiCandidate {
        span,
        kind: PiiKind::Card,
        detector,
        evidence: &evidence,
        score: None,
        disposition,
    });
    let mut doc = Document::new(text);
    let by = Provenance::by("test");
    doc.add(&Key::DEFAULT, pii, by.clone()).unwrap();
    doc.add(&Key::DEFAULT, candidates, by).unwrap();
    let good = export(&doc, &codecs()).unwrap();
    import(&good, &codecs()).unwrap();

    // The label fact: tag 3, span 0..4, phrase 0. Moving its end past the
    // text, or onto no character boundary, is refused on import by the
    // document the layer is restored into.
    let label_bytes = |end: u32| {
        let mut b = vec![3, 0, 0, 0, 0];
        b.extend_from_slice(&end.to_le_bytes());
        b.extend_from_slice(&[0, 0, 0, 0]);
        b
    };
    for layer in [Pii::ID, PiiCandidates::ID] {
        let mut bad = good.clone();
        patch(&mut bad, layer, &label_bytes(4), &label_bytes(40));
        let err = import(&bad, &codecs()).err().expect("refused").to_string();
        assert!(err.contains("label evidence span 0..40"), "{layer}: {err}");
    }
    // The dispositions: a count of 1, then tag 1 (kept) and mention 0.
    // A reference past the one mention is refused before the document
    // sees it, and the error names no value.
    let mut bad = good.clone();
    let dispositions = |mention: u32| {
        let mut b = vec![1u8, 0, 0, 0, 1];
        b.extend_from_slice(&mention.to_le_bytes());
        b
    };
    let (kept, past) = (dispositions(0), dispositions(5));
    patch(&mut bad, PiiCandidates::ID, &kept, &past);
    let err = import(&bad, &codecs()).err().expect("refused").to_string();
    assert!(err.contains("refers to mention 5 of 1") && !err.contains("4111"), "{err}");
}
