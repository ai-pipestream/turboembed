//! Every reject reason of a `turbo:pii-candidates` row is served under
//! its own `PiiRejectReason` value and validates; a value past the known
//! ones is refused (#2286).

use turbo_analysis::{Document, Ix, Key, Provenance, Span, ViewId};
use turbo_analysis_layers::{
    Pii, PiiCandidate, PiiCandidates, PiiDisposition, PiiEvidence, PiiKind, PiiMention, RejectReason,
};
use turbo_kserve::api::TURBO_E_INVALID_ARGUMENT;
use turbo_kserve::offsets::OffsetMode;
use turbo_kserve::proto::result::pii_disposition::Disposition;
use turbo_kserve::proto::result::typed_annotation::Value;
use turbo_kserve::proto::result::{PiiDisposition as WireDisposition, PiiRejectReason};
use turbo_kserve::results::disclosure::Disclosure;
use turbo_kserve::results::typed::{document_to_typed_with, validate_typed_document};

#[test]
fn each_reject_reason_is_served_under_its_own_value() {
    let mut doc = Document::new("a@b.co c@d.co");
    let mut pii = Pii::new(ViewId::SOURCE);
    let detector = pii.detector("test:det", "1").unwrap();
    let (kind, score, evidence) = (PiiKind::Email, None, [PiiEvidence::Shape]);
    pii.push(PiiMention { span: Span::new(0, 6), kind, detector, evidence: &evidence, score, normalized: "a@b.co" });
    doc.add(&Key::<Pii>::DEFAULT, pii, Provenance::by("test:pii")).unwrap();
    let mut rows = PiiCandidates::new(ViewId::SOURCE);
    let detector = rows.detectors.add("test:det", "1").unwrap();
    let rejected = |reason| PiiDisposition::RejectedByRule { reason };
    for (span, disposition) in [
        (Span::new(0, 6), PiiDisposition::Kept { mention: Ix::new(0) }),
        (Span::new(7, 13), rejected(RejectReason::LabelRequired)),
        (Span::new(7, 13), rejected(RejectReason::Threshold)),
        (Span::new(7, 13), rejected(RejectReason::RulePrecedence)),
    ] {
        rows.push(PiiCandidate { span, kind, detector, evidence: &evidence, score, disposition });
    }
    let mut by = Provenance::by("test:pii");
    by.on.push(Key::<Pii>::DEFAULT.id());
    doc.add(&Key::<PiiCandidates>::DEFAULT, rows, by).unwrap();

    let mut wire = document_to_typed_with("r", &doc, ViewId::SOURCE, OffsetMode::Utf8, &Disclosure::FULL).unwrap();
    validate_typed_document(&wire).unwrap();
    let layer = wire.layers.iter_mut().find(|l| l.id == "turbo:pii-candidates").expect("a candidates layer");
    let served: Vec<_> = layer
        .annotations
        .iter()
        .map(|a| {
            let Some(Value::PiiCandidate(c)) = &a.value else { panic!("a candidate") };
            c.disposition.as_ref().and_then(|d| d.disposition)
        })
        .collect();
    assert_eq!(
        served,
        [
            Some(Disposition::Kept(0)),
            Some(Disposition::RejectedByRule(PiiRejectReason::LabelRequired as i32)),
            Some(Disposition::RejectedByRule(PiiRejectReason::Threshold as i32)),
            Some(Disposition::RejectedByRule(PiiRejectReason::RulePrecedence as i32)),
        ]
    );

    let Some(Value::PiiCandidate(c)) = &mut layer.annotations[3].value else { unreachable!() };
    c.disposition = Some(WireDisposition { disposition: Some(Disposition::RejectedByRule(4)) });
    let err = validate_typed_document(&wire).expect_err("a reject reason past the known ones");
    assert_eq!(err.code, TURBO_E_INVALID_ARGUMENT);
    assert!(err.message.contains("turbo:pii-candidates[3]"), "{}", err.message);
}
