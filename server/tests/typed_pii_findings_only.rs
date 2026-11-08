//! A findings-only typed result (#1904): with the text withheld, every
//! layer but the PII findings is listed bare, the result still validates,
//! and a withheld result that carries another layer's annotations is
//! refused.

use turbo_analysis::{Document, Key, Provenance, Span, Tag, TagSet, View, ViewId};
use turbo_analysis_layers::{Categories, QualityMetrics, TextQuality};
use turbo_kserve::api::TURBO_E_INVALID_ARGUMENT;
use turbo_kserve::offsets::OffsetMode;
use turbo_kserve::proto::result::TypedDocumentResult;
use turbo_kserve::results::disclosure::Disclosure;
use turbo_kserve::results::typed::{document_to_typed_with, validate_typed_document};
use turbo_pii::{Pii, PiiEvidence, PiiKind, PiiMention};

#[test]
fn a_findings_only_result_lists_other_layers_bare_and_still_validates() {
    // A text quality summary (one annotation required) and a view-aligned
    // category layer (one per view) beside the PII layer (#1904).
    let mut doc = Document::new("mail jane@example.com end");
    let mut pii = Pii::new(ViewId::SOURCE);
    let detector = pii.detector("test", "1").unwrap();
    pii.push(PiiMention {
        span: Span::new(5, 21),
        kind: PiiKind::Email,
        detector,
        evidence: &[PiiEvidence::Shape],
        score: None,
        normalized: "jane@example.com",
    });
    doc.add(&Key::<Pii>::DEFAULT, pii, Provenance::by("turbo-pii:pii")).unwrap();
    let quality = TextQuality::new(ViewId::SOURCE, doc.text(ViewId::SOURCE).len() as u32, QualityMetrics::default());
    doc.add(&Key::DEFAULT, quality, Provenance::by("test:quality")).unwrap();
    let mut cats = Categories::<View>::new(TagSet::new("sentiment", &["positive", "negative"]));
    cats.push(Tag(0), &[0.9, 0.1]);
    doc.add(&Key::DEFAULT, cats, Provenance::by("test:doccat")).unwrap();

    let full = document_to_typed_with("pii", &doc, ViewId::SOURCE, OffsetMode::Utf8, &Disclosure::FULL).unwrap();
    let findings =
        document_to_typed_with("pii", &doc, ViewId::SOURCE, OffsetMode::Utf8, &Disclosure::FINDINGS).unwrap();
    validate_typed_document(&findings).unwrap();
    for id in ["turbo:text_quality", "turbo:categories"] {
        let listed =
            |w: &TypedDocumentResult| w.layers.iter().find(|l| l.id.starts_with(id)).map(|l| l.annotations.len());
        assert!(listed(&full).is_some_and(|n| n > 0), "{id}: positive control");
        assert_eq!(listed(&findings), Some(0), "{id} is listed bare");
    }
    // A withheld result that still carries another layer's annotations is
    // refused: those annotations are the withheld text's words.
    let mut leaky = findings.clone();
    let donor = full.layers.iter().find(|l| l.id.starts_with("turbo:text_quality")).unwrap();
    let bare = leaky.layers.iter_mut().find(|l| l.id == donor.id).unwrap();
    bare.annotations = donor.annotations.clone();
    let err = validate_typed_document(&leaky).expect_err("annotations beside a withheld text");
    assert_eq!(err.code, TURBO_E_INVALID_ARGUMENT);
    assert!(err.message.contains("while the text is withheld"), "{}", err.message);
}
