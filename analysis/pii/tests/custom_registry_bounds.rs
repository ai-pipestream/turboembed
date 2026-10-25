//! Isolated-process test for the process-global custom `PiiKind` registry
//! (`PiiKind::MAX_CUSTOM_KINDS` = 256, `PiiKind::MAX_CUSTOM_NAME_BYTES` = 64).

use turbo_analysis::export::{DocumentGraph, export, import};
use turbo_analysis::{Document, Key, Layer, Provenance, Span, ViewId};
use turbo_analysis_layers::export::codecs;
use turbo_pii::{Pii, PiiConfidence, PiiDetector, PiiKind, PiiMention, PiiSink, Scanner, SinkError};

const KNOWN_KIND: PiiKind = PiiKind::custom("custom-kind-001");
const OVERFLOW_KIND: PiiKind = PiiKind::custom("kind-overflow-257");

fn check_overflow_sink(_text: &str, sink: &mut PiiSink<'_>) {
    assert_eq!(
        sink.try_add(Span::new(0, 7), OVERFLOW_KIND, PiiConfidence::Format, "BADGE1"),
        Err(SinkError::RegistryFull("kind-overflow-257"))
    );
    assert_eq!(sink.try_add(Span::new(0, 7), KNOWN_KIND, PiiConfidence::Format, "BADGE1"), Ok(()));
}

#[test]
fn custom_kind_registry_enforces_capacity_and_multibyte_utf8_boundaries() {
    // 1. Multibyte UTF-8 64-byte boundary (32 copies of U+00E9 = 64 UTF-8 bytes)
    //    succeeds, whereas 65 bytes is refused.
    let multibyte_64: &'static str = Box::leak("é".repeat(32).into_boxed_str());
    let multibyte_65: &'static str = Box::leak(format!("{multibyte_64}x").into_boxed_str());
    assert_eq!(multibyte_64.len(), PiiKind::MAX_CUSTOM_NAME_BYTES);
    assert_eq!(multibyte_65.len(), PiiKind::MAX_CUSTOM_NAME_BYTES + 1);

    let mut mb_doc = Document::new("BADGE-1");
    let mut mb_pii = Pii::new(ViewId::SOURCE);
    mb_pii.push(PiiMention {
        span: Span::new(0, 7),
        kind: PiiKind::custom(multibyte_64),
        confidence: PiiConfidence::Format,
        normalized: "BADGE1",
    });
    assert_eq!(mb_pii.check(), Ok(()));
    mb_doc.add(&Key::DEFAULT, mb_pii, Provenance::by("test")).unwrap();
    let mb_bytes = export(&mb_doc, &codecs()).unwrap().to_bytes().unwrap();
    let mb_back = import(&DocumentGraph::from_bytes(&mb_bytes).unwrap(), &codecs()).unwrap();
    assert_eq!(mb_back.get::<Pii>().unwrap().kind, [PiiKind::Custom(multibyte_64)]);

    let mut mb_bad = Pii::new(ViewId::SOURCE);
    mb_bad.push(PiiMention {
        span: Span::new(0, 7),
        kind: PiiKind::custom(multibyte_65),
        confidence: PiiConfidence::Format,
        normalized: "BADGE1",
    });
    assert!(mb_bad.check().unwrap_err().contains("65 bytes (max 64)"));

    // 2. Fill the remaining slots up to PiiKind::MAX_CUSTOM_KINDS (256).
    for i in 1..PiiKind::MAX_CUSTOM_KINDS {
        let leaked: &'static str = Box::leak(format!("custom-kind-{i:03}").into_boxed_str());
        assert!(PiiKind::remember_custom(PiiKind::custom(leaked)), "slot {i} should succeed");
    }

    // 3. At capacity (256 entries), an already-registered custom kind still succeeds
    //    at remember_custom, PiiSink, Pii::check, export, and import.
    assert!(PiiKind::remember_custom(KNOWN_KIND));
    let mut known_doc = Document::new("BADGE-1");
    let mut known_pii = Pii::new(ViewId::SOURCE);
    known_pii.push(PiiMention {
        span: Span::new(0, 7),
        kind: KNOWN_KIND,
        confidence: PiiConfidence::Format,
        normalized: "BADGE1",
    });
    assert_eq!(known_pii.check(), Ok(()));
    known_doc.add(&Key::DEFAULT, known_pii, Provenance::by("test")).unwrap();
    let known_bytes = export(&known_doc, &codecs()).unwrap().to_bytes().unwrap();
    let known_back = import(&DocumentGraph::from_bytes(&known_bytes).unwrap(), &codecs()).unwrap();
    assert_eq!(known_back.get::<Pii>().unwrap().kind, [KNOWN_KIND]);

    // 4. A 257th distinct custom kind is refused by remember_custom, intern_custom,
    //    PiiSink, Pii::check, and Document::add.
    assert!(!PiiKind::remember_custom(OVERFLOW_KIND));
    assert_eq!(PiiKind::intern_custom("kind-overflow-257"), None);

    let det = PiiDetector::new(vec![Scanner::custom_fn("OverflowCheck", KNOWN_KIND, check_overflow_sink)]);
    let resolved = det.layer("BADGE-1", ViewId::SOURCE);
    assert_eq!(resolved.kind, [KNOWN_KIND]);

    let mut overflow_pii = Pii::new(ViewId::SOURCE);
    overflow_pii.push(PiiMention {
        span: Span::new(0, 7),
        kind: OVERFLOW_KIND,
        confidence: PiiConfidence::Format,
        normalized: "BADGE1",
    });
    assert!(overflow_pii.check().unwrap_err().contains("exceeds custom kind registry capacity (256)"));

    let mut overflow_doc = Document::new("BADGE-1");
    let add_err = overflow_doc.add(&Key::DEFAULT, overflow_pii, Provenance::by("test")).unwrap_err();
    assert!(add_err.to_string().contains("exceeds custom kind registry capacity (256)"), "{add_err}");
}
