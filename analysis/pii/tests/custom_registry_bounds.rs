//! Custom-kind tables (#1901): bounds checked when a table is built
//! (`CustomKinds::MAX` = 256 names of at most `CustomKinds::MAX_NAME_BYTES` =
//! 64 bytes), no process-wide state, and refusal of unknown kinds by the
//! sink, the detector builder, the rewrites and binary import.

use turbo_analysis::export::{DocumentGraph, export, import};
use turbo_analysis::{Document, Key, Layer, Provenance, Span, ViewId};
use turbo_analysis_layers::export::codecs;
use turbo_pii::rewrite::{RewriteError, StaticReplacer};
use turbo_pii::{
    CustomKindId, CustomKinds, KindError, Pii, PiiConfidence, PiiDetector, PiiKind, PiiMention, PiiSink, Scanner,
    SinkError,
};

const FIRST: PiiKind = PiiKind::Custom(CustomKindId::new(0));
const STRAY: PiiKind = PiiKind::Custom(CustomKindId::new(9));

fn badge(_text: &str, sink: &mut PiiSink<'_>) {
    assert_eq!(
        sink.try_add(Span::new(0, 7), STRAY, PiiConfidence::Format, "BADGE1"),
        Err(SinkError::Kind(KindError::UnknownCustom(9)))
    );
    assert_eq!(sink.try_add(Span::new(0, 7), FIRST, PiiConfidence::Format, "BADGE1"), Ok(()));
}

fn one(kinds: CustomKinds, kind: PiiKind) -> Pii {
    let mut pii = Pii::with_custom_kinds(ViewId::SOURCE, kinds);
    pii.push(PiiMention { span: Span::new(0, 7), kind, confidence: PiiConfidence::Format, normalized: "BADGE1" });
    pii
}

fn round_trip(pii: Pii) -> Document {
    let mut doc = Document::new("BADGE-1");
    doc.add(&Key::DEFAULT, pii, Provenance::by("test")).unwrap();
    let bytes = export(&doc, &codecs()).unwrap().to_bytes().unwrap();
    import(&DocumentGraph::from_bytes(&bytes).unwrap(), &codecs()).unwrap()
}

#[test]
fn table_bounds_are_checked_when_the_table_is_built() {
    let multibyte_64 = "é".repeat(32);
    let multibyte_65 = format!("{multibyte_64}x");
    assert_eq!(multibyte_64.len(), CustomKinds::MAX_NAME_BYTES);
    let kinds = CustomKinds::from_names([multibyte_64.as_str()]).unwrap();
    let doc = round_trip(one(kinds.clone(), FIRST));
    let back = doc.get::<Pii>().unwrap();
    assert_eq!(back.kind, [FIRST]);
    assert_eq!(back.kind_name(FIRST), Some(multibyte_64.as_str()));
    assert_eq!(CustomKinds::from_names([multibyte_65.as_str()]), Err(KindError::NameTooLong(65)));
    assert_eq!(CustomKinds::from_names(["phone"]), Err(KindError::Reserved(PiiKind::Phone)));

    let names: Vec<String> = (0..CustomKinds::MAX).map(|i| format!("custom-kind-{i:03}")).collect();
    let mut full = CustomKinds::from_names(&names).unwrap();
    assert_eq!(full.add("kind-overflow-257"), Err(KindError::Full));
    let last = full.find("custom-kind-255").unwrap();
    let doc = round_trip(one(full.clone(), last));
    let back = doc.get::<Pii>().unwrap();
    assert_eq!(back.kind, [last]);
    assert_eq!(back.custom_kinds, full);
}

#[test]
fn documents_with_different_custom_kinds_share_no_state() {
    // Filling one table leaves no trace on another built afterwards.
    let names: Vec<String> = (0..CustomKinds::MAX).map(|i| format!("k{i}")).collect();
    let _full = CustomKinds::from_names(&names).unwrap();
    let a_kinds = CustomKinds::from_names(["badge"]).unwrap();
    let b_kinds = CustomKinds::from_names(["ticket", "badge"]).unwrap();
    let scan = |name: &'static str| Scanner::custom_fn(name, FIRST, badge);
    let a = PiiDetector::new(vec![]).with_custom_kinds(a_kinds.clone()).try_with_scanner(scan("A")).unwrap();
    let b = PiiDetector::new(vec![]).with_custom_kinds(b_kinds.clone()).try_with_scanner(scan("B")).unwrap();
    let (pa, pb) = (a.layer("BADGE-1", ViewId::SOURCE), b.layer("BADGE-1", ViewId::SOURCE));
    assert_eq!((pa.kind.as_slice(), pb.kind.as_slice()), ([FIRST].as_slice(), [FIRST].as_slice()));
    assert_eq!(pa.kind_name(FIRST), Some("badge"));
    assert_eq!(pb.kind_name(FIRST), Some("ticket"));
    let (ra, rb) = (round_trip(pa), round_trip(pb));
    assert_eq!(ra.get::<Pii>().unwrap().custom_kinds, a_kinds);
    assert_eq!(rb.get::<Pii>().unwrap().custom_kinds, b_kinds);
}

#[test]
fn unknown_kinds_are_refused_where_they_enter() {
    // The builder refuses an extractor declaring a kind outside the table.
    let declared = PiiDetector::new(vec![]).with_custom_fn("Stray", STRAY, badge);
    assert_eq!(declared.err(), Some(KindError::UnknownCustom(9)));
    // Layer check and export refuse a custom kind outside the layer's table.
    let stray = one(CustomKinds::new(), FIRST);
    let err = stray.check().unwrap_err();
    assert!(err.contains("not in the kind table") && !err.contains("BADGE1"), "{err}");
    let mut doc = Document::new("BADGE-1");
    assert!(doc.add(&Key::DEFAULT, stray, Provenance::by("test")).is_err());
    // Every redaction path refuses it too.
    let stray = one(CustomKinds::new(), FIRST);
    assert_eq!(
        StaticReplacer::new("[X]").rewrite("BADGE-1", &stray).err(),
        Some(RewriteError::Kind(Span::new(0, 7), KindError::UnknownCustom(0)))
    );
}

/// The exported body of a one-mention layer of `kind` over "BADGE-1".
fn graph_of(kind: PiiKind, confidence: PiiConfidence) -> DocumentGraph {
    let mut pii = Pii::new(ViewId::SOURCE);
    pii.push(PiiMention { span: Span::new(0, 7), kind, confidence, normalized: "BADGE1" });
    let mut doc = Document::new("BADGE-1");
    doc.add(&Key::DEFAULT, pii, Provenance::by("test")).unwrap();
    export(&doc, &codecs()).unwrap()
}

/// Replaces the one occurrence of `from` in the pii layer's body by `to`.
fn patch(graph: &mut DocumentGraph, from: &[u8], to: &[u8]) {
    let body = graph.layers.iter_mut().find(|l| l.layer == Pii::ID).unwrap().body.as_mut().unwrap();
    let at: Vec<usize> = (0..body.len() - from.len() + 1).filter(|&i| body[i..].starts_with(from)).collect();
    assert_eq!(at.len(), 1, "one place to patch");
    body[at[0]..at[0] + to.len()].copy_from_slice(to);
}

#[test]
fn unknown_kind_codes_and_confidence_ids_are_refused_on_import() {
    // No custom names, one kind code (30, phone), one confidence (2, checksum).
    let tail = |code: u32, conf: u8| {
        let mut b = vec![0, 0, 0, 0, 1, 0, 0, 0];
        b.extend_from_slice(&code.to_le_bytes());
        b.extend_from_slice(&[1, 0, 0, 0, conf]);
        b
    };
    let good = graph_of(PiiKind::Phone, PiiConfidence::Checksum);
    assert_eq!(import(&good, &codecs()).unwrap().get::<Pii>().unwrap().kind, [PiiKind::Phone]);
    for (code, conf, want) in [
        (0, 2, "unknown pii kind code 0"),
        (31, 2, "unknown pii kind code 31"),
        (65_536, 2, "unknown pii kind code 65536"),
        (30, 0, "unknown pii confidence id 0"),
        (30, 5, "unknown pii confidence id 5"),
    ] {
        let mut bad = good.clone();
        patch(&mut bad, &tail(30, 2), &tail(code, conf));
        let err = import(&bad, &codecs()).err().expect("refused").to_string();
        assert!(err.contains(want) && !err.contains("BADGE1"), "{err}");
    }
}
