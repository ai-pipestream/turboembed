use turbo_analysis::{Analyzer, Annotator, Document, NormalizationXray, ViewId, XrayRunKind};
use turbo_pii::{
    PiiDetector,
    rewrite::{MASKED, MaskedView},
};

#[test]
fn actual_detection_and_masking_keep_inspectable_source_ranges() {
    let masker = MaskedView::by_kind('*').unwrap();
    let hash = masker.config_hash();
    let analyzer = Analyzer::builder().add(PiiDetector::cursor()).step(masker.bound()).build().unwrap();
    let mut docs = [Document::new("é Contact jane@example.com 😀")];
    for r in analyzer.run(&mut docs) {
        r.unwrap();
    }
    let doc = &docs[0];
    let masked = doc.get_key(&MASKED).unwrap();
    assert_eq!(masked.text(), "é Contact ****@*******.*** 😀");
    let x = NormalizationXray::between(doc, ViewId::SOURCE, masked.id()).unwrap();
    let changed: Vec<_> = x.iter().enumerate().filter(|(_, r)| r.kind == XrayRunKind::Replaced).collect();
    assert!(!changed.is_empty());
    let p = x.steps().last().unwrap();
    assert_eq!((p.annotator, p.config_hash), (MaskedView::ID, hash));
    let mut covered = (0, 0);
    for (i, r) in x.iter().enumerate() {
        assert_eq!((r.parent.start, r.child.start), covered);
        if r.kind == XrayRunKind::Equal {
            assert_eq!(x.parent_slice(i, doc.source()), x.child_slice(i, masked.text()));
        }
        covered = (r.parent.end, r.child.end);
    }
    assert_eq!(covered, (doc.source().len() as u32, masked.text().len() as u32));
}
