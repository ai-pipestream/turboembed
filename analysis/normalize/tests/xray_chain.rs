use turbo_analysis::{Document, Key, NormalizationXray, Provenance, View, ViewId, XrayRunKind};
use turbo_normalize::{CaseFold, Dehyphenate, Mojibake, Normalizer, UnicodeForm, normalize};

fn step<N: Normalizer>(doc: &mut Document, parent: &Key<View>, name: &'static str, n: N) -> Key<View> {
    let source = doc.get_key(parent).unwrap().clone();
    let (text, edits) = normalize(&n, source.text());
    let key = Key::named(name);
    let mut p = Provenance::by(N::ID);
    p.version = N::VERSION;
    p.config = n.config_hash();
    doc.add(&key, View::derived(&source, text, edits), p).unwrap();
    key
}

#[test]
fn real_normalizer_chain_keeps_text_partitions_and_provenance() {
    let mut doc = Document::new("FranÃ§ais E\u{301}COLE litiga-\ntion Straße 😀");
    let a = step(&mut doc, &Key::DEFAULT, "repair", Mojibake);
    let b = step(&mut doc, &a, "nfc", UnicodeForm::nfc());
    let c = step(&mut doc, &b, "fold", CaseFold::full());
    let d = step(&mut doc, &c, "dehyphenated", Dehyphenate::new());
    let target = doc.get_key(&d).unwrap();
    assert_eq!(target.text(), "français école litigation strasse 😀");
    let x = NormalizationXray::between(&doc, ViewId::SOURCE, target.id()).unwrap();
    assert_eq!(x.steps().len(), 5);
    for (s, key) in x.steps().iter().skip(1).zip([a, b, c, d]) {
        let p = doc.provenance(key.id()).unwrap();
        assert_eq!(s.instance, key.id().name);
        assert_eq!((s.annotator, s.version, s.config_hash), (p.annotator, p.version, p.config));
        assert!(s.edit_count > 0);
    }
    let (mut p, mut c) = (0, 0);
    for (i, r) in x.iter().enumerate() {
        assert_eq!((r.parent.start, r.child.start), (p, c));
        assert!(x.parent_slice(i, doc.source()).is_some());
        assert!(x.child_slice(i, target.text()).is_some());
        if r.kind == XrayRunKind::Equal {
            assert_eq!(x.parent_slice(i, doc.source()), x.child_slice(i, target.text()));
        }
        (p, c) = (r.parent.end, r.child.end);
    }
    assert_eq!((p, c), (doc.source().len() as u32, target.text().len() as u32));
    let projected =
        x.project_utf16(doc.source(), doc.offsets(ViewId::SOURCE), target.text(), target.offsets()).unwrap();
    for (i, r) in x.iter().enumerate() {
        assert_eq!(projected.parent()[i], doc.view_span_to_utf16(ViewId::SOURCE, r.parent).unwrap());
        assert_eq!(projected.child()[i], target.span_to_utf16(r.child));
    }
    let mid = doc.get_key(&b).unwrap();
    let intermediate = NormalizationXray::between(&doc, mid.id(), target.id()).unwrap();
    assert_eq!(
        intermediate.steps().iter().map(|s| s.instance.as_str()).collect::<Vec<_>>(),
        ["nfc", "fold", "dehyphenated"]
    );
}
