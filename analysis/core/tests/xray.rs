use turbo_analysis::{
    Document, Edit, Key, NormalizationXray, OffsetTable, Provenance, Span, View, ViewBuilder, ViewId, XrayRunKind,
};

fn add(doc: &mut Document, parent: ViewId, name: &'static str, change: impl FnOnce(&mut ViewBuilder<'_>)) -> ViewId {
    let key = Key::<View>::named(name);
    let source = doc.get_key(&Key::<View>::named(if parent == ViewId::SOURCE { "" } else { "first" })).unwrap().clone();
    let mut b = ViewBuilder::new(source.text());
    change(&mut b);
    let (text, alignment) = b.finish();
    let mut p = Provenance::by("test:edit");
    p.version = "3";
    p.config = 42;
    doc.add(&key, View::derived(&source, text, alignment), p).unwrap();
    doc.get_key(&key).unwrap().id()
}

fn partition(doc: &Document, a: ViewId, b: ViewId) {
    let x = NormalizationXray::between(doc, a, b).unwrap();
    let (mut p, mut c) = (0, 0);
    for (i, r) in x.iter().enumerate() {
        assert_eq!((r.parent.start, r.child.start), (p, c));
        if r.kind == XrayRunKind::Equal {
            assert_eq!(x.parent_slice(i, doc.text(a)), x.child_slice(i, doc.text(b)));
        }
        (p, c) = (r.parent.end, r.child.end);
    }
    assert_eq!((p, c), (doc.text(a).len() as u32, doc.text(b).len() as u32));
}

#[test]
fn partitions_replacements_insertions_deletions_and_identity() {
    let mut d = Document::new("Aé中😀Z");
    let v = add(&mut d, ViewId::SOURCE, "first", |b| {
        b.replace(0, "!");
        b.keep(1);
        b.replace(3, "e");
        b.keep(6);
        b.replace(10, "");
    });
    let x = NormalizationXray::between(&d, ViewId::SOURCE, v).unwrap();
    assert_eq!(
        x.iter().map(|r| r.kind).collect::<Vec<_>>(),
        [
            XrayRunKind::Inserted,
            XrayRunKind::Equal,
            XrayRunKind::Replaced,
            XrayRunKind::Equal,
            XrayRunKind::Deleted,
            XrayRunKind::Equal
        ]
    );
    partition(&d, ViewId::SOURCE, v);
    partition(&d, v, ViewId::SOURCE);
    partition(&d, v, v);
    assert_eq!(x.steps().last().unwrap().instance, "first");
    assert_eq!(x.steps().last().unwrap().config_hash, 42);
    assert!(x.run(usize::MAX).is_none());
}

#[test]
fn typed_projection_matches_existing_offset_tables() {
    let mut d = Document::new("é中😀a");
    let v = add(&mut d, ViewId::SOURCE, "first", |b| {
        b.keep(2);
        b.replace(5, "XY");
    });
    let x = NormalizationXray::between(&d, ViewId::SOURCE, v).unwrap();
    let (p, c) = (d.source(), d.text(v));
    let (pt, ct) = (OffsetTable::build(p), OffsetTable::build(c));
    let u = x.project_utf16(p, &pt, c, &ct).unwrap();
    let cp = x.project_code_points(p, &pt, c, &ct).unwrap();
    for (i, r) in x.iter().enumerate() {
        assert_eq!(u.parent()[i], pt.span_to_utf16_typed(p, r.parent));
        assert_eq!(u.child()[i], ct.span_to_utf16_typed(c, r.child));
        assert_eq!(cp.parent()[i], pt.span_to_code_point_typed(p, r.parent));
        assert_eq!(cp.child()[i], ct.span_to_code_point_typed(c, r.child));
    }
    assert!(x.project_utf16("", &pt, c, &ct).is_err());
    assert!(x.project_code_points(p, &OffsetTable::build(""), c, &ct).is_err());
}

#[test]
fn empty_text_unknown_views_and_sibling_paths() {
    let empty = Document::new("");
    assert!(NormalizationXray::between(&empty, ViewId::SOURCE, ViewId::SOURCE).unwrap().is_empty());
    assert!(NormalizationXray::between(&empty, ViewId(99), ViewId::SOURCE).is_err());
    let mut d = Document::new("abc");
    let a = add(&mut d, ViewId::SOURCE, "first", |b| b.replace(1, "A"));
    let b = add(&mut d, ViewId::SOURCE, "second", |b| b.replace(3, ""));
    partition(&d, a, b);
    partition(&d, b, a);
    let x = NormalizationXray::between(&d, a, b).unwrap();
    assert_eq!(x.steps().iter().map(|s| s.view).collect::<Vec<_>>(), [a, ViewId::SOURCE, b]);
}

#[test]
fn original_bytes_include_decoder_edits() {
    let bytes = vec![0xe9, b'!'];
    let alignment =
        turbo_analysis::Alignment::from_edits(vec![Edit { parent: Span::new(0, 1), child: Span::new(0, 2) }]).unwrap();
    let d = Document::decoded(bytes.clone(), "é!", alignment).unwrap();
    let x = NormalizationXray::from_source_bytes(&d, ViewId::SOURCE).unwrap();
    assert_eq!(x.len(), 2);
    assert_eq!(x.run(0).unwrap().kind, XrayRunKind::Replaced);
    assert_eq!(x.parent_bytes_slice(0, &bytes), Some(&bytes[..1]));
    assert_eq!(x.steps()[0].parent_bytes, 2);
    assert_eq!(x.steps()[0].child_bytes, 3);
}

#[test]
fn decoder_byte_boundaries_are_not_mistaken_for_utf8_boundaries() {
    let edits = vec![
        Edit { parent: Span::new(0, 1), child: Span::new(0, 2) },
        Edit { parent: Span::new(1, 2), child: Span::new(2, 4) },
    ];
    let d = Document::decoded(vec![0xc3, 0xa9], "Ã©", turbo_analysis::Alignment::from_edits(edits).unwrap()).unwrap();
    let x = NormalizationXray::from_source_bytes(&d, ViewId::SOURCE).unwrap();
    assert!(x.project_utf16("é", &OffsetTable::build("é"), d.source(), d.offsets(ViewId::SOURCE)).is_err());
}

#[test]
fn mixed_width_edit_sequences_partition_both_directions() {
    let source: String = ['a', 'é', '中', '😀'].into_iter().cycle().take(80).collect();
    for seed in 0..64 {
        let mut d = Document::new(source.as_str());
        let parent = d.get::<View>().unwrap().clone();
        let mut b = ViewBuilder::new(&source);
        for (i, (at, c)) in source.char_indices().enumerate() {
            match (i + seed) % 5 {
                0 => {
                    b.keep(at);
                    b.replace(at, "!");
                    b.keep(at + c.len_utf8());
                }
                1 => b.replace(at + c.len_utf8(), ""),
                2 => b.replace(at + c.len_utf8(), "ß😀"),
                _ => b.keep(at + c.len_utf8()),
            }
        }
        let (text, edits) = b.finish();
        let key = Key::named("mixed");
        d.add(&key, View::derived(&parent, text, edits), Provenance::by("test:edit")).unwrap();
        let v = d.get_key(&key).unwrap().id();
        partition(&d, ViewId::SOURCE, v);
        partition(&d, v, ViewId::SOURCE);
        for (a, z) in [(ViewId::SOURCE, v), (v, ViewId::SOURCE)] {
            let x = NormalizationXray::between(&d, a, z).unwrap();
            let projected = x.project_code_points(d.text(a), d.offsets(a), d.text(z), d.offsets(z)).unwrap();
            for (i, r) in x.iter().enumerate() {
                assert_eq!(projected.parent()[i], d.view_span_to_code_point(a, r.parent).unwrap());
                assert_eq!(projected.child()[i], d.view_span_to_code_point(z, r.child).unwrap());
            }
        }
    }
}
