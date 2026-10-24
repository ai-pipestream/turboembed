//! Three separately stored views compared with OpenNLP Alignment.andThen.
use turbo_analysis::{Document, Key, Provenance, Span, View, ViewId};
use turbo_goldens::{Golden, Tolerance, data_dir, matches_golden_key};
use turbo_normalize::{CaseFold, GermanUmlauts, Normalizer, Whitespace, normalize};

fn step<N: Normalizer>(doc: &mut Document, parent: &Key<View>, key: &Key<View>, n: N) {
    let parent = doc.get_key(parent).unwrap().clone();
    let (text, edits) = normalize(&n, parent.text());
    doc.add(key, View::derived(&parent, text, edits), Provenance::by(N::ID)).unwrap();
}

#[test]
fn three_views_match_jvm_text_edits_and_source_offsets() {
    let directory = data_dir().join("normalize/umlaut-full-case-whitespace-chain/norm");
    let mut paths: Vec<_> = std::fs::read_dir(directory).unwrap().map(|e| e.unwrap().path()).collect();
    paths.sort();
    let (a, b, c) = (Key::named("umlaut"), Key::named("full-case"), Key::named("whitespace"));
    let mut count = 0;
    let mut edits = 0;
    for path in paths {
        let golden = Golden::read(&path).unwrap();
        assert_eq!(golden.header.annotator.method, "andThen");
        let source = golden.input_text(data_dir()).unwrap();
        let mut doc = Document::new(source.as_str());
        step(&mut doc, &Key::<View>::DEFAULT, &a, GermanUmlauts::new());
        step(&mut doc, &a, &b, CaseFold::full());
        step(&mut doc, &b, &c, Whitespace::collapse());
        let view = doc.get_key(&c).unwrap();
        assert_eq!(view.parent(), Some(doc.get_key(&b).unwrap().id()));
        matches_golden_key(&doc, &c, &golden, Tolerance::exact()).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
        let composed = doc.view_alignment(ViewId::SOURCE, view.id()).unwrap();
        for edit in composed.edits() {
            assert!(source.is_char_boundary(edit.parent.start as usize));
            assert!(source.is_char_boundary(edit.parent.end as usize));
            assert!(view.text().is_char_boundary(edit.child.start as usize));
            assert!(view.text().is_char_boundary(edit.child.end as usize));
            if !edit.child.is_empty() {
                let back = doc.map_span(view.id(), ViewId::SOURCE, edit.child).unwrap();
                assert_eq!(back, edit.parent, "{}: source projection", path.display());
            }
            if !edit.parent.is_empty() {
                let forward = doc.map_span(ViewId::SOURCE, view.id(), edit.parent).unwrap();
                assert_eq!(forward, edit.child, "{}: child projection", path.display());
            }
        }
        edits += composed.edits().len();
        // A later identity view must not discard changes made by ancestors.
        let again = Key::named("again");
        step(&mut doc, &c, &again, Whitespace::collapse());
        matches_golden_key(&doc, &again, &golden, Tolerance::exact()).unwrap();
        count += 1;
    }
    assert_eq!(count, 61);
    assert!(edits > 100);
}

#[test]
fn expansion_changed_again_and_then_trimmed_keeps_source_bytes() {
    let text = "  MÜLLER  Straße 😀  ";
    let mut doc = Document::new(text);
    let (a, b, c) = (Key::named("umlaut"), Key::named("full-case"), Key::named("whitespace"));
    step(&mut doc, &Key::<View>::DEFAULT, &a, GermanUmlauts::new());
    step(&mut doc, &a, &b, CaseFold::full());
    step(&mut doc, &b, &c, Whitespace::collapse());
    let v = doc.get_key(&c).unwrap();
    assert_eq!(v.text(), "mueller strasse 😀");
    // The first pass expands Ü to Ue, then the second changes U to u.
    let umlaut = Span::new(1, 3);
    let back = doc.map_span(v.id(), ViewId::SOURCE, umlaut).unwrap();
    assert_eq!(&text[back.range()], "Ü");
    assert_eq!(doc.map_span(ViewId::SOURCE, v.id(), back).unwrap(), umlaut);
    let emoji = Span::new(16, 20);
    let back = doc.map_span(v.id(), ViewId::SOURCE, emoji).unwrap();
    assert_eq!(&text[back.range()], "😀");
    assert_eq!(doc.map_span(ViewId::SOURCE, v.id(), back).unwrap(), emoji);
}
