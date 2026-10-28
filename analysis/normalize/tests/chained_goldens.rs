//! Three separately stored views: umlaut → full case fold → whitespace.
//!
//! Snapshots under `tests/snapshots/umlaut-full-case-whitespace-chain/` hold
//! the final view text and the composed source→final edit list
//! (`start\tend\twith`, escaped). Offset round trips and the hand case
//! below are independent of those files. Regenerate with
//! `TURBO_BLESS=1 cargo test -p turbo-normalize --test chained_goldens`.

use std::path::PathBuf;

use turbo_analysis::{Document, Key, Provenance, Span, View, ViewId};
use turbo_goldens::escape::escape;
use turbo_normalize::{CaseFold, GermanUmlauts, Normalizer, Whitespace, normalize};

fn root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests")
}

fn step<N: Normalizer>(doc: &mut Document, parent: &Key<View>, key: &Key<View>, n: N) {
    let parent = doc.get_key(parent).unwrap().clone();
    let (text, edits) = normalize(&n, parent.text());
    doc.add(key, View::derived(&parent, text, edits), Provenance::by(N::ID)).unwrap();
}

#[test]
fn three_views_match_snapshots_and_source_offsets() {
    let bless = std::env::var_os("TURBO_BLESS").is_some();
    let mut ids: Vec<String> = std::fs::read_dir(root().join("inputs"))
        .unwrap()
        .map(|e| e.unwrap().file_name().into_string().unwrap())
        .filter_map(|n| n.strip_suffix(".txt").map(str::to_string))
        .collect();
    ids.sort();
    let (a, b, c) = (Key::named("umlaut"), Key::named("full-case"), Key::named("whitespace"));
    let mut count = 0;
    let mut edits = 0;
    for id in &ids {
        let source = std::fs::read_to_string(root().join("inputs").join(format!("{id}.txt"))).unwrap();
        let mut doc = Document::new(source.as_str());
        step(&mut doc, &Key::<View>::DEFAULT, &a, GermanUmlauts::new());
        step(&mut doc, &a, &b, CaseFold::full());
        step(&mut doc, &b, &c, Whitespace::collapse());
        let final_text = doc.get_key(&c).unwrap().text().to_string();
        let view_id = doc.get_key(&c).unwrap().id();
        assert_eq!(doc.get_key(&c).unwrap().parent(), Some(doc.get_key(&b).unwrap().id()));
        let composed = doc.view_alignment(ViewId::SOURCE, view_id).unwrap();
        let mut body = String::from("===OUT===\n");
        body.push_str(&final_text);
        body.push_str("\n===EDITS===\n");
        for edit in composed.edits() {
            let with = &final_text[edit.child.range()];
            body.push_str(&format!("{}\t{}\t{}\n", edit.parent.start, edit.parent.end, escape(with)));
        }
        let path = root().join("snapshots/umlaut-full-case-whitespace-chain").join(format!("{id}.txt"));
        if bless {
            std::fs::create_dir_all(path.parent().unwrap()).unwrap();
            std::fs::write(&path, &body).unwrap();
        } else {
            let want = std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
            assert_eq!(body, want, "{id}");
        }
        for edit in composed.edits() {
            assert!(source.is_char_boundary(edit.parent.start as usize));
            assert!(source.is_char_boundary(edit.parent.end as usize));
            assert!(final_text.is_char_boundary(edit.child.start as usize));
            assert!(final_text.is_char_boundary(edit.child.end as usize));
            if !edit.child.is_empty() {
                let back = doc.map_span(view_id, ViewId::SOURCE, edit.child).unwrap();
                assert_eq!(back, edit.parent, "{id}: source projection");
            }
            if !edit.parent.is_empty() {
                let forward = doc.map_span(ViewId::SOURCE, view_id, edit.parent).unwrap();
                assert_eq!(forward, edit.child, "{id}: child projection");
            }
        }
        edits += composed.edits().len();
        let again = Key::named("again");
        step(&mut doc, &c, &again, Whitespace::collapse());
        assert_eq!(doc.get_key(&again).unwrap().text(), final_text, "{id}: identity after collapse");
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
    let umlaut = Span::new(1, 3);
    let back = doc.map_span(v.id(), ViewId::SOURCE, umlaut).unwrap();
    assert_eq!(&text[back.range()], "Ü");
    assert_eq!(doc.map_span(ViewId::SOURCE, v.id(), back).unwrap(), umlaut);
    let emoji = Span::new(16, 20);
    let back = doc.map_span(v.id(), ViewId::SOURCE, emoji).unwrap();
    assert_eq!(&text[back.range()], "😀");
    assert_eq!(doc.map_span(ViewId::SOURCE, v.id(), back).unwrap(), emoji);
}
