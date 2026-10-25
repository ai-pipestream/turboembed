use super::comparison::has;
use super::*;

// ---------------------------------------------------------------------
// Options and normalized views: `tests/fixtures/ligature.golden`, a
// hand written `turbo:view` golden over `inputs/test/ligature.txt`. Its
// edits expand a ligature, delete the second of two spaces and fold two
// full-width letters, one row each. No two rows touch, so the tests do
// not depend on how a `ViewBuilder` records touching replacements.

const LIGATURE: &str = "The \u{FB01}nal  report \u{FF21} \u{FF22}.";

fn view_golden() -> Golden {
    Golden::read(fixtures().join("ligature.golden")).unwrap()
}

#[test]
fn config_records_read_unescaped_and_write_back() {
    let g = view_golden();
    assert_eq!(
        g.header.config,
        [
            ConfigEntry { key: "form".into(), value: "NFKC".into() },
            ConfigEntry { key: "separator".into(), value: "a\tb\u{1}".into() },
        ]
    );
    assert!(g.header.models.is_empty());
    assert_eq!(g.input_text(fixtures()).unwrap(), LIGATURE);
    let text = std::fs::read_to_string(fixtures().join("ligature.golden")).unwrap();
    assert_eq!(g.to_text().unwrap(), text);
    let v = g.layer("turbo:view").unwrap();
    assert_eq!(v.anchor, GoldenAnchor::Text);
    assert_eq!(v.strs("with").unwrap().iter().collect::<Vec<_>>(), ["fi", "", "A", "B"]);

    // Keys are sorted, each once, on both sides.
    let unsorted = text.replace("config\tform\tNFKC\n", "config\ttail\tx\n");
    let e = refused(&unsorted, 9);
    assert!(matches!(e, GoldenError::ConfigOrder { .. }), "{e:?}");
    assert!(e.to_string().contains("the config key \"separator\" comes after \"tail\""), "{e}");
    let twice = text.replace("config\tseparator", "config\tform");
    assert!(matches!(refused(&twice, 9), GoldenError::ConfigOrder { .. }));
    let bad = text.replace("a\\tb", "a\\qb");
    assert!(matches!(refused(&bad, 9), GoldenError::Escape { .. }));
    let mut g = view_golden();
    g.header.config.reverse();
    assert_eq!(g.to_text().unwrap_err().kind(), std::io::ErrorKind::InvalidInput);
}

/// A view made from the source by `edits` (parent end, replacement),
/// kept between them, under the name "norm".
fn normalized(edits: &[(usize, usize, &str)]) -> Document {
    let mut doc = Document::new(LIGATURE);
    let mut b = ViewBuilder::new(LIGATURE);
    for &(start, end, with) in edits {
        b.keep(start);
        b.replace(end, with);
    }
    let (text, alignment) = b.finish();
    let view = View::derived(doc.get::<View>().unwrap(), text, alignment);
    doc.add(&Key::named("norm"), view, Provenance::by("test")).unwrap();
    doc
}

const NORM: Key<View> = Key::named("norm");

#[test]
fn a_normalized_view_matches_its_golden() {
    let g = view_golden();
    let doc = normalized(&[(4, 7, "fi"), (11, 12, ""), (19, 22, "A"), (23, 26, "B")]);
    assert_eq!(doc.get_key(&NORM).unwrap().alignment().edits().len(), 4);
    assert_eq!(doc.get_key(&NORM).unwrap().text(), "The final report A B.");
    assert_matches_golden_key(&doc, &NORM, &g, Tolerance::exact());
}

#[test]
fn a_view_with_other_text_or_other_edits() {
    let g = view_golden();
    // The spaces left alone: the text differs from byte 10 of the view.
    let doc = normalized(&[(4, 7, "fi"), (19, 22, "A"), (23, 26, "B")]);
    let m = matches_golden_key(&doc, &NORM, &g, Tolerance::exact()).unwrap_err();
    assert!(matches!(m, Mismatch::Text { at: 10, .. }), "{m:?}");
    has(&m.to_string(), &["turbo:view@norm: the normalized text differs from byte 10", "\"The final  report A B.\""]);

    // The same text, but both spaces replaced by one instead of the second
    // deleted: the edit lists differ at row 1, which is Rust edit 1.
    let doc = normalized(&[(4, 7, "fi"), (10, 12, " "), (19, 22, "A"), (23, 26, "B")]);
    let m = matches_golden_key(&doc, &NORM, &g, Tolerance::exact()).unwrap_err();
    assert!(matches!(m, Mismatch::Item { index: 1, what: "span", .. }), "{m:?}");
    has(
        &m.to_string(),
        &["turbo:view@norm item 1", "rust:   bytes 10..12 \"  \", with \" \"", "golden: bytes 11..12 \" \", with \"\""],
    );

    // A chained view is compared, and reports its actual text difference.
    let mut doc = normalized(&[(4, 7, "fi")]);
    let norm = doc.get_key(&NORM).unwrap().clone();
    let (text, alignment) = ViewBuilder::new(norm.text()).finish();
    doc.add(&Key::named("again"), View::derived(&norm, text, alignment), Provenance::by("test")).unwrap();
    let m = matches_golden_key(&doc, &Key::<View>::named("again"), &g, Tolerance::exact()).unwrap_err();
    assert!(matches!(m, Mismatch::Text { .. }), "{m:?}");
}

#[test]
fn a_view_golden_that_cannot_be_applied() {
    let mut g = view_golden();
    let v = g.layers.iter_mut().find(|l| l.id == "turbo:view").unwrap();
    v.columns[1].data = ColumnData::U32(vec![7, 12, 21, 26]);
    let doc = normalized(&[(4, 7, "fi")]);
    let m = matches_golden_key(&doc, &NORM, &g, Tolerance::exact()).unwrap_err();
    assert!(matches!(m, Mismatch::BadGolden { .. }), "{m:?}");
    has(&m.to_string(), &["row 2, the edit 19..21, cuts a character"]);
}

/// Golden rows and alignment edits are compared one for one: two
/// touching golden rows (one per character, as the JVM writes them) do
/// not match one edit that replaces both characters in one call, and the
/// first row is where they differ.
#[test]
fn touching_rows_do_not_match_one_edit_over_both() {
    let text = "\u{FF21}\u{FF22}";
    let mut g = view_golden();
    g.header.input.utf8_len = text.len() as u32;
    g.header.input.utf16_len = 2;
    g.header.input.sha256 = sha256(text.as_bytes());
    let v = g.layers.iter_mut().find(|l| l.id == "turbo:view").unwrap();
    let mut with = Strs::default();
    with.push("A");
    with.push("B");
    v.columns = vec![
        Column { name: "start".into(), data: ColumnData::U32(vec![0, 3]) },
        Column { name: "end".into(), data: ColumnData::U32(vec![3, 6]) },
        Column { name: "with".into(), data: ColumnData::Str(with) },
    ];
    let mut doc = Document::new(text);
    let mut b = ViewBuilder::new(text);
    b.replace(6, "AB");
    let (folded, alignment) = b.finish();
    assert_eq!(alignment.edits().len(), 1);
    let view = View::derived(doc.get::<View>().unwrap(), folded, alignment);
    doc.add(&NORM, view, Provenance::by("test")).unwrap();
    let m = matches_golden_key(&doc, &NORM, &g, Tolerance::exact()).unwrap_err();
    assert!(matches!(m, Mismatch::Item { index: 0, what: "span", .. }), "{m:?}");
    has(
        &m.to_string(),
        &[
            "turbo:view@norm item 0",
            "rust:   bytes 0..6 \"ＡＢ\", with \"AB\"",
            "golden: bytes 0..3 \"Ａ\", with \"A\"",
        ],
    );
}
