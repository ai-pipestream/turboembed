//! A layer type that is not in turbo-analysis-layers, read from a golden
//! and compared, as a crate that owns such a type does it: the type
//! implements `GoldenCompare`, and its goldens are read with
//! `Golden::read_with`, passing `known_shapes()` with the type's shape
//! added. The type here is a word type per token, `turbo:word-types`, and
//! its golden is the hand written `tests/fixtures/word-types.golden` over
//! the same input as `zurich.golden`.

use std::path::PathBuf;

use turbo_analysis::{Anchor, Document, Key, Layer, Provenance, Span, Spans, ViewId};
use turbo_analysis_layers::{Strs, Tokens};
use turbo_goldens::compare::{Items, Labels, Names, compare_items};
use turbo_goldens::format::ColumnType;
use turbo_goldens::*;

const TEXT: &str = "Dr. Smith went to Zürich. It rained!";
const TOKENS: [(u32, u32); 9] = [(0, 3), (4, 9), (10, 14), (15, 17), (18, 25), (25, 26), (27, 29), (30, 36), (36, 37)];
const KINDS: [&str; 9] = ["abbreviation", "word", "word", "word", "word", "punct", "word", "word", "punct"];

/// A word type per token, as a tokenizer crate would define it.
struct WordTypes {
    kind: Strs,
}

impl Layer for WordTypes {
    const ID: &'static str = "turbo:word-types";
    fn anchor() -> Anchor {
        Anchor::aligned::<Tokens>()
    }
    fn len(&self) -> usize {
        self.kind.len()
    }
}

impl GoldenCompare for WordTypes {
    const COLUMNS: &'static [ColumnSpec] = &[ColumnSpec::new("kind", ColumnType::Str)];

    fn compare(&self, cx: &Cx<'_>, golden: &GoldenLayer, tol: Tolerance) -> Result<(), Mismatch> {
        let rust = Items {
            len: self.kind.len(),
            spans: None,
            context: cx.rust_anchor::<Tokens>(),
            tokens: None,
            units: None,
            labels: Some(Labels::Strs(&self.kind)),
            values: None,
        };
        let golden = Items {
            len: golden.len(),
            spans: None,
            context: cx.golden_spans("opennlp:tokens"),
            tokens: None,
            units: None,
            labels: Some(Labels::Strs(golden.strs("kind").expect("a str column the shape checked"))),
            values: None,
        };
        compare_items(cx, &rust, &golden, Names { label: Some("kind"), value: None }, tol)
    }
}

fn fixture() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/word-types.golden")
}

fn shapes() -> Vec<Shape> {
    let mut shapes = known_shapes();
    shapes.push(shape_of::<WordTypes>());
    shapes
}

fn doc(kinds: &[&str]) -> Document {
    let mut doc = Document::new(TEXT);
    let mut spans = Spans::new(ViewId::SOURCE);
    for &(a, b) in &TOKENS {
        spans.push(Span::new(a, b));
    }
    doc.add(&Key::DEFAULT, Tokens { spans, sentence: None }, Provenance::by("test")).unwrap();
    let mut kind = Strs::default();
    for k in kinds {
        kind.push(k);
    }
    doc.add(&Key::DEFAULT, WordTypes { kind }, Provenance::by("test")).unwrap();
    doc
}

#[test]
fn the_shape_comes_from_the_type() {
    let s = shape_of::<WordTypes>();
    assert_eq!(s.id, "turbo:word-types");
    assert_eq!(s.anchor, GoldenAnchor::Aligned("opennlp:tokens".into()));
    assert_eq!(s.columns, [ColumnSpec::new("kind", ColumnType::Str)]);
}

#[test]
fn read_with_takes_a_layer_the_default_set_does_not_know() {
    // The default set refuses the block, naming the line and the id.
    let e = Golden::read(fixture()).unwrap_err().to_string();
    assert!(e.contains("line 21: no layer type has the id turbo:word-types"), "{e}");

    let g = Golden::read_with(fixture(), &shapes()).unwrap();
    let w = g.layer("turbo:word-types").unwrap();
    assert_eq!(w.anchor, GoldenAnchor::Aligned("opennlp:tokens".into()));
    assert_eq!(w.strs("kind").unwrap().iter().collect::<Vec<_>>(), KINDS);

    // The other forms take the same set, and write back byte for byte.
    let text = std::fs::read_to_string(fixture()).unwrap();
    assert_eq!(Golden::parse_with(&text, &shapes()).unwrap(), g);
    assert_eq!(Golden::parse_bytes_with(text.as_bytes(), &shapes()).unwrap(), g);
    assert!(Golden::parse(&text).is_err());
    assert!(Golden::parse_bytes(text.as_bytes()).is_err());
    assert_eq!(g.to_text().unwrap(), text);
}

#[test]
fn read_with_checks_the_columns_against_the_shape_passed() {
    let text = std::fs::read_to_string(fixture()).unwrap().replace("\tkind:str", "\tkind:u32");
    let e = Golden::parse_with(&text, &shapes()).unwrap_err().to_string();
    assert!(e.contains("turbo:word-types has the columns [kind:u32], expected [kind:str]"), "{e}");
}

#[test]
fn a_layer_from_another_crate_compares_with_its_golden() {
    let g = Golden::read_with(fixture(), &shapes()).unwrap();
    assert_matches_golden::<WordTypes>(&doc(&KINDS), &g, Tolerance::exact());
    assert_matches_golden::<Tokens>(&doc(&KINDS), &g, Tolerance::exact());

    let mut kinds = KINDS;
    kinds[4] = "abbreviation";
    let m = matches_golden::<WordTypes>(&doc(&kinds), &g, Tolerance::exact()).unwrap_err();
    assert!(matches!(m, Mismatch::Item { index: 4, what: "kind", .. }), "{m:?}");
    let shown = m.to_string();
    for part in [
        "turbo:word-types item 4: the kind differs",
        "rust:   bytes 18..25 \"Zürich\", kind \"abbreviation\"",
        "golden: bytes 18..25 \"Zürich\", kind \"word\"",
    ] {
        assert!(shown.contains(part), "{part:?} is not in the report:\n{shown}");
    }
}
