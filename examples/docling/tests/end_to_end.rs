//! `data/report.docling.json`, a small report in docling's JSON form, run
//! through the whole pipeline with the tiny test model, against
//! `data/report.expected.json`. Spans, levels and entities must match
//! exactly; vector values within `report::TOLERANCE`, since kernels may round
//! differently across machines. `BLESS=1` rewrites the expected file.

use std::path::PathBuf;

use serde_json::Value;
use turbo_analysis::{Ix, Key, Layer};
use turbo_analysis_layers::{Block, Chunks, Structure};
use turbo_example_docling::docling::parse;
use turbo_example_docling::report::close;
use turbo_example_docling::{CLEAN, analyze, pipeline, report};

const PLACES: [&str; 3] = ["Zürich", "Lyon", "Oslo"];

fn here() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
}

fn input() -> String {
    std::fs::read_to_string(here().join("data/report.docling.json")).unwrap()
}

fn bundle() -> PathBuf {
    here().join("../../testdata/tiny-bert-bundle")
}

#[test]
fn the_docling_structure_is_read_in_reading_order() {
    let (d, s) = parse(&input()).unwrap();
    s.check().unwrap();
    // The page header is furniture and is not read.
    assert!(!d.text.contains("Internal"));
    let refs: Vec<&str> = d.items.iter().map(|i| i.self_ref.as_str()).collect();
    let want: Vec<String> = (1..=12).map(|i| format!("#/texts/{i}")).collect();
    assert_eq!(refs, want);
    for i in &d.items {
        assert!(!d.text[i.span.range()].is_empty(), "{}", i.self_ref);
    }
    // The list's items are a line apart, other blocks a blank line.
    assert!(d.text.contains("hours.\nLyon, with twelve megawatt hours.\nOslo"));
    assert!(d.text.contains("The sites are:\n\nZürich, with"));
    // The title and the three headers each open a section.
    assert_eq!(s.kind.iter().filter(|&&k| k == Block::Section).count(), 4);
    assert_eq!(s.kind.iter().filter(|&&k| k == Block::ListItem).count(), 3);
    // "2.1 Lyon" (level 2) sits in "2. Operations" (level 1).
    let section_of = |text: &str| {
        let at = d.text.find(text).unwrap() as u32;
        let h = (0..s.kind.len()).find(|&i| s.kind[i] == Block::Heading && s.spans.get(Ix::new(i as u32)).start == at);
        s.parent[h.unwrap()].unwrap().get()
    };
    let (ops, lyon) = (section_of("2. Operations"), section_of("2.1 Lyon"));
    assert_eq!(s.parent[lyon].map(|p| p.get()), Some(ops));
    assert_eq!(s.depth[lyon], 3);
}

#[test]
fn refuses_what_it_cannot_convert() {
    let mut doc: Value = serde_json::from_str(&input()).unwrap();
    doc["body"]["children"].as_array_mut().unwrap().push(serde_json::json!({"$ref": "#/tables/0"}));
    let err = parse(&doc.to_string()).err().unwrap();
    assert!(err.contains("#/tables/0"), "{err}");
    assert!(parse("{\"schema_name\": \"Other\"}").is_err());
}

#[test]
fn docling_to_vectors_end_to_end() {
    let analyzer = pipeline(&bundle(), &PLACES).unwrap();
    let (d, doc) = analyze(&analyzer, &input()).unwrap();
    // The structure was carried onto the clean view, where the chunker
    // read it.
    let clean = doc.get_key(&Key::<turbo_analysis::View>::named(CLEAN)).unwrap();
    assert!(clean.text().len() < d.text.len(), "normalizing removed nothing");
    assert_eq!(doc.get::<Structure>().unwrap().spans.view(), clean.id());
    let checked = report::check_offsets(&d, &doc).unwrap();
    assert!(checked > 100, "{checked}");
    let chunks = doc.get::<Chunks>().unwrap();
    assert!(chunks.len() > 1);
    assert!(chunks.model_tokens.iter().zip(&chunks.over_budget).all(|(&n, &over)| over || n <= 48));

    let got = report::report(&d, &doc).unwrap();
    let path = here().join("data/report.expected.json");
    if std::env::var_os("BLESS").is_some() {
        std::fs::write(&path, serde_json::to_string_pretty(&got).unwrap() + "\n").unwrap();
    }
    let want: Value = serde_json::from_str(&std::fs::read_to_string(&path).unwrap()).unwrap();
    if let Err(e) = close("report", &got, &want) {
        panic!("{e}\nrerun with BLESS=1 to accept, then review the diff of data/report.expected.json");
    }
}
