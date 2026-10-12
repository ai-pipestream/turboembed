//! Parity of quantity extraction with OpenNLP 3.x on the JVM.
//!
//! For every text in `analysis/goldens/data/inputs/quantity/`,
//! `quantity/<notation>/quantity/<id>.golden` holds the mentions OpenNLP's
//! `QuantityAnnotator` found with `CursorQuantityExtractor` and its default
//! units (see `analysis/goldens-tool/README.md`). Each test runs
//! `QuantityExtractor` in an analyzer over every text and compares the
//! `turbo:quantities` layer with the golden row for row. No tolerance is
//! applied.

use std::path::PathBuf;

use turbo_analysis::{Analyzer, Document};
use turbo_goldens::{Golden, Tolerance, data_dir, matches_golden};
use turbo_numeric::{NumberNotation, Quantities, QuantityExtractor};

/// The ids of the texts in `inputs/quantity`, sorted.
fn inputs() -> Vec<String> {
    let mut ids: Vec<String> = std::fs::read_dir(data_dir().join("inputs/quantity"))
        .expect("the quantity input set")
        .map(|e| {
            let name = e.expect("a directory entry").file_name().into_string().expect("a UTF-8 file name");
            name.strip_suffix(".txt").expect("an input set holds only <id>.txt files").to_string()
        })
        .collect();
    ids.sort();
    ids
}

fn golden(notation: &str, id: &str) -> Golden {
    let path: PathBuf = data_dir().join("quantity").join(notation).join("quantity").join(format!("{id}.golden"));
    Golden::read(path).unwrap_or_else(|e| panic!("{e}"))
}

fn config<'g>(golden: &'g Golden, key: &str) -> &'g str {
    let entry = golden.header.config.iter().find(|c| c.key == key);
    &entry.unwrap_or_else(|| panic!("no {key} config record")).value
}

/// Runs extraction in `notation` over every text, compares the layer with
/// the goldens of `dir` and returns how many mentions were compared.
fn check(dir: &str, notation: NumberNotation, name: &str) -> usize {
    let ids = inputs();
    assert!(!ids.is_empty(), "no inputs in the quantity set");
    let analyzer = Analyzer::builder().add(QuantityExtractor::new(notation)).build().expect("the analyzer");
    let mut found = 0;
    let mut failures = Vec::new();
    for id in &ids {
        let g = golden(dir, id);
        assert_eq!(config(&g, "notation"), name, "{id}: the golden was made in another notation");
        let text = g.input_text(data_dir()).unwrap_or_else(|e| panic!("{e}"));
        let mut docs = vec![Document::new(text.as_str())];
        for result in analyzer.run(&mut docs) {
            result.unwrap_or_else(|e| panic!("{id}: {e}"));
        }
        found += docs[0].get::<Quantities>().map_or(0, |q| q.spans.len());
        if let Err(m) = matches_golden::<Quantities>(&docs[0], &g, Tolerance::exact()) {
            failures.push(format!("{id}: {m}"));
        }
    }
    assert!(
        failures.is_empty(),
        "{} texts of {} differ from the JVM:\n{}",
        failures.len(),
        ids.len(),
        failures.join("\n")
    );
    found
}

#[test]
fn latin_us_matches_the_jvm() {
    let found = check("latin-us", NumberNotation::LatinUs, "LATIN_US");
    assert!(found > 120, "only {found} mentions compared");
}

#[test]
fn latin_eu_matches_the_jvm() {
    let found = check("latin-eu", NumberNotation::LatinEu, "LATIN_EU");
    assert!(found > 110, "only {found} mentions compared");
}
