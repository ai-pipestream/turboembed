//! Parity of the number scan with OpenNLP 3.x on the JVM.
//!
//! `analysis/goldens/data/numbers/<notation>/num/<id>.golden` holds, for
//! every text in `analysis/goldens/data/inputs/num/`, the numbers the
//! goldens tool found with `NumberScan.parse` from the opennlp-addons
//! `numeric` module, read in that notation with scale markers applied (see
//! `analysis/goldens-tool/README.md`, "Number goldens"). Each test runs
//! `NumberScanner` over every text and compares its `turbo:numbers` layer
//! with the golden row for row: the span, then the value, where the
//! numeral ends, the sign, the coefficient and the scale. No tolerance is
//! applied.

use std::path::PathBuf;

use turbo_analysis::{Document, Key, Provenance, ViewId};
use turbo_analysis_layers::{NumberNotation, Numbers};
use turbo_goldens::{Golden, Tolerance, data_dir, matches_golden};
use turbo_numeric::NumberScanner;

/// The ids of the texts in `inputs/num`, sorted.
fn inputs() -> Vec<String> {
    let mut ids: Vec<String> = std::fs::read_dir(data_dir().join("inputs/num"))
        .expect("the num input set")
        .map(|e| {
            let name = e.expect("a directory entry").file_name().into_string().expect("a UTF-8 file name");
            name.strip_suffix(".txt").expect("an input set holds only <id>.txt files").to_string()
        })
        .collect();
    ids.sort();
    ids
}

fn golden_path(notation: &str, id: &str) -> PathBuf {
    data_dir().join("numbers").join(notation).join("num").join(format!("{id}.golden"))
}

/// Runs the scanner in `notation` over every text and compares each with
/// its golden; returns how many numbers were compared.
fn check(dir: &str, notation: NumberNotation, config: &str) -> usize {
    let scanner = NumberScanner::new(notation).with_opennlp_parity(true);
    let ids = inputs();
    assert!(!ids.is_empty(), "no inputs in the num set");
    let mut numbers = 0;
    let mut failures = Vec::new();
    for id in &ids {
        let golden = Golden::read(golden_path(dir, id)).unwrap_or_else(|e| panic!("{e}"));
        let entry = golden.header.config.iter().find(|c| c.key == "notation").expect("a notation config record");
        assert_eq!(entry.value, config, "{id}: the golden was made in another notation");
        let text = golden.input_text(data_dir()).unwrap_or_else(|e| panic!("{e}"));
        let mut layer = Numbers::new(ViewId::SOURCE);
        scanner.scan(&text, &mut layer);
        numbers += layer.spans.len();
        let mut doc = Document::new(text.as_str());
        doc.add(&Key::<Numbers>::DEFAULT, layer, Provenance::by("goldens test")).expect("the layer is new");
        if let Err(m) = matches_golden::<Numbers>(&doc, &golden, Tolerance::exact()) {
            failures.push(format!("{id}: {m}"));
        }
    }
    assert!(
        failures.is_empty(),
        "{} of {} texts differ from the JVM:\n{}",
        failures.len(),
        ids.len(),
        failures.join("\n")
    );
    numbers
}

#[test]
fn latin_us_matches_the_jvm() {
    let n = check("latin-us", NumberNotation::LatinUs, "LATIN_US");
    assert!(n > 100, "only {n} numbers compared");
}

#[test]
fn latin_eu_matches_the_jvm() {
    let n = check("latin-eu", NumberNotation::LatinEu, "LATIN_EU");
    assert!(n > 100, "only {n} numbers compared");
}
