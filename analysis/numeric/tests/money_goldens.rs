//! Parity of money extraction and conversion with OpenNLP 3.x on the JVM.
//!
//! For every text in `analysis/goldens/data/inputs/money/`,
//! `money/<notation>/money/<id>.golden` holds the mentions OpenNLP's
//! `MoneyAnnotator` found with `CursorMoneyExtractor`, and
//! `money-convert/<notation>/money/<id>.golden` holds those mentions
//! restated by `MoneyConversionAnnotator` with `EcbFxRates`, using the
//! target, date and rate snapshot named in the golden's `config` records
//! (see `analysis/goldens-tool/README.md`). Each test runs `MoneyExtractor`
//! and `MoneyConverter` in one analyzer over every text and compares both
//! `turbo:money` layers with the goldens row for row. No tolerance is
//! applied.

use std::path::PathBuf;

use turbo_analysis::{Analyzer, Document, Key};
use turbo_goldens::{Golden, Tolerance, data_dir, matches_golden_key};
use turbo_numeric::NumberNotation;
use turbo_numeric::money::{CONVERTED_MONEY, Currency, Date, EcbRates, Money, MoneyConverter, MoneyExtractor};

/// The ids of the texts in `inputs/money`, sorted.
fn inputs() -> Vec<String> {
    let mut ids: Vec<String> = std::fs::read_dir(data_dir().join("inputs/money"))
        .expect("the money input set")
        .map(|e| {
            let name = e.expect("a directory entry").file_name().into_string().expect("a UTF-8 file name");
            name.strip_suffix(".txt").expect("an input set holds only <id>.txt files").to_string()
        })
        .collect();
    ids.sort();
    ids
}

fn golden(annotator: &str, notation: &str, id: &str) -> Golden {
    let path: PathBuf = data_dir().join(annotator).join(notation).join("money").join(format!("{id}.golden"));
    Golden::read(path).unwrap_or_else(|e| panic!("{e}"))
}

fn config<'g>(golden: &'g Golden, key: &str) -> &'g str {
    let entry = golden.header.config.iter().find(|c| c.key == key);
    &entry.unwrap_or_else(|| panic!("no {key} config record")).value
}

/// What the comparison covered: mentions found and mentions converted.
struct Counts {
    found: usize,
    converted: usize,
}

/// Runs extraction and conversion in `notation` over every text and
/// compares both layers with their goldens.
fn check(dir: &str, notation: NumberNotation, name: &str) -> Counts {
    let ids = inputs();
    assert!(!ids.is_empty(), "no inputs in the money set");
    let mut counts = Counts { found: 0, converted: 0 };
    let mut failures = Vec::new();
    for id in &ids {
        let extracted = golden("money", dir, id);
        let converted = golden("money-convert", dir, id);
        for g in [&extracted, &converted] {
            assert_eq!(config(g, "notation"), name, "{id}: the golden was made in another notation");
        }
        let target = Currency::parse(config(&converted, "target")).expect("an ISO 4217 target");
        let as_of = Date::parse(config(&converted, "as_of")).expect("an ISO date");
        let rates = EcbRates::load(data_dir().join(config(&converted, "snapshot"))).expect("the rate snapshot");
        let analyzer = Analyzer::builder()
            .add(MoneyExtractor::new(notation))
            .step(MoneyConverter::new(rates, target, as_of).bound())
            .build()
            .expect("the analyzer");
        let text = extracted.input_text(data_dir()).unwrap_or_else(|e| panic!("{e}"));
        let mut docs = vec![Document::new(text.as_str())];
        for result in analyzer.run(&mut docs) {
            result.unwrap_or_else(|e| panic!("{id}: {e}"));
        }
        let doc = &docs[0];
        counts.found += doc.get_key(&Key::<Money>::DEFAULT).map_or(0, |m| m.spans.len());
        counts.converted += doc.get_key(&CONVERTED_MONEY).map_or(0, |m| m.spans.len());
        if let Err(m) = matches_golden_key(doc, &Key::<Money>::DEFAULT, &extracted, Tolerance::exact()) {
            failures.push(format!("{id} extraction: {m}"));
        }
        if let Err(m) = matches_golden_key(doc, &CONVERTED_MONEY, &converted, Tolerance::exact()) {
            failures.push(format!("{id} conversion: {m}"));
        }
    }
    assert!(
        failures.is_empty(),
        "{} comparisons over {} texts differ from the JVM:\n{}",
        failures.len(),
        ids.len(),
        failures.join("\n")
    );
    counts
}

#[test]
fn latin_us_matches_the_jvm() {
    let c = check("latin-us", NumberNotation::LatinUs, "LATIN_US");
    assert!(c.found > 100, "only {} mentions compared", c.found);
    assert!(c.converted > 100, "only {} conversions compared", c.converted);
}

#[test]
fn latin_eu_matches_the_jvm() {
    let c = check("latin-eu", NumberNotation::LatinEu, "LATIN_EU");
    assert!(c.found > 100, "only {} mentions compared", c.found);
    assert!(c.converted > 100, "only {} conversions compared", c.converted);
}
