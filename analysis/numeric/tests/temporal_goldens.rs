//! Parity of temporal extraction and document date selection with OpenNLP
//! 3.x on the JVM.
//!
//! For every text in `analysis/goldens/data/inputs/temporal/`,
//! `temporal/<mode>/temporal/<id>.golden` holds the mentions OpenNLP's
//! `TemporalAnnotator` found with `CursorTemporalExtractor` and the date
//! `DocumentDateAnnotator` chose from them (see
//! `analysis/goldens-tool/README.md`). In the `dateline` mode the reference
//! date of the relative mentions is each text's first absolute day; in the
//! `fixed` mode it is 2026-03-15. Each test runs `TemporalExtractor` and
//! `DocumentDateSelector` in an analyzer over every text and compares the
//! `turbo:temporal` and `turbo:document-date` layers with the golden row
//! for row. No tolerance is applied.

use std::path::PathBuf;

use turbo_analysis::{Analyzer, Document};
use turbo_goldens::{Golden, Tolerance, data_dir, matches_golden};
use turbo_numeric::{Date, DocumentDate, DocumentDateSelector, Origin, TemporalExtractor, Temporals};

/// The reference date of the `fixed` goldens.
const REFERENCE: &str = "2026-03-15";

/// The ids of the texts in `inputs/temporal`, sorted.
fn inputs() -> Vec<String> {
    let mut ids: Vec<String> = std::fs::read_dir(data_dir().join("inputs/temporal"))
        .expect("the temporal input set")
        .map(|e| {
            let name = e.expect("a directory entry").file_name().into_string().expect("a UTF-8 file name");
            name.strip_suffix(".txt").expect("an input set holds only <id>.txt files").to_string()
        })
        .collect();
    ids.sort();
    ids
}

fn golden(mode: &str, id: &str) -> Golden {
    let path: PathBuf = data_dir().join("temporal").join(mode).join("temporal").join(format!("{id}.golden"));
    Golden::read(path).unwrap_or_else(|e| panic!("{e}"))
}

fn config<'g>(golden: &'g Golden, key: &str) -> &'g str {
    let entry = golden.header.config.iter().find(|c| c.key == key);
    &entry.unwrap_or_else(|| panic!("no {key} config record")).value
}

/// What one mode compared: mentions, relative mentions among them, and
/// texts with a document date.
struct Compared {
    mentions: usize,
    relative: usize,
    dated: usize,
}

/// Runs `extractor` and the selector over every text, compares both layers
/// with the goldens of `mode`, whose `config reference` must be
/// `reference`, and returns what was compared.
fn check(mode: &str, extractor: TemporalExtractor, reference: &str) -> Compared {
    let ids = inputs();
    assert!(!ids.is_empty(), "no inputs in the temporal set");
    let analyzer = Analyzer::builder().add(extractor).add(DocumentDateSelector).build().expect("the analyzer");
    let mut compared = Compared { mentions: 0, relative: 0, dated: 0 };
    let mut failures = Vec::new();
    for id in &ids {
        let g = golden(mode, id);
        assert_eq!(config(&g, "reference"), reference, "{id}: the golden was made with another reference");
        let text = g.input_text(data_dir()).unwrap_or_else(|e| panic!("{e}"));
        let mut docs = vec![Document::new(text.as_str())];
        for result in analyzer.run(&mut docs) {
            result.unwrap_or_else(|e| panic!("{id}: {e}"));
        }
        if let Some(t) = docs[0].get::<Temporals>() {
            compared.mentions += t.spans.len();
            compared.relative += t.origin.iter().filter(|&&o| o == Origin::Relative).count();
        }
        compared.dated += docs[0].get::<DocumentDate>().map_or(0, |d| usize::from(d.date.is_some()));
        if let Err(m) = matches_golden::<Temporals>(&docs[0], &g, Tolerance::exact()) {
            failures.push(format!("{id}: {m}"));
        }
        if let Err(m) = matches_golden::<DocumentDate>(&docs[0], &g, Tolerance::exact()) {
            failures.push(format!("{id}: {m}"));
        }
    }
    assert!(
        failures.is_empty(),
        "{} layers of {} texts differ from the JVM:\n{}",
        failures.len(),
        ids.len(),
        failures.join("\n")
    );
    compared
}

#[test]
fn the_dateline_mode_matches_the_jvm() {
    let c = check("dateline", TemporalExtractor::new(), "dateline");
    assert!(c.mentions > 75, "only {} mentions compared", c.mentions);
    assert!(c.relative >= 5, "only {} relative mentions compared", c.relative);
    assert!(c.dated > 25, "only {} document dates compared", c.dated);
}

#[test]
fn the_fixed_mode_matches_the_jvm() {
    let reference = Date::parse(REFERENCE).expect("the reference date");
    let c = check("fixed", TemporalExtractor::with_reference(reference), REFERENCE);
    assert!(c.mentions > 110, "only {} mentions compared", c.mentions);
    assert!(c.relative > 40, "only {} relative mentions compared", c.relative);
    assert!(c.dated > 25, "only {} document dates compared", c.dated);
}
