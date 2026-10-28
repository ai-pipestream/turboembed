//! Reviewed quantity snapshots and span/value invariants.
//!
//! `tests/snapshots/quantity/<notation>/<id>.tsv`. Regenerate with
//! `TURBO_BLESS=1 cargo test -p turbo-numeric --test quantity_goldens`.
//! JVM goldens are not an oracle.

use std::path::PathBuf;

use turbo_analysis::{Analyzer, Document, Layer};
use turbo_numeric::{NumberNotation, Quantities, QuantityExtractor};

fn root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests")
}

fn input_ids() -> Vec<String> {
    let mut ids: Vec<String> = std::fs::read_dir(root().join("inputs/quantity"))
        .expect("tests/inputs/quantity")
        .map(|e| {
            let name = e.expect("entry").file_name().into_string().expect("utf-8");
            name.strip_suffix(".txt").expect(".txt").to_string()
        })
        .collect();
    ids.sort();
    ids
}

fn text_of(id: &str) -> String {
    std::fs::read_to_string(root().join("inputs/quantity").join(format!("{id}.txt"))).unwrap()
}

fn snap_path(notation: &str, id: &str) -> PathBuf {
    root().join("snapshots/quantity").join(notation).join(format!("{id}.tsv"))
}

fn render(q: &Quantities) -> String {
    let mut out = String::new();
    for (i, span) in q.spans.iter() {
        let m = q.get(i);
        out.push_str(&format!(
            "{}\t{}\t{}\t{}\t{}\t{}\t{}\n",
            span.start,
            span.end,
            m.coefficient,
            u32::from(m.negative),
            m.scale,
            q.value(i).to_plain_string(),
            m.unit,
        ));
    }
    out
}

fn check_invariants(text: &str, q: &Quantities) {
    q.check().unwrap_or_else(|e| panic!("Layer::check: {e}"));
    let len = text.len() as u32;
    let mut prev = 0u32;
    for (_, span) in q.spans.iter() {
        assert!(span.start <= span.end && span.end <= len);
        assert!(span.start >= prev);
        prev = span.end;
    }
}

fn check(dir: &str, notation: NumberNotation) -> usize {
    let bless = std::env::var_os("TURBO_BLESS").is_some();
    let ids = input_ids();
    assert!(!ids.is_empty());
    let analyzer =
        Analyzer::builder().add(QuantityExtractor::new(notation).with_opennlp_parity(true)).build().expect("analyzer");
    let mut found = 0;
    for id in &ids {
        let text = text_of(id);
        let mut docs = vec![Document::new(text.as_str())];
        for result in analyzer.run(&mut docs) {
            result.unwrap_or_else(|e| panic!("{id}: {e}"));
        }
        let empty = Quantities::new(turbo_analysis::ViewId::SOURCE);
        let q = docs[0].get::<Quantities>().unwrap_or(&empty);
        found += q.spans.len();
        check_invariants(&text, q);
        let body = render(q);
        let path = snap_path(dir, id);
        if bless {
            std::fs::create_dir_all(path.parent().unwrap()).unwrap();
            std::fs::write(&path, &body).unwrap();
            continue;
        }
        let want = std::fs::read_to_string(&path)
            .unwrap_or_else(|e| panic!("missing snapshot {} (TURBO_BLESS=1 to write): {e}", path.display()));
        assert_eq!(body, want, "{dir}/{id}");
    }
    found
}

#[test]
fn latin_us_snapshots() {
    let found = check("latin-us", NumberNotation::LatinUs);
    assert!(found > 120, "only {found} mentions compared");
}

#[test]
fn latin_eu_snapshots() {
    let found = check("latin-eu", NumberNotation::LatinEu);
    assert!(found > 110, "only {found} mentions compared");
}
