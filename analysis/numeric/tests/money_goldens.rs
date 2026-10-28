//! Reviewed money extraction/conversion snapshots and invariants.
//!
//! `tests/snapshots/money/{extract,convert}/<notation>/<id>.tsv`.
//! Regenerate with `TURBO_BLESS=1 cargo test -p turbo-numeric --test money_goldens`.
//! Rates: `tests/fixtures/ecb-snapshot.csv`. Target EUR as of 2026-03-15.
//! Snapshots freeze the default (non-OpenNLP-parity) configuration.

use std::path::PathBuf;

use turbo_analysis::{Analyzer, Document, Key, Layer};
use turbo_numeric::NumberNotation;
use turbo_numeric::money::{CONVERTED_MONEY, Currency, Date, EcbRates, Money, MoneyConverter, MoneyExtractor};

const AS_OF: &str = "2026-03-15";
const TARGET: &str = "EUR";

fn root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests")
}

fn input_ids() -> Vec<String> {
    let mut ids: Vec<String> = std::fs::read_dir(root().join("inputs/money"))
        .expect("tests/inputs/money")
        .map(|e| {
            let name = e.expect("entry").file_name().into_string().expect("utf-8");
            name.strip_suffix(".txt").expect(".txt").to_string()
        })
        .collect();
    ids.sort();
    ids
}

fn text_of(id: &str) -> String {
    std::fs::read_to_string(root().join("inputs/money").join(format!("{id}.txt"))).unwrap()
}

fn snap_path(kind: &str, notation: &str, id: &str) -> PathBuf {
    root().join("snapshots/money").join(kind).join(notation).join(format!("{id}.tsv"))
}

fn render(m: &Money) -> String {
    let mut out = String::new();
    for (i, span) in m.spans.iter() {
        let mention = m.get(i);
        out.push_str(&format!(
            "{}\t{}\t{}\t{}\t{}\t{}\t{}\n",
            span.start,
            span.end,
            mention.coefficient,
            u32::from(mention.negative),
            mention.scale,
            m.amount(i).to_plain_string(),
            mention.currency.as_str(),
        ));
    }
    out
}

fn check_invariants(text: &str, m: &Money) {
    m.check().unwrap_or_else(|e| panic!("Layer::check: {e}"));
    let len = text.len() as u32;
    let mut prev = 0u32;
    for (_, span) in m.spans.iter() {
        assert!(span.start <= span.end && span.end <= len);
        assert!(span.start >= prev);
        prev = span.end;
    }
}

fn assert_snap(path: PathBuf, body: &str, bless: bool, label: &str) {
    if bless {
        std::fs::create_dir_all(path.parent().unwrap()).unwrap();
        std::fs::write(&path, body).unwrap();
        return;
    }
    let want = std::fs::read_to_string(&path)
        .unwrap_or_else(|e| panic!("missing snapshot {} ({label}; TURBO_BLESS=1): {e}", path.display()));
    assert_eq!(body, want, "{label}");
}

struct Counts {
    found: usize,
    converted: usize,
}

fn check(dir: &str, notation: NumberNotation) -> Counts {
    let bless = std::env::var_os("TURBO_BLESS").is_some();
    let ids = input_ids();
    assert!(!ids.is_empty());
    let target = Currency::parse(TARGET).expect("EUR");
    let as_of = Date::parse(AS_OF).expect("as_of");
    let rates = EcbRates::load(root().join("fixtures/ecb-snapshot.csv")).expect("rates");
    let analyzer = Analyzer::builder()
        .add(MoneyExtractor::new(notation))
        .step(MoneyConverter::new(rates, target, as_of).bound())
        .build()
        .expect("analyzer");
    let mut counts = Counts { found: 0, converted: 0 };
    for id in &ids {
        let text = text_of(id);
        let mut docs = vec![Document::new(text.as_str())];
        for result in analyzer.run(&mut docs) {
            result.unwrap_or_else(|e| panic!("{id}: {e}"));
        }
        let doc = &docs[0];
        let empty = Money::new(turbo_analysis::ViewId::SOURCE);
        let extracted = doc.get_key(&Key::<Money>::DEFAULT).unwrap_or(&empty);
        let converted = doc.get_key(&CONVERTED_MONEY).unwrap_or(&empty);
        counts.found += extracted.spans.len();
        counts.converted += converted.spans.len();
        check_invariants(&text, extracted);
        check_invariants(&text, converted);
        assert_snap(snap_path("extract", dir, id), &render(extracted), bless, &format!("{dir}/{id} extract"));
        assert_snap(snap_path("convert", dir, id), &render(converted), bless, &format!("{dir}/{id} convert"));
    }
    counts
}

#[test]
fn latin_us_snapshots() {
    let c = check("latin-us", NumberNotation::LatinUs);
    assert!(c.found > 100, "only {} mentions", c.found);
    assert!(c.converted > 100, "only {} conversions", c.converted);
}

#[test]
fn latin_eu_snapshots() {
    let c = check("latin-eu", NumberNotation::LatinEu);
    assert!(c.found > 100, "only {} mentions", c.found);
    assert!(c.converted > 100, "only {} conversions", c.converted);
}
