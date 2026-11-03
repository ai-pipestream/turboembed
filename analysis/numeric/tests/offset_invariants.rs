//! Offset invariants kept after the fuzz differential numeric/temporal runs
//! were removed: `check_document` (source projections, UTF-16/code-point
//! conversions) plus `OffsetTable` round-trips on every span edge, covering
//! numbers, quantity, money (extract + convert) and temporal (dateline +
//! fixed). Not blessable expected-output.

use std::path::PathBuf;

use turbo_analysis::{Analyzer, Document, Key, OffsetTable, Span, ViewId};
use turbo_analysis_layers::{Money, NumberNotation, Numbers, Quantities};
use turbo_goldens::data_dir;
use turbo_goldens::invariants::check_document;
use turbo_numeric::money::{CONVERTED_MONEY, Currency, Date, EcbRates, MoneyConverter, MoneyExtractor};
use turbo_numeric::{DocumentDateSelector, NumberScanner, QuantityExtractor, TemporalExtractor, Temporals};

fn root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests")
}

fn challenging_texts() -> [&'static str; 8] {
    [
        "Paid $1,250.50 for 2,500 kg on 14 July 2026.",
        "Invoice EUR 99.00 due tomorrow.",
        "Stock: 1.5e3 units @ ¥100.",
        // Non-BMP before amounts.
        "😀 Paid $12.00 yesterday.",
        // Precomposed accent.
        "café $3.50",
        // Combining acute.
        "a\u{0301} 100 kg",
        "Dated 0999-12-31, amended 100 days ago.",
        "CHF 1\u{2019}250.50 and 2\u{202F}500 kg.",
    ]
}

fn collect_spans(doc: &Document) -> Vec<Span> {
    let empty_n = Numbers::new(ViewId::SOURCE);
    let empty_q = Quantities::new(ViewId::SOURCE);
    let empty_m = Money::new(ViewId::SOURCE);
    let empty_t = Temporals::new(ViewId::SOURCE);
    let mut spans = Vec::new();
    for (_, s) in doc.get::<Numbers>().unwrap_or(&empty_n).spans.iter() {
        spans.push(s);
    }
    for (_, s) in doc.get::<Quantities>().unwrap_or(&empty_q).spans.iter() {
        spans.push(s);
    }
    for (_, s) in doc.get_key(&Key::<Money>::DEFAULT).unwrap_or(&empty_m).spans.iter() {
        spans.push(s);
    }
    for (_, s) in doc.get_key(&CONVERTED_MONEY).unwrap_or(&empty_m).spans.iter() {
        spans.push(s);
    }
    for (_, s) in doc.get::<Temporals>().unwrap_or(&empty_t).spans.iter() {
        spans.push(s);
    }
    spans
}

fn assert_offsets(doc: &Document, label: &str) {
    let (checked, violations) = check_document(doc);
    assert!(violations.is_empty(), "{label}: {violations:?}");
    assert!(checked > 0, "{label}: no layers checked");
    let text = doc.text(ViewId::SOURCE);
    let table = OffsetTable::build(text);
    for span in collect_spans(doc) {
        for edge in [span.start, span.end] {
            let u16 = table.to_utf16(text, edge);
            let cp = table.to_code_point(text, edge);
            assert_eq!(table.from_utf16(text, u16).unwrap(), edge, "{label}");
            assert_eq!(table.from_code_point(text, cp).unwrap(), edge, "{label}");
        }
    }
}

#[test]
fn local_numeric_inputs_keep_offset_invariants_in_default_mode() {
    let target = Currency::parse("EUR").unwrap();
    let as_of = Date::parse("2026-03-15").unwrap();
    let reference = Date::parse("2026-03-15").unwrap();
    let mut files = 0;
    for notation in [NumberNotation::LatinUs, NumberNotation::LatinEu] {
        let rates = EcbRates::load(root().join("fixtures/ecb-snapshot.csv")).expect("rates");
        let analyzer = Analyzer::builder()
            .add(NumberScanner::new(notation))
            .add(QuantityExtractor::new(notation))
            .add(MoneyExtractor::new(notation))
            .step(MoneyConverter::new(rates, target, as_of).bound())
            .add(TemporalExtractor::with_reference(reference))
            .add(DocumentDateSelector)
            .build()
            .unwrap();
        for set in ["num", "quantity", "money", "temporal"] {
            let dir = data_dir().join("inputs").join(set);
            for entry in std::fs::read_dir(&dir).unwrap() {
                let path = entry.unwrap().path();
                if path.extension().and_then(|e| e.to_str()) != Some("txt") {
                    continue;
                }
                let text = std::fs::read_to_string(&path).unwrap();
                let mut docs = vec![Document::new(text.as_str())];
                for res in analyzer.run(&mut docs) {
                    res.unwrap_or_else(|e| panic!("{}: {e}", path.display()));
                }
                assert_offsets(&docs[0], &format!("{} ({notation:?})", path.display()));
                files += 1;
            }
        }
    }
    assert!(files > 50, "only {files} input files checked");
}

#[test]
fn challenging_texts_keep_offset_invariants_for_numeric_layers() {
    let rates = EcbRates::load(root().join("fixtures/ecb-snapshot.csv")).expect("rates");
    let target = Currency::parse("EUR").unwrap();
    let as_of = Date::parse("2026-03-15").unwrap();
    let analyzer = Analyzer::builder()
        .add(NumberScanner::new(NumberNotation::LatinUs))
        .add(QuantityExtractor::new(NumberNotation::LatinUs))
        .add(MoneyExtractor::new(NumberNotation::LatinUs))
        .step(MoneyConverter::new(rates, target, as_of).bound())
        .add(TemporalExtractor::with_reference(Date::parse("2026-03-15").unwrap()))
        .add(DocumentDateSelector)
        .build()
        .unwrap();
    for text in challenging_texts() {
        let mut docs = vec![Document::new(text)];
        for res in analyzer.run(&mut docs) {
            res.unwrap_or_else(|e| panic!("{text:?}: {e}"));
        }
        assert_offsets(&docs[0], text);
    }
}
