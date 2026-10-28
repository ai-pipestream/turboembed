//! The unchanged MIT-licensed Italian NumberWithUnit corpus at the pinned
//! Recognizers-Text revision. Only localized resolution labels and decimal
//! commas are translated for the shared, exact-span comparator.
#[allow(dead_code)]
#[path = "spec/units.rs"]
mod units;
use std::path::PathBuf;
use turbo_grammar::{ITALIAN_UNITS, Measure, Parser};
fn data() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/data")
}
fn same_number(expected: &str, got: f64) -> bool {
    expected.parse::<f64>().is_ok_and(|want| want == got || (want - got).abs() <= 1e-12 * want.abs().max(got.abs()))
}
fn name(s: &str) -> &str {
    match s {
        "Anno" => "Year",
        "Giorno" => "Day",
        "Mese" => "Month",
        "Settimana" => "Week",
        "Acro" => "Acre",
        "Barile" => "Barrel",
        "Chilometro" => "Kilometer",
        "Chilometro quadrato" => "Square kilometer",
        "Gallone" => "Gallon",
        "Libbra" => "Pound",
        "Litro" => "Liter",
        "Millilitro" => "Milliliter",
        "Metro" => "Meter",
        "Metro quadrato" => "Square meter",
        "Miglio" => "Mile",
        "Miglia all'ora" => "Mile per hour",
        "Millimetro" => "Millimeter",
        "Oncia" => "Ounce",
        "Piede" => "Foot",
        "Piede cubo" => "Cubic foot",
        "Pollice" => "Inch",
        "Tonnellata" => "Ton",
        _ => s,
    }
}
#[test]
fn italian_number_with_unit_specifications() {
    let mut failures = Vec::new();
    for (model, measure, expected_count) in [
        ("Age", Measure::Age, 15),
        ("Currency", Measure::Currency, 106),
        ("Dimension", Measure::Dimension, 55),
        ("Temperature", Measure::Temperature, 34),
    ] {
        let path = data().join(format!("recognizers-text/Italian/NumberWithUnit/{model}Model.json"));
        let raw = std::fs::read_to_string(path).unwrap();
        let mut cases: Vec<serde_json::Value> = serde_json::from_str(raw.trim_start_matches('\u{feff}')).unwrap();
        let (mut eligible, mut passing) = (0, 0);
        let mut parser = Parser::new(&ITALIAN_UNITS);
        for (i, case) in cases.iter_mut().enumerate() {
            eligible += 1;
            for result in case["Results"].as_array_mut().unwrap() {
                let resolution = &mut result["Resolution"];
                if let Some(label) = resolution["unit"].as_str() {
                    resolution["unit"] = name(label).into();
                }
                if let Some(value) = resolution["value"].as_str() {
                    resolution["value"] = value.replace(',', ".").into();
                }
            }
            if case["Results"].as_array().unwrap().iter().any(|r| r.get("Start").is_none()) {
                assert_eq!(model, "Temperature");
                assert!((25..=27).contains(&i));
                if text_only(&mut parser, case) {
                    passing += 1;
                } else {
                    failures.push(format!("{model} #{i}: text-only comparison failed"));
                }
                continue;
            }
            let parsed: units::UnitCase = serde_json::from_value(case.clone()).unwrap();
            let (pass, got) = units::run(&mut parser, measure, &parsed);
            if pass {
                passing += 1;
            } else {
                failures.push(format!(
                    "{model} #{i}: {} expected {} found {}",
                    case["Input"],
                    case["Results"],
                    units::describe(case["Input"].as_str().unwrap(), &got)
                ));
            }
        }
        println!("{model}: {passing}/{eligible}");
        assert_eq!(eligible, expected_count);
    }
    assert!(failures.is_empty(), "{}", failures.join("\n"));
}

// Upstream Temperature cases 25..27 supply Text but no Start/End coordinates.
// Check their complete text/value/unit multiset instead of inventing offsets.
fn text_only(parser: &mut Parser, case: &serde_json::Value) -> bool {
    use turbo_grammar::{Output, Unit, Value};
    let text = case["Input"].as_str().unwrap();
    parser.parse(text);
    let mut mentions = Vec::new();
    parser.select(text, Output::Quantity, &mut mentions);
    let mut got: Vec<_> = mentions
        .into_iter()
        .filter_map(|m| {
            let (unit, value) = match m.value {
                Value::Quantity(q) => (q.unit(), Some(q.value().to_f64())),
                Value::Unit(u) => (u, None),
                _ => return None,
            };
            matches!(unit, Unit::Temperature(_))
                .then(|| (text[m.start as usize..m.end as usize].to_owned(), value, unit.to_string()))
        })
        .collect();
    let mut want: Vec<_> = case["Results"]
        .as_array()
        .unwrap()
        .iter()
        .map(|r| {
            assert_eq!(r["TypeName"], "temperature");
            (
                r["Text"].as_str().unwrap().to_owned(),
                r["Resolution"]["value"].as_str().map(|s| s.parse::<f64>().unwrap()),
                r["Resolution"]["unit"].as_str().unwrap().to_owned(),
            )
        })
        .collect();
    got.sort_by(|a, b| a.0.cmp(&b.0));
    want.sort_by(|a, b| a.0.cmp(&b.0));
    got == want
}
