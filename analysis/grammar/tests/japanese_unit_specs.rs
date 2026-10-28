//! Unchanged pinned Japanese NumberWithUnit fixtures, including entries marked
//! unsupported upstream. Two wrong upstream labels remain explicit differences.
#[allow(dead_code)]
#[path = "spec/units.rs"]
mod units;
use std::path::PathBuf;
use turbo_grammar::{JAPANESE_UNITS, Measure, Parser};
fn data() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/data")
}
fn same_number(expected: &str, got: f64) -> bool {
    expected.parse::<f64>().is_ok_and(|want| want == got || (want - got).abs() <= 1e-12 * want.abs().max(got.abs()))
}
#[test]
fn japanese_all_number_with_unit_specifications() {
    let mut failures = Vec::new();
    let mut parity = 0;
    let mut policies = 0;
    let mut parser = Parser::new(&JAPANESE_UNITS);
    for (model, measure, count) in [
        ("Age", Measure::Age, 18),
        ("Currency", Measure::Currency, 68),
        ("Dimension", Measure::Dimension, 64),
        ("Temperature", Measure::Temperature, 38),
    ] {
        let raw =
            std::fs::read_to_string(data().join(format!("recognizers-text/Japanese/NumberWithUnit/{model}Model.json")))
                .unwrap();
        let cases: Vec<serde_json::Value> = serde_json::from_str(raw.trim_start_matches('\u{feff}')).unwrap();
        assert_eq!(cases.len(), count);
        for (i, case) in cases.iter().enumerate() {
            let mut native = case.clone();
            if model == "Dimension" {
                for item in native["Results"].as_array_mut().unwrap() {
                    let name = item["Resolution"]["unit"].as_str().unwrap();
                    let subtype = match name {
                        "Pound" | "Ounce" | "Ton" | "Metric ton" | "Kilogram" => "Weight",
                        "Milliliter" | "Gallon" | "Cubic foot" => "Volume",
                        "Acre" | "Square meter" | "Square kilometer" | "Mu" => "Area",
                        "Bit" | "Megabit" => "Information",
                        "Mile per hour" | "Meter per second" | "Kilometer per hour" => "Speed",
                        _ => "Length",
                    };
                    item["Resolution"]["subtype"] = subtype.into();
                }
            }
            let parsed: units::UnitCase = serde_json::from_value(native.clone()).unwrap();
            let (pass, got) = units::run(&mut parser, measure, &parsed);
            let divergence = match (model, i + 1) {
                ("Age", 14) => Some("Year"),
                ("Dimension", 46) => Some("Megabyte"),
                _ => None,
            };
            if let Some(label) = divergence {
                assert!(!pass, "wrong upstream unit unexpectedly passed {model} {}", i + 1);
                native["Results"][0]["Resolution"]["unit"] = label.into();
                let native_case = serde_json::from_value(native).unwrap();
                let (correct, _) = units::run(&mut parser, measure, &native_case);
                if correct {
                    policies += 1;
                } else {
                    failures.push(format!(
                        "{model} #{} native-policy {:?}: {}",
                        i + 1,
                        case["Input"],
                        units::describe(case["Input"].as_str().unwrap(), &got)
                    ));
                }
            } else if pass {
                parity += 1;
            } else {
                failures.push(format!(
                    "{model} #{} {:?}: {}",
                    i + 1,
                    case["Input"],
                    units::describe(case["Input"].as_str().unwrap(), &got)
                ));
            }
        }
    }
    assert!(failures.is_empty(), "parity {parity}/186, native policy {policies}/2\n{}", failures.join("\n"));
    assert_eq!((parity, policies), (186, 2));
}
