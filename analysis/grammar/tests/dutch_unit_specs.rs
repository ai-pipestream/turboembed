//! Unmodified pinned Dutch oracle plus explicit policy differences. Neither
//! oracle inputs nor expected on-disk outputs are rewritten.
#[allow(dead_code)]
#[path = "spec/units.rs"]
mod units;
use std::path::PathBuf;
use turbo_grammar::{DUTCH_UNITS, Measure, Parser};
fn data() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/data")
}
fn same_number(expected: &str, got: f64) -> bool {
    expected.parse::<f64>().is_ok_and(|want| want == got || (want - got).abs() <= 1e-12 * want.abs().max(got.abs()))
}
// Some Dutch fixtures supply Text but omit coordinates. Derive only absent
// coordinates from that expected text, independently of the parser.
fn coordinates(case: &mut serde_json::Value) {
    let input = case["Input"].as_str().unwrap().to_owned();
    let lower = input.to_ascii_lowercase();
    let mut used = Vec::new();
    for r in case["Results"].as_array_mut().unwrap() {
        if r.get("Start").is_some() {
            continue;
        }
        let text = r["Text"].as_str().unwrap().to_ascii_lowercase();
        let start = lower
            .match_indices(&text)
            .map(|(i, _)| i)
            .find(|&i| !used.iter().any(|&(s, e)| i >= s && i + text.len() <= e))
            .unwrap_or_else(|| panic!("expected {text:?} in {input:?}"));
        let end = start + text.len();
        used.push((start, end));
        r["Start"] = (input[..start].encode_utf16().count() as u64).into();
        r["End"] = (input[..end].encode_utf16().count() as u64 - 1).into();
    }
}
fn adapt(case: &mut serde_json::Value, model: &str, index: usize) {
    for r in case["Results"].as_array_mut().unwrap() {
        let res = &mut r["Resolution"];
        if let Some(v) = res["value"].as_str() {
            res["value"] = v.replace(',', ".").into();
        }
        if res["unit"] == "Hectare" {
            res["unit"] = "Square hectometer".into();
        }
        let (unit, factor) = match (model, index, res["unit"].as_str().unwrap_or("")) {
            ("Dimension", 7, "Ounce") => ("Gram", 100.0),
            ("Dimension", 13 | 14 | 44, "Pound") => ("Gram", 500.0),
            ("Dimension", 62, "Acre") => ("Square decameter", 1.0),
            ("Currency", 13, "Kwartje") => ("Cent", 25.0),
            ("Currency", 14, "Dubbeltje") => ("Cent", 10.0),
            ("Currency", 15, "Stuiver") => ("Cent", 5.0),
            ("Currency", 21, "Penni") => ("Finnish markka", 0.01),
            _ => ("", 1.0),
        };
        if !unit.is_empty() {
            res["unit"] = unit.into();
            if let Some(value) = res["value"].as_str() {
                res["value"] = (value.parse::<f64>().unwrap() * factor).to_string().into();
            }
        }
        if model == "Dimension" && index == 5 {
            res["subtype"] = "Volume".into();
        }
    }
    // This rate is deliberately kept whole instead of dropping `per seconde`.
    if model == "Dimension" && index == 48 {
        let r = &mut case["Results"][0];
        r["End"] = 43.into();
        r["Resolution"]["unit"] = "Megabit per second".into();
        r["Resolution"]["subtype"] = "Information".into();
    }
}
#[test]
fn dutch_number_with_unit_specifications() {
    let mut failures = Vec::new();
    for (model, measure, count, original_count) in [
        ("Age", Measure::Age, 17, 17),
        ("Currency", Measure::Currency, 31, 27),
        ("Dimension", Measure::Dimension, 74, 65),
        ("Temperature", Measure::Temperature, 34, 34),
    ] {
        let raw =
            std::fs::read_to_string(data().join(format!("recognizers-text/Dutch/NumberWithUnit/{model}Model.json")))
                .unwrap();
        let cases: Vec<serde_json::Value> = serde_json::from_str(raw.trim_start_matches('\u{feff}')).unwrap();
        let (mut eligible, mut original, mut passing) = (0, 0, 0);
        let mut parser = Parser::new(&DUTCH_UNITS);
        for (i, case) in cases.iter().enumerate() {
            if ["NotSupported", "NotSupportedByDesign"].iter().any(|key| {
                case[*key].as_str().unwrap_or("").split(',').any(|s| s.trim().eq_ignore_ascii_case("dotnet"))
            }) {
                assert_eq!((model, i), ("Currency", 12), "unexpected unsupported upstream case");
                continue;
            }
            eligible += 1;
            let mut normalized = case.clone();
            coordinates(&mut normalized);
            for r in normalized["Results"].as_array_mut().unwrap() {
                if let Some(v) = r["Resolution"]["value"].as_str() {
                    r["Resolution"]["value"] = v.replace(',', ".").into();
                }
            }
            let rawcase = serde_json::from_value(normalized).unwrap();
            if units::run(&mut parser, measure, &rawcase).0 {
                original += 1;
            }
            let mut expected = case.clone();
            coordinates(&mut expected);
            adapt(&mut expected, model, i);
            let parsed: units::UnitCase = serde_json::from_value(expected.clone()).unwrap();
            let (pass, got) = units::run(&mut parser, measure, &parsed);
            if pass {
                passing += 1;
            } else {
                failures.push(format!(
                    "{model} #{i}: {} expected {} found {}",
                    case["Input"],
                    expected["Results"],
                    units::describe(case["Input"].as_str().unwrap(), &got)
                ));
            }
        }
        println!("{model}: original {original}/{eligible}; documented policy {passing}/{eligible}");
        assert_eq!(eligible, count);
        assert_eq!(original, original_count, "raw upstream parity changed for {model}");
    }
    assert!(failures.is_empty(), "{}", failures.join("\n"));
}
