//! The unchanged MIT-licensed Portuguese NumberWithUnit corpus at the pinned
//! Recognizers-Text revision. Only localized resolution labels and decimal
//! commas are translated for the shared, exact-span comparator.
#[allow(dead_code)]
#[path = "spec/units.rs"]
mod units;
use std::path::PathBuf;
use turbo_grammar::{Measure, PORTUGUESE_UNITS, Parser};
fn data() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/data")
}
fn same_number(expected: &str, got: f64) -> bool {
    expected.parse::<f64>().is_ok_and(|want| want == got || (want - got).abs() <= 1e-12 * want.abs().max(got.abs()))
}
fn name(s: &str) -> &'static str {
    match s {
        "Ano" => "Year",
        "Dia" => "Day",
        "Mês" => "Month",
        "Semana" => "Week",
        "Bitcoin" => "Bitcoin",
        "Centavo" => "Centavo",
        "Dólar" => "Dollar",
        "Dólar australiano" => "Australian dollar",
        "Dólar canadense" => "Canadian dollar",
        "Dólar de Hong Kong" => "Hong Kong dollar",
        "Dólar estadunidense" => "United States dollar",
        "Dólar neozelandês" => "New Zealand dollar",
        "Euro" => "Euro",
        "Florim holandês" => "Netherlands guilder",
        "Franco" => "Franc",
        "Franco suíço" => "Swiss franc",
        "Libra" => "Pound",
        "Marco finlandês" => "Finnish markka",
        "Peseta" => "Peseta",
        "Peso mexicano" => "Mexican peso",
        "Pêni" => "Penny",
        "Yen" => "Japanese yen",
        "Yuan chinês" => "Chinese yuan",
        "Acre" => "Acre",
        "Barril" => "Barrel",
        "Centímetro cúbico" => "Cubic centimeter",
        "Degree" => "Degree",
        "Galão" => "Gallon",
        "Metro" => "Meter",
        "Metro quadrado" => "Square meter",
        "Micrômetro" => "Micrometer",
        "Milha" => "Mile",
        "Milha por hora" => "Mile per hour",
        "Mililitro" => "Milliliter",
        "Milímetro" => "Millimeter",
        "Onça" => "Ounce",
        "Polegada" => "Inch",
        "Pé" => "Foot",
        "Pé cúbico" => "Cubic foot",
        "Quilômetro" => "Kilometer",
        "Quilômetro quadrado" => "Square kilometer",
        "Radian" => "Radian",
        "Tonelada" => "Ton",
        "Tonelada métrica" => "Metric ton",
        "Turn" => "Turn",
        "bit" => "Bit",
        "Grau" => "Degree",
        "Grau Celsius" => "C",
        "Grau Fahrenheit" => "F",
        "Kelvin" => "K",
        _ => panic!("unmapped upstream label {s}"),
    }
}
#[test]
fn portuguese_number_with_unit_specifications() {
    let mut failures = Vec::new();
    for (model, measure, expected_count) in [
        ("Age", Measure::Age, 18),
        ("Currency", Measure::Currency, 123),
        ("Dimension", Measure::Dimension, 65),
        ("Temperature", Measure::Temperature, 47),
    ] {
        let path = data().join(format!("recognizers-text/Portuguese/NumberWithUnit/{model}Model.json"));
        let raw = std::fs::read_to_string(path).unwrap();
        let mut cases: Vec<serde_json::Value> = serde_json::from_str(raw.trim_start_matches('\u{feff}')).unwrap();
        let (mut eligible, mut passing) = (0, 0);
        let mut parser = Parser::new(&PORTUGUESE_UNITS);
        for (i, case) in cases.iter_mut().enumerate() {
            if ["NotSupported", "NotSupportedByDesign"].iter().any(|key| {
                case[*key].as_str().unwrap_or("").split(',').any(|x| x.trim().eq_ignore_ascii_case("dotnet"))
            }) {
                continue;
            }
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
