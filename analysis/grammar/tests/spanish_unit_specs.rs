//! The unchanged MIT-licensed Spanish NumberWithUnit corpus at the pinned
//! Recognizers-Text revision. Only localized resolution labels and decimal
//! commas are translated for the shared, exact-span comparator.
#[allow(dead_code)]
#[path = "spec/units.rs"]
mod units;
use std::path::PathBuf;
use turbo_grammar::{Measure, Parser, SPANISH_UNITS};
fn data() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/data")
}
fn same_number(expected: &str, got: f64) -> bool {
    expected.parse::<f64>().is_ok_and(|want| want == got || (want - got).abs() <= 1e-12 * want.abs().max(got.abs()))
}
fn name(s: &str, model: &str) -> &'static str {
    match s {
        "Año" => "Year",
        "Mes" => "Month",
        "Semana" => "Week",
        "Día" => "Day",
        "Dólar" => "Dollar",
        "Dólar estadounidense" => "United States dollar",
        "Dólar canadiense" => "Canadian dollar",
        "Dólar australiano" => "Australian dollar",
        "Dólar neozelandés" => "New Zealand dollar",
        "Dólar de Hong Kong" => "Hong Kong dollar",
        "Marco finlandés" => "Finnish markka",
        "Florín neerlandés" => "Netherlands guilder",
        "Franco" => "Franc",
        "Franco suizo" => "Swiss franc",
        "Peso mexicano" => "Mexican peso",
        "Euro" => "Euro",
        "Libra" if model == "Currency" => "Pound",
        "Centavo" => "Centavo",
        "Penique" => "Penny",
        "Yen" => "Japanese yen",
        "Peseta" => "Peseta",
        "Yuan chino" => "Chinese yuan",
        "Mililitro" => "Milliliter",
        "Pulgada" => "Inch",
        "Milla" => "Mile",
        "Pie" => "Foot",
        "Onza" => "Ounce",
        "Acre" => "Acre",
        "Galón" => "Gallon",
        "Libra" => "Pound",
        "Pie cúbico" => "Cubic foot",
        "Metro cuadrado" => "Square meter",
        "Tonelada" => "Ton",
        "Tonelada métrica" => "Metric ton",
        "bit" => "Bit",
        "Milla por hora" => "Mile per hour",
        "Metro" => "Meter",
        "Kilómetro" => "Kilometer",
        "Milímetro" => "Millimeter",
        "Kilómetro cuadrado" => "Square kilometer",
        "Centímetro cúbico" => "Cubic centimeter",
        "Barril" => "Barrel",
        "Degree" => "Degree",
        "Radian" => "Radian",
        "Turn" => "Turn",
        "Grado" => "Degree",
        "Grado Celsius" => "C",
        "Grado Fahrenheit" => "F",
        "Kelvin" => "K",
        _ => panic!("unmapped upstream label {s}"),
    }
}
#[test]
fn spanish_number_with_unit_specifications() {
    let mut failures = Vec::new();
    for (model, measure, expected_count) in [
        ("Age", Measure::Age, 19),
        ("Currency", Measure::Currency, 122),
        ("Dimension", Measure::Dimension, 54),
        ("Temperature", Measure::Temperature, 45),
    ] {
        let path = data().join(format!("recognizers-text/Spanish/NumberWithUnit/{model}Model.json"));
        let raw = std::fs::read_to_string(path).unwrap();
        let mut cases: Vec<serde_json::Value> = serde_json::from_str(raw.trim_start_matches('\u{feff}')).unwrap();
        let (mut eligible, mut passing) = (0, 0);
        let mut parser = Parser::new(&SPANISH_UNITS);
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
                    resolution["unit"] = name(label, model).into();
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
