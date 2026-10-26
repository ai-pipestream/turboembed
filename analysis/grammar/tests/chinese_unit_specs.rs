//! All unchanged pinned Chinese cases, including upstream unsupported cases.
//! Missing optional subtype fields impose no assertion: Chinese omits them even
//! for ordinary metres. Offsets, values, units and supplied ISO/subtypes are checked.
use std::path::PathBuf;
use turbo_analysis::{OffsetTable, Utf16Offset};
use turbo_grammar::unit::Weight;
use turbo_grammar::{CHINESE_UNITS, Measure, Output, Parser, Unit, Value};
fn same_number(expected: &str, got: f64) -> bool {
    expected.parse::<f64>().is_ok_and(|want| want == got || (want - got).abs() <= 1e-12 * want.abs().max(got.abs()))
}
#[test]
fn chinese_number_with_unit_specifications() {
    let mut failures = Vec::new();
    for (model, measure, count) in [
        ("Age", Measure::Age, 10),
        ("Currency", Measure::Currency, 35),
        ("Dimension", Measure::Dimension, 36),
        ("Temperature", Measure::Temperature, 10),
    ] {
        let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join(format!("tests/data/recognizers-text/Chinese/NumberWithUnit/{model}Model.json"));
        let raw = std::fs::read_to_string(path).unwrap();
        let cases: Vec<serde_json::Value> = serde_json::from_str(raw.trim_start_matches('\u{feff}')).unwrap();
        assert_eq!(cases.len(), count);
        let mut parser = Parser::new(&CHINESE_UNITS);
        let mut passing = 0;
        for (i, case) in cases.iter().enumerate() {
            let input = case["Input"].as_str().unwrap();
            let expected = case["Results"].as_array().unwrap();
            let offsets = OffsetTable::build(input);
            parser.parse(input);
            let mut mentions = Vec::new();
            parser.select(input, Output::Quantity, &mut mentions);
            let got: Vec<_> = mentions
                .iter()
                .filter_map(|m| {
                    let (u, n) = match m.value {
                        Value::Unit(u) => (u, None),
                        Value::Quantity(q) => (q.unit(), Some(q.value())),
                        _ => return None,
                    };
                    (u.measure() == measure).then_some((m, u, n))
                })
                .collect();
            let pass = got.len() == expected.len() && got.iter().zip(expected).all(|((m,u,n), e)| {
                let start = offsets.from_utf16(input, Utf16Offset(e["Start"].as_u64().unwrap() as u32)).unwrap();
                let end = offsets.from_utf16(input, Utf16Offset(e["End"].as_u64().unwrap() as u32 + 1)).unwrap();
                let res = &e["Resolution"];
                let name = res["unit"].as_str().unwrap();
                let unit_ok = name == u.to_string() || name == "Ton" && *u == Unit::Weight(Weight::MetricTon);
                let number_ok = match (res["value"].as_str(), n) {
                    (Some(want), Some(n)) => same_number(want, n.to_f64()),
                    (None, None) => true,
                    _ => false,
                };
                let subtype_ok = res["subtype"].as_str().is_none_or(|s| u.dimension().is_some_and(|d| d.name() == s));
                let iso_ok = res["isoCurrency"].as_str().is_none_or(|iso| {
                    matches!(u, Unit::Currency(c) if c.currency().is_some_and(|(code, scale)| code.as_str()==iso && scale==0))
                });
                m.start == start && m.end == end && unit_ok && number_ok && subtype_ok && iso_ok
                    && e["TypeName"].as_str().or_else(||e["Typename"].as_str()) == Some(u.measure().name())
            });
            if pass {
                passing += 1;
            } else {
                let found: Vec<_> = got
                    .iter()
                    .map(|(m, u, n)| {
                        (&input[m.start as usize..m.end as usize], u.to_string(), n.map(|n| n.to_string()))
                    })
                    .collect();
                failures.push(format!("{model} #{i}: {input:?} expected {expected:?} found {found:?}"));
            }
        }
        println!("{model}: {passing}/{count}");
    }
    assert!(failures.is_empty(), "{}", failures.join("\n"));
}
