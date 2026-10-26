use turbo_grammar::unit::{Length, Temperature};
use turbo_grammar::{Output, Parser, SPANISH_UNITS, Unit, Value};

#[test]
fn spanish_units_keep_source_spans_and_values() {
    let text = "📏 Son veintidós kilómetros y 37ºC.";
    let mut parser = Parser::new(&SPANISH_UNITS);
    parser.parse(text);
    let mut got = Vec::new();
    parser.select(text, Output::Quantity, &mut got);
    let got: Vec<_> = got
        .iter()
        .filter_map(|m| match m.value {
            Value::Quantity(q) => Some((&text[m.start as usize..m.end as usize], q.value().to_f64(), q.unit())),
            _ => None,
        })
        .collect();
    assert_eq!(
        got,
        [
            ("veintidós kilómetros", 22.0, Unit::Length(Length::Kilometer)),
            ("37ºC", 37.0, Unit::Temperature(Temperature::Celsius))
        ]
    );
}

#[test]
fn spanish_pipeline_writes_money_and_quantities() {
    use turbo_analysis::{Analyzer, Document, Ix, Layer, OffsetTable};
    use turbo_grammar::annotator::{GRAMMAR_MONEY, GRAMMAR_PERCENTS, GrammarAnnotator};
    let text = "🛒 Cuesta trece euros y cuarenta y cinco céntimos; 10 Mb/s y 20 dólares por hora.";
    let analyzer = Analyzer::builder().step(GrammarAnnotator::spanish().bound()).build().unwrap();
    let mut docs = [Document::new(text)];
    assert!(analyzer.run(&mut docs).iter().all(Result::is_ok));
    let money = docs[0].get_key(&GRAMMAR_MONEY).unwrap();
    let quantities = docs[0].get_key(&GRAMMAR_PERCENTS).unwrap();
    money.check().unwrap();
    quantities.check().unwrap();
    assert_eq!(money.len(), 2);
    assert_eq!(money.amount(Ix::new(0)).to_plain_string(), "13.45");
    assert_eq!(money.get(Ix::new(0)).currency.as_str(), "EUR");
    assert_eq!(money.amount(Ix::new(1)).to_plain_string(), "20");
    assert_eq!(money.get(Ix::new(1)).currency.as_str(), "USD");
    assert_eq!(&text[money.get(Ix::new(1)).span.range()], "20 dólares por hora");
    assert_eq!(quantities.len(), 1);
    assert_eq!(quantities.unit(Ix::new(0)), "Mbit/s");
    let offsets = OffsetTable::build(text);
    for span in money.spans.iter().map(|(_, s)| s).chain(quantities.spans.iter().map(|(_, s)| s)) {
        for byte in [span.start, span.end] {
            assert_eq!(offsets.from_utf16(text, offsets.to_utf16(text, byte)).unwrap(), byte);
            assert_eq!(offsets.from_code_point(text, offsets.to_code_point(text, byte)).unwrap(), byte);
        }
    }
}

fn quantities(text: &str) -> Vec<(String, f64, String)> {
    let mut parser = Parser::new(&SPANISH_UNITS);
    parser.parse(text);
    let mut got = Vec::new();
    parser.select(text, Output::Quantity, &mut got);
    got.into_iter()
        .filter_map(|m| match m.value {
            Value::Quantity(q) => {
                Some((text[m.start as usize..m.end as usize].to_owned(), q.value().to_f64(), q.unit().symbol()))
            }
            _ => None,
        })
        .collect()
}

#[test]
fn spanish_compounds_rates_and_word_senses() {
    for (text, value, unit) in [
        ("6 pies y 3 pulgadas", 6.25, "ft"),
        ("5 libras y 8 onzas", 5.5, "lb"),
        ("2,5 kilómetros por hora", 2.5, "km/h"),
        ("3 metros cuadrados", 3.0, "m²"),
        ("3 toneladas", 3.0, "t"),
        ("3 metros cúbicos", 3.0, "m³"),
        ("10 MB/s", 10.0, "MB/s"),
        ("10 Mb/s", 10.0, "Mbit/s"),
        ("25 euros con 50", 25.5, "EUR"),
        ("-5 grados Celsius", -5.0, "°C"),
        ("media vuelta", 0.5, "tr"),
    ] {
        assert_eq!(quantities(text), [(text.into(), value, unit.into())], "{text}");
    }
    assert_eq!(quantities("girar 45 grados"), [("45 grados".into(), 45.0, "°".into())]);
    assert_eq!(quantities("la temperatura es 45 grados"), [("45 grados".into(), 45.0, "deg".into())]);
}

#[test]
fn spanish_refuses_cross_currency_compounds_and_false_temperature_marks() {
    for text in [
        "3 euros con 40 peniques",
        "3 libras con 40 centavos",
        "3 euros 100",
        "3 euros 5 personas",
        "5 metros por 0 segundos",
        "-5 años",
        "37oc",
        "37 o C",
    ] {
        assert!(!quantities(text).iter().any(|(span, _, _)| span == text), "{text}");
    }
    for text in ["3º", "3.º", "8 o 9", "37oc", "oc", "of", "ok"] {
        assert!(quantities(text).is_empty(), "{text}");
    }
}

#[test]
fn unicode_whitespace_and_decomposed_accents_keep_exact_spans() {
    for separator in [" ", "\u{a0}", "\u{202f}", "\u{3000}", "\t"] {
        let text = format!("📏 veintido\u{301}s{separator}kilo\u{301}metros");
        let values = quantities(&text);
        assert_eq!(values, [(text[5..].into(), 22.0, "km".into())]);
    }
}

#[test]
fn unrelated_counts_and_glued_signs_are_not_currency_hundredths() {
    for text in ["3 euros 5 kilómetros", "3 euros 5 EUR", "3 USD50", "5 $10"] {
        assert!(!quantities(text).iter().any(|(span, _, _)| span == text), "{text}");
    }
    assert_eq!(
        quantities("3 euros 5 kilómetros"),
        [("3 euros".into(), 3.0, "EUR".into()), ("5 kilómetros".into(), 5.0, "km".into())]
    );
    assert_eq!(
        quantities("3 euros 5 EUR"),
        [("3 euros".into(), 3.0, "EUR".into()), ("5 EUR".into(), 5.0, "EUR".into())]
    );
    assert_eq!(quantities("3 USD50"), [("USD50".into(), 50.0, "USD".into())]);
    assert_eq!(quantities("5 $10"), [("$10".into(), 10.0, "dollar".into())]);
}

#[test]
fn currency_nouns_do_not_consume_following_articles() {
    for text in ["euros un ejemplo", "florines una moneda", "dólares un recuerdo"] {
        assert!(quantities(text).is_empty(), "{text}: {:?}", quantities(text));
    }
}
