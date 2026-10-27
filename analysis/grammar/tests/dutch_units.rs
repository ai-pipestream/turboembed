use turbo_grammar::{DUTCH_UNITS, Output, Parser, Value};
fn quantities(text: &str) -> Vec<(String, f64, String)> {
    let mut parser = Parser::new(&DUTCH_UNITS);
    parser.parse(text);
    let mut found = Vec::new();
    parser.select(text, Output::Quantity, &mut found);
    found
        .into_iter()
        .filter_map(|m| match m.value {
            Value::Quantity(q) => {
                Some((text[m.start as usize..m.end as usize].into(), q.value().to_f64(), q.unit().symbol()))
            }
            _ => None,
        })
        .collect()
}
#[test]
fn dutch_units_regression() {
    let text = "tweeëntwintig kilometer";
    assert_eq!(quantities(text), [(text.into(), 22.0, "km".into())]);
}
#[test]
fn dutch_modern_units_do_not_inherit_incorrect_upstream_conversions() {
    for (text, value, unit) in [
        ("2 ons", 200.0, "g"),
        ("2 pond", 1000.0, "g"),
        ("een halve pond", 250.0, "g"),
        ("een halve stuiver", 2.5, "cent"),
        ("een are", 1.0, "dam²"),
        ("een hectare", 1.0, "hm²"),
        ("1,5 kubieke meter", 1.5, "m³"),
        ("20 kilometer per uur", 20.0, "km/h"),
        ("500 Mb per seconde", 500.0, "Mbit/s"),
        ("2 MB/s", 2.0, "MB/s"),
        ("2 m^2", 2.0, "m²"),
        ("een halve draai", 0.5, "tr"),
        ("6-jarige leeftijd", 6.0, "years old"),
    ] {
        assert_eq!(quantities(text), [(text.into(), value, unit.into())], "{text}");
    }
}
#[test]
fn dutch_coins_keep_the_major_currency_unspecified() {
    use turbo_grammar::unit::{CurrencyUnit, Generic};
    let cent = CurrencyUnit::Generic(Generic::Cent);
    assert_eq!(cent.currency(), None);
    assert!(!cent.has_hundredths());
    for (text, value, unit) in [
        ("een kwartje", 25.0, "cent"),
        ("2 dubbeltjes", 20.0, "cent"),
        ("3 stuivers", 15.0, "cent"),
        ("100 penni", 1.0, "FIM"),
        ("een kuai en vijf mao vijf", 1.55, "CNY"),
        ("3 euro en 50 cent", 3.5, "EUR"),
        ("USD 15 en 25 cent", 15.25, "USD"),
    ] {
        assert_eq!(quantities(text), [(text.into(), value, unit.into())], "{text}");
    }
}
#[test]
fn dutch_pipeline_writes_money_and_quantities_with_reversible_offsets() {
    use turbo_analysis::{Analyzer, Document, Ix, Layer, OffsetTable};
    use turbo_grammar::annotator::{GRAMMAR_MONEY, GRAMMAR_PERCENTS, GrammarAnnotator};
    let text = "🛒 dertien euro en vijfenveertig cent; 500 Mb per seconde; een kwartje.";
    let analyzer = Analyzer::builder().step(GrammarAnnotator::dutch().bound()).build().unwrap();
    let mut docs = [Document::new(text)];
    assert!(analyzer.run(&mut docs).iter().all(Result::is_ok));
    let money = docs[0].get_key(&GRAMMAR_MONEY).unwrap();
    let quantities = docs[0].get_key(&GRAMMAR_PERCENTS).unwrap();
    money.check().unwrap();
    quantities.check().unwrap();
    assert_eq!(money.len(), 1, "ambiguous coin must not become USD or EUR");
    assert_eq!(money.amount(Ix::new(0)).to_plain_string(), "13.45");
    assert_eq!(money.get(Ix::new(0)).currency.as_str(), "EUR");
    assert_eq!(&text[money.get(Ix::new(0)).span.range()], "dertien euro en vijfenveertig cent");
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
#[test]
fn dutch_unicode_compounds_keep_the_original_source() {
    for separator in [" ", "\u{a0}", "\u{202f}", "\u{3000}", "\t"] {
        let text = format!("📏 tweee\u{308}ntwintig{separator}kilometer");
        assert_eq!(quantities(&text), [(text[5..].into(), 22.0, "km".into())]);
    }
}
#[test]
fn dutch_money_boundaries_and_unrelated_nouns() {
    for text in [
        "3 euro 5 kilometer",
        "3 euro 5 EUR",
        "3 USD50",
        "5 $10",
        "3 euro en 5 mao",
        "3 euro 100",
        "-5 jaar",
        "dollar een voorbeeld",
        "guldens een waarde",
    ] {
        assert!(!quantities(text).iter().any(|(span, _, _)| span == text), "{text}");
    }
    assert_eq!(
        quantities("3 euro 5 kilometer"),
        [("3 euro".into(), 3.0, "EUR".into()), ("5 kilometer".into(), 5.0, "km".into())]
    );
    assert_eq!(quantities("3 USD50"), [("USD50".into(), 50.0, "USD".into())]);
    assert_eq!(quantities("5 $10"), [("$10".into(), 10.0, "dollar".into())]);
    assert!(quantities("guldens een waarde").is_empty());
}

#[test]
fn unsupported_named_note_has_no_assumed_currency() {
    // Upstream Currency case 12 is explicitly NotSupported for dotnet.
    assert!(quantities("Hij is me nog een tientje schuldig.").is_empty());
}
