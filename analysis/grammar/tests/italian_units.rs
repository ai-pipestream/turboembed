use turbo_grammar::{ITALIAN_UNITS, Output, Parser, Value};

fn quantities(text: &str) -> Vec<(String, f64, String)> {
    let mut parser = Parser::new(&ITALIAN_UNITS);
    parser.parse(text);
    let mut out = Vec::new();
    parser.select(text, Output::Quantity, &mut out);
    out.into_iter()
        .filter_map(|m| match m.value {
            Value::Quantity(q) => {
                Some((text[m.start as usize..m.end as usize].into(), q.value().to_f64(), q.unit().symbol()))
            }
            _ => None,
        })
        .collect()
}

#[test]
fn italian_native_quantity_regression() {
    assert_eq!(quantities("ventidue chilometri"), [("ventidue chilometri".into(), 22.0, "km".into())]);
}

#[test]
fn fused_cardinals_keep_amounts_and_full_spans() {
    for (text, value, unit) in [
        ("centoventitré metri", 123.0, "m"),
        ("centottanta metri", 180.0, "m"),
        ("centottantuno metri", 181.0, "m"),
        ("duecentotto metri", 208.0, "m"),
        ("duemilacento euro", 2100.0, "EUR"),
        ("milleventuno euro", 1021.0, "EUR"),
        ("novecentonovantanovemilanovecentonovantanove euro", 999999.0, "EUR"),
        ("due miliardi di euro", 2_000_000_000.0, "EUR"),
        ("due bilioni di euro", 2_000_000_000_000.0, "EUR"),
        ("0,9 trilioni di piedi cubi", 9e17, "ft³"),
    ] {
        assert_eq!(quantities(text), [(text.into(), value, unit.into())], "{text}");
    }
    for text in ["duemilano metri", "centometrista", "ventilatore", "duetre metri"] {
        assert!(quantities(text).is_empty(), "{text}: {:?}", quantities(text));
    }
}

#[test]
fn compound_measures_rates_and_minor_units() {
    for (text, value, unit) in [
        ("6 piedi e 3 pollici", 6.25, "ft"),
        ("5 libbre e 8 once", 5.5, "lb"),
        ("3 metri quadrati", 3.0, "m²"),
        ("188 miglia all'ora", 188.0, "mi/h"),
        ("188 miglia all’ora", 188.0, "mi/h"),
        ("20 euro per ora", 20.0, "EUR/h"),
        ("10 MB/s", 10.0, "MB/s"),
        ("10 Mb/s", 10.0, "Mbit/s"),
        ("tredici euro e quarantacinque centesimi", 13.45, "EUR"),
        ("15 dollari e 15", 15.15, "dollar"),
        ("-5 gradi Celsius", -5.0, "°C"),
        ("mezzo giro", 0.5, "tr"),
        ("470 pence", 470.0, "pence"),
    ] {
        assert_eq!(quantities(text), [(text.into(), value, unit.into())], "{text}");
    }
}

#[test]
fn currency_and_unit_boundaries_do_not_capture_prose() {
    for text in ["euro un esempio", "dollari una parola", "sterline uno schema", "l'amministrazione", "37 o C", "3º"] {
        assert!(quantities(text).is_empty(), "{text}: {:?}", quantities(text));
    }
    assert_eq!(quantities("$ 5.000 in meno"), [("$ 5.000".into(), 5000.0, "dollar".into())]);
    assert_eq!(quantities("prezzo di 5 libbre"), [("5 libbre".into(), 5.0, "lb".into())]);
    for text in ["3 euro e 40 pence", "3 euro 5 persone", "3 euro 100", "-5 anni"] {
        assert!(!quantities(text).iter().any(|(span, _, _)| span == text), "{text}");
    }
}

#[test]
fn real_pipeline_keeps_unicode_source_offsets() {
    use turbo_analysis::{Analyzer, Document, Ix, Layer, OffsetTable};
    use turbo_grammar::annotator::{GRAMMAR_MONEY, GRAMMAR_PERCENTS, GrammarAnnotator};
    let analyzer = Analyzer::builder().step(GrammarAnnotator::italian().bound()).build().unwrap();
    for space in [" ", "\u{a0}", "\u{202f}", "\u{3000}", "\t"] {
        let text = format!("🛒 centoventitre\u{301}{space}euro; 37ºC e 10 Mb/s.");
        let mut docs = [Document::new(text.as_str())];
        assert!(analyzer.run(&mut docs).iter().all(Result::is_ok));
        let money = docs[0].get_key(&GRAMMAR_MONEY).unwrap();
        let units = docs[0].get_key(&GRAMMAR_PERCENTS).unwrap();
        money.check().unwrap();
        units.check().unwrap();
        assert_eq!(money.len(), 1);
        assert_eq!(units.len(), 2);
        assert_eq!(money.amount(Ix::new(0)).to_plain_string(), "123");
        assert_eq!(money.get(Ix::new(0)).currency.as_str(), "EUR");
        assert_eq!(&text[money.get(Ix::new(0)).span.range()], format!("centoventitre\u{301}{space}euro"));
        let offsets = OffsetTable::build(&text);
        for span in money.spans.iter().map(|(_, s)| s).chain(units.spans.iter().map(|(_, s)| s)) {
            for byte in [span.start, span.end] {
                assert_eq!(offsets.from_utf16(&text, offsets.to_utf16(&text, byte)).unwrap(), byte);
                assert_eq!(offsets.from_code_point(&text, offsets.to_code_point(&text, byte)).unwrap(), byte);
            }
        }
    }
}
