use turbo_analysis::{Analyzer, Document, Ix, Layer, OffsetTable};
use turbo_grammar::annotator::{GRAMMAR_MONEY, GRAMMAR_PERCENTS, GrammarAnnotator};
use turbo_grammar::{Grammar, ITALIAN_UNITS, Output, PORTUGUESE_UNITS, Parser, Value};

fn numbers(grammar: &'static dyn Grammar, cases: &[(&str, f64)]) {
    for &(text, expected) in cases {
        let mut parser = Parser::new(grammar);
        parser.parse(text);
        let mut matches = Vec::new();
        parser.select(text, Output::Number, &mut matches);
        let got: Vec<_> = matches
            .iter()
            .map(|m| {
                let Value::Num(n) = m.value else { panic!("number output") };
                (&text[m.start as usize..m.end as usize], n.value.to_f64())
            })
            .collect();
        assert_eq!(got, [(text, expected)], "{text}");
    }
}

#[test]
fn italian_scale_half_numbers() {
    numbers(
        &ITALIAN_UNITS,
        &[
            ("3 milioni e mezzo", 3_500_000.0),
            ("un milione e mezzo", 1_500_000.0),
            ("un miliardo e mezzo", 1_500_000_000.0),
            ("3 miliardi e mezzo", 3_500_000_000.0),
            ("meno un milione e mezzo", -1_500_000.0),
            ("un miliardo e mezzo milione", 1_000_500_000.0),
            ("tre e mezzo milioni", 3_500_000.0),
            ("trecento e mezzo", 300.5),
            ("venti e mezzo", 20.5),
            ("3 e mezzo", 3.5),
            ("un milione e tre e mezzo", 1_000_003.5),
        ],
    );
}

#[test]
fn portuguese_scale_half_numbers() {
    numbers(
        &PORTUGUESE_UNITS,
        &[
            ("2 milhões e meio", 2_500_000.0),
            ("três milhões e meio", 3_500_000.0),
            ("um milhão e meio", 1_500_000.0),
            ("um bilhão e meio", 1_500_000_000.0),
            ("menos um milhão e meio", -1_500_000.0),
            ("um trilhão e meio milhão", 1_000_000_500_000.0),
            ("três e meio milhões", 3_500_000.0),
            ("trezentos e meio", 300.5),
            ("vinte e meio", 20.5),
            ("3 e meio", 3.5),
            ("um milhão e três e meio", 1_000_003.5),
        ],
    );
}

fn pipeline(grammar: GrammarAnnotator, money_text: &str, quantity_text: &str) {
    let analyzer = Analyzer::builder().step(grammar.bound()).build().unwrap();
    for space in [" ", "\u{a0}", "\u{202f}", "\u{3000}", "\t"] {
        let money_text = money_text.replace(' ', space);
        let quantity_text = quantity_text.replace(' ', space);
        let text = format!("🛒 {money_text}; {quantity_text}.");
        let mut docs = [Document::new(text.as_str())];
        assert!(analyzer.run(&mut docs).iter().all(Result::is_ok));
        let money = docs[0].get_key(&GRAMMAR_MONEY).unwrap();
        let quantities = docs[0].get_key(&GRAMMAR_PERCENTS).unwrap();
        money.check().unwrap();
        quantities.check().unwrap();
        assert_eq!(money.len(), 1, "{text}");
        assert_eq!(quantities.len(), 1, "{text}");
        assert_eq!(money.amount(Ix::new(0)).to_plain_string(), "3500000");
        assert_eq!(money.get(Ix::new(0)).currency.as_str(), "EUR");
        assert_eq!(quantities.value(Ix::new(0)).to_f64(), 1_500_000_000.0);
        assert_eq!(&text[money.get(Ix::new(0)).span.range()], money_text);
        let unit_span = quantities.spans.get(Ix::new(0));
        assert_eq!(&text[unit_span.range()], quantity_text);
        let offsets = OffsetTable::build(&text);
        for span in [money.get(Ix::new(0)).span, unit_span] {
            for byte in [span.start, span.end] {
                assert_eq!(offsets.from_utf16(&text, offsets.to_utf16(&text, byte)).unwrap(), byte);
                assert_eq!(offsets.from_code_point(&text, offsets.to_code_point(&text, byte)).unwrap(), byte);
            }
        }
    }
}

#[test]
fn italian_scale_half_pipeline_offsets() {
    pipeline(GrammarAnnotator::italian(), "3 milioni e mezzo di euro", "un miliardo e mezzo di metri");
}

#[test]
fn portuguese_scale_half_pipeline_offsets() {
    pipeline(GrammarAnnotator::portuguese(), "3 milho\u{303}es e meio de euros", "um bilhão e meio de metros");
}
