use turbo_analysis::{Ix, Layer, ViewId};
use turbo_grammar::ENGLISH_UNITS;
use turbo_grammar::annotator::GrammarAnnotator;
use turbo_grammar::engine::{Output, Parser};
use turbo_grammar::unit::{CurrencyUnit, Generic, Unit};
use turbo_grammar::value::Value;

const MONEY_CASES: [(&str, &str, &str); 3] = [
    ("it costs 5 pounds 8", "5.08", "5 pounds 8"),
    ("it cost 5 pounds 8.", "5.08", "5 pounds 8"),
    ("he paid 2 pounds 6 and left", "2.06", "2 pounds 6"),
];

#[test]
fn pound_money_context_keeps_the_currency_parser_reading() {
    let got: Vec<_> = MONEY_CASES
        .into_iter()
        .map(|(text, _, _)| {
            let mut parser = Parser::new(&ENGLISH_UNITS);
            parser.parse(text);
            let mut mentions = Vec::new();
            parser.select(text, Output::Quantity, &mut mentions);
            assert_eq!(mentions.len(), 1, "{text}");
            let Value::Quantity(q) = mentions[0].value else { panic!("an amount in {text}") };
            (q.unit(), q.value().to_string(), text[mentions[0].start as usize..mentions[0].end as usize].to_owned())
        })
        .collect();
    let expected: Vec<_> = MONEY_CASES
        .into_iter()
        .map(|(_, amount, span)| {
            (Unit::Currency(CurrencyUnit::Generic(Generic::Pound)), amount.to_owned(), span.to_owned())
        })
        .collect();
    assert_eq!(got, expected);
}

#[test]
fn pound_money_context_reaches_money_without_a_weight() {
    let got: Vec<_> = MONEY_CASES
        .into_iter()
        .map(|(text, _, _)| {
            let found = GrammarAnnotator::english().find(text, ViewId::SOURCE);
            assert_eq!(found.money.check(), Ok(()));
            assert_eq!(found.quantities.check(), Ok(()));
            let money: Vec<_> = (0..found.money.len())
                .map(|i| {
                    let at = Ix::new(i as u32);
                    let row = found.money.get(at);
                    (
                        row.currency.as_str().to_owned(),
                        found.money.amount(at).to_plain_string(),
                        text[row.span.range()].to_owned(),
                    )
                })
                .collect();
            (found.quantities.len(), money)
        })
        .collect();
    let expected: Vec<_> = MONEY_CASES
        .into_iter()
        .map(|(_, amount, span)| (0, vec![("GBP".to_owned(), amount.to_owned(), span.to_owned())]))
        .collect();
    assert_eq!(got, expected);
}

#[test]
fn bare_imperial_weight_and_length_readings_remain_available() {
    for (text, amount, unit) in [
        ("a baby of 5 pounds 8", "5.5", "lb"),
        ("she weighs 10 stone 7", "10.5", "st"),
        ("he is 6 foot 3 tall", "6.25", "ft"),
        ("it costs 5 lb 8", "5.5", "lb"),
    ] {
        let found = GrammarAnnotator::english().find(text, ViewId::SOURCE);
        assert_eq!(found.money.len(), 0, "{text}");
        assert_eq!(found.quantities.len(), 1, "{text}");
        assert_eq!(found.quantities.value(Ix::new(0)).to_plain_string(), amount, "{text}");
        assert_eq!(found.quantities.unit(Ix::new(0)), unit, "{text}");
    }
}
