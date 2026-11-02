use turbo_analysis::{Analyzer, Document, Ix, Layer, ViewId};

use super::*;

#[test]
fn layers_hold_what_the_grammar_finds() {
    let text = "minus five percent of the second batch, twenty-one and a half, 1.5k";
    let f = GrammarAnnotator::english().find(text, ViewId::SOURCE);
    assert_eq!(f.numbers.check(), Ok(()));
    assert_eq!(f.percents.check(), Ok(()));
    assert_eq!(f.ordinals.check(), Ok(()));
    let numbers: Vec<(String, String)> = (0..f.numbers.len())
        .map(|i| {
            let s = f.numbers.spans.get(Ix::new(i as u32));
            (text[s.range()].to_string(), f.numbers.value(Ix::new(i as u32)).to_plain_string())
        })
        .collect();
    assert_eq!(
        numbers,
        [
            ("minus five".to_string(), "-5".to_string()),
            ("twenty-one and a half".into(), "21.5".into()),
            ("1.5k".into(), "1500".into()),
        ]
    );
    assert!(f.numbers.negative[0]);
    assert_eq!(&text[f.numbers.numeral(Ix::new(0)).range()], "minus five");

    let signed_text = "-42 and \u{2212}42";
    let signed = GrammarAnnotator::english().find(signed_text, ViewId::SOURCE);
    assert_eq!(signed.numbers.check(), Ok(()));
    assert_eq!(&signed_text[signed.numbers.numeral(Ix::new(0)).range()], "42");
    assert_eq!(&signed_text[signed.numbers.numeral(Ix::new(1)).range()], "42");

    let es_text = "menos cinco y -12";
    let es_found = GrammarAnnotator::spanish().find(es_text, ViewId::SOURCE);
    assert_eq!(es_found.numbers.check(), Ok(()));
    assert_eq!(&es_text[es_found.numbers.numeral(Ix::new(0)).range()], "menos cinco");
    assert_eq!(&es_text[es_found.numbers.numeral(Ix::new(1)).range()], "12");
    assert_eq!(f.percents.len(), 1);
    assert_eq!(f.percents.value(Ix::new(0)).to_plain_string(), "-5");
    assert!(f.percents.is_percent(Ix::new(0)));
    assert_eq!(text[f.percents.spans.get(Ix::new(0)).range()].to_string(), "minus five percent");
    assert_eq!(f.ordinals.len(), 1);
    assert_eq!(f.ordinals.get(Ix::new(0)).base, OrdinalBase::Start);
    assert_eq!(f.ordinals.get(Ix::new(0)).offset, 2);
}

#[test]
fn large_and_fractional_values_fill_the_columns() {
    let text = "1e10 and two thirds";
    let f = GrammarAnnotator::english().find(text, ViewId::SOURCE);
    assert_eq!(f.numbers.check(), Ok(()));
    assert_eq!(f.numbers.coefficient.get(0), "10000000000");
    assert_eq!(f.numbers.scale[0], 0);
    assert_eq!(f.numbers.scale[1], 34);
    assert!(f.numbers.value(Ix::new(1)).to_plain_string().starts_with("0.6666"));
}

#[test]
fn a_pipeline_writes_the_named_instances() {
    let analyzer = Analyzer::builder().step(GrammarAnnotator::english().bound()).build().unwrap();
    let mut docs = vec![Document::new("the next three hundred and five percent, 2 GB for $5")];
    let results = analyzer.run(&mut docs);
    assert!(results.iter().all(Result::is_ok), "{results:?}");
    let doc = &docs[0];
    assert!(doc.get::<Numbers>().is_none(), "the default numbers instance is turbo-numeric's");
    assert_eq!(doc.get_key(&GRAMMAR_NUMBERS).unwrap().len(), 3);
    assert_eq!(doc.get_key(&GRAMMAR_PERCENTS).unwrap().len(), 2);
    assert!(doc.get::<Money>().is_none(), "the default money instance is turbo-numeric's");
    assert_eq!(doc.get_key(&GRAMMAR_MONEY).unwrap().len(), 1);
    let ordinals = doc.get::<Ordinals>().unwrap();
    assert_eq!(ordinals.get(Ix::new(0)).base, OrdinalBase::Current);
}

#[test]
fn spanish_numbers_take_the_decimal_comma() {
    let text = "1.234,56 y veintiún";
    let f = GrammarAnnotator::spanish().find(text, ViewId::SOURCE);
    assert_eq!(f.numbers.check(), Ok(()));
    assert_eq!(f.numbers.len(), 2);
    assert_eq!(f.numbers.value(Ix::new(0)).to_plain_string(), "1234.56");
    assert_eq!(f.numbers.notation[0], NumberNotation::LatinEu);
    assert_eq!(&text[f.numbers.spans.get(Ix::new(1)).range()], "veintiún");
    assert_eq!(f.numbers.value(Ix::new(1)).to_plain_string(), "21");
}

#[test]
fn quantities_carry_their_unit_symbols() {
    let text = "a 10 km run at minus 40 degrees, 5% of it uphill";
    let f = GrammarAnnotator::english().find(text, ViewId::SOURCE);
    assert_eq!(f.quantities.check(), Ok(()));
    let got: Vec<(String, String, String)> = (0..f.quantities.len())
        .map(|i| {
            let i = Ix::new(i as u32);
            let s = f.quantities.spans.get(i);
            (text[s.range()].to_string(), f.quantities.value(i).to_plain_string(), f.quantities.unit(i).to_string())
        })
        .collect();
    assert_eq!(got.len(), 2, "{got:?}");
    assert_eq!(got[0].0, "10 km");
    assert_eq!(got[0].1, "10");
    assert_eq!(got[1].0, "minus 40 degrees");
    assert_eq!(got[1].1, "-40");
    let merged = merge(&f.percents, &f.quantities);
    assert_eq!(merged.len(), 3);
    assert!(merged.is_percent(Ix::new(2)));
}

#[test]
fn rates_reach_the_quantities_and_money_layers() {
    let text = "a 10 Mb/s link at $5 per hour";
    let f = GrammarAnnotator::english().find(text, ViewId::SOURCE);
    assert_eq!(f.quantities.check(), Ok(()));
    assert_eq!(f.quantities.len(), 1, "quantities in {text}");
    assert_eq!(f.quantities.unit(Ix::new(0)), "Mbit/s");
    assert_eq!(&text[f.quantities.spans.get(Ix::new(0)).range()], "10 Mb/s");
    assert_eq!(f.money.check(), Ok(()));
    assert_eq!(f.money.len(), 1, "money in {text}");
    assert_eq!(f.money.get(Ix::new(0)).currency.as_str(), "USD");
    assert_eq!(f.money.amount(Ix::new(0)).to_plain_string(), "5");
    assert_eq!(f.money.per(Ix::new(0)), Some("h"));
    assert_eq!(&text[f.money.spans.get(Ix::new(0)).range()], "$5 per hour", "the span carries the rate");
}

#[test]
fn money_layer_distinguishes_hourly_daily_and_plain_amounts_across_languages() {
    let en = GrammarAnnotator::english().find("USD 5 per hour, USD 5 per day, and USD 5", ViewId::SOURCE);
    assert_eq!(en.money.check(), Ok(()));
    let en_got: Vec<(&str, String, Option<&str>)> = (0..en.money.len())
        .map(|i| {
            let ix = Ix::new(i as u32);
            (en.money.currency[i].as_str(), en.money.amount(ix).to_plain_string(), en.money.per(ix))
        })
        .collect();
    assert_eq!(
        en_got,
        [("USD", "5".to_string(), Some("h")), ("USD", "5".to_string(), Some("d")), ("USD", "5".to_string(), None),]
    );

    let es_text = "5 dólares por hora, 5 dólares por día y 5 dólares";
    let es = GrammarAnnotator::spanish().find(es_text, ViewId::SOURCE);
    assert_eq!(es.money.check(), Ok(()));
    let es_got: Vec<(&str, &str, String, Option<&str>)> = (0..es.money.len())
        .map(|i| {
            let ix = Ix::new(i as u32);
            let m = es.money.get(ix);
            (
                &es_text[m.span.range()],
                es.money.currency[i].as_str(),
                es.money.amount(ix).to_plain_string(),
                es.money.per(ix),
            )
        })
        .collect();
    assert_eq!(
        es_got,
        [
            ("5 dólares por hora", "USD", "5".to_string(), Some("h")),
            ("5 dólares por día", "USD", "5".to_string(), Some("d")),
            ("5 dólares", "USD", "5".to_string(), None),
        ]
    );

    let nl_text = "5 euro per uur, 5 euro per dag en 5 euro";
    let nl = GrammarAnnotator::dutch().find(nl_text, ViewId::SOURCE);
    assert_eq!(nl.money.check(), Ok(()));
    let nl_got: Vec<(&str, &str, String, Option<&str>)> = (0..nl.money.len())
        .map(|i| {
            let ix = Ix::new(i as u32);
            let m = nl.money.get(ix);
            (
                &nl_text[m.span.range()],
                nl.money.currency[i].as_str(),
                nl.money.amount(ix).to_plain_string(),
                nl.money.per(ix),
            )
        })
        .collect();
    assert_eq!(
        nl_got,
        [
            ("5 euro per uur", "EUR", "5".to_string(), Some("h")),
            ("5 euro per dag", "EUR", "5".to_string(), Some("d")),
            ("5 euro", "EUR", "5".to_string(), None),
        ]
    );
}

#[test]
fn tonnes_and_tons_have_their_own_symbols() {
    let text = "3.5 tonnes, 2 tons and 1 long ton";
    let f = GrammarAnnotator::english().find(text, ViewId::SOURCE);
    let units: Vec<&str> = (0..f.quantities.len()).map(|i| f.quantities.unit(Ix::new(i as u32))).collect();
    assert_eq!(units, ["t", "sh tn", "long tn"]);
    let normalized: Vec<(String, &str)> = (0..f.quantities.len())
        .map(|i| {
            let ix = Ix::new(i as u32);
            (f.quantities.normalized_value(ix).to_plain_string(), f.quantities.base_unit(ix))
        })
        .collect();
    assert_eq!(
        normalized,
        [("3500".to_string(), "kg"), ("1814.36948".to_string(), "kg"), ("1016.0469088".to_string(), "kg"),]
    );
}

#[test]
fn grammar_quantities_carry_normalized_base_units() {
    let text = "2.5 kilometers at 100 degrees Celsius and 36 kilometers per hour";
    let f = GrammarAnnotator::english().find(text, ViewId::SOURCE);
    assert_eq!(f.quantities.check(), Ok(()));
    let got: Vec<(String, &str, String, &str)> = (0..f.quantities.len())
        .map(|i| {
            let ix = Ix::new(i as u32);
            (
                f.quantities.value(ix).to_plain_string(),
                f.quantities.unit(ix),
                f.quantities.normalized_value(ix).to_plain_string(),
                f.quantities.base_unit(ix),
            )
        })
        .collect();
    assert_eq!(
        got,
        [
            ("2.5".to_string(), "km", "2500".to_string(), "m"),
            ("100".to_string(), "°C", "373.15".to_string(), "K"),
            ("36".to_string(), "km/h", "10".to_string(), "m/s"),
        ]
    );
}

#[test]
fn money_is_stated_in_its_currency() {
    let text = "it costs $3.5m, 50 cents or 20 francs";
    let f = GrammarAnnotator::english().find(text, ViewId::SOURCE);
    assert_eq!(f.money.check(), Ok(()));
    assert_eq!(f.money.len(), 2, "francs name no one currency");
    let first = f.money.get(Ix::new(0));
    assert_eq!(first.currency.as_str(), "USD");
    assert_eq!(f.money.amount(Ix::new(0)).to_plain_string(), "3500000");
    assert_eq!(f.money.per(Ix::new(0)), None);
    assert_eq!(f.money.amount(Ix::new(1)).to_plain_string(), "0.5");
    assert_eq!(f.money.per(Ix::new(1)), None);
    assert_eq!(text[f.money.spans.get(Ix::new(1)).range()].to_string(), "50 cents");
}

#[test]
fn overlapping_percents_and_quantities_keep_the_longer() {
    let mut p = Quantities::new(ViewId::SOURCE);
    let mut q = Quantities::new(ViewId::SOURCE);
    let m = |start, end, unit| QuantityMention {
        span: Span::new(start, end),
        negative: false,
        coefficient: "1",
        scale: 0,
        unit,
    };
    p.push(m(0, 4, PERCENT));
    p.push(m(10, 12, PERCENT));
    q.push(m(2, 8, "km"));
    q.push(m(10, 11, "m"));
    let all = merge(&p, &q);
    let spans: Vec<(u32, u32)> =
        (0..all.len()).map(|i| all.spans.get(Ix::new(i as u32))).map(|s| (s.start, s.end)).collect();
    assert_eq!(spans, [(2, 8), (10, 12)]);
}

#[test]
fn the_config_hash_names_the_grammar() {
    assert_eq!(GrammarAnnotator::english().config_hash(), GrammarAnnotator::english().config_hash());
    assert_ne!(GrammarAnnotator::english().config_hash(), 0);
}
