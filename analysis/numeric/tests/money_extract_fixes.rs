//! Tests for the default money extraction fixes and OpenNLP-parity mode
//! (`#933`), plus minor-unit rounding on converted amounts.

mod money_common;

use money_common::{cur, day, dec, load, same};
use turbo_analysis::{Analyzer, Annotator, Document, Ix, Layer, Span, ViewId};
use turbo_numeric::NumberNotation::{LatinEu, LatinUs};
use turbo_numeric::money::{
    CONVERTED_MONEY, Currency, DatedMoneyConverter, Decimal, Money, MoneyConverter, MoneyExtractor, is_test_currency,
    minor_units,
};
use turbo_numeric::{DocumentDateSelector, TemporalExtractor};

#[derive(Debug)]
struct Found {
    span: Span,
    amount: Decimal,
    currency: Currency,
}

fn extract(x: &MoneyExtractor, text: &str) -> Vec<Found> {
    let mut m = Money::new(ViewId::SOURCE);
    x.scan(text, &mut m);
    assert_eq!(m.check(), Ok(()), "{text:?}");
    m.spans
        .iter()
        .map(|(i, span)| {
            let a = m.amount(i);
            let amount = Decimal { negative: a.negative, digits: a.digits.into(), scale: a.scale as i32 };
            Found { span, amount, currency: m.currency[i.get()] }
        })
        .collect()
}

fn single(text: &str) -> Found {
    single_with(&MoneyExtractor::default(), text)
}

fn single_with(x: &MoneyExtractor, text: &str) -> Found {
    let mut all = extract(x, text);
    assert_eq!(all.len(), 1, "expected one mention in {text:?}: {all:?}");
    all.pop().unwrap()
}

fn none(x: &MoneyExtractor, text: &str) {
    let found = extract(x, text);
    assert!(found.is_empty(), "{text:?}: {found:?}");
}

#[test]
fn sign_before_code_is_preserved_by_default() {
    let m = single("-USD 5");
    assert_eq!(m.span, Span::new(0, 6));
    assert!(same(&dec("-5"), &m.amount));
    assert_eq!(m.currency, cur("USD"));

    let m_million = single("balance -EUR 1.2 million today");
    assert_eq!(m_million.span, Span::new(8, 24));
    assert!(same(&dec("-1200000"), &m_million.amount));
    assert_eq!(m_million.currency, cur("EUR"));

    // Hyphen in a range or after a word is not a leading minus sign.
    let range = extract(&MoneyExtractor::default(), "USD 10-USD 20");
    assert_eq!(range.len(), 2);
    assert!(same(&dec("10"), &range[0].amount));
    assert_eq!(range[1].span, Span::new(7, 13));
    assert!(same(&dec("20"), &range[1].amount));

    let after_word = single("pre-USD 5 deal");
    assert_eq!(after_word.span, Span::new(4, 9));
    assert!(same(&dec("5"), &after_word.amount));
}

#[test]
fn dollar_prefixed_region_symbols_resolve_by_default() {
    let cases = [
        ("HK$50", "50", "HKD"),
        ("US$50", "50", "USD"),
        ("NZ$50", "50", "NZD"),
        ("R$50", "50", "BRL"),
        ("A$50", "50", "AUD"),
        ("AU$1,200.50", "1200.50", "AUD"),
        ("CA$75", "75", "CAD"),
        ("C$75", "75", "CAD"),
        ("S$20", "20", "SGD"),
        ("SG$20", "20", "SGD"),
        ("NT$300", "300", "TWD"),
        ("TW$300", "300", "TWD"),
        ("MX$90", "90", "MXN"),
        ("Z$10", "10", "ZWL"),
        ("-HK$50", "-50", "HKD"),
        ("US$ 1.5M", "1500000.0", "USD"),
    ];
    for (text, amount, code) in cases {
        let m = single(text);
        assert_eq!(m.span, Span::new(0, text.len() as u32), "{text:?}");
        assert!(same(&dec(amount), &m.amount), "{text:?}: {}", m.amount);
        assert_eq!(m.currency, cur(code), "{text:?}");
    }

    // Multiple prefixed symbols in one sentence.
    let multi = extract(&MoneyExtractor::default(), "HK$50 NZ$50 R$50 US$50 A$50");
    let codes: Vec<_> = multi.iter().map(|f| f.currency).collect();
    assert_eq!(codes, [cur("HKD"), cur("NZD"), cur("BRL"), cur("USD"), cur("AUD")]);

    // A non-ASCII letter or unknown prefix before `$` is rejected.
    for text in ["\u{DC}$5", "FOO$50", "X$50", "wordA$50"] {
        none(&MoneyExtractor::default(), text);
    }

    // If `$` is not in the symbol table, dollar-prefixed symbols do not match.
    let euro_only = MoneyExtractor::with_symbols([('\u{20AC}', cur("EUR"))], LatinUs).unwrap();
    none(&euro_only, "HK$50");
}

#[test]
fn units_of_weight_are_not_read_as_pounds_sterling_from_context() {
    let default = MoneyExtractor::default();
    for weight_text in [
        "3 pounds of flour",
        "bought 5 pounds of apples at the market",
        "1 pound of butter",
        "2 pounds of sugar",
        "a 10 pound bag",
        "costs 4 dollars per pound or 5 pounds per box",
        "he weighs 180 pounds",
        "the parcel weighed 12 pounds",
        "lost 15 pounds last month",
        "gained 5 pounds",
        "lifted 200 pounds",
        "3 pounds 8 ounces",
        "6 pounds in weight",
    ] {
        let found = extract(&default, weight_text);
        assert!(found.iter().all(|m| m.currency != cur("GBP")), "{weight_text:?}: {found:?}");
    }

    // Genuine currency mentions in pounds still extract.
    let cases = [
        ("3 billion pounds", "3000000000", Span::new(0, 16)),
        ("500 thousand pounds of funding", "500000", Span::new(0, 19)),
        ("paid 50 pounds for the ticket", "50", Span::new(5, 14)),
        ("raised 200 pounds of funding", "200", Span::new(7, 17)),
        ("3 pounds sterling", "3", Span::new(0, 17)),
        ("paid 25 pounds sterling yesterday", "25", Span::new(5, 23)),
    ];
    for (text, amount, span) in cases {
        let m = single(text);
        assert_eq!(m.span, span, "{text:?}");
        assert!(same(&dec(amount), &m.amount), "{text:?}");
        assert_eq!(m.currency, cur("GBP"), "{text:?}");
    }
}

#[test]
fn test_codes_xts_and_xxx_are_refused_by_default() {
    assert!(is_test_currency(cur("XTS")));
    assert!(is_test_currency(cur("XXX")));
    assert!(!is_test_currency(cur("USD")));
    assert!(!is_test_currency(cur("DEM")));

    let default = MoneyExtractor::default();
    for text in ["XTS 5", "5 XTS", "XXX 5", "5 XXX", "-XTS 10", "XXX 1.2 million"] {
        none(&default, text);
    }
    // Historical ISO codes like DEM remain accepted.
    assert_eq!(single("DEM 5").currency, cur("DEM"));
}

#[test]
fn opennlp_parity_mode_preserves_jvm_behaviour() {
    let us = MoneyExtractor::default();
    let au = MoneyExtractor::default().with_symbol('$', cur("AUD")).unwrap();
    let parity = MoneyExtractor::default().with_opennlp_parity(true);
    assert!(parity.opennlp_parity());
    assert_ne!(us.config_hash(), au.config_hash());
    assert_ne!(us.config_hash(), MoneyExtractor::new(LatinEu).config_hash());
    assert_ne!(us.config_hash(), parity.config_hash());
    assert_eq!(us.config_hash(), MoneyExtractor::new(LatinUs).config_hash());

    // A minus before a code is dropped, so the amount turns positive.
    let m = single_with(&parity, "-USD 5");
    assert_eq!(m.span, Span::new(1, 6));
    assert!(same(&dec("5"), &m.amount));

    // Only an ASCII letter makes a symbol the tail of a longer one.
    assert_eq!(single_with(&parity, "\u{DC}$5").span, Span::new(2, 4));
    for text in ["HK$50", "NZ$50", "R$50", "US$50"] {
        none(&parity, text);
    }

    // The one gap may be a line break.
    assert_eq!(single_with(&parity, "$\n100").span, Span::new(0, 5));
    assert_eq!(single_with(&parity, "100\nUSD").span, Span::new(0, 7));

    // Withdrawn, testing and no-currency codes are currencies.
    for code in ["DEM", "XTS", "XXX"] {
        assert_eq!(single_with(&parity, &format!("{code} 5")).currency, cur(code));
    }

    // A weight in pounds is sterling.
    assert_eq!(single_with(&parity, "3 pounds of flour").currency, cur("GBP"));
}

#[test]
fn minor_unit_rounding_on_converted_amounts() {
    assert_eq!(minor_units(cur("EUR")), Some(2));
    assert_eq!(minor_units(cur("JPY")), Some(0));
    assert_eq!(minor_units(cur("KWD")), Some(3));
    assert_eq!(minor_units(cur("CLF")), Some(4));
    assert_eq!(minor_units(cur("XAU")), None);

    let rates = load("Date,USD,JPY,BHD\n2026-07-10,1.2000,160.3333,0.45123\n");
    let text = "paid $1 and $10.05";

    // Unrounded default: 1 USD in EUR at 1.2 is 0.8333333333333333.
    let unrounded_analyzer = Analyzer::builder()
        .add(MoneyExtractor::default())
        .step(MoneyConverter::new(rates.clone(), cur("EUR"), day("2026-07-10")).bound())
        .build()
        .unwrap();
    let mut docs = vec![Document::new(text)];
    assert!(unrounded_analyzer.run(&mut docs).iter().all(Result::is_ok));
    let unrounded = docs[0].get_key(&CONVERTED_MONEY).unwrap();
    assert_eq!(unrounded.amount(Ix::new(0)).to_plain_string(), "0.8333333333333333");

    // Minor-unit rounding enabled for EUR (2 fraction digits): 0.83 and 8.38 (8.375 -> 8.38 half-even).
    let eur_converter =
        MoneyConverter::new(rates.clone(), cur("EUR"), day("2026-07-10")).with_minor_unit_rounding(true);
    assert!(eur_converter.minor_unit_rounding());
    assert_ne!(
        eur_converter.config_hash(),
        MoneyConverter::new(rates.clone(), cur("EUR"), day("2026-07-10")).config_hash()
    );
    let rounded_analyzer =
        Analyzer::builder().add(MoneyExtractor::default()).step(eur_converter.bound()).build().unwrap();
    let mut docs = vec![Document::new(text)];
    assert!(rounded_analyzer.run(&mut docs).iter().all(Result::is_ok));
    let rounded = docs[0].get_key(&CONVERTED_MONEY).unwrap();
    assert_eq!(rounded.amount(Ix::new(0)).to_plain_string(), "0.83");
    assert_eq!(rounded.scale[0], 2);
    assert_eq!(rounded.amount(Ix::new(1)).to_plain_string(), "8.38");
    assert_eq!(rounded.scale[1], 2);

    // Minor-unit rounding for JPY (0 fraction digits) on DatedMoneyConverter.
    let dated_jpy = DatedMoneyConverter::new(rates.clone(), cur("JPY")).with_minor_unit_rounding(true);
    assert!(dated_jpy.minor_unit_rounding());
    let jpy_analyzer = Analyzer::builder()
        .add(TemporalExtractor::new())
        .add(DocumentDateSelector)
        .add(MoneyExtractor::default())
        .step(dated_jpy.bound())
        .build()
        .unwrap();
    let mut jpy_docs = vec![Document::new("2026-07-10: invoice EUR 1")];
    assert!(jpy_analyzer.run(&mut jpy_docs).iter().all(Result::is_ok));
    let jpy_conv = jpy_docs[0].get_key(&CONVERTED_MONEY).unwrap();
    assert_eq!(jpy_conv.amount(Ix::new(0)).to_plain_string(), "160");
    assert_eq!(jpy_conv.scale[0], 0);

    // Minor-unit rounding for BHD (3 fraction digits): 1 EUR -> 0.451 BHD.
    let bhd_analyzer = Analyzer::builder()
        .add(MoneyExtractor::default())
        .step(MoneyConverter::new(rates, cur("BHD"), day("2026-07-10")).with_minor_unit_rounding(true).bound())
        .build()
        .unwrap();
    let mut bhd_docs = vec![Document::new("EUR 1")];
    assert!(bhd_analyzer.run(&mut bhd_docs).iter().all(Result::is_ok));
    let bhd_conv = bhd_docs[0].get_key(&CONVERTED_MONEY).unwrap();
    assert_eq!(bhd_conv.amount(Ix::new(0)).to_plain_string(), "0.451");
    assert_eq!(bhd_conv.scale[0], 3);
}
