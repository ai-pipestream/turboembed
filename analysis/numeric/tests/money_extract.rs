//! The money extractor: the cases of OpenNLP's `CursorMoneyExtractorTest`
//! and `MoneyAnnotatorTest`, then edge cases of the port.
//!
//! Java spans count UTF-16 units; here they are UTF-8 bytes, so a span in
//! a text with a non-ASCII character is written out in bytes, with the
//! Java span beside it. Cases that pass null or a malformed code point or
//! code to the Java have no counterpart: the Rust types cannot hold them.

mod money_common;

use money_common::{
    cur, dec, extract_money as extract, none_money as none, same, single_money as single,
    single_money_with as single_with, unscaled, whole,
};
use turbo_analysis::{Analyzer, Document, Ix, Layer, Span};
use turbo_numeric::NumberNotation::{LatinEu, LatinUs};
use turbo_numeric::money::{DEFAULT_SYMBOLS, Money, MoneyExtractor, SymbolError, is_iso_currency};

fn eu() -> MoneyExtractor {
    MoneyExtractor::new(LatinEu)
}

#[test]
fn recognized_shapes_cover_the_full_mention() {
    let cases = [
        ("$1,234.56", "1234.56", "USD"),
        ("$1,234,567.89", "1234567.89", "USD"),
        ("$ 100", "100", "USD"),
        ("\u{20AC}50", "50", "EUR"),
        ("50\u{20AC}", "50", "EUR"),
        ("\u{00A3}2.5k", "2500.0", "GBP"),
        ("\u{00A5}1000", "1000", "JPY"),
        ("USD 100", "100", "USD"),
        ("100 USD", "100", "USD"),
        ("$1.2M", "1200000.0", "USD"),
        ("$3bn", "3000000000", "USD"),
        ("$3 billion", "3000000000", "USD"),
        ("USD 1.2 million", "1200000.0", "USD"),
        ("1.2 million USD", "1200000.0", "USD"),
        ("-$5", "-5", "USD"),
        ("-5 USD", "-5", "USD"),
        ("$0.00", "0.00", "USD"),
        ("CHF 42", "42", "CHF"),
    ];
    for (text, amount, currency) in cases {
        let m = single(text);
        assert_eq!(m.span, whole(text), "{text:?}");
        assert!(same(&dec(amount), &m.amount), "{text:?}: {}", m.amount);
        assert_eq!(m.currency, cur(currency), "{text:?}");
    }
}

#[test]
fn mention_inside_sentence_has_exact_span() {
    let m = single("they paid $1.2M for it.");
    assert_eq!(m.span, Span::new(10, 15));
    assert!(same(&dec("1200000"), &m.amount));
}

#[test]
fn symbol_after_letter_is_a_longer_symbol_and_yields_no_mention() {
    let parity = MoneyExtractor::default().with_opennlp_parity(true);
    for text in ["HK$50", "NZ$50", "R$50", "US$50"] {
        none(&parity, text);
    }
    for text in ["FOO$50", "Rs$50"] {
        none(&MoneyExtractor::default(), text);
    }
}

#[test]
fn bare_symbol_still_matches_at_ordinary_boundaries() {
    assert_eq!(single("$50").currency, cur("USD"));
    assert_eq!(single("pay $50").span, Span::new(4, 7));
}

#[test]
fn multiple_mentions() {
    let found = extract(&MoneyExtractor::default(), "paid $5 and \u{20AC}3 today");
    let currencies: Vec<_> = found.iter().map(|f| f.currency).collect();
    assert_eq!(currencies, [cur("USD"), cur("EUR")]);
}

#[test]
fn hyphenated_range_yields_two_positive_mentions() {
    let found = extract(&MoneyExtractor::default(), "$100-$200");
    assert_eq!(found.len(), 2);
    assert_eq!(found[0].span, Span::new(0, 4));
    assert!(same(&dec("100"), &found[0].amount));
    assert_eq!(found[1].span, Span::new(5, 9));
    assert!(same(&dec("200"), &found[1].amount));
}

#[test]
fn hyphenated_symbol_suffix_range_yields_two_positive_mentions() {
    // Java spans 0..3 and 4..7; the euro sign is three bytes.
    let found = extract(&MoneyExtractor::default(), "50\u{20AC}-60\u{20AC}");
    assert_eq!(found.len(), 2);
    assert_eq!(found[0].span, Span::new(0, 5));
    assert!(same(&dec("50"), &found[0].amount));
    assert_eq!(found[0].currency, cur("EUR"));
    assert_eq!(found[1].span, Span::new(6, 11));
    assert!(same(&dec("60"), &found[1].amount));
    assert_eq!(found[1].currency, cur("EUR"));
}

#[test]
fn hyphen_between_suffix_and_prefix_mentions_is_not_a_minus() {
    // Java span 4..7 for the second mention.
    let found = extract(&MoneyExtractor::default(), "50\u{20AC}-$60");
    assert_eq!(found.len(), 2);
    assert!(same(&dec("50"), &found[0].amount));
    assert_eq!(found[0].currency, cur("EUR"));
    assert_eq!(found[1].span, Span::new(6, 9));
    assert!(same(&dec("60"), &found[1].amount));
    assert_eq!(found[1].currency, cur("USD"));
}

#[test]
fn hyphen_after_letter_is_not_a_minus() {
    let m = single("pre-$5 deal");
    assert_eq!(m.span, Span::new(4, 6));
    assert!(same(&dec("5"), &m.amount));
}

#[test]
fn negative_symbol_first_at_boundaries_keeps_the_sign() {
    let at_start = single("-$5");
    assert_eq!(at_start.span, Span::new(0, 3));
    assert!(same(&dec("-5"), &at_start.amount));
    let after_space = single("balance -$5");
    assert_eq!(after_space.span, Span::new(8, 11));
    assert!(same(&dec("-5"), &after_space.amount));
    let after_parenthesis = single("(-$5)");
    assert_eq!(after_parenthesis.span, Span::new(1, 4));
    assert!(same(&dec("-5"), &after_parenthesis.amount));
}

#[test]
fn grouping_the_scanner_cannot_parse_yields_no_mention() {
    for (text, cur_code) in [("\u{20B9}1,00,000", "INR"), ("1,00,000 USD", "USD"), ("USD 1,00,000", "USD")] {
        let m = single(text);
        assert!(same(&dec("100000"), &m.amount), "{text:?}");
        assert_eq!(m.currency, cur(cur_code), "{text:?}");
        none(&MoneyExtractor::default().with_opennlp_parity(true), text);
    }
    for text in ["$1,23", "\u{20AC}1.234,56"] {
        none(&MoneyExtractor::default(), text);
    }
}

#[test]
fn trailing_comma_without_digit_still_ends_the_match() {
    let m = single("$5, sure");
    assert_eq!(m.span, Span::new(0, 2));
    assert!(same(&dec("5"), &m.amount));
}

#[test]
fn mention_at_text_start_and_end() {
    assert_eq!(single("$5 up front").span, Span::new(0, 2));
    assert_eq!(single("the fee is $7").span, Span::new(11, 13));
}

#[test]
fn trailing_punctuation_is_not_included() {
    let text = "it cost $3.20.";
    assert_eq!(&text[single(text).span.range()], "$3.20");
}

#[test]
fn rejected_shapes() {
    for text in
        ["5%", "the 100 things", "USD alone", "$x", "$5x", "100 USDX", "chapter USD", "USD -5", "call 555,1234 now"]
    {
        none(&MoneyExtractor::default(), text);
    }
}

#[test]
fn spelled_out_currency_words_cover_the_full_mention() {
    let cases = [
        ("50 dollars", "50", "USD"),
        ("1 dollar", "1", "USD"),
        ("1.2 million dollars", "1200000.0", "USD"),
        ("50 euros", "50", "EUR"),
        ("1 euro", "1", "EUR"),
        ("3 billion pounds", "3000000000", "GBP"),
        ("40 million sterling", "40000000", "GBP"),
        ("1,200 yen", "1200", "JPY"),
        ("500 rupees", "500", "INR"),
        ("3.5k dollars", "3500.0", "USD"),
        ("-20 dollars", "-20", "USD"),
        ("12 trillion yuan", "12000000000000", "CNY"),
        ("80 renminbi", "80", "CNY"),
        ("300 roubles", "300", "RUB"),
        ("25 shekels", "25", "ILS"),
        ("900 baht", "900", "THB"),
    ];
    for (text, amount, currency) in cases {
        let m = single(text);
        assert_eq!(m.span, whole(text), "{text:?}");
        assert!(same(&dec(amount), &m.amount), "{text:?}: {}", m.amount);
        assert_eq!(m.currency, cur(currency), "{text:?}");
    }
}

#[test]
fn currency_word_is_matched_case_insensitively() {
    assert_eq!(single("50 Dollars").currency, cur("USD"));
    assert_eq!(single("50 EUROS").currency, cur("EUR"));
}

#[test]
fn currency_word_mention_inside_sentence_has_exact_span() {
    let m = single("the fund raised 1.2 million dollars last year");
    assert_eq!(m.span, Span::new(16, 35));
    assert!(same(&dec("1200000"), &m.amount));
}

#[test]
fn currency_word_in_the_european_notation() {
    let m = single_with(&eu(), "1.234,56 euros");
    assert!(same(&dec("1234.56"), &m.amount));
    assert_eq!(m.currency, cur("EUR"));
}

#[test]
fn words_that_are_not_currencies_yield_no_mention() {
    for text in [
        "5 apples",
        "10 cents",
        "3 pence",
        "12 items",
        "500 pesos",
        "20 francs",
        "75 kronor",
        "3 won",
        "50 dollarsx",
        "50 dollar5",
        "50 euro-cent",
        "dollars 50",
        "50  dollars",
    ] {
        none(&MoneyExtractor::default(), text);
    }
}

#[test]
fn european_notation_shapes_cover_the_full_mention() {
    let cases = [
        ("1.234,56 EUR", "1234.56", "EUR"),
        ("1.234.567,89 EUR", "1234567.89", "EUR"),
        ("\u{20AC}1.234,56", "1234.56", "EUR"),
        ("\u{20AC}50,25", "50.25", "EUR"),
        ("\u{20AC}1.500", "1500", "EUR"),
        ("EUR 2.400.000", "2400000", "EUR"),
        ("-\u{20AC}5,50", "-5.50", "EUR"),
        ("\u{20AC}2,5 million", "2500000.0", "EUR"),
        ("1.250,5 CHF", "1250.5", "CHF"),
    ];
    for (text, amount, currency) in cases {
        let m = single_with(&eu(), text);
        assert_eq!(m.span, whole(text), "{text:?}");
        assert!(same(&dec(amount), &m.amount), "{text:?}: {}", m.amount);
        assert_eq!(m.currency, cur(currency), "{text:?}");
    }
}

#[test]
fn amount_in_the_other_notation_yields_no_mention() {
    for text in ["$1,234.56", "1,234.56 USD", "USD 1,234.56", "1,234,567.89 USD"] {
        none(&eu(), text);
    }
    for text in ["\u{20AC}1.234,56", "1.234,56 EUR", "EUR 1.234,56", "1.234.567,89 EUR"] {
        none(&MoneyExtractor::default(), text);
    }
}

#[test]
fn amount_valid_in_both_notations_means_what_the_notation_says() {
    assert!(same(&dec("1500"), &single_with(&eu(), "1.500 EUR").amount));
    assert!(same(&dec("1.500"), &single("1.500 EUR").amount));
}

/// `testForRegionResolvesTheNotationOfTheRegion`, with the extractors
/// `forRegion` builds for Germany, France and Australia written out: the
/// euro sign already denotes EUR, and `$` denotes AUD in Australia.
#[test]
fn region_extractors_written_out() {
    let german = eu();
    assert!(same(&dec("1234.56"), &single_with(&german, "1.234,56 EUR").amount));
    none(&german, "$1,234.56");
    let french = eu();
    assert!(same(&dec("50.25"), &single_with(&french, "50,25 EUR").amount));
    let australian = MoneyExtractor::new(LatinUs).with_symbol('$', cur("AUD")).unwrap();
    let m = single_with(&australian, "$1,234.56");
    assert!(same(&dec("1234.56"), &m.amount));
    assert_eq!(m.currency, cur("AUD"));
    // Canada, from MoneyConversionAnnotatorTest; South Africa keeps the
    // defaults, its rand being written with a letter.
    let canadian = MoneyExtractor::default().with_symbol('$', cur("CAD")).unwrap();
    assert_eq!(single_with(&canadian, "$5").currency, cur("CAD"));
    assert_eq!(single_with(&MoneyExtractor::default(), "$5").currency, cur("USD"));
}

#[test]
fn custom_symbol_table() {
    let canadian = MoneyExtractor::with_symbols([('$', cur("CAD"))], LatinUs).unwrap();
    assert_eq!(single_with(&canadian, "$5").currency, cur("CAD"));
}

#[test]
fn custom_symbol_table_keeps_codes_aligned_after_sorting() {
    let custom = MoneyExtractor::with_symbols(
        [('\u{20B9}', cur("INR")), ('$', cur("CAD")), ('\u{20AC}', cur("EUR")), ('\u{00A3}', cur("GBP"))],
        LatinUs,
    )
    .unwrap();
    let found = extract(&custom, "$1 \u{00A3}2 \u{20AC}3 \u{20B9}4");
    let currencies: Vec<_> = found.iter().map(|f| f.currency).collect();
    assert_eq!(currencies, [cur("CAD"), cur("GBP"), cur("EUR"), cur("INR")]);
}

#[test]
fn custom_symbol_table_validation() {
    assert_eq!(MoneyExtractor::with_symbols([], LatinUs), Err(SymbolError::Empty));
    assert_eq!(MoneyExtractor::with_symbols([('A', cur("USD"))], LatinUs), Err(SymbolError::NotACurrencySymbol('A')));
    assert_eq!(
        MoneyExtractor::with_symbols([('$', cur("USX"))], LatinUs),
        Err(SymbolError::UnknownCurrency(cur("USX")))
    );
    // `DOLLARS` is not a code at all, so it cannot reach the table.
    assert_eq!(turbo_numeric::Currency::parse("DOLLARS"), None);
}

#[test]
fn repeated_decimal_separator_yields_no_mention() {
    for text in ["$1.2.3", "USD 1.2.3", "1.2.3 USD"] {
        none(&MoneyExtractor::default(), text);
    }
}

#[test]
fn supplementary_plane_neighbors() {
    // U+1F4B0 is four bytes (two Java chars): Java span 2..4.
    let m = single("\u{1F4B0}$5");
    assert_eq!(m.span, Span::new(4, 6));
    assert!(same(&dec("5"), &m.amount));
    // U+1D7D9 is a digit, so no number starts after it.
    none(&MoneyExtractor::default(), "\u{1D7D9}5 USD");
}

#[test]
fn unicode_whitespace_gap_matches_like_a_space() {
    for gap in ["\u{00A0}", "\u{202F}", "\u{2009}", "\u{3000}", "\t"] {
        for shape in ["$ 100", "USD 100", "100 USD", "USD 1.2 million", "1.2 million dollars", "$3 billion"] {
            let expected = single(shape);
            let text = shape.replace(' ', gap);
            let m = single(&text);
            // The gap may be longer than a space in bytes.
            let extra = (gap.len() - 1) * shape.matches(' ').count();
            assert_eq!(m.span, Span::new(expected.span.start, expected.span.end + extra as u32), "{text:?}");
            assert!(same(&expected.amount, &m.amount), "{text:?}");
            assert_eq!(expected.currency, m.currency, "{text:?}");
        }
    }
}

/// `MoneyAnnotatorTest.testMoneyLayerOnRawText`: the layer in a pipeline.
#[test]
fn money_layer_on_raw_text() {
    let analyzer = Analyzer::builder().add(MoneyExtractor::default()).build().unwrap();
    let mut docs = vec![Document::new("the deal closed at $2.3 billion after weeks of talks"), Document::new("")];
    let results = analyzer.run(&mut docs);
    assert!(results.iter().all(Result::is_ok), "{results:?}");
    let money = docs[0].get::<Money>().unwrap();
    assert_eq!(money.len(), 1);
    assert_eq!(&docs[0].source()[money.spans.get(Ix::new(0)).range()], "$2.3 billion");
    assert!(same(&dec("2300000000"), &unscaled(money.coefficient.get(0), money.scale[0])));
    assert_eq!(money.currency[0], cur("USD"));
    assert_eq!(docs[1].get::<Money>().unwrap().len(), 0);
}

// Edge cases of the port.

#[test]
fn the_default_table_is_the_javas() {
    let x = MoneyExtractor::default();
    assert_eq!(x.notation(), LatinUs);
    assert_eq!(x.symbols().len(), DEFAULT_SYMBOLS.len());
    assert!(x.symbols().windows(2).all(|w| w[0].0 < w[1].0));
    for text in ["\u{20A9}5", "\u{20BD}5", "\u{20BA}5", "\u{20AA}5", "\u{0E3F}5", "\u{20AB}5"] {
        assert!(is_iso_currency(single(text).currency), "{text:?}");
    }
    assert_eq!(single("\u{20A9}5").currency, cur("KRW"));
    assert_eq!(single("\u{20AB}5").currency, cur("VND"));
}

#[test]
fn a_later_entry_replaces_an_earlier_one() {
    let x = MoneyExtractor::with_symbols([('$', cur("USD")), ('$', cur("AUD"))], LatinUs).unwrap();
    assert_eq!(x.symbols(), [('$', cur("AUD"))]);
}

#[test]
fn a_symbol_outside_the_basic_plane() {
    // U+1ECB0 INDIC SIYAQ RUPEE MARK is a currency symbol of four bytes.
    let x = MoneyExtractor::with_symbols([('\u{1ECB0}', cur("INR"))], LatinUs).unwrap();
    let found = extract(&x, "pay \u{1ECB0}5 and 7\u{1ECB0}.");
    assert_eq!(found.len(), 2);
    assert_eq!(found[0].span, Span::new(4, 9));
    assert_eq!(found[1].span, Span::new(14, 19));
    // After an ASCII letter it is the tail of a longer symbol.
    none(&x, "Rs\u{1ECB0}5");
}

#[test]
fn a_negative_zero_is_zero() {
    let m = single("-$0.00");
    assert_eq!(m.span, Span::new(0, 6));
    assert!(!m.amount.negative);
    assert_eq!(m.amount.to_plain_string(), "0.00");
}

#[test]
fn the_text_ends_inside_a_shape() {
    for text in ["$", "-$", "USD", "USD ", "5 ", "5 USD5", "-", "$ ", "$\u{00A0}", "5 dollar-"] {
        none(&MoneyExtractor::default(), text);
    }
    assert_eq!(single("5$").span, Span::new(0, 2));
}

#[test]
fn a_code_must_stand_alone() {
    for text in ["XUSD 5", "USD5", "5USD", "5 USDs", "5 USD\u{0301}", "_USD 5"] {
        none(&MoneyExtractor::default(), text);
    }
    assert_eq!(single("(USD 5)").span, Span::new(1, 6));
    // An unknown code before an amount does not hide a symbol after it.
    assert_eq!(single("ABC 5$").span, Span::new(4, 6));
}

#[test]
fn letters_that_lowercase_to_a_currency_word() {
    // U+0130 lowercases to `i` code point by code point, as in the Java.
    let m = single("80 RENM\u{0130}NB\u{0130}");
    assert_eq!(m.currency, cur("CNY"));
    assert_eq!(m.span, whole("80 RENM\u{0130}NB\u{0130}"));
}

#[test]
fn a_scale_suffix_then_a_symbol() {
    let m = single("5k\u{20AC}");
    assert!(same(&dec("5000"), &m.amount));
    assert_eq!(m.span, whole("5k\u{20AC}"));
    none(&MoneyExtractor::default(), "5kg\u{20AC}");
}
