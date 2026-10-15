//! The English rules on texts of their own, one group per kind of
//! mention. The Recognizers-Text cases are in `tests/spec.rs`.

use crate::engine::{Output, Parser};
use crate::value::Value;

use super::ENGLISH;

fn show(v: Value) -> String {
    match v {
        Value::Num(n) => {
            let s = n.value.to_string();
            let s = if s.len() > 20 { format!("{:.9}", n.value.to_f64()) } else { s };
            format!("{s} {}", n.kind.name())
        }
        Value::Ordinal(o) => format!("#{}", o.value),
        Value::Relative(r) => format!("{}{:+}", r.to.name(), r.offset),
        Value::Percent(p) => format!("{p}%"),
        v => format!("{v:?}"),
    }
}

fn found(text: &str, output: Output) -> Vec<(String, String)> {
    let mut p = Parser::new(&ENGLISH);
    p.parse(text);
    let mut out = Vec::new();
    p.select(text, output, &mut out);
    out.iter().map(|m| (text[m.start as usize..m.end as usize].to_string(), show(m.value))).collect()
}

fn numbers(text: &str) -> Vec<(String, String)> {
    found(text, Output::Number)
}

fn values(text: &str) -> Vec<String> {
    numbers(text).into_iter().map(|n| n.1).collect()
}

fn pair(a: &str, b: &str) -> Vec<(String, String)> {
    vec![(a.to_string(), b.to_string())]
}

/// The value of the one number that covers all of `text`.
fn one(text: &str) -> String {
    let all = numbers(text);
    assert_eq!(all.len(), 1, "{text:?}: {all:?}");
    assert_eq!(all[0].0, text, "{text:?}: the whole text");
    all[0].1.clone()
}

#[test]
fn spelled_integers() {
    assert_eq!(one("seven"), "7 integer");
    assert_eq!(one("twenty-one"), "21 integer");
    assert_eq!(one("Twenty One"), "21 integer");
    assert_eq!(one("one hundred and six"), "106 integer");
    assert_eq!(one("nineteen hundred"), "1900 integer");
    assert_eq!(one("a thousand"), "1000 integer");
    assert_eq!(one("two thousand and two hundred"), "2200 integer");
    assert_eq!(one("two hundred and two thousand"), "202000 integer");
    assert_eq!(one("one million two thousand three"), "1002003 integer");
    assert_eq!(one("three lakh crore"), "3000000000000 integer");
    assert_eq!(one("twenty-one trillion three hundred"), "21000000000300 integer");
    assert_eq!(one("zero"), "0 integer");
}

#[test]
fn malformed_sequences_split() {
    assert_eq!(values("three hundred and two hundred"), ["300 integer", "200 integer"]);
    // A scale word takes only a number below it; a bare scale is no number.
    assert_eq!(values("two million thousand"), ["2000000 integer"]);
    assert_eq!(values("seven seven"), ["7 integer", "7 integer"]);
    // An addend does not stop short of its own multiplier.
    assert_eq!(values("three hundred and two dozen"), ["324 integer"]);
}

#[test]
fn digit_numerals() {
    assert_eq!(one("1,234,567"), "1234567 integer");
    assert_eq!(one("12,34,567"), "1234567 integer");
    assert_eq!(one("1 234 567"), "1234567 integer");
    assert_eq!(one("4.800"), "4.8 decimal");
    assert_eq!(one(".5"), "0.5 decimal");
    assert_eq!(one("5.00"), "5 integer");
    let parts: Vec<String> = numbers("192.168.1.2").into_iter().map(|n| n.0).collect();
    assert_eq!(parts, ["192", "168", "1", "2"]);
    assert_eq!(numbers("22,24,26").len(), 3);
}

#[test]
fn digits_glued_to_letters_are_no_number() {
    assert!(numbers("29km").is_empty());
    assert!(numbers("1mm thick").is_empty());
    assert!(numbers("1m tall").is_empty());
    assert!(numbers("abc123").is_empty());
    assert_eq!(numbers("SCP-173"), pair("173", "173 integer"));
}

#[test]
fn scale_suffixes() {
    assert_eq!(one("10k"), "10000 integer");
    assert_eq!(one("10K"), "10000 integer");
    assert_eq!(one("2mil"), "2000000 integer");
    assert_eq!(one("1MM"), "1000000 integer");
    assert_eq!(one("10M"), "10000000 integer");
    assert_eq!(one("1.2b"), "1200000000 integer");
    assert_eq!(one("3 T"), "3000000000000 integer");
    assert_eq!(one("1,674.5 million"), "1674500000 integer");
    assert_eq!(one("2.5 thousand"), "2500 integer");
    assert_eq!(one("1.25 hundred"), "125 integer");
    assert_eq!(one("4 thousand 3 hundred and 21"), "4321 integer");
    assert_eq!(one("322 hundred"), "32200 integer");
}

#[test]
fn negatives() {
    assert_eq!(one("-5"), "-5 integer");
    assert_eq!(one("- 5.5"), "-5.5 decimal");
    assert_eq!(one("minus five"), "-5 integer");
    assert_eq!(one("negative 2.5"), "-2.5 decimal");
    assert_eq!(one("\u{2212}7"), "-7 integer");
    assert_eq!(numbers("(-3)"), pair("-3", "-3 integer"));
    // A minus between numbers is a range.
    assert_eq!(values("pages 5 - 10"), ["5 integer", "10 integer"]);
    assert_eq!(values("+1-222"), ["1 integer", "222 integer"]);
}

#[test]
fn spelled_fractions() {
    assert_eq!(one("two thirds"), "0.666666667 fraction");
    assert_eq!(one("a half"), "0.5 fraction");
    assert_eq!(one("one-eighth"), "0.125 fraction");
    assert_eq!(one("three quarters"), "0.75 fraction");
    assert_eq!(one("one and a half"), "1.5 fraction");
    assert_eq!(one("forty two and a third"), "42.333333333 fraction");
    assert_eq!(one("two hundred and fifty-four hundredths"), "200.54 fraction");
    assert_eq!(one("one hundred and thirty-two fifths"), "26.4 fraction");
    assert_eq!(one("three twenty-firsts"), "0.142857143 fraction");
    assert_eq!(one("a hundred thousand trillionths"), "0.0000001 fraction");
    // One takes the singular, more the plural.
    assert_eq!(values("two fifth"), ["2 integer"]);
    assert_eq!(values("one fifths"), ["1 integer"]);
    // `a second` is no half.
    assert!(numbers("a second").is_empty());
}

#[test]
fn fractions_of_scales() {
    assert_eq!(one("half a million"), "500000 fraction");
    assert_eq!(one("quarter million"), "250000 fraction");
    assert_eq!(one("two thirds of a million"), "666666.666666667 fraction");
    assert_eq!(one("two and a half million"), "2500000 fraction");
    assert_eq!(one("one million and a half"), "1500000 fraction");
    assert_eq!(one("five billion and three quarters"), "5750000000 fraction");
    assert_eq!(one("one billion and a half million"), "1000500000 fraction");
    assert_eq!(one("half a dozen"), "6 integer");
    assert_eq!(one("three dozen"), "36 integer");
    assert_eq!(one("a dozen"), "12 integer");
}

#[test]
fn digit_fractions() {
    assert_eq!(one("3/4"), "0.75 fraction");
    assert_eq!(one("-4/5"), "-0.8 fraction");
    assert_eq!(one("2 1/4"), "2.25 fraction");
    assert_eq!(one("½"), "0.5 fraction");
    assert_eq!(one("1½"), "1.5 fraction");
    // Dates and phone numbers are runs, not fractions.
    assert_eq!(numbers("5/17/2018").len(), 3);
    assert_eq!(numbers("222-2222/2222").len(), 3);
}

#[test]
fn fractions_with_prepositions() {
    assert_eq!(one("one over three"), "0.333333333 fraction");
    assert_eq!(one("2000 over 3"), "666.666666667 fraction");
    assert_eq!(one("1 in 5"), "0.2 fraction");
    assert_eq!(one("two out of five hundred"), "0.004 fraction");
    // A part above its whole, money before it, or a year after `in`
    // leave two numbers.
    assert_eq!(numbers("12 out of 10").len(), 2);
    assert_eq!(numbers("$2000 over 3 years").len(), 2);
    assert_eq!(numbers("3 in 2009").len(), 2);
}

#[test]
fn spoken_decimals() {
    assert_eq!(one("two point five"), "2.5 decimal");
    assert_eq!(one("nine point nought four"), "9.04 decimal");
    assert_eq!(one("two hundred point seventy-one"), "200.71 decimal");
    assert_eq!(numbers("at one point"), pair("one", "1 integer"));
}

#[test]
fn exponents() {
    assert_eq!(one("1.3e6"), "1300000 power");
    assert_eq!(one("1e-3"), "0.001 power");
    assert_eq!(one("-12e-1"), "-1.2 power");
    assert_eq!(one("2^10"), "1024 power");
    assert_eq!(one("2.5^-1"), "0.4 power");
    assert_eq!(one("1.1^--2"), "1.21 power");
    assert_eq!(one("6.02x10^8"), "602000000 power");
    assert_eq!(one("6.23x10^-4"), "0.000623 power");
    // A caret power is bounded; past the bound the parts stay apart.
    assert_eq!(values("2^100"), ["2 integer", "100 integer"]);
}

#[test]
fn lone_one_after_a_determiner_is_a_pronoun() {
    assert!(numbers("the one you mentioned").is_empty());
    assert!(numbers("which one").is_empty());
    assert!(numbers("no one came").is_empty());
    assert_eq!(numbers("one apple").len(), 1);
    assert_eq!(values("the one hundred days"), ["100 integer"]);
}

#[test]
fn ordinals() {
    let ord = |t: &str| found(t, Output::Ordinal);
    let o = |t: &str| {
        let all = ord(t);
        assert_eq!(all.len(), 1, "{t:?}: {all:?}");
        all[0].1.clone()
    };
    assert_eq!(o("21st"), "#21");
    assert_eq!(o("22nd"), "#22");
    assert_eq!(o("13th"), "#13");
    assert_eq!(o("111th"), "#111");
    assert!(ord("21th").is_empty());
    assert!(ord("3st").is_empty());
    assert_eq!(o("twenty-first"), "#21");
    assert_eq!(o("one hundred twenty fifth"), "#125");
    assert_eq!(o("two hundredth"), "#200");
    assert_eq!(o("the hundredth"), "#100");
    assert_eq!(o("nineth"), "#9");
    // `a hundredth` is a fraction; a plural is a denominator.
    assert!(ord("a hundredth").is_empty());
    assert!(ord("fifths").is_empty());
}

#[test]
fn relative_ordinals() {
    let o = |t: &str| found(t, Output::Ordinal);
    assert_eq!(o("the last page"), pair("last", "end+0"));
    assert_eq!(o("the next one"), pair("next one", "current+1"));
    assert_eq!(o("the previous one"), pair("previous one", "current-1"));
    assert_eq!(o("the penultimate"), pair("penultimate", "end-1"));
    assert_eq!(o("the antepenultimate"), pair("antepenultimate", "end-2"));
    assert_eq!(o("the third to last"), pair("the third to last", "end-2"));
    assert_eq!(o("next to last"), pair("next to last", "end-1"));
    assert_eq!(o("the last but one"), pair("the last but one", "end-1"));
    assert_eq!(o("the one before the last"), pair("the one before the last", "end-1"));
}

#[test]
fn percentages() {
    let p = |t: &str| found(t, Output::Percent);
    assert_eq!(p("up 45% today"), pair("45%", "45%"));
    assert_eq!(p("45 %"), pair("45 %", "45%"));
    assert_eq!(p("fifty percent"), pair("fifty percent", "50%"));
    assert_eq!(p("12.5 per cent"), pair("12.5 per cent", "12.5%"));
    assert_eq!(p("per cent of 10"), pair("per cent of 10", "10%"));
    assert_eq!(p("minus 3 percent"), pair("minus 3 percent", "-3%"));
    // The number of a percentage is a number too.
    assert_eq!(numbers("45%").len(), 1);
}

#[test]
fn web_addresses_hold_nothing() {
    assert!(found("see https://example.invalid/a?q=30%25%2020%", Output::Percent).is_empty());
    assert!(numbers("www.example.invalid/2020/5").is_empty());
}

#[test]
fn spans_are_byte_offsets_past_non_ascii_text() {
    let text = "café – twenty-one ½";
    let mut p = Parser::new(&ENGLISH);
    p.parse(text);
    let mut out = Vec::new();
    p.select(text, Output::Number, &mut out);
    let spans: Vec<(u32, u32)> = out.iter().map(|m| (m.start, m.end)).collect();
    assert_eq!(spans, [(10, 20), (21, 23)]);
}

#[test]
fn a_parser_is_reused_and_deterministic() {
    let texts = ["one hundred thirty fifths", "the second to last of 3 1/2", "", "  ", "minus"];
    let run = |p: &mut Parser, t: &str| {
        p.parse(t);
        let mut out = Vec::new();
        p.select(t, Output::Number, &mut out);
        out
    };
    let mut p = Parser::new(&ENGLISH);
    let first: Vec<_> = texts.iter().map(|t| run(&mut p, t)).collect();
    for (t, want) in texts.iter().zip(&first) {
        assert_eq!(&run(&mut p, t), want, "{t:?}");
        assert_eq!(&run(&mut Parser::new(&ENGLISH), t), want, "{t:?}");
    }
}

#[test]
fn ties_prefer_a_compound_ordinal_denominator() {
    // 100/35, not 130/5.
    assert_eq!(one("one hundred thirty fifths"), "2.857142857 fraction");
    // A mixed number wins over an improper fraction.
    assert_eq!(one("one hundred and three quarters"), "100.75 fraction");
}

#[test]
fn long_runs_of_number_words_stay_bounded() {
    let text = "one ".repeat(2000) + &"and a half ".repeat(200) + &"hundred thousand ".repeat(500);
    let mut p = Parser::new(&ENGLISH);
    p.parse(&text);
    assert!(p.chart().nodes.len() < 200_000, "{} nodes", p.chart().nodes.len());
    let mut out = Vec::new();
    p.select(&text, Output::Number, &mut out);
    assert!(!out.is_empty());
}

#[test]
fn the_lexicon_is_lowercase_without_repeats() {
    use crate::engine::Grammar;
    let mut seen = std::collections::BTreeSet::new();
    for e in super::lexicon::ENTRIES {
        assert_eq!(e.word, e.word.to_lowercase(), "{}", e.word);
        assert!(seen.insert((e.word, e.dim)), "{} twice in one class", e.word);
        assert_eq!(ENGLISH.lexicon().lookup(e.word).iter().filter(|x| x.dim == e.dim).count(), 1);
    }
    assert!(ENGLISH.lexicon().lookup("eleventy").is_empty());
}
