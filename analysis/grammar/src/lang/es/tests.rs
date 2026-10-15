//! The Spanish rules on texts of their own. The Recognizers-Text cases are
//! in `tests/spec/spanish.rs`.

use crate::engine::{Output, Parser};
use crate::value::Value;

use super::{SPANISH, split};

fn show(v: Value) -> String {
    match v {
        Value::Num(n) => format!("{} {}", n.value, n.kind.name()),
        Value::Ordinal(o) => format!("#{}", o.value),
        Value::Relative(r) => format!("{}{:+}", r.to.name(), r.offset),
        Value::Percent(p) => format!("{p}%"),
        v => format!("{v:?}"),
    }
}

fn found(text: &str, output: Output) -> Vec<(String, String)> {
    let mut p = Parser::new(&SPANISH);
    p.parse(text);
    let mut out = Vec::new();
    p.select(text, output, &mut out);
    out.iter().map(|m| (text[m.start as usize..m.end as usize].to_string(), show(m.value))).collect()
}

fn pair(a: &str, b: &str) -> Vec<(String, String)> {
    vec![(a.to_string(), b.to_string())]
}

/// The value of the one number that covers all of `text`.
fn one(text: &str) -> String {
    let all = found(text, Output::Number);
    assert_eq!(all.len(), 1, "{text:?}: {all:?}");
    assert_eq!(all[0].0, text, "{text:?}: the whole text");
    all[0].1.clone()
}

fn ord(text: &str) -> String {
    let all = found(text, Output::Ordinal);
    assert_eq!(all.len(), 1, "{text:?}: {all:?}");
    assert_eq!(all[0].0, text, "{text:?}: the whole text");
    all[0].1.clone()
}

/// The pieces `split` cuts `word` into.
fn pieces(word: &str) -> Vec<&str> {
    let mut cuts = Vec::new();
    split::split(super::Grammar::lexicon(&SPANISH), word, &mut cuts);
    let mut out = Vec::new();
    let mut at = 0;
    for c in cuts {
        out.push(&word[at..c as usize]);
        at = c as usize;
    }
    out.push(&word[at..]);
    out
}

#[test]
fn spelled_integers_agree_in_gender() {
    assert_eq!(one("veintiuna"), "21 integer");
    assert_eq!(one("veintiún"), "21 integer");
    assert_eq!(one("cuarenta y dos"), "42 integer");
    assert_eq!(one("doscientas una"), "201 integer");
    assert_eq!(one("ciento uno"), "101 integer");
    assert_eq!(one("mil y una"), "1001 integer");
    assert_eq!(one("dos mil veintiuno"), "2021 integer");
    assert_eq!(one("un millón"), "1000000 integer");
    assert_eq!(one("mil millones"), "1000000000 integer");
    assert_eq!(one("dos billones"), "2000000000000 integer");
}

#[test]
fn a_scale_agrees_with_its_count() {
    assert!(found("un millones", Output::Number).iter().all(|n| n.0 != "un millones"));
    assert!(found("dos millón", Output::Number).iter().all(|n| n.0 != "dos millón"));
}

#[test]
fn digits_take_a_decimal_comma() {
    assert_eq!(one("1.234,56"), "1234.56 decimal");
    assert_eq!(one("3,5"), "3.5 decimal");
    assert_eq!(one("1.234"), "1234 integer");
    assert_eq!(one("1.000.000"), "1000000 integer");
    assert_eq!(one("2,5 millones"), "2500000 integer");
}

#[test]
fn spoken_decimals_and_fractions() {
    assert_eq!(one("siete coma cinco"), "7.5 decimal");
    assert_eq!(one("tres quintos"), "0.6 fraction");
    assert_eq!(one("un quinto"), "0.2 fraction");
    assert_eq!(one("cinco con un cuarto"), "5.25 fraction");
    assert_eq!(one("medio millón"), "500000 fraction");
    // A plural ordinal after a count is a denominator.
    assert!(found("tres quintos", Output::Ordinal).is_empty());
}

#[test]
fn menos_negates_only_where_it_opens() {
    assert_eq!(found("menos cinco grados", Output::Number), pair("menos cinco", "-5 integer"));
    assert_eq!(found("al menos cinco", Output::Number), pair("cinco", "5 integer"));
}

#[test]
fn ordinals() {
    assert_eq!(ord("1.º"), "#1");
    assert_eq!(ord("2da"), "#2");
    assert_eq!(ord("decimotercero"), "#13");
    assert_eq!(ord("vigésimoprimero"), "#21");
    assert_eq!(ord("Vigésimo Primera"), "#21");
    assert_eq!(ord("tresmillonésimo"), "#3000000");
    assert_eq!(found("el primer paso", Output::Ordinal), pair("primer", "#1"));
    // `un quinto` is a fraction.
    assert!(found("un quinto", Output::Ordinal).is_empty());
}

#[test]
fn relative_ordinals() {
    assert_eq!(ord("penúltimo"), "end-1");
    assert_eq!(found("el último", Output::Ordinal), pair("último", "end+0"));
}

#[test]
fn percentages() {
    assert_eq!(found("el diez por ciento", Output::Percent), pair("diez por ciento", "10%"));
    assert_eq!(found("cien por cien", Output::Percent), pair("cien por cien", "100%"));
    assert_eq!(found("15 %", Output::Percent), pair("15 %", "15%"));
    // The hundred of `por ciento` is no number of its own.
    assert_eq!(found("el diez por ciento", Output::Number), pair("diez", "10 integer"));
}

#[test]
fn accents_fold_and_spans_stay_exact() {
    let text = "el décimo séptimo día";
    assert_eq!(found(text, Output::Ordinal), pair("décimo séptimo", "#17"));
}

#[test]
fn compounds_split_into_number_words() {
    assert_eq!(pieces("tresmillonesimo"), ["tres", "millonesimo"]);
    assert_eq!(pieces("doscientosavos"), ["doscientos", "avos"]);
    assert_eq!(pieces("treintaicincoavos"), ["treinta", "i", "cinco", "avos"]);
    assert_eq!(pieces("vigesimoprimero"), ["vigesimo", "primero"]);
    // A word of the lexicon, a word with no number in it and a stray `i`
    // stay whole.
    assert_eq!(pieces("dieciseis"), ["dieciseis"]);
    assert_eq!(pieces("casamiento"), ["casamiento"]);
    assert_eq!(pieces("doscientosiavos"), ["doscientosiavos"]);
}
