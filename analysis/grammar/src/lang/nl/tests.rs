//! The Dutch rules on texts of their own. The Recognizers-Text cases are
//! in `tests/spec/dutch.rs`.

use crate::engine::{Output, Parser};
use crate::value::Value;

use super::DUTCH;

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
    let mut p = Parser::new(&DUTCH);
    p.parse(text);
    let mut out = Vec::new();
    p.select(text, output, &mut out);
    out.iter().map(|m| (text[m.start as usize..m.end as usize].to_string(), show(m.value))).collect()
}

fn values(text: &str) -> Vec<String> {
    found(text, Output::Number).into_iter().map(|n| n.1).collect()
}

/// The value of the one number that covers all of `text`.
fn one(text: &str) -> String {
    let all = found(text, Output::Number);
    assert_eq!(all.len(), 1, "{text:?}: {all:?}");
    assert_eq!(all[0].0, text, "{text:?}: the whole text");
    all[0].1.clone()
}

#[test]
fn compound_integers() {
    assert_eq!(one("zevenenveertig"), "47 integer");
    assert_eq!(one("Tweeëntachtig"), "82 integer");
    assert_eq!(one("negenhonderdnegenennegentig"), "999 integer");
    assert_eq!(one("achttienhonderd"), "1800 integer");
    assert_eq!(one("duizend"), "1000 integer");
    assert_eq!(one("drie miljard vierhonderdduizend en elf"), "3000400011 integer");
}

fn ordinals(text: &str) -> Vec<(String, String)> {
    found(text, Output::Ordinal)
}

fn pair(a: &str, b: &str) -> (String, String) {
    (a.to_string(), b.to_string())
}

#[test]
fn numbers_in_sentences() {
    let n = found("Hij is drieënnegentig jaar oud en heeft vijfhonderdduizend euro.", Output::Number);
    assert_eq!(n, [pair("drieënnegentig", "93 integer"), pair("vijfhonderdduizend", "500000 integer")]);
    assert_eq!(values("Het duurde tweeënhalf uur, ongeveer anderhalve dag."), ["2.5 fraction", "1.5 fraction"]);
    assert_eq!(values("hoofdstuk zevenendertig, 1,5 miljoen euro"), ["37 integer", "1500000 integer"]);
    assert_eq!(one("negentienhonderdvierentachtig"), "1984 integer");
    assert_eq!(one("3,2 mln"), "3200000 integer");
    assert_eq!(one("2.500,50"), "2500.5 decimal");
}

#[test]
fn spaced_parts_of_a_compound_stay_apart() {
    // `acht en twintig` written apart is no compound.
    assert_eq!(values("acht en twintig"), ["8 integer", "20 integer"]);
    // Words that hold number words are no numbers.
    assert!(values("het eenmalige achten van tienen").is_empty());
}

#[test]
fn een_is_the_article_and_een_with_accents_the_number() {
    assert_eq!(values("Een man en één vrouw"), ["1 integer"]);
    assert!(values("de één en de ander").is_empty());
    assert_eq!(one("duizend en één"), "1001 integer");
    assert_eq!(one("min een"), "-1 integer");
}

#[test]
fn fractions() {
    assert_eq!(one("een kwart"), "0.25 fraction");
    assert_eq!(one("drie vijfde"), "0.6 fraction");
    assert_eq!(one("driekwart"), "0.75 fraction");
    assert_eq!(one("twintig drie vijfde"), "20.6 fraction");
    assert_eq!(one("vier op de tien"), "0.4 fraction");
    assert_eq!(one("zes miljoen en twee derde"), "6666666.666666667 fraction");
    // A ratio after `en` is no part of a mixed number.
    assert_eq!(values("vier en vier op de tien"), ["4 integer", "0.4 fraction"]);
    // An ordinal glued to a number is a position, not a denominator.
    assert!(values("honderdachtste").is_empty());
    assert_eq!(one("honderdduizend biljoensten"), "0.0000001 fraction");
}

#[test]
fn spoken_decimals() {
    assert_eq!(one("zeven komma nul vier"), "7.04 decimal");
    assert_eq!(one("min vijf komma vijf"), "-5.5 decimal");
}

#[test]
fn ordinals_in_words_and_digits() {
    let o = ordinals("tweede helft; 2de plaats, 21ste eeuw, 3e keer, de eenendertigste");
    let want = [("tweede", "#2"), ("2de", "#2"), ("21ste", "#21"), ("3e", "#3"), ("eenendertigste", "#31")];
    assert_eq!(o, want.map(|(a, b)| pair(a, b)));
    // The suffix has to be the one Dutch gives the number.
    assert!(ordinals("2ste 1de 11ste").is_empty());
    assert_eq!(ordinals("100ste 102de 8ste"), [pair("100ste", "#100"), pair("102de", "#102"), pair("8ste", "#8")]);
    assert_eq!(ordinals("vier miljoen vierhonderdduizendste"), [pair("vier miljoen vierhonderdduizendste", "#4400000")]);
    // `een miljoenste` is a fraction.
    assert!(ordinals("een miljoenste").is_empty());
}

#[test]
fn relative_ordinals() {
    assert_eq!(ordinals("de op twee na laatste"), [pair("de op twee na laatste", "end-2")]);
    assert_eq!(ordinals("de huidige"), [pair("huidige", "current+0")]);
}

#[test]
fn percentages() {
    let p = found("dertig procent, 4,5% en percentage van 12", Output::Percent);
    assert_eq!(p, [pair("dertig procent", "30%"), pair("4,5%", "4.5%"), pair("percentage van 12", "12%")]);
}
