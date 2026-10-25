//! The English unit rules on texts of their own.

use super::ENGLISH_UNITS;
use crate::engine::{Output, Parser};
use crate::value::Value;

/// Renders each selected mention as `span = value unit`, or `span = unit` for a unit named alone.
fn found(text: &str) -> Vec<String> {
    let mut parser = Parser::new(&ENGLISH_UNITS);
    parser.parse(text);
    let mut out = Vec::new();
    parser.select(text, Output::Quantity, &mut out);
    out.iter()
        .map(|m| {
            let span = &text[m.start as usize..m.end as usize];
            match m.value {
                Value::Quantity(q) => format!("{span} = {} {}", q.value(), q.unit()),
                Value::Unit(u) => format!("{span} = {u}"),
                v => format!("{span} = {v:?}"),
            }
        })
        .collect()
}

fn check(text: &str, expected: &[&str]) {
    assert_eq!(found(text), expected, "{text}");
}

#[test]
fn lengths() {
    check("a 10 km run", &["10 km = 10 Kilometer"]);
    check("it is 3.5 meters long", &["3.5 meters = 3.5 Meter"]);
    check("two miles away", &["two miles = 2 Mile"]);
    check("a 12-inch pizza", &["12-inch = 12 Inch"]);
}

#[test]
fn feet_and_inches() {
    check("the 6 ft 2 in player", &["6 ft 2 in = 6.1666666666666666666666666666666667 Foot"]);
    check("he is 5'11\" tall", &["5'11\" = 5.9166666666666666666666666666666667 Foot"]);
}

#[test]
fn areas_and_volumes() {
    check("a 40 square meter flat", &["40 square meter = 40 Square meter"]);
    check("50 km2 of forest", &["50 km2 = 50 Square kilometer"]);
    check("add 2 cubic feet of sand", &["2 cubic feet = 2 Cubic foot"]);
    check("drink 3 liters a day", &["3 liters = 3 Liter"]);
}

#[test]
fn weights() {
    check("a 20 kg bag", &["20 kg = 20 Kilogram"]);
    check("2 pounds of apples", &["2 pounds = 2 Pound"]);
    check("five tonnes of steel", &["five tonnes = 5 Metric ton"]);
}

#[test]
fn tons_say_which_ton() {
    check("3.5 tonnes of grain", &["3.5 tonnes = 3.5 Metric ton"]);
    check("a 100t crane", &["100t = 100 Metric ton"]);
    check("16,000 metric tons", &["16,000 metric tons = 16000 Metric ton"]);
    check("a three-ton truck", &["three-ton = 3 Short ton (US)"]);
    check("2 short tons", &["2 short tons = 2 Short ton (US)"]);
    check("2 US tons", &["2 US tons = 2 Short ton (US)"]);
    check("2 long tons", &["2 long tons = 2 Long ton (British)"]);
    check("2 British tons", &["2 British tons = 2 Long ton (British)"]);
}

#[test]
fn imperial_measures_when_the_text_says_so() {
    check("5 gallons", &["5 gallons = 5 Gallon"]);
    check("5 US gallons", &["5 US gallons = 5 Gallon"]);
    check("5 imperial gallons", &["5 imperial gallons = 5 Imperial gallon"]);
    check("2 UK pints", &["2 UK pints = 2 Imperial pint"]);
    check("10 imperial fluid ounces", &["10 imperial fluid ounces = 10 Imperial fluid ounce"]);
    check("give us 2 pints", &["2 pints = 2 Pint"]);
}

#[test]
fn nautical_miles() {
    check("12 nautical miles offshore", &["12 nautical miles = 12 Nautical mile"]);
    check("12 nmi", &["12 nmi = 12 Nautical mile"]);
    check("12 miles", &["12 miles = 12 Mile"]);
}

#[test]
fn speeds_information_and_angles() {
    check("drove at 120 km/h", &["120 km/h = 120 Kilometer per hour"]);
    check("a 2 GB file", &["2 GB = 2 Gigabyte"]);
    check("turn 90 degrees to the left", &["90 degrees = 90 Degree"]);
}

#[test]
fn rates_other_than_speeds() {
    check("a 10 Mb/s link", &["10 Mb/s = 10 Megabit per second"]);
    check("5 megabytes per second", &["5 megabytes per second = 5 Megabyte per second"]);
    check("uploads at 2 GB an hour", &["2 GB an hour = 2 Gigabyte per hour"]);
    check("he earns $5 per hour", &["$5 per hour = 5 Dollar per hour"]);
    check("rent is 20 euros a day", &["20 euros a day = 20 Euro per day"]);
    check("paid 100 USD/month", &["100 USD/month = 100 United States dollar per month"]);
    check("up from $5 a year earlier", &["$5 = 5 Dollar"]);
    check("it moved 2 GB a year ago", &["2 GB = 2 Gigabyte"]);
}

#[test]
fn bare_compound_imperial_measures() {
    check("he is 6 foot 3", &["6 foot 3 = 6.25 Foot"]);
    check("he is 6 foot 3 tall", &["6 foot 3 = 6.25 Foot"]);
    check("she weighs 10 stone 7", &["10 stone 7 = 10.5 Stone"]);
    check("a baby of 5 pounds 8", &["5 pounds 8 = 5.5 Pound"]);
    check("say 6 foot 2 times a week", &["6 foot = 6 Foot"]);
    check("a 6 foot 12 board", &["6 foot = 6 Foot"]);
}

#[test]
fn temperatures() {
    check("it was minus 40 degrees outside", &["minus 40 degrees = -40 Degree"]);
    check("heat to 180 °C", &["180 °C = 180 C"]);
    check("water boils at 212 degrees fahrenheit", &["212 degrees fahrenheit = 212 F"]);
}

#[test]
fn units_named_alone() {
    check("how many kilometers is that", &["kilometers = Kilometer"]);
}

#[test]
fn ages() {
    check("a 5-year-old boy", &["5-year-old = 5 Year"]);
    check("she is 30 years old", &["30 years old = 30 Year"]);
}

#[test]
fn money_with_signs_and_codes() {
    check("it costs $3.5m", &["$3.5m = 3500000 Dollar"]);
    check("pay 20 EUR now", &["20 EUR = 20 Euro"]);
    check("a £5 note", &["£5 = 5 Pound"]);
}

#[test]
fn money_in_two_units() {
    check("it was 5 dollars and 50 cents", &["5 dollars and 50 cents = 5.5 Dollar"]);
    check("pay 15 USD 50.", &["15 USD 50 = 15.5 United States dollar"]);
    check("from 5 $10 bills", &["$10 = 10 Dollar"]);
}

#[test]
fn minor_units() {
    check("only 50 cents", &["50 cents = 50 Cent"]);
}

#[test]
fn words_that_are_not_units() {
    check("3 t-shirts", &[]);
    check("2 turn signals", &[]);
    check("all of them", &[]);
}
