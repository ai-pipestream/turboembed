//! #1886: United States street addresses and post office boxes
//! (`UsAddressPiiExtractor`), and ages in years (`AgePiiExtractor`).
//! Positive and negative cases per scanner, where each mention starts and
//! ends, the normalized age, the packs that report them, and masking.
//! Addresses and ages are made up.

mod support;

use support::*;
use turbo_pii::rewrite::mask_by_kind;
use turbo_pii::{Extractor, Pack, PiiDetector, PiiKind, Scanner};

fn only(e: Extractor) -> PiiDetector {
    PiiDetector::new(vec![Scanner::all(e)])
}

fn of_kind(d: &PiiDetector, text: &str, kind: &str) -> Vec<String> {
    rows(d, text).into_iter().filter(|r| r.2 == kind).map(|r| r.3).collect()
}

#[test]
fn street_addresses_run_from_the_house_number_to_the_suffix_or_unit() {
    let d = only(Extractor::UsAddress);
    for (text, want) in [
        ("Send it to 1600 Pennsylvania Avenue NW, Washington, DC 20500.", "1600 Pennsylvania Avenue NW"),
        ("She lives at 42 Elm St. in Springfield.", "42 Elm St"),
        ("Ship to 350 Fifth Ave, Suite 3300, New York", "350 Fifth Ave, Suite 3300"),
        ("Office: 12 N Main Street Apt. 4B", "12 N Main Street Apt. 4B"),
        ("1234 W 42nd St # 12, Chicago", "1234 W 42nd St # 12"),
        ("Visit 221B Baker Street today", "221B Baker Street"),
        ("at 77 Martin Luther King Jr Blvd", "77 Martin Luther King Jr Blvd"),
        ("Find us at 9 O'Connell Rd", "9 O'Connell Rd"),
        ("address 18 Mill Creek Road", "18 Mill Creek Road"),
        ("MAILING: 500 MARKET ST UNIT 7", "500 MARKET ST UNIT 7"),
        ("Avenida: 3 Calle Peña Way", "3 Calle Peña Way"),
        ("its 5 Park Avenue office", "5 Park Avenue"),
        ("THE HSUS, 2100 L ST., NW, WASHINGTON, DC", "2100 L ST., NW"),
        ("at 9 Elm St. North of the river", "9 Elm St"),
        ("at 9 Elm St, North Carolina", "9 Elm St"),
        ("at 3 Point Lane", "3 Point Lane"),
        ("Ship to 12 Main Street Boston", "12 Main Street"),
        ("at 1999 Main Street", "1999 Main Street"),
        ("at 2016 Central Park West", "2016 Central Park West"),
        ("at 2016 Central Park, Apt 4B", "2016 Central Park, Apt 4B"),
        ("at 2016 Central Park, New York", "2016 Central Park"),
        ("12\u{00A0}Main\u{00A0}St", "12\u{00A0}Main\u{00A0}St"),
        ("12\u{202F}Main\u{202F}St", "12\u{202F}Main\u{202F}St"),
    ] {
        assert_eq!(of_kind(&d, text, "street-address"), [want], "{text}");
    }
    for not in [
        "We sold 1600 units on the street.", // lower case words
        "Turn left on Main Street.",         // no house number
        "Route 66 is long",                  // the number is after the words
        "Call 555 Smith",                    // no suffix
        "In 1999 the court ruled",           // lower case suffix word
        "Table 4 Summary",                   // no suffix
        "Order 0 Main St",                   // a house number never starts with 0
        "12.5 Main St",                      // inside a decimal
        "1234567 Main Street",               // seven digits
        "the 5th Avenue store",              // an ordinal is no house number
        "12\nMain Street",                   // a line break
        "2016 Central Park",                 // year before a noun suffix
        "2018 Apple Park",                   // year before a noun suffix
        "2024 Annual Bay Summit",            // year before a noun suffix
        "2016 Central Park, where we met",   // comma before a lower case word, not a city
        "the 3 Point Plan",                  // suffix word with no street name before it
        "the 3 Five Point Plan",             // suffix word inside a capitalized phrase
        "the 12 Wall Street Journal",        // suffix word inside an article-led phrase
        "the 3 North Point Plan",            // direction and suffix word inside a phrase
    ] {
        assert!(of_kind(&d, not, "street-address").is_empty(), "{not}");
    }
}

#[test]
fn post_office_boxes_are_street_addresses() {
    let d = only(Extractor::UsAddress);
    for (text, want) in [
        ("Mail: PO Box 1234, Austin", "PO Box 1234"),
        ("P.O. Box 77", "P.O. Box 77"),
        ("p. o. box 9", "p. o. box 9"),
        ("P O BOX 300", "P O BOX 300"),
        ("Post Office Box 4410.", "Post Office Box 4410"),
        ("PO\u{00A0}Box\u{00A0}1234", "PO\u{00A0}Box\u{00A0}1234"),
        ("Post\u{202F}Office\u{202F}Box\u{202F}4410", "Post\u{202F}Office\u{202F}Box\u{202F}4410"),
    ] {
        assert_eq!(of_kind(&d, text, "street-address"), [want], "{text}");
    }
    for not in ["PO Box", "PO Box ABC", "POBox 12", "Polo Box 12", "SPO Box 12", "PO Box 12-3"] {
        assert!(of_kind(&d, not, "street-address").is_empty(), "{not}");
    }
}

#[test]
fn ages_are_the_number_with_its_age_words_outside() {
    let d = only(Extractor::Age);
    for (text, want, at) in [
        ("The patient is 34 years old.", "34", 15),
        ("a 34-year-old man", "34", 2),
        ("a 7-yr-old girl", "7", 2),
        ("He is 102 years old", "102", 6),
        ("Aged 61, retired", "61", 5),
        ("age: 47", "47", 5),
        ("Age 9", "9", 4),
        ("at the age of 18 she left", "18", 14),
        ("45 y/o female", "45", 0),
        ("45 y.o. female", "45", 0),
        ("aged 034", "34", 5),
        ("born 1990, 36 Years Old now", "36", 11),
    ] {
        let r = rows(&d, text);
        assert_eq!(r.len(), 1, "{text}: {r:?}");
        assert_eq!((r[0].2, r[0].3.as_str(), r[0].0), ("age", want, at), "{text}");
    }
    for not in [
        "page 34",         // not the word age
        "usage: 47",       // nor here
        "34 years ago",    // not old
        "aged 3 months",   // not years
        "age 3.5",         // a decimal
        "aged 200",        // over 125
        "1,034 years old", // a grouped number
        "34 years older",  // older
        "Ages 5 to 10",    // ages
        "34yo",            // no space before y/o
        "agent 34",        // agent
    ] {
        assert!(rows(&d, not).is_empty(), "{not}: {:?}", rows(&d, not));
    }
}

#[test]
fn contact_reports_street_addresses_and_all_structured_reports_both() {
    let text = "Lena, 52 years old, lives at 8 Oak Lane, Unit 2 (tel 202-555-0143)";
    let contact = pack(Pack::Contact);
    assert_eq!(of_kind(&contact, text, "street-address"), ["8 Oak Lane, Unit 2"]);
    assert!(of_kind(&contact, text, "age").is_empty(), "Contact has no age scanner");
    let wide = wide();
    assert_eq!(of_kind(&wide, text, "street-address"), ["8 Oak Lane, Unit 2"]);
    assert_eq!(of_kind(&wide, text, "age"), ["52"]);
    assert_eq!(PiiKind::parse("age").and_then(PiiKind::builtin_id), Some(39));
    assert_eq!(PiiKind::parse("street-address").and_then(PiiKind::builtin_id), Some(40));
}

#[test]
fn masking_keeps_the_age_words_and_the_address_separators() {
    let text = "a 34-year-old at 8 Oak Lane";
    let masked = mask_by_kind(text, &layer(&wide(), text), '*').unwrap();
    assert_eq!(masked, "a **-year-old at * *** ****");
}
