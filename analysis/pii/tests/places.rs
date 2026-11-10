//! #1886: United States states (`UsStatePiiExtractor`) and counties
//! (`UsCountyPiiExtractor`), the opt-in Places pack. Where each mention
//! starts and ends, its normalized value, what is not a mention, and
//! masking. Places are real; the sentences are made up.

mod support;

use support::*;
use turbo_pii::rewrite::mask_by_kind;
use turbo_pii::{Pack, PiiKind};

/// The written text, kind and normalized value of each mention.
fn found(text: &str) -> Vec<(String, &'static str, String)> {
    rows(&pack(Pack::Places), text)
        .into_iter()
        .map(|(s, e, kind, value)| (text[s as usize..e as usize].to_owned(), kind, value))
        .collect()
}

fn one(text: &str) -> (String, &'static str, String) {
    let mut f = found(text);
    assert_eq!(f.len(), 1, "{text:?}: {f:?}");
    f.remove(0)
}

#[test]
fn state_names_read_as_written_or_in_capitals() {
    for (text, written, code) in [
        ("She moved to Illinois last year.", "Illinois", "IL"),
        ("Born in New York, raised abroad", "New York", "NY"),
        ("from WEST VIRGINIA", "WEST VIRGINIA", "WV"),
        ("Residence: Puerto Rico", "Puerto Rico", "PR"),
        ("a clinic in the U.S. Virgin Islands.", "U.S. Virgin Islands", "VI"),
        ("visited the Virgin Islands", "Virgin Islands", "VI"),
        ("works in Washington, D.C. now", "Washington, D.C.", "DC"),
        ("works in Washington DC now", "Washington DC", "DC"),
        ("the District of Columbia office", "District of Columbia", "DC"),
        ("Texas's border", "Texas", "TX"),
        // A county word that makes no county leaves the state.
        ("We live in the Texas area now.", "Texas", "TX"),
        ("Moved to the Washington area last year.", "Washington", "WA"),
        ("The Ohio area office", "Ohio", "OH"),
        ("Illinois County does not exist.", "Illinois", "IL"),
        ("Florida Parish records", "Florida", "FL"),
        ("New York borough presidents", "New York", "NY"),
        ("the Virginia municipality rules", "Virginia", "VA"),
    ] {
        assert_eq!(one(text), (written.to_owned(), "state", code.to_owned()), "{text:?}");
    }
    let f = found("Born in New York, raised in Ohio");
    assert_eq!(f[1], ("Ohio".to_owned(), "state", "OH".to_owned()));
}

#[test]
fn what_is_no_state_name() {
    for text in [
        "new york in small letters",
        "Indianapolis is a city",
        "a Kansan farmer",
        "New York City traffic",
        "Washington County records",
        "Georgian wine",
        "NewYork",
    ] {
        let states: Vec<_> = found(text).into_iter().filter(|m| m.1 == "state").collect();
        assert!(states.is_empty(), "{text:?}: {states:?}");
    }
}

#[test]
fn a_code_needs_a_state_label_or_a_zip_code() {
    assert_eq!(one("Springfield, IL 62704"), ("IL".to_owned(), "state", "IL".to_owned()));
    assert_eq!(one("State: OR"), ("OR".to_owned(), "state", "OR".to_owned()));
    assert_eq!(one("state of residence: ME"), ("ME".to_owned(), "state", "ME".to_owned()));
    assert_eq!(one("Springfield, IL 62704-1234."), ("IL".to_owned(), "state", "IL".to_owned()));
    assert_eq!(one("Portland, OR 97201."), ("OR".to_owned(), "state", "OR".to_owned()));
    for text in [
        "IN the morning, OR later, ME too, OK?",
        "Springfield, IL",
        "IL 627041",
        "XX 62704",
        "il 62704",
        "State: Il",
        "LOG IN 12345abc",
        "LOG IN 12345\u{e9}t\u{e9}",
        "IN 12345-",
        "IN 12345_x",
        "IN 12345-12",
        "LOG IN 12345.67",
        "IN 12345,678",
        "IN 12345/6",
        "AMOUNT PAID IN 25000.00",
    ] {
        assert!(found(text).is_empty(), "{text:?}: {:?}", found(text));
    }
}

#[test]
fn county_names_end_in_their_own_word() {
    for (text, written, name) in [
        ("Seen at Cook County Hospital", "Cook County", "Cook County"),
        ("lives in Orleans Parish.", "Orleans Parish", "Orleans Parish"),
        ("Matanuska-Susitna Borough", "Matanuska-Susitna Borough", "Matanuska-Susitna Borough"),
        ("the Bethel Census Area clinic", "Bethel Census Area", "Bethel Census Area"),
        ("Juneau City and Borough", "Juneau City and Borough", "Juneau City and Borough"),
        ("(Miami-Dade County)", "Miami-Dade County", "Miami-Dade County"),
        ("PRINCE GEORGE'S COUNTY", "PRINCE GEORGE'S COUNTY", "Prince George's County"),
        ("Prince George\u{2019}s County", "Prince George\u{2019}s County", "Prince George's County"),
        ("in Cook county", "Cook county", "Cook County"),
        ("St. Louis County", "St. Louis County", "St. Louis County"),
        ("from Mayagüez Municipio", "Mayagüez Municipio", "Mayagüez Municipio"),
        ("Doña Ana County", "Doña Ana County", "Doña Ana County"),
        ("Washington County records", "Washington County", "Washington County"),
        ("in\u{a0}Cook County", "Cook County", "Cook County"),
        ("in\u{2003}Orleans Parish", "Orleans Parish", "Orleans Parish"),
        ("County:Cook County", "Cook County", "Cook County"),
        ("see/Cook County", "Cook County", "Cook County"),
        ("O'Brien County", "O'Brien County", "O'Brien County"),
    ] {
        assert_eq!(one(text), (written.to_owned(), "county", name.to_owned()), "{text:?}");
    }
}

#[test]
fn what_is_no_county_name() {
    for text in [
        "the county fair",
        "Gotham County",
        "cook county in small letters",
        "Cook Countywide",
        "Baltimore city",
        "a big Area",
        "Cook  County",
        "the\u{a0}bay area",
        "\u{3000}Area",
        "x_Cook County",
        "e\u{301}Cook County",
        "\u{e9}Cook County",
    ] {
        assert!(found(text).is_empty(), "{text:?}: {:?}", found(text));
    }
}

#[test]
fn places_are_low_opt_in_and_masked_keeping_their_spaces() {
    let text = "Cook County, Illinois";
    assert_eq!(
        found(text).into_iter().map(|m| (m.1, m.2)).collect::<Vec<_>>(),
        [("county", "Cook County".to_owned()), ("state", "IL".to_owned())]
    );
    assert_eq!(mask_by_kind(text, &layer(&pack(Pack::Places), text), '*').unwrap(), "**** ******, ********");
    // The wide pack does not include them.
    assert!(rows(&pack(Pack::AllStructured), text).is_empty());
    assert_eq!(PiiKind::State.builtin_name(), Some("state"));
    assert_eq!(PiiKind::County.builtin_name(), Some("county"));
}

#[test]
fn no_word_character_touches_either_end_of_a_place_name() {
    // #2089: a letter, digit, combining mark or `_` at either end joins the
    // name to another word, for states and counties alike.
    for text in [
        "x_Illinois",
        "Illinois_x",
        "e\u{301}Illinois",
        "Illinois\u{301}",
        "x_Cook County",
        "Cook County_x",
        "e\u{301}Cook County",
        "Cook County\u{301}",
        "Washington_County",
        "Washington County_x",
        "New York City_x",
        "x_IL 62704",
        "State: IL_x",
        "State: IL\u{301}",
        "State: e\u{301}IL",
        // #2096: a District suffix joined to another word is no District,
        // and the `Washington` before it is no state.
        "Washington DC_x",
        "Washington DC9",
        "Washington, DC\u{301}",
        "Washington, D.C.x",
        "Washington D.C._x",
    ] {
        assert!(found(text).is_empty(), "{text:?}: {:?}", found(text));
    }
    // The District and the plain state keep their readings.
    for (text, written, code) in [
        ("Washington DC", "Washington DC", "DC"),
        ("Washington, D.C.", "Washington, D.C.", "DC"),
        ("(Washington, DC)", "Washington, DC", "DC"),
        ("Washington D.C.-based", "Washington D.C.", "DC"),
        ("Washington", "Washington", "WA"),
        ("Washington, Dana said", "Washington", "WA"),
        ("Washington D", "Washington", "WA"),
    ] {
        assert_eq!(one(text), (written.to_owned(), "state", code.to_owned()), "{text:?}");
    }
    // Punctuation is no word character, at either end.
    for (text, written, kind) in [
        ("Illinois-based", "Illinois", "state"),
        ("pre-Illinois", "Illinois", "state"),
        ("(Illinois)", "Illinois", "state"),
        ("Cook County-based", "Cook County", "county"),
        ("\"Cook County\"", "Cook County", "county"),
        ("State: IL.", "IL", "state"),
    ] {
        let (w, k, _) = one(text);
        assert_eq!((w.as_str(), k), (written, kind), "{text:?}");
    }
}

#[test]
fn decomposed_county_spellings_read_as_the_table_writes_them() {
    // #2081: `n` + U+0303 and `u` + U+0308 rather than `ñ` and `ü`. The
    // mention keeps the source's bytes; the value is the table's NFC name.
    for (text, written, name) in [
        ("from Don\u{303}a Ana County.", "Don\u{303}a Ana County", "Doña Ana County"),
        ("in Mayagu\u{308}ez Municipio", "Mayagu\u{308}ez Municipio", "Mayagüez Municipio"),
        ("MAYAGU\u{308}EZ MUNICIPIO", "MAYAGU\u{308}EZ MUNICIPIO", "Mayagüez Municipio"),
        ("DON\u{303}A ANA county", "DON\u{303}A ANA county", "Doña Ana County"),
    ] {
        assert_eq!(one(text), (written.to_owned(), "county", name.to_owned()), "{text:?}");
    }
    // A mark the name does not hold is still no name, and a mark joined
    // to its end keeps the #2089 rule.
    for text in ["Don\u{301}a Ana County", "Doña Ana County\u{303}", "e\u{301}Doña Ana County"] {
        assert!(found(text).is_empty(), "{text:?}: {:?}", found(text));
    }
}
