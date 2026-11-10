//! #1886: cities (`CityPiiExtractor`) in the opt-in Places pack. A city
//! is a place of the bundled gazetteer named before its state or after a
//! city label. Places are real; the sentences are made up.

mod support;

use support::*;
use turbo_pii::Pack;

/// The written text, kind and normalized value of each city mention.
fn cities(text: &str) -> Vec<(String, String)> {
    rows(&pack(Pack::Places), text)
        .into_iter()
        .filter(|r| r.2 == "city")
        .map(|(s, e, _, value)| (text[s as usize..e as usize].to_owned(), value))
        .collect()
}

fn one(text: &str) -> (String, String) {
    let mut f = cities(text);
    assert_eq!(f.len(), 1, "{text:?}: {f:?}");
    f.remove(0)
}

#[test]
fn a_city_before_its_state_is_a_mention() {
    for (text, written, name) in [
        ("Springfield, IL 62704", "Springfield", "Springfield"),
        ("born in Springfield, Illinois.", "Springfield", "Springfield"),
        ("moved to Mobile, Alabama last year", "Mobile", "Mobile"),
        ("Phoenix, AZ", "Phoenix", "Phoenix"),
        ("SPRINGFIELD, MO", "SPRINGFIELD", "Springfield"),
        ("in Salt Lake City, Utah", "Salt Lake City", "Salt Lake City"),
        ("St. Louis, Missouri", "St. Louis", "St. Louis"),
        ("Winston-Salem, NC", "Winston-Salem", "Winston-Salem"),
        ("(Kansas City, MO)", "Kansas City", "Kansas City"),
        ("City: Springfield, IL", "Springfield", "Springfield"),
    ] {
        assert_eq!(one(text), (written.to_owned(), name.to_owned()), "{text:?}");
    }
}

#[test]
fn a_city_after_a_city_label_is_a_mention() {
    for (text, written, name) in [
        ("City: Phoenix", "Phoenix", "Phoenix"),
        ("the city of Mobile", "Mobile", "Mobile"),
        ("City of residence: Chicago", "Chicago", "Chicago"),
        ("Place of birth: Lyon", "Lyon", "Lyon"),
        ("town of Springfield", "Springfield", "Springfield"),
        ("City: Rio de Janeiro", "Rio de Janeiro", "Rio de Janeiro"),
        ("Birthplace: S\u{e3}o Paulo", "S\u{e3}o Paulo", "Sao Paulo"),
    ] {
        assert_eq!(one(text), (written.to_owned(), name.to_owned()), "{text:?}");
    }
}

#[test]
fn what_is_no_city_mention() {
    for text in [
        // A bare name, however well known, needs a state or a label.
        "Phoenix rose from the ashes",
        "my Mobile phone",
        "Chicago is large",
        // The state must hold a place of that name.
        "Springfield, Alaska",
        "Mobile, IL",
        // No label without its colon.
        "the city Council met",
        "town Hall",
        // Small letters, joined characters and a gazetteer miss.
        "springfield, IL",
        "x_Springfield, IL",
        "Springfield_x, IL",
        "Gotham, NY",
        "City: Gotham",
        "Springfield, ILL",
        "Springfield,IL",
        "Springfield , IL",
        "Kansas  City, MO",
        "the Kansas City Chiefs",
        "City, MO",
    ] {
        assert!(cities(text).is_empty(), "{text:?}: {:?}", cities(text));
    }
}

#[test]
fn a_city_and_its_state_are_two_mentions() {
    let rows = rows(&pack(Pack::Places), "Springfield, Illinois");
    let kinds: Vec<_> = rows.iter().map(|r| (r.2, r.3.as_str())).collect();
    assert_eq!(kinds, [("city", "Springfield"), ("state", "IL")]);
}
