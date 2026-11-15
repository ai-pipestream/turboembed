//! #1886: cities (`CityPiiExtractor`) in the opt-in Places pack. A city
//! is a place of the bundled gazetteer named before its state or after a
//! city label, or a bare name that is no common word
//! (`places_city_bare.rs`). Places are real; the sentences are made up.

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
        ("Winston Salem, NC", "Winston Salem", "Winston-Salem"),
        ("Nashville-Davidson, TN", "Nashville-Davidson", "Nashville"),
        ("Phoenix-Mesa, AZ", "Phoenix-Mesa", "Phoenix"),
        ("(Kansas City, MO)", "Kansas City", "Kansas City"),
        ("City: Springfield, IL", "Springfield", "Springfield"),
        ("Truth or Consequences, NM", "Truth or Consequences", "Truth or Consequences"),
        ("in Truth or Consequences, New Mexico", "Truth or Consequences", "Truth or Consequences"),
        ("Fond du Lac, WI", "Fond du Lac", "Fond du Lac"),
        ("moved to Fond du Lac, Wisconsin", "Fond du Lac", "Fond du Lac"),
        ("Coeur d'Alene, ID", "Coeur d'Alene", "Coeur d'Alene"),
        ("in Coeur d'Alene, Idaho", "Coeur d'Alene", "Coeur d'Alene"),
        ("COEUR D'ALENE, ID", "COEUR D'ALENE", "Coeur d'Alene"),
        // The territories, which the gazetteer files under their own codes.
        ("San Juan, PR", "San Juan", "San Juan"),
        ("in San Juan, Puerto Rico", "San Juan", "San Juan"),
        ("Ponce, PR 00731", "Ponce", "Ponce"),
        ("Mayag\u{fc}ez, PR", "Mayag\u{fc}ez", "Mayaguez"),
        ("Mayagu\u{308}ez, PR", "Mayagu\u{308}ez", "Mayaguez"),
        ("Agana, GU", "Agana", "Agana"),
        ("Pago Pago, AS", "Pago Pago", "Pago Pago"),
        ("Christiansted, VI", "Christiansted", "Christiansted"),
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
        ("City: Coeur d'Alene", "Coeur d'Alene", "Coeur d'Alene"),
        ("town of Coeur d'Alene", "Coeur d'Alene", "Coeur d'Alene"),
        ("City: Val d'Or", "Val d'Or", "Val d'Or"),
        ("City: Winston Salem", "Winston Salem", "Winston-Salem"),
        ("City: Beersheba", "Beersheba", "Beer Sheva"),
        ("City: Dhahran", "Dhahran", "Az Zahran"),
        ("City: Southend", "Southend", "Southend-on-Sea"),
        ("City: Nashville-Davidson", "Nashville-Davidson", "Nashville"),
        ("City: Phoenix-Mesa", "Phoenix-Mesa", "Phoenix"),
        ("Birthplace: S\u{e3}o Paulo", "S\u{e3}o Paulo", "Sao Paulo"),
        ("place of birth Lyon", "Lyon", "Lyon"),
        ("CITY OF MOBILE", "MOBILE", "Mobile"),
        ("City : Phoenix", "Phoenix", "Phoenix"),
        ("Town:\tSpringfield", "Springfield", "Springfield"),
        ("the city of Mobile is on the bay", "Mobile", "Mobile"),
        ("City: Springfield IL 62704", "Springfield", "Springfield"),
        ("City: Mobile Alabama", "Mobile", "Mobile"),
        ("Birthplace: Chicago Illinois", "Chicago", "Chicago"),
        ("City: Truth or Consequences New Mexico", "Truth or Consequences", "Truth or Consequences"),
        ("City: Coeur d'Alene ID 83814", "Coeur d'Alene", "Coeur d'Alene"),
        ("City: Coeur d'Alene Idaho", "Coeur d'Alene", "Coeur d'Alene"),
        ("City: San Juan PR 00901", "San Juan", "San Juan"),
    ] {
        assert_eq!(one(text), (written.to_owned(), name.to_owned()), "{text:?}");
    }
}

#[test]
fn what_is_no_city_mention() {
    for text in [
        // A bare name that is a common word needs a state or a label
        // (bare names are in `places_city_bare.rs`).
        "Phoenix rose from the ashes",
        "my Mobile phone",
        // The state must hold a place of that name.
        "Springfield, Alaska",
        "Mobile, IL",
        "City: Mobile, IL",
        "City: Mobile Illinois",
        "City: Mobile IL 62704",
        "City: Coeur d'Alene IL 62704",
        // No label without its colon.
        "the city Council met",
        "town Hall",
        // Small letters, joined characters and a gazetteer miss.
        "springfield, IL",
        "coeur d'alene, ID",
        "City: coeur d'Alene",
        "Fond du lac, WI",
        "Salt lake City, Utah",
        "Winston salem, NC",
        "Winston-salem, NC",
        "Nashville-davidson, TN",
        "Phoenix-mesa, AZ",
        "x_Springfield, IL",
        "Springfield_x, IL",
        "Gotham, NY",
        "City: Gotham",
        // A malformed state anchors nothing; Mobile is a common word, so
        // no bare reading takes it either.
        "Mobile, ALA",
        "Mobile,AL",
        "Mobile , AL",
        "Kansas  City, MO",
        "the Kansas City Chiefs",
        "City, MO",
        // A country's code is no state's: Toronto is in Canada.
        "Toronto, CA",
        // After a label, a known name that a capitalized word follows is
        // the start of a longer one, which the gazetteer does not hold.
        "City: San Germ\u{e1}n",
        "City: San Foo",
        "the city of Paris Hilton",
        "City: Springfield IL",
        "City: Coeur d'Alene ID",
        "City: coeur d'Alene ID 83814",
    ] {
        assert!(cities(text).is_empty(), "{text:?}: {:?}", cities(text));
    }
}

#[test]
fn a_city_and_its_state_are_two_mentions() {
    for (text, city, state) in [
        ("Springfield, Illinois", "Springfield", "IL"),
        ("Coeur d'Alene, Idaho", "Coeur d'Alene", "ID"),
        ("City: Springfield IL 62704", "Springfield", "IL"),
        ("City: Mobile Alabama", "Mobile", "AL"),
        ("City: Coeur d'Alene Idaho", "Coeur d'Alene", "ID"),
    ] {
        let rows = rows(&pack(Pack::Places), text);
        let kinds: Vec<_> = rows.iter().map(|r| (r.2, r.3.as_str())).collect();
        assert_eq!(kinds, [("city", city), ("state", state)], "{text:?}");
    }
}

#[test]
fn a_run_of_separators_is_walked_once() {
    // Linear step scaling across state, county and city is checked at n and 8n
    // in `scan::places_tests` (#2112); here we check the separator forms
    // themselves at a bounded length so a regression fails fast rather than
    // hanging until the CI job timeout.
    let n = 4_000;
    for text in [":".repeat(n), format!("City{}", ":".repeat(n)), " : ".repeat(n / 3), "City #= ".repeat(n / 8)] {
        assert!(cities(&text).is_empty());
    }
    assert_eq!(one(&format!("City{}Phoenix", ": ".repeat(1000))), ("Phoenix".to_owned(), "Phoenix".to_owned()));
    assert_eq!(one("City #: Phoenix"), ("Phoenix".to_owned(), "Phoenix".to_owned()));
    assert_eq!(one("City::Phoenix"), ("Phoenix".to_owned(), "Phoenix".to_owned()));
}
