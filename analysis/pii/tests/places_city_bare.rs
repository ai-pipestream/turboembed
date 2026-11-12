//! #2097: bare city names in running prose, with no state after them and
//! no label before them. A bare name is read only where it is unlikely to
//! be a common word, a person or part of a longer name. Places are real;
//! the sentences are made up.

mod support;

use support::*;
use turbo_pii::Pack;

/// The written text and normalized value of each city mention.
fn cities(text: &str) -> Vec<(String, String)> {
    rows(&pack(Pack::Places), text)
        .into_iter()
        .filter(|r| r.2 == "city")
        .map(|(s, e, _, value)| (text[s as usize..e as usize].to_owned(), value))
        .collect()
}

fn written(text: &str) -> Vec<String> {
    cities(text).into_iter().map(|(w, _)| w).collect()
}

#[test]
fn a_bare_name_in_prose_is_a_mention() {
    for (text, want) in [
        ("they stopped for lunch in Rouen.", "Rouen"),
        ("her long-time home in the Boston area", "Boston"),
        ("When the Pasadena native moved back", "Pasadena"),
        ("Chicago is large", "Chicago"),
        ("In Paris, the rain stopped.", "Paris"),
        ("The Atlanta annex is closed.", "Atlanta"),
        ("three San Francisco chefs", "San Francisco"),
        ("SAN FRANCISCO -- How does it feel", "SAN FRANCISCO"),
        ("Thursday at San Francisco's tony restaurant", "San Francisco"),
        ("Thursday at San Francisco\u{2019}s tony restaurant", "San Francisco"),
        ("winters spent in Santa Barbara.", "Santa Barbara"),
        ("Harry Denton, San Francisco nightclub owner", "San Francisco"),
        ("(Rouen)", "Rouen"),
        ("\"Rouen\"", "Rouen"),
        ("a trip to Kansas City last May", "Kansas City"),
        ("born in S\u{e3}o Paulo in 1990", "S\u{e3}o Paulo"),
        ("Indianapolis traffic", "Indianapolis"),
        // A verb of speech after a name in a place phrase leaves it a
        // place (#2225 review).
        ("people in Paris said so", "Paris"),
        ("a source in Boston told reporters", "Boston"),
        ("they wrote from Rouen", "Rouen"),
        ("letters to Chicago wrote back", "Chicago"),
        // A hyphenated name is one name, never its last word alone.
        ("a Winston-Salem address", "Winston-Salem"),
    ] {
        assert_eq!(written(text), [want], "{text:?}");
    }
    assert_eq!(cities("born in S\u{e3}o Paulo")[0].1, "Sao Paulo");
}

#[test]
fn a_list_of_bare_names_is_one_mention_each() {
    let text = "sail between Istanbul and Barcelona, Rome and Athens, or Athens and Istanbul.";
    assert_eq!(written(text), ["Istanbul", "Barcelona", "Rome", "Athens", "Athens", "Istanbul"]);
}

#[test]
fn a_common_word_or_name_is_no_bare_mention() {
    for text in [
        "Price shown is for an outside cabin.",
        "the Guide to the museum",
        "Reading is fun",
        "the Mobile app",
        "Phoenix rose from the ashes",
        "Ask Jackson about it.",
        "Florence said no.",
        "Nice work",
        "Split the bill.",
        // A state's name is the state scanner's, alone or in capitals.
        "In Virginia, residents may ask.",
        "moved to New York last year",
        "IN FLORIDA, A COPY",
        "the Georgia office",
        // A country's name, though a city shares it.
        "IBAN formats of Luxembourg and Kuwait",
        "DJIBOUTI",
        // The Antarctic stations carry country names, and small places
        // fall below the population floor.
        "Australia was the first nation to",
        "Ende of the line",
        // Short one-word names.
        "Van de Sande",
        "a PO Box",
        "the UK office",
    ] {
        assert!(cities(text).is_empty(), "{text:?}: {:?}", cities(text));
    }
}

#[test]
fn a_bare_name_inside_a_longer_name_is_no_mention() {
    for text in [
        // An organisation and a longer place name.
        "MUST CREDIT THE SAN FRANCISCO CHRONICLE",
        "c. 2002 The San Francisco Chronicle",
        "New York City buses",
        "moved to Orleans Parish, Louisiana",
        "she lived in Prince George\u{2019}s County, Maryland",
        "a copy from the Bethel Census Area clinic",
        "the Paris Hilton story",
        // A person: a capitalized word before the name.
        "said Harry Denton.",
        "University of Bristol",
        "the Federal Reserve Bank of Boston",
        "chefs like Alice Waters",
        "Mr. Lincoln arrived",
        // A person: a verb of speech after the name.
        "Madison said the vote would wait.",
        "Paris told her to wait",
        "When Columbus wrote back, nobody answered.",
        // Joined to a word, a path or an address.
        "the Grand-Rouen plan",
        "see example.com/Rouen for more",
        "Rouen.example.com",
        "mail Rouen@example.com",
        "the #Rouen tag",
        "Rouen-based",
        "x_Rouen",
    ] {
        assert!(cities(text).is_empty(), "{text:?}: {:?}", cities(text));
    }
}

#[test]
fn a_bare_name_before_a_state_is_left_to_the_state_rule() {
    // The anchored reading refuses a state that holds no place of that
    // name, and the bare reading does not take it back.
    for text in ["Rouen, IL", "Paris, Alaska", "City: Rome, NY"] {
        assert!(cities(text).is_empty(), "{text:?}: {:?}", cities(text));
    }
    assert_eq!(written("Springfield, IL 62704"), ["Springfield"]);
}
