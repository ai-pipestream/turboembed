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

#[test]
fn a_person_who_shares_a_city_name_is_no_mention() {
    // #2295: person context around a lone bare name.
    for text in [
        // A verb that takes a person as its object.
        "we met Madison at the desk",
        "please ask Columbus first",
        "we thanked Dayton for the flowers",
        "the firm hired Salem last spring",
        "they invited Paris to the party",
        // `who`, a reflexive pronoun, or a subject pronoun one word on.
        "Salem, who runs the class, is out.",
        "the note from a friend: Paris who knew",
        "Madison herself drove the van",
        "Dayton knew she would be late",
        "then Columbus thought he could help",
        "Salem realized she was late",
        "we will introduce Madison tomorrow",
        // A verb of each first letter the verb rule looks for.
        "we congratulated Madison",
        "we emailed Madison",
        "they fired Madison",
        "we greeted Madison",
        "they kissed Madison",
        "please remind Madison",
        "we texted Madison",
        "they phoned Madison",
        "they promote Dayton next year",
        // A possessive before a kin or body noun.
        "Paris's mother called",
        "Madison\u{2019}s sister wrote",
        "Columbus's hair was wet",
    ] {
        assert!(cities(text).is_empty(), "{text:?}: {:?}", cities(text));
    }
}

#[test]
fn a_city_in_a_place_phrase_stays_a_city_beside_person_words() {
    // #2295: each person rule leaves a city in a place phrase alone.
    for (text, want) in [
        ("we met in Madison last year", "Madison"),
        ("they asked to fly to Columbus", "Columbus"),
        ("hired from Dayton in May", "Dayton"),
        ("a donor in Columbus who funds it", "Columbus"),
        ("people from Paris, who wrote letters", "Paris"),
        ("a trip to Salem where he grew up", "Salem"),
        ("Boston where he lived", "Boston"),
        ("Paris is where she grew up", "Paris"),
        ("visited Paris last May", "Paris"),
        ("left Madison for good", "Madison"),
        ("Paris's museums are open", "Paris"),
        ("in Dayton she found work", "Dayton"),
        // A line break between the place word and the name.
        ("people in\nParis who wrote letters", "Paris"),
        ("a source in\r\nBoston told reporters", "Boston"),
        // A conjunction between the name and a pronoun starts a new clause.
        ("we loved Boston and she stayed", "Boston"),
        ("they toured Dayton but he left", "Dayton"),
        ("a week in Rome or he goes home", "Rome"),
        // Only a verb of thought or speech makes the pronoun the name's.
        ("we left Atlanta then she moved", "Atlanta"),
        ("I miss Boston whenever he travels", "Boston"),
        ("we toured Chicago once he arrived", "Chicago"),
        ("we drove past Boston yet he never stopped", "Boston"),
        ("we love Paris though she does not", "Paris"),
        ("Dayton unless he calls", "Dayton"),
        ("Paris's partner company", "Paris"),
        ("from Madison to Columbus for the conference", "Madison"),
    ] {
        assert_eq!(written(text).first().map(String::as_str), Some(want), "{text:?}: {:?}", cities(text));
    }
    assert_eq!(written("from Madison to Columbus for the conference"), ["Madison", "Columbus"]);
}

#[test]
fn a_line_break_after_a_multi_byte_character_reads_without_a_panic() {
    // #2296 review: the word before a line break ends in a character of
    // more than one byte, so its start is found on character boundaries.
    for text in [
        "caf\u{e9}\nParis said so",
        "\u{201C}\nSalem, who runs it",
        "na\u{ef}ve \u{65e5}\r\nColumbus herself",
        "\u{1F600}\nDayton knew she would",
    ] {
        assert!(cities(text).is_empty(), "{text:?}: {:?}", cities(text));
    }
    for (text, want) in [("met in\nParis\u{2019}s caf\u{e9}", "Paris"), ("\u{e9}t\u{e9} in\nSalem, who knew", "Salem")]
    {
        assert_eq!(written(text), [want], "{text:?}");
    }
}
