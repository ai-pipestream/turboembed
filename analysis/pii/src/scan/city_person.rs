//! Person context around a lone bare city name (#2295): a few cheap
//! rules, with no model, that read a name sharing a city's as a person's
//! (`we met Madison`, `Paris's mother`, `Salem, who`). Each applies only
//! where the bare reading takes the name at all, and the rules after the
//! name give way when a place word stands before it (`people in Paris
//! who`), as [`super::city_bare`] decides.

use super::city_bare::word_before;

/// The verbs that take a person as their object and a city seldom
/// (`we met Madison`, `ask Columbus`, `they hired Dayton`). A verb that
/// takes a place as readily (`visited`, `left`, `called`, `joined`,
/// `loves`) is not one.
const PERSON_OBJECT_VERBS: [&str; 39] = [
    "ask",
    "asked",
    "asking",
    "asks",
    "congratulate",
    "congratulated",
    "email",
    "emailed",
    "fire",
    "fired",
    "greet",
    "greeted",
    "hire",
    "hired",
    "hug",
    "hugged",
    "interview",
    "interviewed",
    "introduced",
    "invite",
    "invited",
    "kiss",
    "kissed",
    "marry",
    "married",
    "meet",
    "meeting",
    "meets",
    "met",
    "phone",
    "phoned",
    "promoted",
    "remind",
    "reminded",
    "text",
    "texted",
    "thank",
    "thanked",
    "thanks",
];

/// The kin and body nouns a person's possessive stands before and a
/// city's seldom does (`Paris's mother`, `Madison's hair`).
const KIN_AND_BODY_NOUNS: [&str; 30] = [
    "aunt",
    "boyfriend",
    "brother",
    "cousin",
    "dad",
    "daughter",
    "eyes",
    "face",
    "father",
    "fiance",
    "fiancee",
    "girlfriend",
    "grandfather",
    "grandmother",
    "hair",
    "hand",
    "hands",
    "husband",
    "mom",
    "mother",
    "nephew",
    "niece",
    "parents",
    "partner",
    "shoulder",
    "sister",
    "smile",
    "son",
    "uncle",
    "wife",
];

/// The words that, between a name and a pronoun, start a clause about the
/// place (`Boston where he lived`), so the pronoun does not refer back.
const CLAUSE_WORDS: [&str; 13] =
    ["after", "as", "because", "before", "if", "since", "so", "than", "that", "until", "when", "where", "while"];

/// Whether a verb of [`PERSON_OBJECT_VERBS`] stands one space before the
/// name at `start`, as written in lower case (`we met Madison`).
pub(super) fn after_person_verb(text: &str, start: usize) -> bool {
    word_before(text, start).is_some_and(|(_, w)| PERSON_OBJECT_VERBS.contains(&w))
}

/// Whether the words after the name ending at `end` speak of a person:
/// `who` after a space or a comma (`Salem, who`), a reflexive pronoun
/// (`Madison herself`), a subject pronoun one word on (`Madison knew
/// she`), or a possessive before a kin or body noun (`Paris's mother`).
pub(super) fn person_after(text: &str, end: usize) -> bool {
    let rest = &text[end..];
    if let Some(noun) = rest.strip_prefix("'s ").or_else(|| rest.strip_prefix("\u{2019}s ")) {
        return KIN_AND_BODY_NOUNS.contains(&first_word(noun));
    }
    let after_comma = rest.strip_prefix(", ");
    if after_comma.is_some_and(|r| first_word(r) == "who") {
        return true;
    }
    let Some(rest) = rest.strip_prefix(' ') else {
        return false;
    };
    let next = first_word(rest);
    if matches!(next, "who" | "herself" | "himself") {
        return true;
    }
    if next.is_empty() || CLAUSE_WORDS.contains(&next) || !next.bytes().all(|c| c.is_ascii_lowercase()) {
        return false;
    }
    rest[next.len()..].strip_prefix(' ').is_some_and(|r| matches!(first_word(r), "she" | "he"))
}

/// The ASCII letters `text` starts with.
fn first_word(text: &str) -> &str {
    &text[..text.find(|c: char| !c.is_ascii_alphabetic()).unwrap_or(text.len())]
}
