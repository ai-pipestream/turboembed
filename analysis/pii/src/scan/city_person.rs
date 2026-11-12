//! Person context around a lone bare city name (#2295): a few cheap
//! rules, with no model, that read a name sharing a city's as a person's
//! (`we met Madison`, `Paris's mother`, `Salem, who`). Each applies only
//! where the bare reading takes the name at all, and the rules after the
//! name give way when a place word stands before it (`people in Paris
//! who`), as [`super::city_bare`] decides.

/// The verbs that take a person as their object and a city seldom
/// (`we met Madison`, `ask Columbus`, `they hired Dayton`). A verb that
/// takes a place as readily (`visited`, `left`, `called`, `joined`,
/// `loves`) is not one.
fn person_object_verb(word: &str) -> bool {
    // By length first, so a word is compared only with the words of its length.
    match word.len() {
        3 => matches!(word, "ask" | "hug" | "met"),
        4 => matches!(word, "asks" | "fire" | "hire" | "kiss" | "meet" | "text"),
        5 => matches!(word, "asked" | "email" | "fired" | "greet" | "hired" | "marry" | "meets" | "phone" | "thank"),
        6 => matches!(word, "asking" | "hugged" | "invite" | "kissed" | "phoned" | "remind" | "texted" | "thanks"),
        7 => matches!(word, "emailed" | "greeted" | "invited" | "married" | "meeting" | "promote" | "thanked"),
        8 => matches!(word, "promoted" | "reminded"),
        9 => matches!(word, "interview" | "introduce"),
        10 => matches!(word, "introduced"),
        11 => matches!(word, "interviewed"),
        12 => matches!(word, "congratulate"),
        13 => matches!(word, "congratulated"),
        _ => false,
    }
}

/// The kin and body nouns a person's possessive stands before and a
/// city's seldom does (`Paris's mother`, `Madison's hair`).
fn kin_or_body_noun(word: &str) -> bool {
    // By length first, so a word is compared only with the words of its length.
    match word.len() {
        3 => matches!(word, "dad" | "mom" | "son"),
        4 => matches!(word, "aunt" | "eyes" | "face" | "hair" | "hand" | "wife"),
        5 => matches!(word, "hands" | "niece" | "smile" | "uncle"),
        6 => matches!(word, "cousin" | "father" | "fiance" | "mother" | "nephew" | "sister"),
        7 => matches!(word, "brother" | "fiancee" | "husband" | "parents"),
        8 => matches!(word, "daughter" | "shoulder"),
        9 => matches!(word, "boyfriend"),
        10 => matches!(word, "girlfriend"),
        11 => matches!(word, "grandfather" | "grandmother"),
        _ => false,
    }
}

/// The verbs of thought, feeling and speech that, between a name and a
/// subject pronoun, make the pronoun the name's (`Dayton knew she`,
/// `Columbus thought he`). Any other word there leaves the name a place
/// (`we left Atlanta then she moved`, `Boston where he lived`).
fn pronoun_verb(word: &str) -> bool {
    // By length first, so a word is compared only with the words of its length.
    match word.len() {
        4 => matches!(word, "felt" | "knew" | "said"),
        5 => matches!(word, "hoped" | "swore"),
        6 => matches!(word, "agreed" | "feared" | "forgot" | "wished"),
        7 => matches!(word, "claimed" | "decided" | "figured" | "guessed" | "noticed" | "thought"),
        8 => matches!(word, "admitted" | "believed" | "insisted" | "promised" | "realised" | "realized"),
        9 => matches!(word, "explained" | "suspected"),
        10 => matches!(word, "remembered"),
        _ => false,
    }
}

/// Whether `word`, the word one space before a name, is a
/// [`person_object_verb`] as written in lower case (`we met Madison`).
pub(super) fn after_person_verb(word: &str) -> bool {
    // The letters the verbs start with, as one mask, so most words before
    // a name (`in`, `between`, `from`) cost one probe.
    const FIRST: u32 = mask(b"acefghikmprt");
    word.as_bytes().first().is_some_and(|&c| c.is_ascii_lowercase() && FIRST & 1 << (c - b'a') != 0)
        && person_object_verb(word)
}

/// One bit per lower-case ASCII letter of `letters`.
const fn mask(letters: &[u8]) -> u32 {
    let mut m = 0;
    let mut i = 0;
    while i < letters.len() {
        m |= 1 << (letters[i] - b'a');
        i += 1;
    }
    m
}

/// Whether the words after the name ending at `end` speak of a person:
/// `who` after a space or a comma (`Salem, who`), a reflexive pronoun
/// (`Madison herself`), a subject pronoun after a [`pronoun_verb`]
/// (`Madison knew she`), or a possessive before a kin or body noun (`Paris's mother`).
pub(super) fn person_after(text: &str, end: usize) -> bool {
    let rest = &text[end..];
    // Every rule starts with a space, a comma or an apostrophe.
    if !rest.as_bytes().first().is_some_and(|&c| matches!(c, b' ' | b',' | b'\'' | 0xE2)) {
        return false;
    }
    if let Some(noun) = rest.strip_prefix("'s ").or_else(|| rest.strip_prefix("\u{2019}s ")) {
        return kin_or_body_noun(first_word(noun));
    }
    let after_comma = rest.strip_prefix(", ");
    if after_comma.is_some_and(|r| first_word(r) == "who") {
        return true;
    }
    let Some(rest) = rest.strip_prefix(' ') else {
        return false;
    };
    let next = first_word(rest);
    if next.starts_with(['w', 'h']) && matches!(next, "who" | "herself" | "himself") {
        return true;
    }
    // The pronoun is looked for first, as it is rarer than the verb.
    let after = rest.as_bytes()[next.len()..].strip_prefix(b" ");
    after.is_some_and(|r| matches!(r, [b's', b'h', b'e', ..] | [b'h', b'e', ..]))
        && matches!(first_word(&rest[next.len() + 1..]), "she" | "he")
        && pronoun_verb(next)
}

/// The ASCII letters `text` starts with.
fn first_word(text: &str) -> &str {
    &text[..text.bytes().position(|c| !c.is_ascii_alphabetic()).unwrap_or(text.len())]
}
