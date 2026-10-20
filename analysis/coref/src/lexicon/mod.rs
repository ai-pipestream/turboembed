//! The English word knowledge mention detection and the sieves consult
//! (`CorefLexicon`): pronoun attributes, gendered and animate nouns,
//! titles, first-name genders, stop words, indefinite openers, compound
//! name heads, pleonastic `it` cues, speech verbs and quotation marks.
//!
//! The first names come from `first-names-en.txt`, the list OpenNLP
//! bundles: names given to one gender at least 90% of the time in the
//! United States Social Security Administration's yearly baby name
//! tables for 1920 to 2008, a public-domain work.

use std::collections::HashMap;
use std::sync::OnceLock;

use crate::layer::{Animacy, Gender, Number, Person};

mod animate;
mod closed;
mod nouns;
mod words;

/// One pronoun's attributes (`CorefLexicon.Pronoun`).
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct Pronoun {
    pub person: Person,
    pub number: Number,
    pub gender: Gender,
    pub animacy: Animacy,
}

/// The pronoun classes of the lexicon, named for their citation form.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
#[allow(clippy::upper_case_acronyms)]
enum Forms {
    HE,
    SHE,
    IT,
    THEY,
    I,
    WE,
    YOU,
    THIS,
    THESE,
}

impl Forms {
    const fn pronoun(self) -> Pronoun {
        use Animacy::*;
        use Gender::*;
        use Number::*;
        use Person::*;
        let (person, number, gender, animacy) = match self {
            Forms::HE => (Third, Singular, Male, Animate),
            Forms::SHE => (Third, Singular, Female, Animate),
            Forms::IT => (Third, Singular, Neutral, Inanimate),
            Forms::THEY => (Third, Plural, Gender::Unknown, Animacy::Unknown),
            Forms::I => (First, Singular, Gender::Unknown, Animate),
            Forms::WE => (First, Plural, Gender::Unknown, Animate),
            Forms::YOU => (Second, Number::Unknown, Gender::Unknown, Animate),
            Forms::THIS => (Third, Singular, Neutral, Inanimate),
            Forms::THESE => (Third, Plural, Neutral, Inanimate),
        };
        Pronoun { person, number, gender, animacy }
    }
}

fn find<'a, T: Copy>(table: &'a [(&'a str, T)], word: &str) -> Option<T> {
    table.binary_search_by(|(w, _)| (*w).cmp(word)).ok().map(|i| table[i].1)
}

fn has(table: &[&str], word: &str) -> bool {
    table.binary_search(&word).is_ok()
}

/// The attributes of a personal pronoun, `None` for any other word.
pub fn pronoun(word: &str) -> Option<Pronoun> {
    find(closed::PRONOUNS, word).map(Forms::pronoun)
}

/// The attributes of a demonstrative used as a pronoun
/// (`That was costly`), `None` for any other word.
pub fn demonstrative(word: &str) -> Option<Pronoun> {
    find(closed::DEMONSTRATIVES, word).map(Forms::pronoun)
}

/// The gender a noun or title implies, `Unknown` for a word without one.
pub fn noun_gender(word: &str) -> Gender {
    if has(nouns::MALE_NOUNS, word) {
        Gender::Male
    } else if has(nouns::FEMALE_NOUNS, word) {
        Gender::Female
    } else {
        find(closed::TITLES, word).unwrap_or(Gender::Unknown)
    }
}

/// Whether a noun names a person, a group of people or an animal.
pub fn animate_noun(word: &str) -> bool {
    has(animate::ANIMATE_NOUNS, word) || has(nouns::MALE_NOUNS, word) || has(nouns::FEMALE_NOUNS, word)
}

/// Whether a word is a title that precedes a name.
pub fn title(word: &str) -> bool {
    find(closed::TITLES, word).is_some()
}

/// The gender of a lowercased first name, `Unknown` for an unlisted name.
pub fn first_name_gender(name: &str) -> Gender {
    static NAMES: OnceLock<HashMap<&'static str, Gender>> = OnceLock::new();
    let names = NAMES.get_or_init(|| parse_first_names(include_str!("first-names-en.txt")));
    names.get(name).copied().unwrap_or(Gender::Unknown)
}

/// Reads the bundled list: `#` comments, then one `name<TAB>m|f` per line.
fn parse_first_names(text: &'static str) -> HashMap<&'static str, Gender> {
    let mut names = HashMap::new();
    for line in text.lines() {
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let (name, gender) = line.split_once('\t').expect("a first name entry is name, tab, gender");
        assert!(!name.is_empty() && !gender.is_empty(), "malformed first name entry: {line}");
        names.insert(name, if gender.starts_with('m') { Gender::Male } else { Gender::Female });
    }
    names
}

/// Whether a word carries no name content: a stop word, or a token with
/// no letter or digit (`Character.isLetterOrDigit`).
pub fn stop_word(word: &str) -> bool {
    has(words::STOP_WORDS, word) || !word.chars().any(turbo_unicode::java::is_letter_or_digit)
}

/// Whether a word is a verb of speech or thought.
pub fn speech_verb(word: &str) -> bool {
    has(words::SPEECH_VERBS, word)
}

/// Whether a token opens a quotation.
pub fn opens_quote(token: &str) -> bool {
    has(words::OPENING_QUOTES, token)
}

/// Whether a token closes a quotation.
pub fn closes_quote(token: &str) -> bool {
    has(words::CLOSING_QUOTES, token)
}

/// Whether a word opens an indefinite noun phrase.
pub fn indefinite_word(word: &str) -> bool {
    has(words::INDEFINITE_WORDS, word)
}

/// Whether a word heads a compound place or institution name (`city`
/// in `Kansas City`).
pub fn compound_name_head(word: &str) -> bool {
    has(words::COMPOUND_NAME_HEADS, word)
}

/// Whether a verb can make a preceding `it` pleonastic.
pub fn pleonastic_verb(word: &str) -> bool {
    has(words::PLEONASTIC_VERBS, word)
}

/// Whether a word completes a pleonastic `it` construction.
pub fn pleonastic_complement(word: &str) -> bool {
    has(words::PLEONASTIC_COMPLEMENTS, word)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tables_are_sorted_for_binary_search() {
        for table in [
            nouns::MALE_NOUNS,
            nouns::FEMALE_NOUNS,
            animate::ANIMATE_NOUNS,
            words::STOP_WORDS,
            words::SPEECH_VERBS,
            words::OPENING_QUOTES,
            words::CLOSING_QUOTES,
            words::INDEFINITE_WORDS,
            words::COMPOUND_NAME_HEADS,
            words::PLEONASTIC_VERBS,
            words::PLEONASTIC_COMPLEMENTS,
        ] {
            assert!(table.windows(2).all(|w| w[0] < w[1]), "{table:?}");
        }
        assert!(closed::TITLES.windows(2).all(|w| w[0].0 < w[1].0));
        assert!(closed::PRONOUNS.windows(2).all(|w| w[0].0 < w[1].0));
        assert!(closed::DEMONSTRATIVES.windows(2).all(|w| w[0].0 < w[1].0));
    }

    #[test]
    fn pronouns_titles_and_names_carry_their_attributes() {
        assert_eq!(pronoun("she").unwrap().gender, Gender::Female);
        assert_eq!(pronoun("themselves").unwrap().number, Number::Plural);
        assert_eq!(pronoun("you").unwrap().person, Person::Second);
        assert!(pronoun("this").is_none());
        assert_eq!(demonstrative("those").unwrap().number, Number::Plural);
        assert_eq!(noun_gender("mrs."), Gender::Female);
        assert_eq!(noun_gender("dr."), Gender::Unknown);
        assert!(title("dr."));
        assert_eq!(first_name_gender("alice"), Gender::Female);
        assert_eq!(first_name_gender("john"), Gender::Male);
        assert_eq!(first_name_gender("kowalczyk"), Gender::Unknown);
        assert!(animate_noun("teacher") && animate_noun("widow") && !animate_noun("company"));
    }

    #[test]
    fn stop_words_include_tokens_without_letters_or_digits() {
        assert!(stop_word("the") && stop_word(",") && stop_word("--"));
        assert!(!stop_word("acme") && !stop_word("\u{1D7D9}"));
    }

    #[test]
    fn quotes_and_cues() {
        assert!(opens_quote("\u{201c}") && opens_quote("``") && closes_quote("''") && closes_quote("\""));
        assert!(pleonastic_verb("seems") && pleonastic_complement("whether"));
        assert!(speech_verb("said") && indefinite_word("some") && compound_name_head("university"));
    }
}
