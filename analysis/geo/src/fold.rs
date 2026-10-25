//! The match key of a name: the one folding rule every gazetteer here
//! applies to the names it indexes and to the queries it answers.
//!
//! OpenNLP's `GazetteerIndex.foldKey` runs a `TermAnalyzer` built with
//! `nfc().caseFold().accentFold()`: the UAX #29 word tokens of the name
//! (OpenNLP's `WordTokenizer`, spaces and punctuation dropped, tokens cut
//! at 255 UTF-16 units), each put through NFC, `toLowerCase(Locale.ROOT)`
//! and the accent fold in that order, joined by one space. A name with no
//! word token has the empty key, which no lookup reaches.

use std::num::NonZeroU32;

use turbo_analysis::Span;
use turbo_normalize::{AccentFold, FoldScratch};
use turbo_tokenize::{WordTokenizer, WordType, word_tokens};
use turbo_unicode::java::to_lowercase;
use turbo_unicode::normalize::{Form, is_normalized, normalize};

/// Folds names to match keys. It keeps its buffers between calls, so one
/// folder folds any number of names with no allocation once the buffers
/// have grown; it is not shared between threads.
pub struct Folder {
    accent: AccentFold,
    spans: Vec<Span>,
    types: Vec<WordType>,
    nfc: String,
    lower: String,
    scratch: FoldScratch,
}

/// The maximum token length of OpenNLP's `WordTokenizer`, in UTF-16 units.
const MAX_TOKEN_LENGTH: NonZeroU32 = WordTokenizer::DEFAULT_MAX_TOKEN_LENGTH;

impl Default for Folder {
    fn default() -> Folder {
        Folder::new()
    }
}

impl Folder {
    pub fn new() -> Folder {
        Folder {
            accent: AccentFold::new(),
            spans: Vec::new(),
            types: Vec::new(),
            nfc: String::new(),
            lower: String::new(),
            scratch: FoldScratch::default(),
        }
    }

    /// Writes the match key of `name` into `key`, replacing what it held.
    /// The key is empty when the name has no word token.
    pub fn fold_into(&mut self, name: &str, key: &mut String) {
        key.clear();
        if fold_plain_ascii(name, key) {
            return;
        }
        self.spans.clear();
        self.types.clear();
        word_tokens(name, MAX_TOKEN_LENGTH, &mut self.spans, &mut self.types);
        for i in 0..self.spans.len() {
            let Span { start, end } = self.spans[i];
            // OpenNLP appends the separator only after a non-empty key, so
            // a token that folds to nothing at the start adds no space.
            if !key.is_empty() {
                key.push(' ');
            }
            self.fold_token(&name[start as usize..end as usize], key);
        }
    }

    /// The match key of `name`, as a new string.
    pub fn fold(&mut self, name: &str) -> String {
        let mut key = String::with_capacity(name.len());
        self.fold_into(name, &mut key);
        key
    }

    /// The word tokens of a running text, each folded as a match key folds
    /// it, joined by one space into `folded`: token `i` is `spans[i]` in
    /// `text` and `starts[i]..ends[i]` in `folded`. A name made of tokens
    /// `i..=j` has the key `folded[starts[i]..ends[j]]`.
    pub(crate) fn fold_text(
        &mut self,
        text: &str,
        folded: &mut String,
        spans: &mut Vec<Span>,
        starts: &mut Vec<u32>,
        ends: &mut Vec<u32>,
    ) {
        folded.clear();
        spans.clear();
        starts.clear();
        ends.clear();
        self.types.clear();
        word_tokens(text, MAX_TOKEN_LENGTH, spans, &mut self.types);
        for (i, s) in spans.iter().enumerate() {
            if i > 0 {
                folded.push(' ');
            }
            starts.push(folded.len() as u32);
            self.fold_token(&text[s.start as usize..s.end as usize], folded);
            ends.push(folded.len() as u32);
        }
    }

    /// Appends NFC, then `toLowerCase(Locale.ROOT)`, then the accent fold
    /// of one token to `out`. An ASCII token is in NFC and has nothing to
    /// fold, so it is only lowercased.
    fn fold_token(&mut self, token: &str, out: &mut String) {
        if token.is_ascii() {
            out.extend(token.bytes().map(|b| char::from(b.to_ascii_lowercase())));
            return;
        }
        let nfc: &str = if is_normalized(Form::Nfc, token) {
            token
        } else {
            self.nfc.clear();
            normalize(Form::Nfc, token, &mut self.nfc);
            &self.nfc
        };
        self.lower.clear();
        to_lowercase(nfc, &mut self.lower);
        let accent = self.accent;
        accent.fold_str(&self.lower, &mut self.scratch, out);
    }
}

/// Folds a name made only of ASCII letters, digits and spaces, with no
/// word longer than the token limit, and returns whether it did. UAX #29
/// keeps a run of ASCII letters and digits together as one word and
/// breaks at every space, so the key is the runs lowercased and joined by
/// one space; any other name goes through the tokenizer.
fn fold_plain_ascii(name: &str, key: &mut String) -> bool {
    let bytes = name.as_bytes();
    if !bytes.iter().all(|&b| b.is_ascii_alphanumeric() || b == b' ') {
        return false;
    }
    if bytes.split(|&b| b == b' ').any(|word| word.len() > MAX_TOKEN_LENGTH.get() as usize) {
        return false;
    }
    for word in bytes.split(|&b| b == b' ').filter(|w| !w.is_empty()) {
        if !key.is_empty() {
            key.push(' ');
        }
        key.extend(word.iter().map(|&b| char::from(b.to_ascii_lowercase())));
    }
    true
}

#[cfg(test)]
mod tests {
    use super::*;

    fn key(name: &str) -> String {
        Folder::new().fold(name)
    }

    #[test]
    fn ascii_names_are_lowercased_word_by_word() {
        assert_eq!(key("New  York"), "new york");
        assert_eq!(key("  Saint-Louis\n"), "saint louis");
        assert_eq!(key("St. John's"), "st john's");
    }

    #[test]
    fn accents_case_and_composition_fold_away() {
        assert_eq!(key("São Paulo"), "sao paulo");
        assert_eq!(key("Sa\u{303}o PAULO"), "sao paulo");
        assert_eq!(key("Zürich"), "zurich");
        assert_eq!(key("Łódź"), "lodz");
        assert_eq!(key("Ærøskøbing"), "aeroskobing");
        assert_eq!(key("Großbeeren"), "grossbeeren");
    }

    #[test]
    fn dotted_capital_i_folds_to_i_and_dotless_i_to_i_by_its_stroke_letter() {
        // toLowerCase(ROOT) maps U+0130 to "i" U+0307; the mark then folds.
        assert_eq!(key("İstanbul"), "istanbul");
        // U+0131 is in OpenNLP's stroke letter list.
        assert_eq!(key("Diyarbakır"), "diyarbakir");
    }

    #[test]
    fn marks_on_other_scripts_stay() {
        assert_eq!(key("नई दिल्ली"), "नई दिल्ली");
        // UAX #29 breaks between ideographs.
        assert_eq!(key("東京"), "東 京");
    }

    /// The key the tokenizer gives, without the plain ASCII shortcut.
    fn tokenized_key(name: &str) -> String {
        let mut folder = Folder::new();
        let mut key = String::new();
        word_tokens(name, MAX_TOKEN_LENGTH, &mut folder.spans, &mut folder.types);
        for &Span { start, end } in &folder.spans.clone() {
            if !key.is_empty() {
                key.push(' ');
            }
            folder.fold_token(&name[start as usize..end as usize], &mut key);
        }
        key
    }

    #[test]
    fn the_plain_ascii_shortcut_agrees_with_the_tokenizer() {
        let long = "a".repeat(255);
        let longer = "B".repeat(256);
        let names = [
            "",
            " ",
            "Berlin",
            "New York",
            "  New   York  ",
            "1st Street",
            "A1 B2 33",
            "123",
            "x 1 y",
            long.as_str(),
            longer.as_str(),
            "ZZ9 Plural Z Alpha",
        ];
        for name in names {
            let mut key = String::new();
            let plain = fold_plain_ascii(name, &mut key);
            assert_eq!(plain, name != longer, "{name}");
            if plain {
                assert_eq!(key, tokenized_key(name), "{name}");
            }
            assert_eq!(Folder::new().fold(name), tokenized_key(name), "{name}");
        }
    }

    /// Keys that `GazetteerIndex.foldKey` gave on JDK 21 at the goldens'
    /// commits, for case mappings the golden inputs do not reach: Georgian
    /// Mtavruli, Cherokee and Deseret capitals, the JDK's final sigma
    /// condition, U+0130 with a combining dot, the Kelvin sign, the DŽ
    /// digraphs and the capital sharp s.
    #[test]
    fn case_mappings_fold_as_on_the_jvm() {
        let cases = [
            ("ᲗᲑᲘᲚᲘᲡᲘ", "თბილისი"),
            ("Ꮳ ᏣᎳᎩ", "ꮳ ꮳꮃꭹ"),
            ("𐐔𐐯𐑅𐐨𐑉𐐯𐐻", "𐐼𐐯𐑅𐐨𐑉𐐯𐐻"),
            ("Ꭰ𐐀ꭰ", "ꭰ𐐨ꭰ"),
            ("ΟΔΟΣ", "οδος"),
            ("ΣΟΦΙΑ", "σοφια"),
            ("ΑΣ1", "ας1"),
            ("ΑΣ1Α", "ασ1α"),
            ("Σ", "σ"),
            ("ΑΣ.Β", "ασ.β"),
            ("İSTANBUL İ\u{307}", "istanbul i"),
            ("Kelvin \u{212A}", "kelvin k"),
            ("ǄǅǆD\u{17D}", "ǆǆǆdz"),
            ("ẞTRASSE", "sstrasse"),
        ];
        for (name, jvm) in cases {
            assert_eq!(key(name), jvm, "{name}");
        }
    }

    #[test]
    fn a_name_without_word_tokens_has_the_empty_key() {
        assert_eq!(key(""), "");
        assert_eq!(key(" - , ."), "");
    }
}
