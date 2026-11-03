//! `DictionaryNameFinder`: names that are dictionary entries.
//!
//! The dictionary is our own line format, one typed entry per line:
//!
//! ```text
//! # comment
//! case-insensitive
//! place   New York
//! place   New York City
//! person  John Smith
//! ```
//!
//! The optional `case-sensitive` or `case-insensitive` directive (default
//! `case-sensitive`) sets how lookups fold case. Every other non-empty
//! line is one entry: its type, whitespace, then the entry's tokens
//! separated by ASCII spaces. Lines are read in order; in a
//! case-insensitive dictionary two entries that fold to the same tokens
//! keep the first, and a token that holds U+0000 is refused.
//!
//! From each token the finder tries every run of tokens up to the longest
//! entry (`max_tokens`), keeps the longest run that is an entry, and
//! resumes after it; with none, it resumes at the next token. So names
//! never overlap, and a long entry hides a shorter one that starts inside
//! it. A run is looked up as written when the dictionary is case-sensitive
//! and lowercased per character with the Java `Character.toLowerCase`
//! mapping when it is case-insensitive, the folding `turbo_unicode`
//! provides.

use std::collections::HashMap;
use std::sync::Arc;

use turbo_analysis_layers::{Tag, TagSet};
use turbo_unicode::java::{lowercase_code_points_changes as needs_lowering, to_lowercase_code_points};

use crate::error::Error;
use crate::name::{Name, TokenNameFinder, kinds};

/// One typed entry while a dictionary is built: its type and its tokens.
type GivenEntry = (Box<str>, Box<[Box<str>]>);

/// Joins the tokens of a multi-token entry in the map. The line format
/// cannot hold U+0000, so no token contains it and a joined entry can
/// never equal a single token.
const SEPARATOR: char = '\0';

/// A typed entry set, for membership tests. See the module documentation
/// for the line format.
#[derive(Debug, Clone)]
pub struct Dictionary {
    case_sensitive: bool,
    /// Joined tokens to the type of the first entry that gave them.
    entries: HashMap<Box<str>, u16>,
    /// The types of the entries, in first-appearance order.
    types: Box<[Box<str>]>,
    max_tokens: usize,
}

impl Dictionary {
    /// Reads the line format of the module documentation. `at` in the
    /// errors is the 0-based line number.
    pub fn read(text: &str) -> Result<Dictionary, Error> {
        let mut case_sensitive = true;
        let mut seen_directive = false;
        let mut given: Vec<GivenEntry> = Vec::new();
        for (at, line) in text.lines().enumerate() {
            let line = line.trim_end_matches('\r');
            if line.trim().is_empty() || line.trim_start().starts_with('#') {
                continue;
            }
            match line.trim() {
                "case-sensitive" | "case-insensitive" => {
                    if seen_directive {
                        return Err(Error::Dictionary { at, message: "the case directive is given twice" });
                    }
                    seen_directive = true;
                    case_sensitive = line.trim() == "case-sensitive";
                    continue;
                }
                _ => {}
            }
            let mut fields = line.split_whitespace();
            let (Some(ty), Some(first)) = (fields.next(), fields.next()) else {
                return Err(Error::Dictionary { at, message: "an entry needs a type and at least one token" });
            };
            let mut tokens: Vec<Box<str>> = vec![first.into()];
            tokens.extend(fields.map(Box::from));
            if tokens.iter().any(|t| t.contains(SEPARATOR)) {
                return Err(Error::Dictionary { at, message: "a token must not contain U+0000" });
            }
            given.push((Box::from(ty), tokens.into()));
        }
        Ok(Dictionary::build(case_sensitive, given))
    }

    /// `from_lines` from typed entries already in memory: each item is the
    /// type and the tokens of one entry, in order. Entries keep their
    /// tokens as given; a case-insensitive dictionary folds for lookup.
    pub fn from_entries(case_sensitive: bool, entries: impl IntoIterator<Item = (String, Vec<String>)>) -> Dictionary {
        Dictionary::build(
            case_sensitive,
            entries.into_iter().map(|(t, ts)| (t.into(), ts.into_iter().map(Into::into).collect())),
        )
    }

    fn build(case_sensitive: bool, given: impl IntoIterator<Item = GivenEntry>) -> Dictionary {
        let mut types: Vec<Box<str>> = Vec::new();
        let mut entries: HashMap<Box<str>, u16> = HashMap::new();
        let mut max_tokens = 0;
        for (ty, tokens) in given {
            if tokens.is_empty() {
                continue;
            }
            let ty = match types.iter().position(|t| **t == *ty) {
                Some(i) => i,
                None => {
                    types.push(ty);
                    types.len() - 1
                }
            };
            max_tokens = max_tokens.max(tokens.len());
            let mut key = String::new();
            push_joined(&mut key, tokens.iter().map(Box::as_ref), case_sensitive);
            // Of entries that fold equal the first is kept, as with the
            // entry sets the finders used before.
            entries.entry(key.into()).or_insert(ty as u16);
        }
        Dictionary { case_sensitive, entries, types: types.into(), max_tokens }
    }

    /// Whether lookups match case exactly.
    pub fn is_case_sensitive(&self) -> bool {
        self.case_sensitive
    }

    /// The number of entries.
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Whether the dictionary has no entries.
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// The most tokens in an entry; 0 for an empty dictionary.
    pub fn max_tokens(&self) -> usize {
        self.max_tokens
    }

    /// The types of the entries, in first-appearance order. The finder
    /// gives its names these types; [`Name::kind`](crate::Name::kind)
    /// indexes this list.
    pub fn types(&self) -> impl Iterator<Item = &str> {
        self.types.iter().map(Box::as_ref)
    }

    /// Each entry's type index in [`types`](Self::types) and tokens joined
    /// with U+0000, in no particular order. In a case-insensitive
    /// dictionary the tokens are the lowercased forms the lookups use.
    pub fn entries(&self) -> impl Iterator<Item = (u16, &str)> {
        self.entries.iter().map(|(k, &t)| (t, k.as_ref()))
    }

    /// Whether the dictionary holds the entry made of exactly these
    /// tokens, in order, and the index of its type in [`types`](Self::types)
    /// when it does. `scratch` holds the joined probe.
    pub fn match_tokens(&self, tokens: &[&str], scratch: &mut String) -> Option<u16> {
        if tokens.is_empty() || tokens.iter().any(|t| t.contains(SEPARATOR)) {
            return None;
        }
        if let [token] = tokens
            && (self.case_sensitive || !needs_lowering(token))
        {
            return self.entries.get(*token).copied();
        }
        scratch.clear();
        push_joined(scratch, tokens.iter().copied(), self.case_sensitive);
        self.entries.get(scratch.as_str()).copied()
    }
}

fn push_joined<'a>(out: &mut String, tokens: impl Iterator<Item = &'a str>, case_sensitive: bool) {
    let mut first = true;
    for token in tokens {
        if !first {
            out.push(SEPARATOR);
        }
        first = false;
        if case_sensitive {
            out.push_str(token);
        } else {
            to_lowercase_code_points(token, out);
        }
    }
}

/// `opennlp.tools.namefind.DictionaryNameFinder` over a [`Dictionary`]:
/// every name has the type of the entry it matched.
#[derive(Clone, Debug)]
pub struct DictionaryNameFinder {
    dictionary: Dictionary,
    kinds: Arc<TagSet>,
}

impl DictionaryNameFinder {
    /// A finder over `dictionary`. Its entity types are the dictionary's,
    /// in first-appearance order. An empty dictionary is refused: there is
    /// no type a name could carry.
    pub fn new(dictionary: Dictionary) -> Result<DictionaryNameFinder, Error> {
        if dictionary.is_empty() {
            return Err(Error::EmptyDictionary);
        }
        let kinds = kinds(dictionary.types())?;
        Ok(DictionaryNameFinder { dictionary, kinds })
    }

    pub fn dictionary(&self) -> &Dictionary {
        &self.dictionary
    }

    /// The entity types the dictionary gives its names, in first-appearance
    /// order.
    pub fn types(&self) -> impl Iterator<Item = &str> {
        self.dictionary.types()
    }
}

impl TokenNameFinder for DictionaryNameFinder {
    fn kinds(&self) -> &Arc<TagSet> {
        &self.kinds
    }

    fn find(&self, tokens: &[&str], out: &mut Vec<Name>) -> Result<(), Error> {
        let longest = self.dictionary.max_tokens();
        let mut scratch = String::new();
        let mut from = 0;
        while from < tokens.len() {
            let last = tokens.len().min(from + longest);
            let found = (from + 1..=last)
                .rev()
                .find_map(|end| self.dictionary.match_tokens(&tokens[from..end], &mut scratch).map(|ty| (end, ty)));
            match found {
                Some((end, ty)) => {
                    out.push(Name { start: from as u32, end: end as u32, kind: Tag(ty) });
                    from = end;
                }
                None => from += 1,
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn find(text: &str, tokens: &[&str]) -> Vec<String> {
        let f = DictionaryNameFinder::new(Dictionary::read(text).unwrap()).unwrap();
        let mut out = Vec::new();
        f.find(tokens, &mut out).unwrap();
        out.iter().map(|n| format!("{}:{}={}", n.start, n.end, f.kinds().label(n.kind))).collect()
    }

    const PLACES: &str = "case-insensitive\n\
        place   New York\n\
        place   New York City\n\
        place   York City Hall\n\
        place   City\n";

    #[test]
    fn longest_entry_wins_and_hides_overlaps() {
        assert_eq!(find(PLACES, &["in", "new", "YORK", "City", "Hall"]), ["1:4=place"]);
        assert_eq!(find(PLACES, &["York", "City", "Hall", "City"]), ["0:3=place", "3:4=place"]);
        assert_eq!(find(PLACES, &["New", "Jersey", "City"]), ["2:3=place"]);
        assert_eq!(find(PLACES, &[]), Vec::<String>::new());
    }

    #[test]
    fn entries_carry_their_type() {
        let text = "city    York\nperson  John Smith\ncity    City Hall\n";
        assert_eq!(find(text, &["John", "Smith", "at", "City", "Hall"]), ["0:2=person", "3:5=city"]);
        let finders = DictionaryNameFinder::new(Dictionary::read(text).unwrap()).unwrap();
        assert_eq!(finders.types().collect::<Vec<_>>(), ["city", "person"]);
    }

    #[test]
    fn case_folding_is_java_lowercase_per_character() {
        // U+0130 folds to i, U+212A (Kelvin) to k, as Java lowercases.
        let text = "case-insensitive\nword    k\nword    strasse\nword    i\n";
        assert_eq!(find(text, &["K"]), ["0:1=word"]);
        assert_eq!(find(text, &["\u{212A}"]), ["0:1=word"]);
        assert_eq!(find(text, &["STRASSE"]), ["0:1=word"]);
        assert_eq!(find(text, &["\u{0130}"]), ["0:1=word"]);
        let text = "case-sensitive\nword    k\n";
        assert_eq!(find(text, &["K"]), Vec::<String>::new());
        assert_eq!(find(text, &["k"]), ["0:1=word"]);
    }

    #[test]
    fn first_of_case_equal_entries_is_kept() {
        let text = "case-insensitive\nplace   New York\nperson  NEW YORK\n";
        assert_eq!(find(text, &["New", "York"]), ["0:2=place"]);
    }

    #[test]
    fn refused_lines_name_the_line() {
        let err = Dictionary::read("place\n").unwrap_err();
        assert_eq!(err, Error::Dictionary { at: 0, message: "an entry needs a type and at least one token" });
        let err = Dictionary::read("case-sensitive\ncase-insensitive\na b\n").unwrap_err();
        assert_eq!(err, Error::Dictionary { at: 1, message: "the case directive is given twice" });
        let err = Dictionary::read("ok a\u{0}b\n").unwrap_err();
        assert_eq!(err, Error::Dictionary { at: 0, message: "a token must not contain U+0000" });
        assert_eq!(
            DictionaryNameFinder::new(Dictionary::read("# nothing\n").unwrap()).unwrap_err(),
            Error::EmptyDictionary
        );
    }
}
