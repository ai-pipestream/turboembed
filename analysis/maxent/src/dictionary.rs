//! OpenNLP's dictionary XML, as `DictionaryEntryPersistor` reads it:
//!
//! ```xml
//! <dictionary case_sensitive="false">
//!   <entry operation="MOVE_LEFT"><token>)</token></entry>
//!   <entry><token>New</token><token>York</token></entry>
//! </dictionary>
//! ```
//!
//! [`Dictionary`] is the set OpenNLP's `Dictionary` class builds, for
//! abbreviation lookups. [`Entries`] keeps every entry with its
//! attributes, in document order, for the detokenizer dictionary, whose
//! entries carry an `operation`.
//!
//! The reading follows the Java content handler exactly: the text of each
//! `token` element is trimmed with `String.trim` (every character at or
//! below U+0020 at either end), an `entry` ending takes the tokens read
//! since the previous entry, and `case_sensitive` is true unless the
//! attribute is present and is not `true` in any case. Other elements and
//! text outside `token` are ignored.

use std::collections::HashSet;

use turbo_unicode::java::to_lower_case;

use crate::error::{Error, Result};
use crate::hash::BuildFoldHasher;
use crate::xml::{self, Attribute, Handler};

/// Joins the tokens of a multi-token entry in the set. XML cannot hold
/// U+0000, so no token contains it and a joined entry can never equal a
/// single token.
const SEPARATOR: char = '\0';

/// The entries of a dictionary as a set, for membership tests.
///
/// In a case-insensitive dictionary every entry and every probe is
/// lowercased per character with Java's `Character.toLowerCase(int)`, as
/// OpenNLP's `StringList` hashes them. (Java then confirms a match with
/// `String.compareToIgnoreCase`, which agrees with comparing the
/// lowercased forms for every character outside a few with irregular case
/// mappings, such as U+0130.)
#[derive(Debug, Clone)]
pub struct Dictionary {
    case_sensitive: bool,
    entries: HashSet<Box<str>, BuildFoldHasher>,
    min_tokens: usize,
    max_tokens: usize,
}

impl Dictionary {
    /// Reads a dictionary from its XML.
    pub fn read(xml: &[u8]) -> Result<Dictionary> {
        // OpenNLP's `Dictionary(InputStream)` constructor inserts entries
        // before its `isCaseSensitive` field is assigned, so every entry goes
        // in as case-insensitive: of entries that differ only in case, the
        // first is kept and the others are dropped, even in a case-sensitive
        // dictionary. This reproduces that, so lookups agree with the JVM.
        let mut lowered: HashSet<Box<str>, BuildFoldHasher> = HashSet::default();
        let mut originals: Vec<Box<str>> = Vec::new();
        let mut min_tokens = usize::MAX;
        let mut max_tokens = 0;
        let mut key = String::new();
        let case_sensitive = read_entries(xml, |tokens, _| {
            key.clear();
            push_joined(&mut key, tokens.iter().map(String::as_str), false);
            if !lowered.contains(key.as_str()) {
                lowered.insert(key.as_str().into());
                let mut original = String::new();
                push_joined(&mut original, tokens.iter().map(String::as_str), true);
                originals.push(original.into());
            }
            min_tokens = min_tokens.min(tokens.len());
            max_tokens = max_tokens.max(tokens.len());
        })?;
        let entries = if case_sensitive { originals.into_iter().collect() } else { lowered };
        if entries.is_empty() {
            min_tokens = 0;
        }
        Ok(Dictionary { case_sensitive, entries, min_tokens, max_tokens })
    }

    /// Whether lookups match case exactly.
    pub fn is_case_sensitive(&self) -> bool {
        self.case_sensitive
    }

    /// The number of distinct entries.
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// The fewest tokens in an entry; 0 for an empty dictionary (where
    /// OpenNLP reports 99999).
    pub fn min_tokens(&self) -> usize {
        self.min_tokens
    }

    /// The most tokens in an entry; 0 for an empty dictionary.
    pub fn max_tokens(&self) -> usize {
        self.max_tokens
    }

    /// Whether the dictionary holds the one-token entry `token`. In a
    /// case-insensitive dictionary a probe that lowercases to something
    /// other than itself is lowercased into a new string; use
    /// [`contains_with`](Self::contains_with) to reuse a buffer instead.
    pub fn contains(&self, token: &str) -> bool {
        if self.case_sensitive || !needs_lowering(token) {
            return !token.contains(SEPARATOR) && self.entries.contains(token);
        }
        let mut scratch = String::new();
        self.contains_with(token, &mut scratch)
    }

    /// [`contains`](Self::contains), lowercasing into `scratch` when the
    /// dictionary is case-insensitive.
    pub fn contains_with(&self, token: &str, scratch: &mut String) -> bool {
        self.contains_tokens(&[token], scratch)
    }

    /// Whether the dictionary holds the entry made of exactly these
    /// tokens, in order. `scratch` holds the joined probe.
    pub fn contains_tokens(&self, tokens: &[&str], scratch: &mut String) -> bool {
        if tokens.is_empty() || tokens.iter().any(|t| t.contains(SEPARATOR)) {
            return false;
        }
        if let [token] = tokens
            && (self.case_sensitive || !needs_lowering(token))
        {
            return self.entries.contains(*token);
        }
        scratch.clear();
        push_joined(scratch, tokens.iter().copied(), self.case_sensitive);
        self.entries.contains(scratch.as_str())
    }

    /// Each entry's tokens, in no particular order. In a case-insensitive
    /// dictionary the tokens are lowercased.
    pub fn entries(&self) -> impl Iterator<Item = impl Iterator<Item = &str>> {
        self.entries.iter().map(|e| e.split(SEPARATOR))
    }

    /// The first token of each entry, in no particular order: entries that
    /// share a first token give it once each, as OpenNLP's
    /// `Dictionary.asStringSet` iterates them. In a case-insensitive
    /// dictionary the tokens here are lowercased, where `asStringSet`
    /// gives each entry's original casing; the lowercased form is what
    /// `SentenceDetectorME`'s `AbbreviationIndex` builds from it, by
    /// lowercasing each first token with `StringUtil.toLowerCase`.
    pub fn as_set(&self) -> impl Iterator<Item = &str> {
        self.entries.iter().map(|e| e.split(SEPARATOR).next().unwrap_or(""))
    }
}

/// Whether lowercasing `token` per character with Java's mapping changes it.
fn needs_lowering(token: &str) -> bool {
    if token.is_ascii() {
        return token.bytes().any(|b| b.is_ascii_uppercase());
    }
    token.chars().any(|c| to_lower_case(c as u32) != c as u32)
}

/// Appends the tokens joined by [`SEPARATOR`], each character lowercased
/// with Java's mapping unless `case_sensitive`.
fn push_joined<'t>(out: &mut String, tokens: impl Iterator<Item = &'t str>, case_sensitive: bool) {
    for (i, token) in tokens.enumerate() {
        if i > 0 {
            out.push(SEPARATOR);
        }
        if case_sensitive {
            out.push_str(token);
        } else {
            out.extend(token.chars().map(|c| char::from_u32(to_lower_case(c as u32)).unwrap_or(c)));
        }
    }
}

/// One dictionary entry with its attributes.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Entry {
    /// The entry's tokens, trimmed, in order. Never empty.
    pub tokens: Vec<Box<str>>,
    /// The entry element's attributes by local name, in document order.
    pub attributes: Vec<(Box<str>, Box<str>)>,
}

impl Entry {
    /// The value of an attribute.
    pub fn attribute(&self, name: &str) -> Option<&str> {
        self.attributes.iter().find(|(k, _)| &**k == name).map(|(_, v)| &**v)
    }
}

/// Every entry of a dictionary, with its attributes, in document order.
/// Entries are kept as written, duplicates included, and are not
/// lowercased.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Entries {
    case_sensitive: bool,
    entries: Vec<Entry>,
}

impl Entries {
    /// Reads the entries from dictionary XML.
    pub fn read(xml: &[u8]) -> Result<Entries> {
        let mut entries = Vec::new();
        let case_sensitive = read_entries(xml, |tokens, attributes| {
            entries.push(Entry {
                tokens: tokens.iter().map(|t| t.as_str().into()).collect(),
                attributes: attributes.to_vec(),
            });
        })?;
        Ok(Entries { case_sensitive, entries })
    }

    /// The dictionary element's `case_sensitive` setting.
    pub fn is_case_sensitive(&self) -> bool {
        self.case_sensitive
    }

    pub fn as_slice(&self) -> &[Entry] {
        &self.entries
    }

    pub fn iter(&self) -> std::slice::Iter<'_, Entry> {
        self.entries.iter()
    }

    pub fn len(&self) -> usize {
        self.entries.len()
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }
}

/// Reads dictionary XML, calling `sink` with each entry's tokens and its
/// attributes. Returns the dictionary's case setting.
fn read_entries<F>(xml: &[u8], sink: F) -> Result<bool>
where
    F: FnMut(&[String], &[(Box<str>, Box<str>)]),
{
    let text = xml::decode(xml)?;
    let mut reader = EntryReader {
        sink,
        case_sensitive: true,
        inside_token: false,
        token: String::new(),
        tokens: Vec::new(),
        attributes: Vec::new(),
    };
    xml::parse(&text, &mut reader)?;
    Ok(reader.case_sensitive)
}

/// `DictionaryEntryPersistor.DictionaryContenthandler`.
struct EntryReader<F> {
    sink: F,
    case_sensitive: bool,
    inside_token: bool,
    /// The text of the token being read.
    token: String,
    /// The tokens read since the last entry ended.
    tokens: Vec<String>,
    /// The attributes of the entry being read.
    attributes: Vec<(Box<str>, Box<str>)>,
}

impl<F> Handler for EntryReader<F>
where
    F: FnMut(&[String], &[(Box<str>, Box<str>)]),
{
    fn start_element(&mut self, name: &str, attributes: &[Attribute<'_>], _offset: usize) -> Result<()> {
        match name {
            "dictionary" => {
                // Two attributes with the same local name (under different
                // prefixes) leave the last value, as the Java map does.
                if let Some(a) = attributes.iter().rev().find(|a| a.name == "case_sensitive") {
                    self.case_sensitive = a.value.eq_ignore_ascii_case("true");
                }
            }
            "entry" => {
                self.attributes.clear();
                for a in attributes {
                    match self.attributes.iter_mut().find(|(k, _)| **k == *a.name) {
                        Some(slot) => slot.1 = (*a.value).into(),
                        None => self.attributes.push((a.name.into(), (*a.value).into())),
                    }
                }
            }
            "token" => self.inside_token = true,
            _ => {}
        }
        Ok(())
    }

    fn end_element(&mut self, name: &str, offset: usize) -> Result<()> {
        match name {
            "token" => {
                self.tokens.push(java_trim(&self.token).to_string());
                self.token.clear();
                self.inside_token = false;
            }
            "entry" => {
                // StringList refuses an empty token list, which fails the load.
                if self.tokens.is_empty() {
                    return Err(Error::Dictionary { offset, message: "an entry has no tokens" });
                }
                (self.sink)(&self.tokens, &self.attributes);
                self.tokens.clear();
                self.attributes.clear();
            }
            _ => {}
        }
        Ok(())
    }

    fn characters(&mut self, text: &str) {
        if self.inside_token {
            self.token.push_str(text);
        }
    }
}

/// `String.trim`: removes every character at or below U+0020 from both
/// ends.
fn java_trim(s: &str) -> &str {
    s.trim_matches(|c: char| c <= '\u{20}')
}

#[cfg(test)]
mod tests {
    use super::*;

    const XML: &str = r#"<?xml version="1.0" encoding="UTF-8"?>
<dictionary case_sensitive="false">
  <entry><token> Dr. </token></entry>
  <entry kind="place"><token>New</token><token>York</token></entry>
  <entry><token>ÉTÉ</token></entry>
  <entry><token>dr.</token></entry>
</dictionary>"#;

    #[test]
    fn case_insensitive_lookups_lowercase_entries_and_probes() {
        let d = Dictionary::read(XML.as_bytes()).unwrap();
        assert!(!d.is_case_sensitive());
        assert_eq!(d.len(), 3);
        assert!(d.contains("DR."));
        assert!(d.contains("dr."));
        assert!(d.contains("été"));
        assert!(d.contains("Été"));
        assert!(!d.contains("New"));
        let mut s = String::new();
        assert!(d.contains_tokens(&["new", "YORK"], &mut s));
        assert!(!d.contains_tokens(&["York", "New"], &mut s));
        assert!(!d.contains("new\0york"));
        assert_eq!((d.min_tokens(), d.max_tokens()), (1, 2));
        let mut firsts: Vec<&str> = d.as_set().collect();
        firsts.sort();
        assert_eq!(firsts, ["dr.", "new", "été"]);
    }

    #[test]
    fn case_sensitive_is_the_default() {
        let d = Dictionary::read(b"<dictionary><entry><token>Mr.</token></entry></dictionary>").unwrap();
        assert!(d.is_case_sensitive());
        assert!(d.contains("Mr."));
        assert!(!d.contains("mr."));
        let d = Dictionary::read(b"<dictionary case_sensitive='TRUE'><entry><token>A</token></entry></dictionary>")
            .unwrap();
        assert!(d.is_case_sensitive());
        let d =
            Dictionary::read(b"<dictionary case_sensitive='yes'><entry><token>A</token></entry></dictionary>").unwrap();
        assert!(!d.is_case_sensitive());
        assert!(d.contains("a"));
    }

    #[test]
    fn entries_differing_only_in_case_keep_the_first_even_when_case_sensitive() {
        // What OpenNLP 3 does (checked on the JVM): "dr." is dropped on load.
        let xml = b"<dictionary case_sensitive='true'><entry><token>Dr.</token></entry><entry><token>dr.</token></entry></dictionary>";
        let d = Dictionary::read(xml).unwrap();
        assert_eq!(d.len(), 1);
        assert!(d.contains("Dr."));
        assert!(!d.contains("dr."));
    }

    #[test]
    fn dotted_capital_i_lowercases_to_i() {
        let xml = "<dictionary case_sensitive='false'><entry><token>\u{130}</token></entry><entry><token>\u{17F}</token></entry></dictionary>";
        let d = Dictionary::read(xml.as_bytes()).unwrap();
        assert!(d.contains("i"));
        assert!(d.contains("\u{130}"));
        assert!(!d.contains("s"));
        assert!(!d.contains("S"));
    }

    #[test]
    fn a_one_token_entry_with_a_space_is_not_two_tokens() {
        let d = Dictionary::read(b"<dictionary><entry><token>a b</token></entry></dictionary>").unwrap();
        assert!(d.contains("a b"));
        assert!(!d.contains_tokens(&["a", "b"], &mut String::new()));
    }

    #[test]
    fn entries_keep_attributes_and_order() {
        let e = Entries::read(XML.as_bytes()).unwrap();
        assert_eq!(e.len(), 4);
        assert_eq!(&*e.as_slice()[0].tokens[0], "Dr.");
        assert_eq!(e.as_slice()[1].tokens, vec![Box::from("New"), Box::from("York")]);
        assert_eq!(e.as_slice()[1].attribute("kind"), Some("place"));
        assert_eq!(e.as_slice()[0].attribute("kind"), None);
    }

    #[test]
    fn an_entry_without_tokens_is_refused() {
        assert!(matches!(
            Dictionary::read(b"<dictionary><entry a='1'/></dictionary>"),
            Err(Error::Dictionary { offset: 12, message: "an entry has no tokens" })
        ));
    }

    #[test]
    fn empty_dictionaries_report_zero_tokens() {
        let d = Dictionary::read(b"<dictionary/>").unwrap();
        assert!(d.is_empty());
        assert_eq!((d.min_tokens(), d.max_tokens()), (0, 0));
    }

    #[test]
    fn trim_follows_java() {
        assert_eq!(java_trim("\u{1}\t a b\n\u{20}"), "a b");
        assert_eq!(java_trim("\u{A0}x\u{A0}"), "\u{A0}x\u{A0}");
    }
}
