//! `DictionaryNameFinder`: names that are dictionary entries.
//!
//! From each token it tries every run of tokens up to the longest entry
//! (`getMaxTokenCount`), keeps the longest run that is an entry, and
//! resumes after it; with none, it resumes at the next token. So names
//! never overlap, and a long entry hides a shorter one that starts inside
//! it. A run is looked up as OpenNLP's `Dictionary.contains` looks up a
//! `StringList`, which `turbo_maxent::Dictionary` reproduces: case
//! folded per character with Java's lowercase mapping when the dictionary
//! is case-insensitive, exact when it is case-sensitive.

use std::sync::Arc;

use turbo_analysis_layers::{Tag, TagSet};
use turbo_maxent::Dictionary;

use crate::error::Error;
use crate::name::{Name, TokenNameFinder, kinds};

/// `opennlp.tools.namefind.DictionaryNameFinder`.
#[derive(Clone, Debug)]
pub struct DictionaryNameFinder {
    dictionary: Dictionary,
    kinds: Arc<TagSet>,
}

impl DictionaryNameFinder {
    /// The type `DictionaryNameFinder(Dictionary)` gives its names.
    pub const DEFAULT_TYPE: &'static str = "default";

    /// `new DictionaryNameFinder(dictionary, type)`.
    pub fn new(dictionary: Dictionary, kind: &str) -> DictionaryNameFinder {
        let kinds = kinds(std::iter::once(kind)).expect("one type");
        DictionaryNameFinder { dictionary, kinds }
    }

    pub fn dictionary(&self) -> &Dictionary {
        &self.dictionary
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
            let found =
                (from + 1..=last).rev().find(|&end| self.dictionary.contains_tokens(&tokens[from..end], &mut scratch));
            match found {
                Some(end) => {
                    out.push(Name { start: from as u32, end: end as u32, kind: Tag(0) });
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

    fn find(xml: &str, tokens: &[&str]) -> Vec<(u32, u32)> {
        let f = DictionaryNameFinder::new(Dictionary::read(xml.as_bytes()).unwrap(), "place");
        let mut out = Vec::new();
        f.find(tokens, &mut out).unwrap();
        out.iter().map(|n| (n.start, n.end)).collect()
    }

    const XML: &str = "<dictionary case_sensitive='false'>\
        <entry><token>New</token><token>York</token></entry>\
        <entry><token>New</token><token>York</token><token>City</token></entry>\
        <entry><token>York</token><token>City</token><token>Hall</token></entry>\
        <entry><token>City</token></entry></dictionary>";

    #[test]
    fn longest_entry_wins_and_hides_overlaps() {
        assert_eq!(find(XML, &["in", "new", "YORK", "City", "Hall"]), [(1, 4)]);
        assert_eq!(find(XML, &["York", "City", "Hall", "City"]), [(0, 3), (3, 4)]);
        assert_eq!(find(XML, &["New", "Jersey", "City"]), [(2, 3)]);
        assert_eq!(find(XML, &[]), []);
    }
}
