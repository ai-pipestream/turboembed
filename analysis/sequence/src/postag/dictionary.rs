//! `POSDictionary`, the tag dictionary a part-of-speech model may store
//! as `tags.tagdict`, and `DefaultPOSSequenceValidator`, its only reader
//! when tagging.
//!
//! The dictionary restricts the tags of the words it lists; it adds no
//! features. `POSTaggerFactory.getSequenceValidator` hands it to the
//! validator, and neither context generator reads it (the n-gram
//! dictionary of models before 1.8 is a different entry,
//! `ngram.dictionary`). When a model is loaded OpenNLP does not check that
//! the dictionary's tags are outcomes of the model; it checks that only
//! when it builds a model from parts it did not read from a file.

use std::collections::HashMap;
use std::collections::hash_map::Entry as MapEntry;
use std::sync::Arc;

use turbo_maxent::{Entries, ModelFile};
use turbo_unicode::java::code_point::to_lower_case;

use super::TAG_DICTIONARY_ENTRY_NAME;
use crate::beam::SequenceValidator;
use crate::error::{Error, Result};
use crate::java::{lowercase_into, split_trimmed};

/// `POSDictionary`: the tags each word may have.
#[derive(Debug, Clone, Default)]
pub struct TagDictionary {
    case_sensitive: bool,
    tags: HashMap<String, Vec<String>>,
}

impl TagDictionary {
    /// Reads the dictionary XML (`POSDictionary.create`): one token per
    /// entry, its tags in the `tags` attribute split at single spaces as
    /// `StringUtil.split` splits them (an empty field between two spaces
    /// is an empty tag, trailing empty fields are dropped). A word listed
    /// twice keeps its last tags. The dictionary is case sensitive unless
    /// the root element says `case_sensitive="false"`.
    ///
    /// In a case-insensitive dictionary every word is lowercased
    /// (`StringUtil.toLowerCase`, per code point). Java does this after
    /// reading, by walking the `HashMap` of the words as written and
    /// putting each into a new map under its lowercase form, so when
    /// several words lowercase to the same one, the word Java's hash order
    /// visits last gives the tags, not the word last in the file. That
    /// order is reproduced here ([`java_hash_order`]).
    pub fn read(xml: &[u8]) -> Result<TagDictionary> {
        let entries = Entries::read(xml)?;
        let case_sensitive = entries.is_case_sensitive();
        // The words as written, in the order they were first put into
        // Java's map, each with its last tags.
        let mut words: Vec<(&str, Vec<String>)> = Vec::new();
        let mut index: HashMap<&str, usize> = HashMap::new();
        for entry in entries.iter() {
            if entry.tokens.len() != 1 {
                return Err(Error::TagDictionary(format!(
                    "Each entry must have exactly one token! {:?}",
                    entry.tokens
                )));
            }
            let Some(list) = entry.attribute("tags") else {
                return Err(Error::TagDictionary(format!("the entry {:?} has no tags", entry.tokens[0])));
            };
            let list: Vec<String> = split_trimmed(list, ' ').into_iter().map(str::to_owned).collect();
            let word: &str = &entry.tokens[0];
            match index.entry(word) {
                MapEntry::Occupied(o) => words[*o.get()].1 = list,
                MapEntry::Vacant(v) => {
                    v.insert(words.len());
                    words.push((word, list));
                }
            }
        }
        let mut tags = HashMap::with_capacity(words.len());
        if case_sensitive {
            for (word, list) in words {
                tags.insert(word.to_owned(), list);
            }
        } else {
            for i in java_hash_order(words.iter().map(|(w, _)| *w)) {
                let (word, list) = std::mem::take(&mut words[i]);
                let mut lower = String::with_capacity(word.len());
                lowercase_into(&mut lower, word);
                tags.insert(lower, list);
            }
        }
        Ok(TagDictionary { case_sensitive, tags })
    }

    /// The model's tag dictionary, if it has one.
    pub fn for_model(model: &ModelFile) -> Result<Option<TagDictionary>> {
        match model.entry(TAG_DICTIONARY_ENTRY_NAME) {
            Ok(bytes) => TagDictionary::read(&bytes).map(Some),
            Err(turbo_maxent::Error::MissingEntry(_)) => Ok(None),
            Err(e) => Err(e.into()),
        }
    }

    pub fn is_case_sensitive(&self) -> bool {
        self.case_sensitive
    }

    /// The number of words.
    pub fn len(&self) -> usize {
        self.tags.len()
    }

    pub fn is_empty(&self) -> bool {
        self.tags.is_empty()
    }

    /// Each word with its tags, in no particular order. In a
    /// case-insensitive dictionary the words are lowercased.
    pub fn iter(&self) -> impl Iterator<Item = (&str, &[String])> {
        self.tags.iter().map(|(w, t)| (w.as_str(), t.as_slice()))
    }

    /// `getTags`: the tags a word may have, or `None` when the dictionary
    /// does not list it. A case-insensitive dictionary lowercases the
    /// word first, into a new string only when lowercasing changes it.
    pub fn tags(&self, word: &str) -> Option<&[String]> {
        if self.case_sensitive || word.chars().all(|c| to_lower_case(c as u32) == c as u32) {
            return self.tags.get(word).map(Vec::as_slice);
        }
        let mut lower = String::with_capacity(word.len());
        lowercase_into(&mut lower, word);
        self.tags.get(&lower).map(Vec::as_slice)
    }
}

/// The order in which a `java.util.HashMap`, filled by putting `keys` one
/// by one into `new HashMap<>()`, iterates them: the indices of `keys`,
/// by bucket and, within a bucket, by insertion. Keys must be distinct.
///
/// The table starts with 16 buckets and doubles whenever the map holds
/// more than three quarters as many keys; a key's bucket is the low bits
/// of `h ^ (h >>> 16)`, where `h` is `String.hashCode()`. Doubling keeps
/// the keys of a bucket in insertion order. Java turns a bucket of more
/// than eight keys into a tree once the table has 64 buckets, which can
/// change the order within that bucket; that is not reproduced. A
/// bucket that full needs keys chosen to collide.
fn java_hash_order<'a>(keys: impl ExactSizeIterator<Item = &'a str>) -> Vec<usize> {
    let n = keys.len();
    let mut buckets = 16usize;
    while n > buckets / 4 * 3 {
        buckets *= 2;
    }
    let mask = buckets as u32 - 1;
    let mut order: Vec<(u32, usize)> = keys
        .enumerate()
        .map(|(i, key)| {
            let h = java_string_hash(key) as u32;
            ((h ^ (h >> 16)) & mask, i)
        })
        .collect();
    order.sort_unstable();
    order.into_iter().map(|(_, i)| i).collect()
}

/// `String.hashCode()`: `s[0]*31^(n-1) + ... + s[n-1]` over the UTF-16
/// code units, wrapping.
fn java_string_hash(s: &str) -> i32 {
    s.encode_utf16().fold(0i32, |h, u| h.wrapping_mul(31).wrapping_add(i32::from(u)))
}

/// `DefaultPOSSequenceValidator`: a word the tag dictionary lists may
/// only take one of its tags; any other word, or any word without a
/// dictionary, any tag.
#[derive(Debug, Clone, Default)]
pub struct PosValidator {
    dictionary: Option<Arc<TagDictionary>>,
}

impl PosValidator {
    pub fn new(dictionary: Option<Arc<TagDictionary>>) -> PosValidator {
        PosValidator { dictionary }
    }
}

impl<'a> SequenceValidator<&'a str> for PosValidator {
    fn valid_sequence(&self, index: usize, sequence: &[&'a str], _prior: &[&str], outcome: &str) -> bool {
        match &self.dictionary {
            None => true,
            Some(d) => match d.tags(sequence[index]) {
                None => true,
                Some(tags) => tags.iter().any(|t| t == outcome),
            },
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tag_dictionaries_restrict_listed_words() {
        let d = TagDictionary::read(
            b"<dictionary case_sensitive=\"false\"><entry tags=\"NN VB\"><token>Run</token></entry></dictionary>",
        )
        .unwrap();
        assert_eq!(d.tags("RUN").unwrap(), ["NN", "VB"]);
        let v = PosValidator::new(Some(Arc::new(d)));
        assert!(v.valid_sequence(0, &["run"], &[], "VB"));
        assert!(!v.valid_sequence(0, &["run"], &[], "JJ"));
        assert!(v.valid_sequence(0, &["walk"], &[], "JJ"));
    }

    #[test]
    fn string_hashes_are_java_ones() {
        assert_eq!(java_string_hash(""), 0);
        assert_eq!(java_string_hash("a"), 97);
        // "Aa" and "BB" are the classic colliding pair.
        assert_eq!(java_string_hash("Aa"), java_string_hash("BB"));
        assert_eq!(java_string_hash("hello world"), 1_794_106_052);
        // U+10400 is the two units D801 DC00.
        assert_eq!(java_string_hash("\u{10400}"), 0xD801 * 31 + 0xDC00);
    }

    #[test]
    fn hash_order_follows_buckets_then_insertion() {
        // "Aa" and "BB" share bucket 0 and keep their insertion order
        // there, ahead of "a" in bucket 1.
        assert_eq!(java_hash_order(["BB", "a", "Aa"].into_iter()), [0, 2, 1]);
    }

    #[test]
    fn entries_with_several_tokens_are_refused() {
        let xml = b"<dictionary><entry tags=\"NN\"><token>a</token><token>b</token></entry></dictionary>";
        assert!(TagDictionary::read(xml).is_err());
        assert!(TagDictionary::read(b"<dictionary><entry><token>a</token></entry></dictionary>").is_err());
    }
}
