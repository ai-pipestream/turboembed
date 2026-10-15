//! Token and character n-gram frequency table (`opennlp.tools.ngram.NGramModel`).
//!
//! [`NGramModel`] stores case-sensitive token sequences (`StringList` in OpenNLP)
//! with their `i32` occurrence counts in insertion order (`LinkedHashMap` in the
//! JVM). Lookups, count updates, and removals operate directly on `&[S]` slices
//! (`S: AsRef<str>`) without allocating a key vector.

use std::collections::HashMap;
use std::fmt::{self, Write as _};
use std::hash::{BuildHasher, Hasher};

use turbo_jvm::text::parse_int;
use turbo_maxent::{Dictionary, Entries};
use turbo_unicode::java::code_point::to_lower_case;

use crate::error::{Error, Result};

const COUNT_ATTR: &str = "count";

/// A counted collection of token n-grams, porting `opennlp.tools.ngram.NGramModel`.
///
/// Entries are ordered by first insertion (`LinkedHashMap` semantics): updating an
/// existing n-gram's count via [`add`](Self::add) or [`set_count`](Self::set_count)
/// preserves its position, while [`remove`](Self::remove) followed by [`add`](Self::add)
/// places the re-inserted n-gram at the end.
#[derive(Debug, Clone, Default)]
pub struct NGramModel {
    /// Slots in insertion order; `None` marks a removed entry.
    entries: Vec<Option<(Vec<String>, i32)>>,
    /// Hash-to-slot-index lookup table so `&[&str]` probes never allocate.
    buckets: HashMap<u64, Vec<usize>>,
    /// Hasher builder used to compute 64-bit hashes for token sequences.
    hasher_builder: std::collections::hash_map::RandomState,
    /// Number of active (`Some`) entries.
    active_len: usize,
}

impl NGramModel {
    /// Creates an empty [`NGramModel`].
    pub fn new() -> Self {
        Self::default()
    }

    /// Reads a serialized [`NGramModel`] from OpenNLP dictionary XML bytes.
    ///
    /// Every `<entry>` must have a valid integer `count` attribute and at least
    /// one `<token>` child. As in `new NGramModel(InputStream)`, each entry is
    /// added as a case-sensitive token sequence and its count is set to the parsed
    /// `count` attribute.
    pub fn read(xml: &[u8]) -> Result<Self> {
        let parsed = Entries::read(xml).map_err(Error::DictionaryXml)?;
        let mut model = Self::new();
        for (entry_index, entry) in parsed.iter().enumerate() {
            let Some(raw_count) = entry.attribute(COUNT_ATTR) else {
                return Err(Error::MissingCountAttribute { entry_index });
            };
            let Ok(count) = parse_int(raw_count) else {
                return Err(Error::InvalidCountAttribute { entry_index, value: raw_count.to_string() });
            };
            model.add(&entry.tokens)?;
            model.set_count(&entry.tokens, count)?;
        }
        Ok(model)
    }

    /// Returns the count of `ngram`, or `0` if `ngram` is not present.
    pub fn get_count<S: AsRef<str>>(&self, ngram: &[S]) -> i32 {
        self.find_slot(ngram).and_then(|(_, slot)| self.entries[slot].as_ref()).map_or(0, |(_, count)| *count)
    }

    /// Sets the count of an existing `ngram` to `count`.
    ///
    /// # Errors
    ///
    /// Returns [`Error::NoSuchNGram`] if `ngram` is not present in the model
    /// (matching OpenNLP's `NoSuchElementException`).
    pub fn set_count<S: AsRef<str>>(&mut self, ngram: &[S], count: i32) -> Result<()> {
        let Some((_, slot)) = self.find_slot(ngram) else {
            return Err(Error::NoSuchNGram);
        };
        if let Some((_, existing_count)) = &mut self.entries[slot] {
            *existing_count = count;
            Ok(())
        } else {
            Err(Error::NoSuchNGram)
        }
    }

    /// Adds one occurrence of `ngram` to the model, inserting it with count `1`
    /// if absent or incrementing its count by `1` if already present.
    ///
    /// # Errors
    ///
    /// Returns [`Error::EmptyNGram`] if `ngram` is empty.
    pub fn add<S: AsRef<str>>(&mut self, ngram: &[S]) -> Result<()> {
        if ngram.is_empty() {
            return Err(Error::EmptyNGram);
        }
        let hash = self.hash_tokens(ngram);
        if let Some(bucket) = self.buckets.get(&hash) {
            for &slot in bucket {
                if let Some((tokens, _)) = &self.entries[slot]
                    && tokens_eq(tokens, ngram)
                {
                    if let Some((_, count)) = &mut self.entries[slot] {
                        *count = count.wrapping_add(1);
                    }
                    return Ok(());
                }
            }
        }
        let slot = self.entries.len();
        let owned: Vec<String> = ngram.iter().map(|s| s.as_ref().to_string()).collect();
        self.entries.push(Some((owned, 1)));
        self.buckets.entry(hash).or_default().push(slot);
        self.active_len += 1;
        Ok(())
    }

    /// Adds all contiguous token sub-spans of `ngram` whose lengths are between
    /// `min_length` and `max_length` (inclusive).
    ///
    /// # Errors
    ///
    /// Returns [`Error::InvalidLengthRange`] if `min_length < 1`, `max_length < 1`,
    /// or `min_length > max_length`, or [`Error::EmptyNGram`] if `ngram` is empty.
    pub fn add_range<S: AsRef<str>>(&mut self, ngram: &[S], min_length: usize, max_length: usize) -> Result<()> {
        if min_length < 1 || max_length < 1 || min_length > max_length {
            return Err(Error::InvalidLengthRange { min_length, max_length });
        }
        if ngram.is_empty() {
            return Err(Error::EmptyNGram);
        }
        for length_index in min_length..=max_length {
            if length_index > ngram.len() {
                break;
            }
            for text_index in 0..=ngram.len() - length_index {
                self.add(&ngram[text_index..text_index + length_index])?;
            }
        }
        Ok(())
    }

    /// Adds character n-grams of lengths `min_length..=max_length` (in UTF-16 code
    /// units) from `chars`, after lowercasing each code point with Java's
    /// `Character.toLowerCase(int)` (`StringUtil.toLowerCaseCharBuffer`).
    ///
    /// Each character n-gram is added as a single-token entry.
    pub fn add_chars(&mut self, chars: &str, min_length: usize, max_length: usize) -> Result<()> {
        if min_length < 1 || max_length < 1 || min_length > max_length {
            return Err(Error::InvalidLengthRange { min_length, max_length });
        }
        let units = lowercase_utf16_units(chars);
        let mut gram_buf = String::new();
        for length_index in min_length..=max_length {
            if length_index > units.len() {
                break;
            }
            for text_index in 0..=units.len() - length_index {
                decode_utf16_slice(&units[text_index..text_index + length_index], &mut gram_buf);
                self.add(&[gram_buf.as_str()])?;
            }
        }
        Ok(())
    }

    /// Removes `ngram` from the model if present.
    ///
    /// Returns `true` if the n-gram was present and removed.
    pub fn remove<S: AsRef<str>>(&mut self, ngram: &[S]) -> bool {
        let Some((hash, slot)) = self.find_slot(ngram) else {
            return false;
        };
        self.entries[slot] = None;
        self.active_len -= 1;
        if let Some(bucket) = self.buckets.get_mut(&hash) {
            bucket.retain(|&s| s != slot);
            if bucket.is_empty() {
                self.buckets.remove(&hash);
            }
        }
        true
    }

    /// Returns `true` if `ngram` is present in the model.
    pub fn contains<S: AsRef<str>>(&self, ngram: &[S]) -> bool {
        self.find_slot(ngram).is_some()
    }

    /// Returns the number of distinct n-grams in the model (`NGramModel.size()`).
    pub fn size(&self) -> usize {
        self.active_len
    }

    /// Returns the number of distinct n-grams in the model.
    pub fn len(&self) -> usize {
        self.active_len
    }

    /// Returns `true` if the model contains no n-grams.
    pub fn is_empty(&self) -> bool {
        self.active_len == 0
    }

    /// Returns the sum of counts across all n-grams in the model
    /// (`NGramModel.numberOfGrams()`).
    pub fn number_of_grams(&self) -> i32 {
        self.entries.iter().filter_map(|slot| slot.as_ref().map(|(_, count)| *count)).fold(0_i32, i32::wrapping_add)
    }

    /// Removes every n-gram whose count is strictly less than `cutoff_under` or
    /// strictly greater than `cutoff_over`.
    pub fn cutoff(&mut self, cutoff_under: i32, cutoff_over: i32) {
        if cutoff_under <= 0 && cutoff_over == i32::MAX {
            return;
        }
        for slot in 0..self.entries.len() {
            let should_remove = match &self.entries[slot] {
                Some((_, count)) => *count < cutoff_under || *count > cutoff_over,
                None => false,
            };
            if should_remove {
                let (tokens, _) = self.entries[slot].take().expect("checked Some");
                self.active_len -= 1;
                let hash = self.hash_tokens(&tokens);
                if let Some(bucket) = self.buckets.get_mut(&hash) {
                    bucket.retain(|&s| s != slot);
                    if bucket.is_empty() {
                        self.buckets.remove(&hash);
                    }
                }
            }
        }
    }

    /// Converts this model's n-grams into a [`Dictionary`], folding case when
    /// `case_sensitive` is `false` (`NGramModel.toDictionary(boolean)`).
    pub fn to_dictionary(&self, case_sensitive: bool) -> Dictionary {
        Dictionary::from_entries(case_sensitive, self.iter())
    }

    /// Iterates over the distinct n-grams in insertion order.
    pub fn iter(&self) -> impl Iterator<Item = &[String]> {
        self.entries.iter().filter_map(|slot| slot.as_ref().map(|(tokens, _)| tokens.as_slice()))
    }

    /// Iterates over `(ngram, count)` pairs in insertion order.
    pub fn iter_with_counts(&self) -> impl Iterator<Item = (&[String], i32)> {
        self.entries.iter().filter_map(|slot| slot.as_ref().map(|(tokens, count)| (tokens.as_slice(), *count)))
    }

    /// Serializes the model to OpenNLP dictionary XML bytes
    /// (`NGramModel.serialize(OutputStream)`).
    ///
    /// Note that `NGramModel.serialize` always writes `case_sensitive="false"` on
    /// the root `<dictionary>` element (`DictionaryEntryPersistor.serialize(..., false)`).
    pub fn serialize(&self) -> Result<Vec<u8>> {
        let mut out = String::from("<?xml version=\"1.0\" encoding=\"UTF-8\"?><dictionary case_sensitive=\"false\"");
        let mut any = false;
        for (tokens, count) in self.iter_with_counts() {
            if !any {
                out.push_str(">\n");
                any = true;
            }
            writeln!(out, "    <entry count=\"{count}\">").expect("writing to String");
            for token in tokens {
                out.push_str("        <token>");
                push_xml_text(&mut out, token)?;
                out.push_str("</token>\n");
            }
            out.push_str("    </entry>\n");
        }
        out.push_str(if any { "</dictionary>\n" } else { "/>\n" });
        Ok(out.into_bytes())
    }

    fn hash_tokens<S: AsRef<str>>(&self, ngram: &[S]) -> u64 {
        let mut hasher = self.hasher_builder.build_hasher();
        hasher.write_usize(ngram.len());
        for token in ngram {
            let bytes = token.as_ref().as_bytes();
            hasher.write_usize(bytes.len());
            hasher.write(bytes);
        }
        hasher.finish()
    }

    fn find_slot<S: AsRef<str>>(&self, ngram: &[S]) -> Option<(u64, usize)> {
        if ngram.is_empty() {
            return None;
        }
        let hash = self.hash_tokens(ngram);
        let bucket = self.buckets.get(&hash)?;
        for &slot in bucket {
            if let Some((tokens, _)) = &self.entries[slot]
                && tokens_eq(tokens, ngram)
            {
                return Some((hash, slot));
            }
        }
        None
    }
}

impl PartialEq for NGramModel {
    fn eq(&self, other: &Self) -> bool {
        if self.active_len != other.active_len {
            return false;
        }
        for (tokens, count) in self.iter_with_counts() {
            let Some((_, slot)) = other.find_slot(tokens) else {
                return false;
            };
            if other.entries[slot].as_ref().map(|(_, c)| *c) != Some(count) {
                return false;
            }
        }
        true
    }
}

impl Eq for NGramModel {}

impl fmt::Display for NGramModel {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Size: {}", self.size())
    }
}

fn tokens_eq<S: AsRef<str>>(stored: &[String], probe: &[S]) -> bool {
    stored.len() == probe.len() && stored.iter().zip(probe).all(|(a, b)| a.as_str() == b.as_ref())
}

/// Converts `text` to UTF-16 code units after lowercasing each code point with
/// `Character.toLowerCase(int)` (`StringUtil.toLowerCaseCharBuffer`).
pub(crate) fn lowercase_utf16_units(text: &str) -> Vec<u16> {
    let mut units = Vec::with_capacity(text.len());
    for c in text.chars() {
        let lower = to_lower_case(u32::from(c));
        if lower <= 0xFFFF {
            units.push(lower as u16);
        } else {
            let v = lower - 0x1_0000;
            units.push(0xD800 | (v >> 10) as u16);
            units.push(0xDC00 | (v & 0x3FF) as u16);
        }
    }
    units
}

pub(crate) fn decode_utf16_slice(units: &[u16], out: &mut String) {
    out.clear();
    for decoded in char::decode_utf16(units.iter().copied()) {
        out.push(decoded.unwrap_or(char::REPLACEMENT_CHARACTER));
    }
}

fn xml_can_carry(c: char) -> bool {
    !(((c as u32) < 0x20 && !matches!(c, '\t' | '\n' | '\r')) || matches!(c, '\u{FFFE}' | '\u{FFFF}'))
}

fn push_xml_text(out: &mut String, text: &str) -> Result<()> {
    for c in text.chars() {
        match c {
            '&' => out.push_str("&amp;"),
            '<' => out.push_str("&lt;"),
            '>' => out.push_str("&gt;"),
            '\r' => out.push_str("&#13;"),
            '\t' | '\n' => out.push(c),
            c if !xml_can_carry(c) => return Err(Error::InvalidXmlCharacter(c as u32)),
            c if (0x7F..=0x9F).contains(&(c as u32)) || (c as u32) > 0xFFFF => {
                write!(out, "&#{};", c as u32).expect("writing to String");
            }
            c => out.push(c),
        }
    }
    Ok(())
}
