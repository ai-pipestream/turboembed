//! The binary model format of OpenNLP's `SymSpellModelSerializer`, read
//! and written directly.
//!
//! The format holds the source dictionary and the configuration, not the
//! deletes index, which is rebuilt on load. Big-endian, as Java's
//! `DataOutputStream` writes it:
//!
//! ```text
//! int    magic = 0x53594D53 ("SYMS")
//! int    format version = 1
//! UTF    language, UTF name, UTF version
//! int    maxDictionaryEditDistance
//! int    prefixLength
//! long   countThreshold
//! UTF    edit distance id ("damerau-osa" or "levenshtein")
//! long   corpusWordCount (0 derives N from the counts)
//! int    unigram count, then that many (UTF word, vlong count)
//! int    bigram count, then that many (UTF w1, UTF w2, vlong count)
//! ```
//!
//! `UTF` is `DataOutputStream.writeUTF`: an unsigned 16-bit byte length,
//! then Java's modified UTF-8 (U+0000 as two bytes, a supplementary
//! character as its two surrogates, three bytes each). `vlong` is seven
//! bits per byte, least significant first, the high bit set on every byte
//! but the last.

use std::collections::HashMap;
use std::fmt;

use turbo_unicode::java;

use super::config::{ConfigError, MAX_COUNT, SymSpellConfig};
use super::distance::Metric;
use super::symspell::SymSpell;

pub const MAGIC: u32 = 0x5359_4D53;
pub const FORMAT_VERSION: u32 = 1;

/// The most unigram or bigram rows a model may declare: OpenNLP's default
/// `ResourceLimits.MAX_ENTRIES`, which keeps a corrupt count from forcing
/// a huge allocation.
pub const MAX_ENTRIES: usize = 10_000_000;

/// OpenNLP's default model name.
pub const DEFAULT_MODEL_NAME: &str = "spellcheck-symspell";
/// OpenNLP's default model version.
pub const DEFAULT_MODEL_VERSION: &str = "1.0";

/// A spelling model: metadata, a configuration and the source dictionary,
/// as OpenNLP's `SymSpellModel` holds them. The unigrams keep the order
/// they were first given in, and a repeated word keeps its first position
/// with its last count, as a Java `LinkedHashMap` does; bigrams are keyed
/// by `"w1 w2"` the same way.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SymSpellModel {
    language: String,
    name: String,
    version: String,
    config: SymSpellConfig,
    unigrams: Vec<(String, u64)>,
    bigrams: Vec<(String, u64)>,
}

/// Why a model could not be read, built or written.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ModelError {
    /// The bytes ended in the middle of a field.
    Truncated,
    BadMagic(u32),
    UnsupportedVersion(u32),
    /// A `UTF` field that is not modified UTF-8, or holds an unpaired
    /// surrogate, which a Rust string cannot.
    MalformedUtf,
    /// A string longer than the 65535 bytes a `UTF` field holds.
    StringTooLong,
    /// A configuration field out of range, or an unknown edit distance id.
    CorruptConfig(String),
    /// A unigram or bigram row count below 0 or above the limit; the field
    /// is `"unigram count"` or `"bigram count"`.
    EntryCount {
        field: &'static str,
        value: i64,
    },
    /// A `vlong` of more than ten bytes.
    VLongTooLong,
    /// A count above `Long.MAX_VALUE`, which Java reads as negative.
    NegativeCount,
    /// A bigram key that is not two words around a space.
    InvalidBigramKey(String),
    /// A blank language, name or version.
    Blank(&'static str),
    /// Bytes after the last bigram.
    TrailingBytes(usize),
}

impl fmt::Display for ModelError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ModelError::Truncated => write!(f, "unexpected end of the SymSpell model stream"),
            ModelError::BadMagic(m) => {
                write!(f, "not a SymSpell model stream (magic was 0x{m:08X}, expected 0x{MAGIC:08X})")
            }
            ModelError::UnsupportedVersion(v) => {
                write!(f, "unsupported SymSpell model format version: {v} (this build reads version {FORMAT_VERSION})")
            }
            ModelError::MalformedUtf => write!(f, "malformed modified UTF-8 string"),
            ModelError::StringTooLong => write!(f, "encoded string too long"),
            ModelError::CorruptConfig(m) => write!(f, "corrupt SymSpell model config: {m}"),
            ModelError::EntryCount { field, value } => write!(f, "{field} out of range [0, {MAX_ENTRIES}]: {value}"),
            ModelError::VLongTooLong => write!(f, "variable-length long is too long (corrupt stream)"),
            ModelError::NegativeCount => write!(f, "count must not be negative"),
            ModelError::InvalidBigramKey(k) => write!(f, "invalid bigram key (expected 'w1 w2'): {k}"),
            ModelError::Blank(field) => write!(f, "{field} must not be null or blank"),
            ModelError::TrailingBytes(n) => write!(f, "{n} bytes after the end of the SymSpell model"),
        }
    }
}

impl std::error::Error for ModelError {}

impl From<ConfigError> for ModelError {
    fn from(e: ConfigError) -> ModelError {
        ModelError::CorruptConfig(e.to_string())
    }
}

fn require_non_blank(value: &str, field: &'static str) -> Result<(), ModelError> {
    if value.chars().all(java::is_whitespace) { Err(ModelError::Blank(field)) } else { Ok(()) }
}

impl SymSpellModel {
    /// A model from its parts. Repeated unigrams and bigrams are merged as
    /// described on the type; counts above [`MAX_COUNT`] are refused.
    pub fn new(
        language: &str,
        name: &str,
        version: &str,
        config: SymSpellConfig,
        unigrams: impl IntoIterator<Item = (String, u64)>,
        bigrams: impl IntoIterator<Item = (String, u64)>,
    ) -> Result<SymSpellModel, ModelError> {
        require_non_blank(language, "language")?;
        require_non_blank(name, "name")?;
        require_non_blank(version, "version")?;
        let unigrams = linked(unigrams)?;
        let bigrams = linked(bigrams)?;
        for (key, _) in &bigrams {
            split_bigram(key)?;
        }
        Ok(SymSpellModel {
            language: language.to_owned(),
            name: name.to_owned(),
            version: version.to_owned(),
            config,
            unigrams,
            bigrams,
        })
    }

    pub fn language(&self) -> &str {
        &self.language
    }

    pub fn name(&self) -> &str {
        &self.name
    }

    pub fn version(&self) -> &str {
        &self.version
    }

    pub fn config(&self) -> &SymSpellConfig {
        &self.config
    }

    /// The unigrams, in order.
    pub fn unigrams(&self) -> &[(String, u64)] {
        &self.unigrams
    }

    /// The bigrams, keyed `"w1 w2"`, in order.
    pub fn bigrams(&self) -> &[(String, u64)] {
        &self.bigrams
    }

    /// The engine, built by adding the unigrams and then the bigrams in
    /// order, as OpenNLP builds it.
    pub fn build(&self) -> SymSpell {
        let mut engine = SymSpell::new(self.config);
        for (word, count) in &self.unigrams {
            engine.add(word, *count);
        }
        for (key, count) in &self.bigrams {
            let (w1, w2) = split_bigram(key).expect("checked when the model was made");
            engine.add_bigram(w1, w2, *count);
        }
        engine
    }

    /// Reads a model from the bytes OpenNLP's serializer writes.
    pub fn read(bytes: &[u8]) -> Result<SymSpellModel, ModelError> {
        SymSpellModel::read_with_limit(bytes, MAX_ENTRIES)
    }

    /// `read`, with another limit on the row counts.
    pub fn read_with_limit(bytes: &[u8], max_entries: usize) -> Result<SymSpellModel, ModelError> {
        let mut r = Reader { bytes, at: 0 };
        let magic = r.u32()?;
        if magic != MAGIC {
            return Err(ModelError::BadMagic(magic));
        }
        let version = r.u32()?;
        if version != FORMAT_VERSION {
            return Err(ModelError::UnsupportedVersion(version));
        }
        let language = r.utf()?;
        let name = r.utf()?;
        let model_version = r.utf()?;
        let config = r.config()?;
        let unigram_rows = r.rows("unigram count", max_entries)?;
        let mut unigrams = Vec::with_capacity(unigram_rows);
        for _ in 0..unigram_rows {
            let word = r.utf()?;
            unigrams.push((word, r.count()?));
        }
        let bigram_rows = r.rows("bigram count", max_entries)?;
        let mut bigrams = Vec::with_capacity(bigram_rows);
        for _ in 0..bigram_rows {
            let w1 = r.utf()?;
            let w2 = r.utf()?;
            bigrams.push((format!("{w1} {w2}"), r.count()?));
        }
        if r.at < bytes.len() {
            return Err(ModelError::TrailingBytes(bytes.len() - r.at));
        }
        SymSpellModel::new(&language, &name, &model_version, config, unigrams, bigrams)
    }

    /// The bytes OpenNLP's serializer writes for this model.
    pub fn write(&self) -> Result<Vec<u8>, ModelError> {
        let mut out = Vec::new();
        out.extend_from_slice(&MAGIC.to_be_bytes());
        out.extend_from_slice(&FORMAT_VERSION.to_be_bytes());
        for s in [&self.language, &self.name, &self.version] {
            write_utf(&mut out, s)?;
        }
        let c = &self.config;
        out.extend_from_slice(&(c.max_dictionary_edit_distance() as i32).to_be_bytes());
        out.extend_from_slice(&(c.prefix_length() as i32).to_be_bytes());
        out.extend_from_slice(&c.count_threshold().to_be_bytes());
        write_utf(&mut out, c.metric().id())?;
        out.extend_from_slice(&c.corpus_word_count().to_be_bytes());
        out.extend_from_slice(&(self.unigrams.len() as i32).to_be_bytes());
        for (word, count) in &self.unigrams {
            write_utf(&mut out, word)?;
            write_vlong(&mut out, *count);
        }
        out.extend_from_slice(&(self.bigrams.len() as i32).to_be_bytes());
        for (key, count) in &self.bigrams {
            let (w1, w2) = split_bigram(key)?;
            write_utf(&mut out, w1)?;
            write_utf(&mut out, w2)?;
            write_vlong(&mut out, *count);
        }
        Ok(out)
    }
}

/// Merges repeated keys as a Java `LinkedHashMap.put` does: the first
/// position, the last value.
fn linked(rows: impl IntoIterator<Item = (String, u64)>) -> Result<Vec<(String, u64)>, ModelError> {
    let mut out: Vec<(String, u64)> = Vec::new();
    let mut at: HashMap<String, usize> = HashMap::new();
    for (key, count) in rows {
        if count > MAX_COUNT {
            return Err(ModelError::NegativeCount);
        }
        match at.get(&key) {
            Some(&i) => out[i].1 = count,
            None => {
                at.insert(key.clone(), out.len());
                out.push((key, count));
            }
        }
    }
    Ok(out)
}

/// Splits a bigram key at its first space, which must have a character on
/// each side.
fn split_bigram(key: &str) -> Result<(&str, &str), ModelError> {
    match key.find(' ') {
        Some(space) if space >= 1 && space + 1 < key.len() => Ok((&key[..space], &key[space + 1..])),
        _ => Err(ModelError::InvalidBigramKey(key.to_owned())),
    }
}

struct Reader<'a> {
    bytes: &'a [u8],
    at: usize,
}

impl Reader<'_> {
    fn take(&mut self, n: usize) -> Result<&[u8], ModelError> {
        let end = self.at.checked_add(n).filter(|&e| e <= self.bytes.len()).ok_or(ModelError::Truncated)?;
        let s = &self.bytes[self.at..end];
        self.at = end;
        Ok(s)
    }

    fn u8(&mut self) -> Result<u8, ModelError> {
        Ok(self.take(1)?[0])
    }

    fn u32(&mut self) -> Result<u32, ModelError> {
        let (b, _) = self.take(4)?.as_chunks::<4>();
        Ok(u32::from_be_bytes(b[0]))
    }

    fn i32(&mut self) -> Result<i32, ModelError> {
        Ok(self.u32()? as i32)
    }

    fn i64(&mut self) -> Result<i64, ModelError> {
        let (b, _) = self.take(8)?.as_chunks::<8>();
        Ok(i64::from_be_bytes(b[0]))
    }

    fn config(&mut self) -> Result<SymSpellConfig, ModelError> {
        let max_edit = self.i32()?;
        let prefix_length = self.i32()?;
        let count_threshold = self.i64()?;
        let metric_id = self.utf()?;
        let corpus_word_count = self.i64()?;
        let corrupt = |m: String| ModelError::CorruptConfig(m);
        if max_edit < 0 {
            return Err(corrupt(format!("maxDictionaryEditDistance must be >= 0: {max_edit}")));
        }
        if prefix_length < 1 {
            return Err(corrupt(format!("prefixLength must be >= 1: {prefix_length}")));
        }
        if count_threshold < 1 {
            return Err(corrupt(format!("countThreshold must be >= 1: {count_threshold}")));
        }
        let metric =
            Metric::from_id(&metric_id).ok_or_else(|| corrupt(format!("unknown edit-distance id: {metric_id}")))?;
        if corpus_word_count < 0 {
            return Err(corrupt(format!("corpusWordCount must be >= 0: {corpus_word_count}")));
        }
        Ok(SymSpellConfig::builder()
            .max_dictionary_edit_distance(max_edit as usize)
            .prefix_length(prefix_length as usize)
            .count_threshold(count_threshold as u64)
            .metric(metric)
            .corpus_word_count(corpus_word_count as u64)
            .build()?)
    }

    fn rows(&mut self, field: &'static str, max_entries: usize) -> Result<usize, ModelError> {
        let value = self.i32()?;
        match usize::try_from(value) {
            Ok(n) if n <= max_entries => Ok(n),
            _ => Err(ModelError::EntryCount { field, value: i64::from(value) }),
        }
    }

    fn count(&mut self) -> Result<u64, ModelError> {
        let mut value: u64 = 0;
        let mut shift = 0;
        loop {
            let b = self.u8()?;
            value |= u64::from(b & 0x7F) << shift;
            if b & 0x80 == 0 {
                break;
            }
            shift += 7;
            if shift >= 64 {
                return Err(ModelError::VLongTooLong);
            }
        }
        if value > MAX_COUNT {
            return Err(ModelError::NegativeCount);
        }
        Ok(value)
    }

    /// `DataInputStream.readUTF`, then the UTF-16 code units as a string.
    fn utf(&mut self) -> Result<String, ModelError> {
        let len = usize::from(u16::from_be_bytes([self.u8()?, self.u8()?]));
        let bytes = self.take(len)?;
        let mut units: Vec<u16> = Vec::with_capacity(len);
        let mut i = 0;
        while i < bytes.len() {
            let c = bytes[i];
            let (unit, width) = match c >> 4 {
                0..=7 => (u16::from(c), 1),
                12 | 13 => {
                    let c2 = *bytes.get(i + 1).ok_or(ModelError::MalformedUtf)?;
                    if c2 & 0xC0 != 0x80 {
                        return Err(ModelError::MalformedUtf);
                    }
                    ((u16::from(c & 0x1F) << 6) | u16::from(c2 & 0x3F), 2)
                }
                14 => {
                    let (c2, c3) = match bytes.get(i + 1..i + 3) {
                        Some(&[c2, c3]) => (c2, c3),
                        _ => return Err(ModelError::MalformedUtf),
                    };
                    if c2 & 0xC0 != 0x80 || c3 & 0xC0 != 0x80 {
                        return Err(ModelError::MalformedUtf);
                    }
                    ((u16::from(c & 0x0F) << 12) | (u16::from(c2 & 0x3F) << 6) | u16::from(c3 & 0x3F), 3)
                }
                _ => return Err(ModelError::MalformedUtf),
            };
            units.push(unit);
            i += width;
        }
        String::from_utf16(&units).map_err(|_| ModelError::MalformedUtf)
    }
}

/// `DataOutputStream.writeUTF`.
fn write_utf(out: &mut Vec<u8>, s: &str) -> Result<(), ModelError> {
    let len: usize = s
        .encode_utf16()
        .map(|u| match u {
            0x0001..=0x007F => 1,
            0x0000 | 0x0080..=0x07FF => 2,
            _ => 3,
        })
        .sum();
    let len = u16::try_from(len).map_err(|_| ModelError::StringTooLong)?;
    out.extend_from_slice(&len.to_be_bytes());
    for u in s.encode_utf16() {
        match u {
            0x0001..=0x007F => out.push(u as u8),
            0x0000 | 0x0080..=0x07FF => {
                out.push(0xC0 | (u >> 6) as u8);
                out.push(0x80 | (u & 0x3F) as u8);
            }
            _ => {
                out.push(0xE0 | (u >> 12) as u8);
                out.push(0x80 | ((u >> 6) & 0x3F) as u8);
                out.push(0x80 | (u & 0x3F) as u8);
            }
        }
    }
    Ok(())
}

fn write_vlong(out: &mut Vec<u8>, mut v: u64) {
    while v & !0x7F != 0 {
        out.push((v & 0x7F) as u8 | 0x80);
        v >>= 7;
    }
    out.push(v as u8);
}
