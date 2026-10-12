//! The part-of-speech tagger's context generators, tag dictionary and
//! sequence validator (`opennlp.tools.postag`).
//!
//! `POSTaggerFactory` picks the context generator by the version of
//! OpenNLP that trained the model: models from before 1.8 use the fixed
//! [`DefaultPosContext`], later ones the generators of their descriptor
//! (the model's `generator.featuregen`, or OpenNLP's default descriptor
//! when the model has none) behind a [`ConfigurablePosContext`].
//! [`PosContext::for_model`] makes the same choice.

use std::collections::{HashMap, VecDeque};
use std::sync::Arc;

use turbo_jvm::chars::utf16_len;
use turbo_maxent::{Dictionary, Entries, ModelFile};

use crate::beam::{ContextGenerator, SequenceValidator};
use crate::error::{Error, Result};
use crate::featuregen::{BoxedGenerator, descriptor};
use crate::features::Features;
use crate::java::{lowercase_into, parse_int, split_trimmed};
use crate::pattern::PatternMode;

/// The model entry of the classifier.
pub const MODEL_ENTRY_NAME: &str = "pos.model";
/// The model entry of the tag dictionary.
pub const TAG_DICTIONARY_ENTRY_NAME: &str = "tags.tagdict";
/// The model entry of the word dictionary of models before 1.8.
pub const NGRAM_DICTIONARY_ENTRY_NAME: &str = "ngram.dictionary";
/// `POSTaggerME.DEFAULT_BEAM_SIZE`.
pub const DEFAULT_BEAM_SIZE: usize = 3;

/// OpenNLP's default part-of-speech descriptor
/// (`opennlp/tools/postag/pos-default-features.xml`), for models from 1.8
/// on that carry none.
pub const DEFAULT_DESCRIPTOR: &str = include_str!("descriptors/pos-default-features.xml");

/// `DefaultPOSContextGenerator`: the word, its first and last four UTF-16
/// code units (unless the word is in the dictionary), whether it holds a
/// hyphen, an ASCII capital or an ASCII digit, the two words and tags
/// before it and the two words after it.
pub struct DefaultPosContext {
    dictionary: Option<Arc<Dictionary>>,
    scratch: String,
}

impl DefaultPosContext {
    /// Sentence end and beginning, written in place of missing words.
    const SE: &'static str = "*SE*";
    const SB: &'static str = "*SB*";

    pub fn new(dictionary: Option<Arc<Dictionary>>) -> DefaultPosContext {
        DefaultPosContext { dictionary, scratch: String::new() }
    }

    /// The features of `tokens[index]` after `tags`, the tags of the
    /// tokens before it.
    pub fn features(&mut self, out: &mut Features, index: usize, tokens: &[&str], tags: &[&str]) {
        let lex = tokens[index];
        out.add("default");
        out.push_str("w=").push_str(lex).finish();
        let known = match &self.dictionary {
            Some(d) => d.contains_with(lex, &mut self.scratch),
            None => false,
        };
        if !known {
            let units = utf16_len(lex);
            for li in 0..4 {
                out.push_str("suf=").push_units(lex, units.saturating_sub(li + 1), units).finish();
            }
            for li in 0..4 {
                out.push_str("pre=").push_units(lex, 0, (li + 1).min(units)).finish();
            }
            if lex.contains('-') {
                out.add("h");
            }
            if lex.bytes().any(|b| b.is_ascii_uppercase()) {
                out.add("c");
            }
            if lex.bytes().any(|b| b.is_ascii_digit()) {
                out.add("d");
            }
        }
        if index == 0 {
            out.push_str("p=").push_str(Self::SB).finish();
        } else {
            out.push_str("p=").push_str(tokens[index - 1]).finish();
            out.push_str("t=").push_str(tags[index - 1]).finish();
            if index == 1 {
                out.push_str("pp=").push_str(Self::SB).finish();
            } else {
                out.push_str("pp=").push_str(tokens[index - 2]).finish();
                out.push_str("t2=").push_str(tags[index - 2]).push_str(",").push_str(tags[index - 1]).finish();
            }
        }
        if index + 1 < tokens.len() {
            out.push_str("n=").push_str(tokens[index + 1]).finish();
            out.push_str("nn=").push_str(tokens.get(index + 2).copied().unwrap_or(Self::SE)).finish();
        } else {
            out.push_str("n=").push_str(Self::SE).finish();
        }
    }
}

impl<'a> ContextGenerator<&'a str> for DefaultPosContext {
    fn context(&mut self, index: usize, sequence: &[&'a str], prior: &[&str], out: &mut Features) {
        self.features(out, index, sequence, prior);
    }
}

/// `ConfigurablePOSContextGenerator`: the features of a generator built
/// from a descriptor.
///
/// With a cache size above zero it keeps the contexts of the current
/// sentence by the key `index + tagprev + tagprevprev` (the index and the
/// two previous tags written one after the other, `null` for a missing
/// tag), dropping the oldest beyond the size, as Java does; `POSTaggerME`
/// gives it the beam size. Two keys can only collide if a tag set holds
/// tags that run into each other, and the cache answers what Java's does
/// even then.
pub struct ConfigurablePosContext {
    generator: BoxedGenerator,
    cache_size: usize,
    cache: VecDeque<(String, Features)>,
    key: String,
}

impl ConfigurablePosContext {
    pub fn new(generator: BoxedGenerator, cache_size: usize) -> ConfigurablePosContext {
        ConfigurablePosContext { generator, cache_size, cache: VecDeque::new(), key: String::new() }
    }

    /// The generator, for `update_adaptive_data` and the like.
    pub fn generator_mut(&mut self) -> &mut BoxedGenerator {
        &mut self.generator
    }
}

impl<'a> ContextGenerator<&'a str> for ConfigurablePosContext {
    fn start_sequence(&mut self) {
        self.cache.clear();
        self.generator.start_sequence();
    }

    fn context(&mut self, index: usize, sequence: &[&'a str], prior: &[&str], out: &mut Features) {
        if self.cache_size == 0 {
            self.generator.create_features(out, sequence, index, prior);
            return;
        }
        let tag_prev = index.checked_sub(1).map(|i| prior[i]);
        let tag_prev_prev = index.checked_sub(2).map(|i| prior[i]);
        self.key.clear();
        self.key.push_str(&index.to_string());
        self.key.push_str(tag_prev.unwrap_or("null"));
        self.key.push_str(tag_prev_prev.unwrap_or("null"));
        if let Some((_, cached)) = self.cache.iter().find(|(k, _)| *k == self.key) {
            out.extend_from(cached);
            return;
        }
        let mut features = Features::new();
        self.generator.create_features(&mut features, sequence, index, prior);
        out.extend_from(&features);
        self.cache.push_back((self.key.clone(), features));
        if self.cache.len() > self.cache_size {
            self.cache.pop_front();
        }
    }
}

/// The context generator `POSTaggerFactory` picks for a model.
pub enum PosContext {
    Default(DefaultPosContext),
    Configurable(ConfigurablePosContext),
}

impl PosContext {
    /// `POSTaggerFactory.getPOSContextGenerator(cacheSize)`: the fixed
    /// generator for a model trained by OpenNLP before 1.8, with the
    /// model's `ngram.dictionary` if it has one; otherwise the model's
    /// descriptor, or the default descriptor.
    pub fn for_model(model: &ModelFile, cache_size: usize, mode: PatternMode) -> Result<PosContext> {
        if is_before_1_8(model.opennlp_version())? {
            let dictionary = model.dictionary(NGRAM_DICTIONARY_ENTRY_NAME)?.map(Arc::new);
            return Ok(PosContext::Default(DefaultPosContext::new(dictionary)));
        }
        let generator = match descriptor::from_model(model, mode)? {
            Some(g) => g,
            None => descriptor::build(DEFAULT_DESCRIPTOR.as_bytes(), Some(model), mode)?,
        };
        Ok(PosContext::Configurable(ConfigurablePosContext::new(generator, cache_size)))
    }
}

impl<'a> ContextGenerator<&'a str> for PosContext {
    fn start_sequence(&mut self) {
        match self {
            PosContext::Default(_) => {}
            PosContext::Configurable(c) => ContextGenerator::<&str>::start_sequence(c),
        }
    }

    fn context(&mut self, index: usize, sequence: &[&'a str], prior: &[&str], out: &mut Features) {
        match self {
            PosContext::Default(c) => c.context(index, sequence, prior, out),
            PosContext::Configurable(c) => c.context(index, sequence, prior, out),
        }
    }
}

/// Whether an `OpenNLP-Version` names a version before 1.8, as
/// `Version.parse` reads it: the numbers before the first and second dots.
fn is_before_1_8(version: &str) -> Result<bool> {
    let bad = || Error::Manifest { property: "OpenNLP-Version".into(), value: version.to_owned() };
    let mut parts = version.splitn(3, '.');
    let major = parts.next().and_then(parse_int).ok_or_else(bad)?;
    let minor = parts.next().and_then(parse_int).ok_or_else(bad)?;
    if parts.next().is_none() {
        return Err(bad());
    }
    Ok(major == 1 && minor < 8)
}

/// `POSDictionary`: the tags each word may have.
#[derive(Debug, Clone, Default)]
pub struct TagDictionary {
    case_sensitive: bool,
    tags: HashMap<String, Vec<String>>,
}

impl TagDictionary {
    /// Reads the dictionary XML (`POSDictionary.create`): one token per
    /// entry, its tags in the `tags` attribute separated by spaces. In a
    /// case-insensitive dictionary the words are lowercased
    /// (`StringUtil.toLowerCase`). A word listed twice keeps its last
    /// tags.
    ///
    /// Java lowercases a case-insensitive dictionary by walking a
    /// `HashMap`, so when two entries lowercase to the same word, which
    /// one wins depends on Java's hash order; here the later entry in the
    /// file wins.
    pub fn read(xml: &[u8]) -> Result<TagDictionary> {
        let entries = Entries::read(xml)?;
        let case_sensitive = entries.is_case_sensitive();
        let mut tags = HashMap::new();
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
            let mut word = String::new();
            if case_sensitive {
                word.push_str(&entry.tokens[0]);
            } else {
                lowercase_into(&mut word, &entry.tokens[0]);
            }
            tags.insert(word, list);
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

    /// `getTags`: the tags a word may have, or `None` when the dictionary
    /// does not list it.
    pub fn tags(&self, word: &str) -> Option<&[String]> {
        if self.case_sensitive {
            return self.tags.get(word).map(Vec::as_slice);
        }
        let mut lower = String::with_capacity(word.len());
        lowercase_into(&mut lower, word);
        self.tags.get(&lower).map(Vec::as_slice)
    }
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
    fn versions_before_1_8() {
        assert!(is_before_1_8("1.5.0").unwrap());
        assert!(is_before_1_8("1.7.2-SNAPSHOT").unwrap());
        assert!(!is_before_1_8("1.8.0").unwrap());
        assert!(!is_before_1_8("2.5.4").unwrap());
        assert!(is_before_1_8("1.5").is_err());
    }

    #[test]
    fn the_default_context_of_a_short_sentence() {
        let mut g = DefaultPosContext::new(None);
        let mut out = Features::new();
        g.features(&mut out, 1, &["The", "co-op", "won"], &["DT"]);
        let got: Vec<String> = (0..out.len()).map(|i| out.to_string_lossy(i)).collect();
        assert_eq!(
            got,
            [
                "default", "w=co-op", "suf=p", "suf=op", "suf=-op", "suf=o-op", "pre=c", "pre=co", "pre=co-",
                "pre=co-o", "h", "p=The", "t=DT", "pp=*SB*", "n=won", "nn=*SE*",
            ]
        );
    }

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
}
