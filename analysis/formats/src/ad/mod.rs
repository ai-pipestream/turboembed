//! Floresta Sinta(c)tica in the Arvores Deitadas format: OpenNLP's
//! `opennlp.tools.formats.ad` readers of the Portuguese treebank.
//!
//! An Arvores Deitadas ("lying trees") file holds sentences in `<s>`
//! elements, grouped into texts (`<ext>`), paragraphs (`<p>`), titles
//! (`<t>`) and boxes (`<caixa>`). A sentence has a `SOURCE` line, a line
//! with its id and text, and its tree, one element per line, the depth
//! given by leading `=`:
//!
//! ```text
//! <s id="7">
//! SOURCE: ref="CP3-1" source="exemplo"
//! CP3-1 O gato da vizinha dorme.
//! A1
//! STA:fcl
//! =SUBJ:np
//! ==>N:art("o" <artd> DET M S)  O
//! ==H:n("gato" <np-def> M S)  gato
//! ==N<:pp
//! ===H:prp("de" <sam->)  de
//! ===P<:np
//! ====>N:art("o" <-sam> DET F S)  a
//! ====H:n("vizinha" <NER:hum> F S)  vizinha
//! =P:v-fin("dormir" PR 3S IND VFIN)  dorme
//! .
//! </s>
//! ```
//!
//! A node line is a syntactic function and a phrase type (`SUBJ:np`); a
//! leaf line a function, a word class, and in parentheses the lemma,
//! secondary tags in angle brackets and the inflection, then the word.
//! The tree is read into a typed structure of nodes and leaves that the
//! readers walk; it is not public.
//!
//! | Samples | Reader | OpenNLP | Registry format |
//! |---|---|---|---|
//! | names | [`AdNameReader`] | `ADNameSampleStream` | `ad` ([`AdNames`]) |
//! | part-of-speech | [`AdPosReader`] | `ADPOSSampleStream` | `ad` ([`AdPos`]) |
//! | chunks | [`AdChunkReader`] | `ADChunkSampleStream` | `ad` ([`AdChunks`]) |
//! | sentences | [`AdSentenceReader`] | `ADSentenceSampleStream` | `ad` ([`AdSentences`]) |
//! | tokens | [`ToTokenSamples`](crate::convert::ToTokenSamples) over [`AdNameReader`] | `ADTokenSampleStreamFactory` | `ad` ([`AdTokens`]) |
//!
//! The readers take decoded text ([`crate::decode`]); the factories read
//! the corpus in the encoding they are given (older Floresta releases are
//! Latin-1). Tokens are fields of the text with their byte spans, except
//! a joined contraction, whose text is its own. The readers parse one
//! sentence at a time.
//!
//! Floresta's data has its own terms, and none of it is in this
//! repository: the tests read files written for them in the format.
//! The parse samples of the parser are not ported here.
//!
//! # What OpenNLP does here
//!
//! These are reproduced, and the tests compare the readers with the `ad`
//! goldens the JVM wrote:
//!
//! - A sentence the parser gives up on (no `SOURCE` line, a `SOURCE` line
//!   of six characters, or nothing after it) is `null` from the sentence
//!   stream, which the name, part-of-speech and chunk streams return: the
//!   reading ends there. The sentence sample stream ends its sample there
//!   and reads on at the next call.
//! - The text and the metadata are kept from the sentence before when a
//!   sentence repeats one with `&&`, and the metadata when the text line
//!   has no space after its first character. Before any sentence set
//!   them they are missing, and the name and sentence sample streams
//!   throw `NullPointerException` ([`AdError::NullMetadata`],
//!   [`AdError::NullText`]).
//! - The left part of a contraction carries over from the end of one
//!   sentence into the next. A contraction whose right part is only
//!   underscores throws `ArrayIndexOutOfBoundsException`
//!   ([`AdError::EmptyContraction`]) and keeps the left part for the next
//!   leaf. A pair not in the table is two tokens. A left part is
//!   lowercased only when the right part has no underscore.
//! - A word cut down to nothing after a `«` throws
//!   `StringIndexOutOfBoundsException` ([`AdError::EmptyToken`]), and one
//!   cut down to nothing between a `«` and a `»` is an empty token.
//! - A named entity tag on the right part of a contraction makes an empty
//!   name, which `NameSample` refuses as overlapping when a name starts
//!   at the next token.
//! - `currency` is a `numeric` name, the second of the two types the
//!   table gives it.
//! - In the literary and scientific corpora (metadata starting with `LIT`
//!   or `CIE`) the text of a sentence, which clears adaptive data when it
//!   changes, is named by the letters before its id or by the metadata's
//!   `source` attribute; elsewhere it is the text id's number.
//!
//! Where this port differs, deliberately:
//!
//! - A `SOURCE` line whose seventh UTF-16 unit is the second half of a
//!   surrogate pair leaves U+FFFD in the metadata where Java keeps half
//!   the pair.
//! - After [`AdError::NullText`] the sentence sample reader ends; Java's
//!   stream throws the same exception on every later read.
//! - Java's parser logs the lines it cannot read as a leaf ("Couldn't
//!   parse leaf") and the sentences it gives up on; nothing is logged
//!   here.

mod chunks;
mod contraction;
mod metadata;
mod names;
mod pos;
mod sentence;
mod sentences;
mod tree;

use std::fmt;
use std::sync::Arc;

use turbo_tokenize::detok::{Detokenizer, DetokenizerDictionary};

pub use chunks::AdChunkReader;
pub use names::AdNameReader;
pub use pos::AdPosReader;
pub use sentences::AdSentenceReader;

use crate::convert::ToTokenSamples;
use crate::error::FormatError;
use crate::registry::{
    ChunkSamples, Format, NameSamples, PosSamples, ReadOptions, Samples, SentenceSamples, TokenSamples,
};

/// The registry name of the Arvores Deitadas formats, for every sample
/// kind.
pub const AD_FORMAT: &str = "ad";

/// Which call a missing metadata string fails in.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub enum NullAt {
    /// The name stream's `meta.startsWith`.
    StartsWith,
    /// The sentence sample stream's `meta.length()`, in `ADMetadata`.
    Length,
}

/// What the Arvores Deitadas readers refuse. `Display` is the message of
/// the exception OpenNLP throws, and [`AdError::java_class`] its class.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum AdError {
    /// Metadata without a text id and a paragraph id, or with one that
    /// does not fit a Java `int`; in the literary corpora, without the
    /// letters before the id, and in the scientific ones without a
    /// `source` attribute.
    InvalidMetadata { metadata: String },
    /// A sentence before any sentence had metadata.
    NullMetadata { at: NullAt },
    /// A sentence before any sentence had a text.
    NullText,
    /// A contraction whose right part is only underscores.
    EmptyContraction,
    /// A word that is nothing after a `«` is cut off.
    EmptyToken,
}

impl AdError {
    /// The class of the exception OpenNLP throws in the same place.
    pub fn java_class(&self) -> &'static str {
        match self {
            AdError::InvalidMetadata { .. } => "java.lang.RuntimeException",
            AdError::NullMetadata { .. } | AdError::NullText => "java.lang.NullPointerException",
            AdError::EmptyContraction => "java.lang.ArrayIndexOutOfBoundsException",
            AdError::EmptyToken => "java.lang.StringIndexOutOfBoundsException",
        }
    }
}

impl fmt::Display for AdError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            AdError::InvalidMetadata { metadata } => write!(f, "Invalid metadata: {metadata}"),
            AdError::NullMetadata { at: NullAt::StartsWith } => {
                f.write_str("Cannot invoke \"String.startsWith(String)\" because \"meta\" is null")
            }
            AdError::NullMetadata { at: NullAt::Length } => {
                f.write_str("Cannot invoke \"String.length()\" because \"meta\" is null")
            }
            AdError::NullText => f.write_str("Cannot invoke \"String.trim()\" because \"text\" is null"),
            AdError::EmptyContraction => f.write_str("Index 0 out of bounds for length 0"),
            AdError::EmptyToken => f.write_str("Index -1 out of bounds for length 0"),
        }
    }
}

impl std::error::Error for AdError {}

impl From<AdError> for FormatError {
    fn from(e: AdError) -> FormatError {
        FormatError::Ad(e)
    }
}

/// Name samples from Arvores Deitadas (`ADNameSampleStreamFactory`),
/// named [`AD_FORMAT`]. Its `split` parameter is true by default.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub struct AdNames {
    pub split_hyphenated_tokens: bool,
}

impl Default for AdNames {
    fn default() -> AdNames {
        AdNames { split_hyphenated_tokens: true }
    }
}

impl Format<NameSamples> for AdNames {
    fn name(&self) -> &str {
        AD_FORMAT
    }

    fn open<'a>(&'a self, source: &'a str, _options: ReadOptions) -> Result<Samples<'a, NameSamples>, FormatError> {
        Ok(Box::new(AdNameReader::new(source, self.split_hyphenated_tokens)?))
    }
}

/// Part-of-speech samples from Arvores Deitadas (`ADPOSSampleStreamFactory`,
/// whose `expandME` and `includeFeatures` are false by default), named
/// [`AD_FORMAT`].
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq, Hash)]
pub struct AdPos {
    pub expand_me: bool,
    pub include_features: bool,
}

impl Format<PosSamples> for AdPos {
    fn name(&self) -> &str {
        AD_FORMAT
    }

    fn open<'a>(&'a self, source: &'a str, _options: ReadOptions) -> Result<Samples<'a, PosSamples>, FormatError> {
        Ok(Box::new(AdPosReader::new(source, self.expand_me, self.include_features)?))
    }
}

/// Chunk samples from Arvores Deitadas (`ADChunkSampleStreamFactory`, whose
/// `start` and `end` are optional), named [`AD_FORMAT`].
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq, Hash)]
pub struct AdChunks {
    pub start: Option<u32>,
    pub end: Option<u32>,
}

impl Format<ChunkSamples> for AdChunks {
    fn name(&self) -> &str {
        AD_FORMAT
    }

    fn open<'a>(&'a self, source: &'a str, _options: ReadOptions) -> Result<Samples<'a, ChunkSamples>, FormatError> {
        Ok(Box::new(AdChunkReader::new(source, self.start, self.end)?))
    }
}

/// Sentence samples from Arvores Deitadas (`ADSentenceSampleStreamFactory`,
/// whose `includeTitles` is false by default), named [`AD_FORMAT`].
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq, Hash)]
pub struct AdSentences {
    pub include_titles: bool,
}

impl Format<SentenceSamples> for AdSentences {
    fn name(&self) -> &str {
        AD_FORMAT
    }

    fn open<'a>(&'a self, source: &'a str, _options: ReadOptions) -> Result<Samples<'a, SentenceSamples>, FormatError> {
        Ok(Box::new(AdSentenceReader::new(source, self.include_titles)?))
    }
}

/// Token samples from Arvores Deitadas name samples, detokenized with a
/// dictionary (`ADTokenSampleStreamFactory`, which reads the names with
/// `split` true unless told otherwise), named [`AD_FORMAT`].
#[derive(Clone, Debug)]
pub struct AdTokens {
    pub dictionary: Arc<DetokenizerDictionary>,
    pub split_hyphenated_tokens: bool,
}

impl Format<TokenSamples> for AdTokens {
    fn name(&self) -> &str {
        AD_FORMAT
    }

    fn open<'a>(&'a self, source: &'a str, _options: ReadOptions) -> Result<Samples<'a, TokenSamples>, FormatError> {
        let names = AdNameReader::new(source, self.split_hyphenated_tokens)?;
        Ok(Box::new(ToTokenSamples::new(names, Detokenizer::new(&self.dictionary))))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::registry::Registry;

    const TEXT: &str = "<ext id=\"1\">\n<p>\n<s id=\"1\">\nSOURCE: ref=\"CP3-1\"\nCP3-1 O gato da vizinha dorme.\nA1\nSTA:fcl\n=SUBJ:np\n==>N:art(\"o\" <artd> DET M S)  O\n==H:n(\"gato\" <np-def> M S)  gato\n==N<:pp\n===H:prp(\"de\" <sam->)  de\n===P<:np\n====>N:art(\"o\" <-sam> DET F S)  a\n====H:n(\"vizinha\" <NER:hum> F S)  vizinha\n=P:v-fin(\"dormir\" PR 3S IND VFIN)  dorme\n.\n</s>\n</ext>\n";

    fn first<K: crate::registry::SampleKind>(r: &Registry<K>) -> String
    where
        for<'a> K::Sample<'a>: fmt::Display,
    {
        r.open(AD_FORMAT, TEXT, ReadOptions::default()).unwrap().next().unwrap().unwrap().to_string()
    }

    #[test]
    fn formats_register_by_name() {
        let mut names = Registry::<NameSamples>::with_native();
        names.register(AdNames::default()).unwrap();
        assert_eq!(first(&names), "\nO gato da <START:person> vizinha <END> dorme .");
        let mut pos = Registry::<PosSamples>::with_native();
        pos.register(AdPos::default()).unwrap();
        assert_eq!(first(&pos), "O_art gato_n de_prp a_art vizinha_n dorme_v-fin ._.");
        let mut chunks = Registry::<ChunkSamples>::with_native();
        chunks.register(AdChunks::default()).unwrap();
        assert!(first(&chunks).starts_with("O art B-NP\ngato n I-NP\nde prp B-PP\n"));
        let mut sentences = Registry::<SentenceSamples>::with_native();
        sentences.register(AdSentences::default()).unwrap();
        assert_eq!(first(&sentences), "O gato da vizinha dorme.\n");
        let mut tokens = Registry::<TokenSamples>::with_native();
        let dictionary = Arc::new(DetokenizerDictionary::new());
        tokens.register(AdTokens { dictionary, split_hyphenated_tokens: true }).unwrap();
        assert_eq!(first(&tokens), "O gato da vizinha dorme .");
        assert_eq!(names.names().collect::<Vec<_>>(), ["opennlp", "ad"]);
    }
}
