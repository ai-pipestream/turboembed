//! The format registry: the formats a kind of sample can be read from,
//! by name, as OpenNLP's `StreamFactoryRegistry` keeps them.
//!
//! A sample kind ([`SampleKind`]) is a type that names one sample type,
//! such as [`PosSamples`] for [`PosSample`]. A [`Format`] of that kind
//! opens a source text as a stream of its samples, and a [`Registry`]
//! holds the formats of one kind under their names. OpenNLP's registry is
//! one map from sample class to format name to factory, with untyped
//! parameters parsed from the command line; here each kind has its own
//! registry, a format's parameters are fields of the format value, and
//! the only parameter every format shares is in [`ReadOptions`].
//!
//! [`Registry::with_native`] holds OpenNLP's own format of the kind under
//! [`OPENNLP_FORMAT`]. The conversion formats need a detokenizer
//! dictionary, so they are registered by whoever has one (OpenNLP names
//! them `namefinder` and `pos`, after the samples they read). Readers of
//! other corpora (CoNLL, CoNLL-U, OntoNotes) register the same way.
//!
//! A duplicate name is refused and the first format kept, as
//! `StreamFactoryRegistry.registerFactory` keeps it.

use std::fmt;
use std::sync::Arc;

use turbo_tokenize::detok::{Detokenizer, DetokenizerDictionary};
use turbo_unicode::WhitespaceMode;

use crate::convert::{ChunkSize, ToSentenceSamples, ToTokenSamples};
use crate::error::FormatError;
use crate::native::{
    ChunkReader, DocumentReader, LanguageReader, LemmaReader, NameReader, SentenceReader, TokenReader, WordTagReader,
};
use crate::sample::{
    ChunkSample, DocumentSample, LanguageSample, LemmaSample, NameSample, PosSample, SentenceSample, TokenSample,
};

/// The name of OpenNLP's own format of every kind
/// (`StreamFactoryRegistry.DEFAULT_FORMAT`).
pub const OPENNLP_FORMAT: &str = "opennlp";

/// A kind of sample: the type that names one sample type for the
/// registry.
pub trait SampleKind: 'static {
    /// The sample type, borrowing the source text for `'a`.
    type Sample<'a>;
    /// The OpenNLP class of the samples.
    const JAVA_CLASS: &'static str;
}

macro_rules! kinds {
    ($($(#[$doc:meta])* $kind:ident => $sample:ident, $class:literal;)*) => {$(
        $(#[$doc])*
        #[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
        pub struct $kind;

        impl SampleKind for $kind {
            type Sample<'a> = $sample<'a>;
            const JAVA_CLASS: &'static str = $class;
        }
    )*};
}

kinds! {
    /// [`NameSample`]s.
    NameSamples => NameSample, "opennlp.tools.namefind.NameSample";
    /// [`PosSample`]s.
    PosSamples => PosSample, "opennlp.tools.postag.POSSample";
    /// [`LemmaSample`]s.
    LemmaSamples => LemmaSample, "opennlp.tools.lemmatizer.LemmaSample";
    /// [`ChunkSample`]s.
    ChunkSamples => ChunkSample, "opennlp.tools.chunker.ChunkSample";
    /// [`TokenSample`]s.
    TokenSamples => TokenSample, "opennlp.tools.tokenize.TokenSample";
    /// [`SentenceSample`]s.
    SentenceSamples => SentenceSample, "opennlp.tools.sentdetect.SentenceSample";
    /// [`DocumentSample`]s.
    DocumentSamples => DocumentSample, "opennlp.tools.doccat.DocumentSample";
    /// [`LanguageSample`]s.
    LanguageSamples => LanguageSample, "opennlp.tools.langdetect.LanguageSample";
}

/// A stream of samples of kind `K` read from a source text.
pub type Samples<'a, K> = Box<dyn Iterator<Item = Result<<K as SampleKind>::Sample<'a>, FormatError>> + 'a>;

/// The options every format is opened with.
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq, Hash)]
pub struct ReadOptions {
    /// What splits whitespace-separated lines, for the formats that split
    /// them (`WhitespaceTokenizer`, which follows OpenNLP's process-wide
    /// `WhitespaceMode`).
    pub whitespace: WhitespaceMode,
}

/// A format samples of kind `K` are read from.
pub trait Format<K: SampleKind>: Send + Sync {
    /// The name the format is registered under.
    fn name(&self) -> &str;

    /// The samples of `source`.
    fn open<'a>(&'a self, source: &'a str, options: ReadOptions) -> Result<Samples<'a, K>, FormatError>;
}

/// A format name registered twice.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DuplicateFormat {
    pub kind: &'static str,
    pub name: String,
}

impl fmt::Display for DuplicateFormat {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "a format named {} is already registered for {}", self.name, self.kind)
    }
}

impl std::error::Error for DuplicateFormat {}

/// Why [`Registry::open`] gave no samples.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum OpenError {
    /// No format of that name.
    Unknown { kind: &'static str, name: String },
    /// The format refused the source.
    Format(FormatError),
}

impl fmt::Display for OpenError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            OpenError::Unknown { kind, name } => write!(f, "no format named {name} for {kind}"),
            OpenError::Format(e) => e.fmt(f),
        }
    }
}

impl std::error::Error for OpenError {}

/// The formats of one sample kind, by name, in the order registered.
pub struct Registry<K: SampleKind> {
    formats: Vec<Box<dyn Format<K>>>,
}

impl<K: SampleKind> Registry<K> {
    /// A registry with no formats.
    pub fn empty() -> Registry<K> {
        Registry { formats: Vec::new() }
    }

    /// Adds `format` under its name.
    ///
    /// # Errors
    ///
    /// [`DuplicateFormat`] when the name is taken; the registry keeps the
    /// format it had.
    pub fn register(&mut self, format: impl Format<K> + 'static) -> Result<(), DuplicateFormat> {
        if self.get(format.name()).is_some() {
            return Err(DuplicateFormat { kind: K::JAVA_CLASS, name: format.name().to_owned() });
        }
        self.formats.push(Box::new(format));
        Ok(())
    }

    /// The format named `name`.
    pub fn get(&self, name: &str) -> Option<&dyn Format<K>> {
        self.formats.iter().find(|f| f.name() == name).map(|f| f.as_ref())
    }

    /// The names of the formats, in the order registered.
    pub fn names(&self) -> impl Iterator<Item = &str> {
        self.formats.iter().map(|f| f.name())
    }

    /// The samples of `source` in the format named `name`.
    pub fn open<'a>(&'a self, name: &str, source: &'a str, options: ReadOptions) -> Result<Samples<'a, K>, OpenError> {
        let format = self.get(name).ok_or_else(|| OpenError::Unknown { kind: K::JAVA_CLASS, name: name.into() })?;
        format.open(source, options).map_err(OpenError::Format)
    }
}

/// A sample kind with an OpenNLP format of its own.
pub trait NativeFormat: SampleKind + Sized {
    /// OpenNLP's format of this kind, named [`OPENNLP_FORMAT`].
    fn native() -> Box<dyn Format<Self>>;
}

impl<K: NativeFormat> Registry<K> {
    /// A registry with OpenNLP's own format of the kind.
    pub fn with_native() -> Registry<K> {
        Registry { formats: vec![K::native()] }
    }
}

/// OpenNLP's own format of a kind, read by `$reader`.
macro_rules! native {
    ($($(#[$doc:meta])* $format:ident: $kind:ident => |$source:ident, $options:ident| $open:expr;)*) => {$(
        $(#[$doc])*
        #[derive(Copy, Clone, Debug, Default, PartialEq, Eq, Hash)]
        pub struct $format;

        impl Format<$kind> for $format {
            fn name(&self) -> &str {
                OPENNLP_FORMAT
            }

            fn open<'a>(&'a self, $source: &'a str, $options: ReadOptions) -> Result<Samples<'a, $kind>, FormatError> {
                Ok(Box::new($open?))
            }
        }

        impl NativeFormat for $kind {
            fn native() -> Box<dyn Format<Self>> {
                Box::new($format)
            }
        }
    )*};
}

native! {
    /// The name format ([`NameReader`]).
    NativeNames: NameSamples => |source, options| NameReader::new(source, options.whitespace);
    /// The word and tag format ([`WordTagReader`]).
    NativePos: PosSamples => |source, options| WordTagReader::new(source, options.whitespace);
    /// The lemma format ([`LemmaReader`]).
    NativeLemmas: LemmaSamples => |source, _options| LemmaReader::new(source);
    /// The chunk format ([`ChunkReader`]).
    NativeChunks: ChunkSamples => |source, _options| ChunkReader::new(source);
    /// The token format with `<SPLIT>` ([`TokenReader`]).
    NativeTokens: TokenSamples => |source, options| TokenReader::new(source, options.whitespace);
    /// The sentence format ([`SentenceReader`]).
    NativeSentences: SentenceSamples => |source, _options| SentenceReader::new(source);
    /// The document category format ([`DocumentReader`]).
    NativeDocuments: DocumentSamples => |source, options| DocumentReader::new(source, options.whitespace);
    /// The language format ([`LanguageReader`]).
    NativeLanguages: LanguageSamples => |source, _options| LanguageReader::new(source);
}

/// Which samples a conversion format reads: the name format (OpenNLP's
/// `namefinder` conversion formats) or the word and tag format (`pos`).
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub enum ConvertFrom {
    Names,
    Pos,
}

impl ConvertFrom {
    /// The name OpenNLP registers the conversion format under.
    pub fn format_name(self) -> &'static str {
        match self {
            ConvertFrom::Names => "namefinder",
            ConvertFrom::Pos => "pos",
        }
    }
}

/// Token samples converted from name or word and tag samples with a
/// detokenizer dictionary: `NameToTokenSampleStreamFactory` and
/// `POSToTokenSampleStreamFactory`.
#[derive(Clone, Debug)]
pub struct ConvertedTokens {
    pub from: ConvertFrom,
    pub dictionary: Arc<DetokenizerDictionary>,
}

impl Format<TokenSamples> for ConvertedTokens {
    fn name(&self) -> &str {
        self.from.format_name()
    }

    fn open<'a>(&'a self, source: &'a str, options: ReadOptions) -> Result<Samples<'a, TokenSamples>, FormatError> {
        let detokenizer = Detokenizer::new(&self.dictionary);
        Ok(match self.from {
            ConvertFrom::Names => {
                Box::new(ToTokenSamples::new(NameReader::new(source, options.whitespace)?, detokenizer))
            }
            ConvertFrom::Pos => {
                Box::new(ToTokenSamples::new(WordTagReader::new(source, options.whitespace)?, detokenizer))
            }
        })
    }
}

/// Sentence samples converted from name or word and tag samples with a
/// detokenizer dictionary: `NameToSentenceSampleStreamFactory` and
/// `POSToSentenceSampleStreamFactory`, which use a chunk size of 30
/// ([`ChunkSize::COMMAND_LINE`]).
#[derive(Clone, Debug)]
pub struct ConvertedSentences {
    pub from: ConvertFrom,
    pub dictionary: Arc<DetokenizerDictionary>,
    pub chunk: ChunkSize,
}

impl Format<SentenceSamples> for ConvertedSentences {
    fn name(&self) -> &str {
        self.from.format_name()
    }

    fn open<'a>(&'a self, source: &'a str, options: ReadOptions) -> Result<Samples<'a, SentenceSamples>, FormatError> {
        let detokenizer = Detokenizer::new(&self.dictionary);
        Ok(match self.from {
            ConvertFrom::Names => {
                Box::new(ToSentenceSamples::new(NameReader::new(source, options.whitespace)?, detokenizer, self.chunk))
            }
            ConvertFrom::Pos => Box::new(ToSentenceSamples::new(
                WordTagReader::new(source, options.whitespace)?,
                detokenizer,
                self.chunk,
            )),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn native_formats_open_by_name() {
        let r = Registry::<PosSamples>::with_native();
        assert_eq!(r.names().collect::<Vec<_>>(), [OPENNLP_FORMAT]);
        let samples: Vec<_> = r.open("opennlp", "a_X b_Y\nc_Z", ReadOptions::default()).unwrap().collect();
        assert_eq!(samples.len(), 2);
        assert!(matches!(r.open("conllu", "", ReadOptions::default()), Err(OpenError::Unknown { .. })));
    }

    #[test]
    fn a_duplicate_keeps_the_first() {
        let mut r = Registry::<TokenSamples>::with_native();
        let dictionary = Arc::new(DetokenizerDictionary::new());
        r.register(ConvertedTokens { from: ConvertFrom::Names, dictionary: dictionary.clone() }).unwrap();
        r.register(ConvertedTokens { from: ConvertFrom::Pos, dictionary }).unwrap();
        let e = r.register(NativeTokens).unwrap_err();
        assert_eq!(e.name, "opennlp");
        assert_eq!(r.names().collect::<Vec<_>>(), ["opennlp", "namefinder", "pos"]);
        let samples: Vec<_> =
            r.open("namefinder", "<START:x> a <END> b", ReadOptions::default()).unwrap().map(Result::unwrap).collect();
        assert_eq!(samples[0].text(), "a b");
    }
}
