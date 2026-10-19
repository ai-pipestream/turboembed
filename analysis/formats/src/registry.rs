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
//! them `namefinder` and `pos`, after the samples they read). The CoNLL-U
//! formats ([`ConlluPos`], [`ConlluLemmas`], [`ConlluSentences`] and
//! [`ConlluTokens`]) are named [`CONLLU_FORMAT`], and the CoNLL formats
//! [`Conll02Names`], [`Conll03Names`], [`ConllXPos`], [`ConllXTokens`] and
//! [`ConllXSentences`] `conll02`, `conll03` and `conllx`; readers of other
//! corpora register the same way.
//!
//! OpenNLP has no format of its own for dependency samples: it registers
//! the CoNLL-U dependency format both as the default
//! ([`NativeDependencies`], [`OPENNLP_FORMAT`]) and as [`CONLLU_FORMAT`]
//! ([`ConlluDependencies`]).
//!
//! A duplicate name is refused and the first format kept, as
//! `StreamFactoryRegistry.registerFactory` keeps it.

mod formats;

use std::fmt;

use turbo_unicode::WhitespaceMode;

pub use self::formats::{
    Conll02Names, Conll03Names, ConllXPos, ConllXSentences, ConllXTokens, ConlluDependencies, ConlluLemmas, ConlluPos,
    ConlluSentences, ConlluTokens, ConvertFrom, ConvertedSentences, ConvertedTokens, NativeDependencies,
};
use crate::error::FormatError;
use crate::native::{
    ChunkReader, DocumentReader, LanguageReader, LemmaReader, NameReader, SentenceReader, TokenReader, WordTagReader,
};
use crate::sample::{
    ChunkSample, DependencySample, DocumentSample, LanguageSample, LemmaSample, NameSample, PosSample, SentenceSample,
    TokenSample,
};

/// The name of OpenNLP's own format of every kind
/// (`StreamFactoryRegistry.DEFAULT_FORMAT`).
pub const OPENNLP_FORMAT: &str = "opennlp";

/// The name of the CoNLL-U formats (`ConlluPOSSampleStreamFactory.CONLLU_FORMAT`).
pub const CONLLU_FORMAT: &str = "conllu";

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
    /// [`DependencySample`]s.
    DependencySamples => DependencySample, "opennlp.tools.depparse.DependencySample";
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

impl std::error::Error for OpenError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            OpenError::Unknown { .. } => None,
            OpenError::Format(e) => e.source(),
        }
    }
}

impl From<FormatError> for OpenError {
    fn from(e: FormatError) -> Self {
        OpenError::Format(e)
    }
}

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

impl<K: NativeFormat> Default for Registry<K> {
    fn default() -> Self {
        Self::with_native()
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
