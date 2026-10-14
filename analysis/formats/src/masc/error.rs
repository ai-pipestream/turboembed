//! What the MASC readers refuse, with the exception OpenNLP throws in the
//! same place.

use std::fmt;
use std::path::PathBuf;

use crate::error::FormatError;

/// One of a document's stand-off annotation files.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub enum AnnotationFile {
    /// The quarks, `f.seg`.
    Segments,
    /// The sentence regions, `f.s`.
    Sentences,
    /// The Penn tokens and tags, `f.penn`.
    Penn,
    /// The named entities, `f.ne`.
    NamedEntities,
}

impl AnnotationFile {
    /// The `f.id` the document header names the file with.
    pub fn id(self) -> &'static str {
        match self {
            AnnotationFile::Segments => "f.seg",
            AnnotationFile::Sentences => "f.s",
            AnnotationFile::Penn => "f.penn",
            AnnotationFile::NamedEntities => "f.ne",
        }
    }

    /// The message of the `IOException` `MascDocument` throws when the
    /// file cannot be parsed, whatever was wrong in it.
    fn failure(self) -> &'static str {
        match self {
            AnnotationFile::Segments => "Could not parse the region annotation file",
            AnnotationFile::Sentences => "Could not parse the sentence annotation file",
            AnnotationFile::Penn => "Could not parse the Penn tag annotation file",
            AnnotationFile::NamedEntities => "Could not parse the named entity annotation file",
        }
    }
}

/// A sample stream of the MASC readers, for the messages that name it.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub enum MascStream {
    PartOfSpeech,
    Tokens,
    NamedEntities,
}

/// An error of the MASC readers.
///
/// [`MascError::java_class`] is the class of the exception OpenNLP throws
/// in the same place, and `Display` is its message; the messages of
/// `NullPointerException` are the JDK's helpful messages as JDK 21 writes
/// them for OpenNLP's classes. A `MascDocumentStream` constructor skips a
/// document whose parse throws an `IOException` and fails on anything
/// else ([`MascError::skips_document`]).
#[derive(Debug)]
pub enum MascError {
    /// The corpus path is not a directory.
    NotADirectory { path: PathBuf },
    /// A file could not be read. Java names no class here: OpenNLP checks
    /// the files exist before it opens them.
    Io { path: PathBuf, error: std::io::Error },
    /// A directory is inside itself through a symbolic link. OpenNLP walks
    /// such a corpus until it runs out of memory.
    DirectoryCycle { path: PathBuf },
    /// A document header is not well-formed XML.
    Header { path: PathBuf, cause: turbo_maxent::Error },
    /// A document header has no `annotation` or `primaryData` element.
    NoAnnotations,
    /// An `annotation` or `primaryData` element has no `loc`.
    NoLocation,
    /// A file the header names is not there.
    MissingFile { path: PathBuf },
    /// The header names no primary data (`f.text`).
    NoPrimaryData,
    /// The header names no segmentation or no sentence file. OpenNLP
    /// wraps the missing stream in a `BufferedInputStream` and reads it.
    NoAnnotationStream { file: AnnotationFile },
    /// An annotation file cannot be parsed: not XML, an element whose
    /// attributes the readers refuse, or a link the Penn or named entity
    /// file breaks. `cause` says which; Java's message does not.
    Annotation { file: AnnotationFile, cause: turbo_maxent::Error },
    /// The quarks end inside a sentence that is not the last.
    QuarksEndInSentence { end: i32 },
    /// `List.get` past the end of a list: of the sentence regions when
    /// the sentence file has none (`index` -1), or of the quarks when there
    /// are none or a sentence starts after the last quark is used.
    IndexOutOfBounds { index: i64, len: usize },
    /// A Penn token names a quark the segmentation does not have.
    MissingQuark { id: i32 },
    /// `String.substring` out of the text's range, in UTF-16 units.
    Substring { begin: i64, end: i64, len: usize },
    /// A span of the text starts or ends between the two halves of a
    /// surrogate pair, which a Rust string cannot hold; Java cuts the pair.
    SplitsSurrogatePair { unit: i64 },
    /// The text is too long for byte spans to index.
    TooLong { len: usize },
    /// `read` after the document stream is used up: OpenNLP's sample
    /// streams leave the document they read null and call it again.
    NullDocument,
    /// A sentence of a document without Penn tokens asked for its tokens
    /// (`method` is the `List` method called on the null token list).
    Untokenized { method: &'static str },
    /// No document has what the sample stream reads.
    NoDocuments { stream: MascStream },
    /// A token starts before the sentence's first token, so its offset in
    /// the sample is negative.
    NegativeTokenStart { start: i64 },
    /// A token reaches past the sample's text (UTF-16 units).
    TokenOutOfBounds { start: i64, end: i64, len: usize },
    /// A token has no tag.
    NullTag,
    /// A sample could not be made (overlapping names).
    Sample(FormatError),
}

impl MascError {
    /// The class of the exception OpenNLP throws in the same place, for
    /// the errors that have one.
    pub fn java_class(&self) -> Option<&'static str> {
        use MascError::*;
        match self {
            NotADirectory { .. }
            | Header { .. }
            | MissingFile { .. }
            | NoAnnotationStream { .. }
            | Annotation { .. }
            | QuarksEndInSentence { .. }
            | MissingQuark { .. }
            | NoDocuments { .. } => Some("java.io.IOException"),
            NoAnnotations | NoLocation | NoPrimaryData | NullDocument | Untokenized { .. } => {
                Some("java.lang.NullPointerException")
            }
            IndexOutOfBounds { .. } => Some("java.lang.IndexOutOfBoundsException"),
            Substring { .. } => Some("java.lang.StringIndexOutOfBoundsException"),
            NegativeTokenStart { .. } | TokenOutOfBounds { .. } | NullTag => Some("java.lang.IllegalArgumentException"),
            Sample(e) => e.java_class(),
            Io { .. } | DirectoryCycle { .. } | SplitsSurrogatePair { .. } | TooLong { .. } => None,
        }
    }

    /// Whether `MascDocumentStream` logs this error of a document and goes
    /// on with the next one (an `IOException` from `parseDocument`), rather
    /// than failing as a whole.
    pub fn skips_document(&self) -> bool {
        matches!(
            self,
            MascError::NoAnnotationStream { .. }
                | MascError::Annotation { .. }
                | MascError::QuarksEndInSentence { .. }
                | MascError::MissingQuark { .. }
        )
    }
}

impl fmt::Display for MascError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        use MascError::*;
        match self {
            NotADirectory { .. } => {
                f.write_str("Input corpus directory must be a directory according to File.isDirectory()!")
            }
            Io { path, error } => write!(f, "{}: {error}", path.display()),
            DirectoryCycle { path } => write!(f, "{} is inside itself", path.display()),
            Header { path, .. } => write!(f, "Invalid corpus format. Could not parse the header: {}", path.display()),
            NoAnnotations => {
                f.write_str("Cannot invoke \"java.util.HashMap.entrySet()\" because \"annotationFiles\" is null")
            }
            NoLocation | NoPrimaryData => Ok(()),
            MissingFile { path } => {
                write!(f, "Corpus integrity violated. Annotation file {} is missing.", path.display())
            }
            NoAnnotationStream { .. } => f.write_str("Stream closed"),
            Annotation { file, .. } => f.write_str(file.failure()),
            QuarksEndInSentence { end } => write!(
                f,
                "Sentence ends and word ends do not match.First sentence not completed ends at character: {end}"
            ),
            IndexOutOfBounds { index, len } => write!(f, "Index {index} out of bounds for length {len}"),
            MissingQuark { id } => write!(f, "Could not attach POS tags to words. Word{id} not found in the document."),
            Substring { begin, end, len } => write!(f, "Range [{begin}, {end}) out of bounds for length {len}"),
            SplitsSurrogatePair { unit } => write!(f, "UTF-16 offset {unit} is inside a surrogate pair"),
            TooLong { len } => write!(f, "a text of {len} bytes is longer than spans can index"),
            NullDocument => f.write_str(
                "Cannot invoke \"opennlp.tools.formats.masc.MascDocument.read()\" because \"this.buffer\" is null",
            ),
            Untokenized { method } => {
                write!(f, "Cannot invoke \"java.util.List.{method}()\" because \"this.sentenceTokens\" is null")
            }
            NoDocuments { stream } => {
                let (what, method) = match stream {
                    MascStream::PartOfSpeech => ("POS tags", "hasPennTags"),
                    MascStream::Tokens => ("Penn tokenization", "hasPennTags"),
                    MascStream::NamedEntities => ("named entity labels", "hasNamedEntities"),
                };
                write!(
                    f,
                    "None of the documents has {what}Cannot invoke \
                     \"opennlp.tools.formats.masc.MascDocument.{method}()\" because \"this.buffer\" is null"
                )
            }
            NegativeTokenStart { start } => write!(f, "start index must be zero or greater: {start}"),
            TokenOutOfBounds { start, end, len } => {
                write!(f, "Span [{start}..{end}) is out of bounds, text length: {len}!")
            }
            NullTag => f.write_str("null elements are not allowed in tags!"),
            Sample(e) => e.fmt(f),
        }
    }
}

impl std::error::Error for MascError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            MascError::Io { error, .. } => Some(error),
            MascError::Header { cause, .. } | MascError::Annotation { cause, .. } => Some(cause),
            MascError::Sample(e) => Some(e),
            _ => None,
        }
    }
}

impl From<FormatError> for MascError {
    fn from(e: FormatError) -> MascError {
        MascError::Sample(e)
    }
}
