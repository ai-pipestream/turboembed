//! What a reader refuses, and what it skips.
//!
//! A [`FormatError`] is a sample OpenNLP's stream throws on. Its
//! `Display` is the message of the exception Java throws there, and
//! [`FormatError::java_class`] the exception's class, so a test compares
//! both with the JVM's. A [`Warning`] is a line OpenNLP's stream logs and
//! skips or replaces; the reader carries on as Java does and keeps the
//! warning for the caller.

use std::fmt;

use turbo_analysis::Span;

use crate::ad::AdError;
use crate::conllu::coref::CorefError;
use crate::corpora::CorpusError;
use crate::ner::NerError;

/// A sample a reader cannot read, or a sample value that cannot be built.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum FormatError {
    /// A `<START>` tag inside a name: `NameSample.parse` throws an
    /// `IOException`. `context` is the tag with up to two parts of the
    /// line before and after it, the tag marked with `###`.
    NestedName { line: u32, context: String },
    /// `<START:>`, a start tag with an empty type.
    MissingNameType { line: u32, context: String },
    /// An `<END>` tag outside a name.
    UnexpectedEnd { line: u32, context: String },
    /// Two names of one sample overlap after `NameSample` sorts them: the
    /// `RuntimeException` of its constructor. The parser makes this only
    /// from an empty name followed at the same token by a longer one.
    OverlappingNames { line: u32, first: String, second: String, sentence: String },
    /// A document category line with no text after the category, or an
    /// empty line: `DocumentSampleStream` throws an `IOException`.
    NoDocumentText { line: u32 },
    /// A chunk tag that starts a phrase but has fewer than two UTF-16
    /// code units, so `substring(2)` throws a
    /// `StringIndexOutOfBoundsException`; `len` is the tag's length in
    /// code units.
    ShortChunkTag { token: usize, len: usize },
    /// A chunk tag whose first two UTF-16 code units end inside a
    /// surrogate pair. Java cuts the pair and keeps half of it in the
    /// phrase type; a Rust string cannot hold half a pair, so this port
    /// refuses the tag instead.
    SplitsSurrogatePair { token: usize },
    /// Columns of different lengths given to a sample's constructor.
    Lengths { what: &'static str, expected: usize, found: usize },
    /// A name built over tokens a sample does not have.
    NameOutsideTokens { name: String, tokens: usize },
    /// A span that starts after it ends: the `IllegalArgumentException` of
    /// `Span`'s constructor.
    BackwardsSpan { start: u32, end: u32 },
    /// A span outside the text it indexes, or not on its character
    /// boundaries.
    OutOfBounds { span: Span, len: usize },
    /// A source text longer than `u32::MAX` bytes, which [`Span`] cannot
    /// index.
    TooLong { len: usize },
    /// What a reader of a named entity corpus ([`crate::ner`]) refuses.
    Ner(NerError),
    /// What an Arvores Deitadas reader ([`crate::ad`]) refuses.
    Ad(AdError),
    /// What a reader of a sentence, language or document corpus
    /// ([`crate::corpora`]) refuses.
    Corpus(CorpusError),
    /// A CoNLL-U word line without exactly ten tab-separated columns
    /// (after `StringUtil.split` drops the empty columns at its end).
    /// `text` is the line.
    ConlluColumns { line: u32, text: String },
    /// A CoNLL-U `# text_<language> = ...` comment whose key has no two
    /// or three lowercase ASCII letters after a `text_`. `key` is the
    /// trimmed key.
    InvalidLanguage { line: u32, key: String },
    /// A CoNLL-U multiword token id that is not two ASCII digit runs
    /// without a leading zero, each fitting a Java `int`, joined by one
    /// hyphen.
    InvalidMultiwordId { line: u32, id: String },
    /// A CoNLL-U multiword token id whose end is less than its start.
    BackwardsMultiwordId { line: u32, id: String },
    /// A CoNLL-U multiword token whose range names a word id no line of
    /// the sentence has.
    MissingMultiwordLine { line: u32, id: String, missing: u64 },
    /// The `IndexOutOfBoundsException` of `ArrayList.remove` when
    /// `ConlluStream` deletes the word lines of multiword tokens whose
    /// ranges overlap or repeat: it removes the lines by their places
    /// before any removal, so it removes more lines than there are.
    /// `line` is the sentence's first line.
    MultiwordRemoval { line: u32, index: usize, len: usize },
    /// A CoNLL-U word form `ConlluTokenSampleStream` cannot find in the
    /// sentence's text after the forms before it. `text` is the text with
    /// the `<SPLIT>` markers inserted so far, and `sentence` the
    /// `sent_id` comment.
    TokenNotInText { line: u32, token: String, sentence: Option<String>, text: String },
    /// A CoNLL-U sentence with no `# text = ...` comment, which
    /// `ConlluTokenSampleStream` needs.
    NoSentenceText { line: u32 },
    /// A CoNLL-U word line that `ConlluDependencySampleStream` splits into
    /// other than ten columns (empty columns at its end count). `count`
    /// is the number of columns and `text` the line.
    DependencyColumns { line: u32, count: usize, text: String },
    /// What the CorefUD reader ([`crate::conllu::coref`]) refuses in a
    /// document.
    Coref(CorefError),
    /// A CoNLL-U `HEAD` that is not `0`, `_` or the id of another word of
    /// the sentence, which the word-preserving converter refuses rather
    /// than reading as a root. `head` is the column.
    ConlluHead { line: u32, head: String },
    /// A layer the CoNLL-U converter built that the document refused.
    /// `detail` is the document's reason.
    ConlluDocument { line: u32, detail: String },
    /// A dependency head in another sentence than its word, which a
    /// CoNLL-U sentence cannot hold. Both are word indexes.
    HeadOutsideSentence { word: usize, head: usize },
    /// A document with no `Tokens` layer to write as CoNLL-U.
    NoTokens,
    /// A CoNLL-2002 line without exactly three space-separated fields
    /// (after `StringUtil.split` drops the empty fields at its end).
    /// `count` is the number of fields and `text` the line.
    Conll02Fields { line: u32, count: usize, text: String },
    /// A CoNLL-2003 line without the number of fields its language has
    /// (four in English, five in German). `text` is the line.
    Conll03Fields { line: u32, text: String },
    /// A name tag that is not `O` and does not start with `B-` or `I-`,
    /// in CoNLL-2002, CoNLL-2003, BioNLP 2004 and Evalita (GermEval reads
    /// such a tag as `O`).
    InvalidNameTag { line: u32, tag: String },
    /// A name tag whose type the stream has no name for, found when its
    /// name ends: `kind` is the tag after its prefix. CoNLL knows `PER`,
    /// `ORG`, `LOC` and `MISC`, Evalita `PER`, `ORG`, `LOC` and `GPE`,
    /// GermEval its twelve types (BioNLP takes any type as written).
    UnknownNameType { line: u32, kind: String },
    /// A CoNLL-2003 or Evalita `-DOCSTART-` line followed by a line that
    /// is not empty; `line` and `text` are that line.
    DocstartNotFollowedByEmpty { line: u32, text: String },
    /// A CoNLL-2003 or Evalita `-DOCSTART-` line at the end of the text:
    /// the stream asks `StringUtil.isEmpty` about the null line after it.
    DocstartAtEnd { line: u32 },
    /// A CoNLL-X line with fewer than five tab-separated columns (after
    /// `StringUtil.split` drops the empty columns at its end). `text` is
    /// the line.
    ConllXColumns { line: u32, text: String },
}

impl FormatError {
    /// The class of the exception OpenNLP throws in the same place, for
    /// the errors that have one.
    pub fn java_class(&self) -> Option<&'static str> {
        use FormatError::*;
        match self {
            NestedName { .. } | MissingNameType { .. } | UnexpectedEnd { .. } | NoDocumentText { .. } => {
                Some("java.io.IOException")
            }
            OverlappingNames { .. } => Some("java.lang.RuntimeException"),
            Ner(e) => Some(e.java_class()),
            Ad(e) => Some(e.java_class()),
            Corpus(e) => e.java_class(),
            ShortChunkTag { .. } => Some("java.lang.StringIndexOutOfBoundsException"),
            Lengths { .. } | OutOfBounds { .. } | BackwardsSpan { .. } => Some("java.lang.IllegalArgumentException"),
            ConlluColumns { .. }
            | InvalidLanguage { .. }
            | InvalidMultiwordId { .. }
            | BackwardsMultiwordId { .. }
            | MissingMultiwordLine { .. } => Some("opennlp.tools.util.InvalidFormatException"),
            MultiwordRemoval { .. } => Some("java.lang.IndexOutOfBoundsException"),
            DependencyColumns { .. } | Coref(_) => Some("opennlp.tools.util.InvalidFormatException"),
            TokenNotInText { .. } | NoSentenceText { .. } => Some("java.io.IOException"),
            ConlluHead { .. } | ConlluDocument { .. } | HeadOutsideSentence { .. } | NoTokens => None,
            Conll02Fields { .. } | Conll03Fields { .. } | InvalidNameTag { .. } | DocstartNotFollowedByEmpty { .. } => {
                Some("java.io.IOException")
            }
            UnknownNameType { .. } | ConllXColumns { .. } => Some("opennlp.tools.util.InvalidFormatException"),
            DocstartAtEnd { .. } => Some("java.lang.NullPointerException"),
            SplitsSurrogatePair { .. } | NameOutsideTokens { .. } | TooLong { .. } => None,
        }
    }

    /// The source line the error is on, for the errors a reader makes.
    pub fn line(&self) -> Option<u32> {
        use FormatError::*;
        match *self {
            Ner(ref e) => e.line(),
            Coref(ref e) => Some(e.line()),
            NestedName { line, .. }
            | MissingNameType { line, .. }
            | UnexpectedEnd { line, .. }
            | OverlappingNames { line, .. }
            | NoDocumentText { line }
            | ConlluColumns { line, .. }
            | InvalidLanguage { line, .. }
            | InvalidMultiwordId { line, .. }
            | BackwardsMultiwordId { line, .. }
            | MissingMultiwordLine { line, .. }
            | MultiwordRemoval { line, .. }
            | DependencyColumns { line, .. }
            | TokenNotInText { line, .. }
            | NoSentenceText { line }
            | ConlluHead { line, .. }
            | ConlluDocument { line, .. }
            | Conll02Fields { line, .. }
            | Conll03Fields { line, .. }
            | InvalidNameTag { line, .. }
            | UnknownNameType { line, .. }
            | DocstartNotFollowedByEmpty { line, .. }
            | DocstartAtEnd { line }
            | ConllXColumns { line, .. } => Some(line),
            _ => None,
        }
    }
}

impl fmt::Display for FormatError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        use FormatError::*;
        match self {
            Ner(e) => e.fmt(f),
            Ad(e) => e.fmt(f),
            Corpus(e) => e.fmt(f),
            NestedName { context, .. } => {
                write!(f, "Found unexpected annotation while handling a name sequence: {context}")
            }
            MissingNameType { context, .. } => write!(f, "Missing a name type: {context}"),
            UnexpectedEnd { context, .. } => write!(f, "Found unexpected annotation: {context}"),
            OverlappingNames { first, second, sentence, .. } => {
                write!(f, "name spans {first} and {second} are overlapped in file: null, sentence: {sentence}")
            }
            NoDocumentText { .. } => f.write_str("Empty lines, or lines with only a category string are not allowed!"),
            ShortChunkTag { len, .. } => write!(f, "Range [2, {len}) out of bounds for length {len}"),
            SplitsSurrogatePair { token } => {
                write!(f, "the chunk tag of token {token} has a surrogate pair across its second code unit")
            }
            Lengths { what, expected, found } => {
                write!(f, "{what}: expected {expected} values, found {found}")
            }
            BackwardsSpan { start, end } => {
                write!(f, "start index must not be larger than end index: start={start}, end={end}")
            }
            NameOutsideTokens { name, tokens } => write!(f, "the name {name} reaches past the {tokens} tokens"),
            OutOfBounds { span, len } => {
                write!(
                    f,
                    "the span {}..{} is outside a text of {len} bytes or not on its characters",
                    span.start, span.end
                )
            }
            TooLong { len } => write!(f, "a source of {len} bytes is longer than spans can index"),
            ConlluColumns { text, .. } => write!(f, "Line [{text}] must have exactly 10 fields"),
            InvalidLanguage { key, .. } => write!(f, "Locale language code is invalid: {key}"),
            InvalidMultiwordId { id, .. } => write!(f, "Invalid multiword token id: {id}"),
            BackwardsMultiwordId { id, .. } => write!(f, "Multiword token id runs backwards: {id}"),
            MissingMultiwordLine { id, missing, .. } => {
                write!(f, "Multiword token {id} has no word line for id {missing}")
            }
            MultiwordRemoval { index, len, .. } => write!(f, "Index {index} out of bounds for length {len}"),
            DependencyColumns { count, text, .. } => {
                write!(f, "CoNLL-U word line has {count} columns, expected 10: {text}")
            }
            Coref(e) => e.fmt(f),
            TokenNotInText { token, sentence, text, .. } => write!(
                f,
                "Failed to match token [{token}] in sentence [{}] with text [{text}]",
                sentence.as_deref().unwrap_or("null")
            ),
            NoSentenceText { .. } => f.write_str("Sentence is missing raw text sample!"),
            ConlluHead { head, .. } => write!(f, "a HEAD [{head}] that is not 0, _ or another word of the sentence"),
            ConlluDocument { detail, .. } => write!(f, "the document refused the sentence's layers: {detail}"),
            HeadOutsideSentence { word, head } => {
                write!(f, "word {word} has its head {head} in another sentence")
            }
            NoTokens => f.write_str("the document has no tokens to write"),
            Conll02Fields { count, text, .. } => {
                write!(f, "Expected three fields per line in training data, got {count} for line '{text}'!")
            }
            Conll03Fields { text, .. } => write!(f, "Incorrect number of fields per line for language: '{text}'!"),
            InvalidNameTag { tag, .. } => write!(f, "Invalid tag: {tag}"),
            UnknownNameType { kind, .. } => write!(f, "Unknown type: {kind}"),
            DocstartNotFollowedByEmpty { text, .. } => write!(f, "Empty line after -DOCSTART- not empty: '{text}'!"),
            DocstartAtEnd { .. } => {
                f.write_str("Cannot invoke \"java.lang.CharSequence.length()\" because \"theString\" is null")
            }
            ConllXColumns { text, .. } => write!(f, "Every non-empty line must have at least 5 fields: '{text}'!"),
        }
    }
}

impl std::error::Error for FormatError {}

/// A line a reader skipped or replaced, as OpenNLP's stream logs it and
/// carries on.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Warning {
    /// A word and tag line with a token that has no `_`.
    /// `WordTagSampleStream` logs "Error during parsing, ignoring
    /// sentence" and returns a sample with no tokens in its place.
    NoTagSeparator { line: u32, token: Span },
    /// A lemma or chunk line without exactly three columns, skipped
    /// ("Skipping corrupt line").
    CorruptLine { line: u32, columns: usize },
    /// A language sample line without a tab after a non-empty language
    /// code, which `LanguageDetectorSampleStream` passes over silently.
    NoLanguage { line: u32 },
}
