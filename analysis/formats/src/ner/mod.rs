//! The named entity corpora OpenNLP reads besides CoNLL: GermEval 2014,
//! BioNLP 2004 (JNLPBA), Evalita 2007, MUC-6 and brat names, and the 1990
//! US Census surname lists.
//!
//! | Corpus | Reader | OpenNLP | Registry name |
//! |---|---|---|---|
//! | GermEval 2014 | [`GermEvalReader`] | `GermEval2014NameSampleStream` | `germeval2014` ([`GermEvalNames`]) |
//! | BioNLP 2004 | [`BioNlpReader`] | `BioNLP2004NameSampleStream` | `bionlp2004` ([`BioNlpNames`]) |
//! | Evalita 2007 | [`EvalitaReader`] | `EvalitaNameSampleStream` | `evalita` ([`EvalitaNames`]) |
//! | 1990 US Census surnames | [`CensusReader`] | `NameFinderCensus90NameStream` | none: it reads names for a dictionary, not samples |
//! | MUC-6 | [`MucReader`], [`MucSamples`] for a directory | `MucNameSampleStream` | `muc6` ([`Muc6Names`], one file) |
//! | brat standoff | [`brat::BratParser`], [`brat::BratSamples`] for a directory | `BratNameSampleStream` | none: a corpus is a directory of files |
//!
//! The readers take decoded text ([`crate::decode`]); OpenNLP reads each of
//! these corpora as UTF-8, and the census lists in the encoding the
//! dictionary tool is given (UTF-8 unless told otherwise). Tokens are
//! fields of the text with their byte spans. MUC-6 and brat text is split
//! into tokens by a tokenizer ([`TokenSpans`], [`RuleTokenizer`]) and brat
//! text into sentences ([`SentenceSpans`], [`NewlineSentences`]); the
//! [`MucReader`] and [`brat`] documentation lists what OpenNLP does
//! with them.
//!
//! The three tagged corpora have a sentence per block of lines, ended by
//! an empty line, a word per line and a name tag per word. A name starts
//! at a `B-` tag and takes each `I-` tag after it, whatever its type,
//! until an `O` or the next `B-`; an `I-` tag with no name open is read
//! as `O`. A name's type is its `B-` tag after the prefix, mapped to
//! OpenNLP's type names where the stream has a map.
//!
//! # What OpenNLP does here
//!
//! - A line with the wrong number of fields ends the read at that line,
//!   and the next read starts at the line after it, in the middle of the
//!   sentence: the rest of the sentence is read as a sample of its own. A
//!   bad tag or type is found after the whole sentence is read, so the
//!   next read starts at the next sentence.
//! - The types left out are matched by the tag's end in BioNLP and
//!   Evalita (`B-XPER` is left out with `PER`), by its start in GermEval
//!   (`B-PERderiv` is left out with `PER`, and `B-PERfoo` is kept and
//!   then refused as an unknown type).
//! - GermEval reads any tag that is not `O` and not of a type it keeps
//!   as `O`, so a tag such as `X` or `B-ABC` is no error: the "Invalid
//!   tag" branch of `GermEval2014NameSampleStream` cannot be reached. Its
//!   inner tag column is trimmed (`String.trim`) and its outer one is
//!   not, so `B-PER ` with a space is an unknown type in the outer layer.
//!   A comment line (`#`) anywhere in a sentence clears adaptive data,
//!   but one followed by an empty line does not: the empty sentence is
//!   skipped and its flag with it.
//! - BioNLP keeps `cell_line` names only when `cell_type` names are kept:
//!   the stream tests the `cell_type` flag for both, and its own
//!   `cell_line` flag is never read. A line of spaces ends a sentence
//!   (BioNLP trims the line before testing it, the others do not). The
//!   line after a `###MEDLINE:` line is skipped whatever it is: in the
//!   corpus it is empty, and when it is not, a word is lost. A line that
//!   contains `ABSTRACT TRUNCATED` anywhere is skipped.
//! - Evalita clears adaptive data for every sample. A `-DOCSTART-` line
//!   must be followed by a line of length zero
//!   ([`NerError::DocstartNotFollowedByEmpty`]), and at the end of the
//!   text it throws `NullPointerException` ([`NerError::DocstartAtEnd`]),
//!   losing the sentence read so far.
//! - The census stream ends at the first empty line or line without a
//!   space: `read` returns null there, and the dictionary tool stops.
//!   A line that starts with a space throws
//!   `StringIndexOutOfBoundsException`. The name is recased by UTF-16
//!   code units, so a first letter outside the Basic Multilingual Plane
//!   is not upper-cased.

mod bionlp;
pub mod brat;
mod census;
mod evalita;
mod formats;
mod germeval;
mod muc;
mod tokens;

use std::fmt;

use turbo_analysis::Span;

pub use bionlp::{BioNlpReader, BioNlpTypes};
pub use census::CensusReader;
pub use evalita::{EvalitaLanguage, EvalitaReader, EvalitaTypes};
pub use formats::{BioNlpNames, EvalitaNames, GermEvalNames, Muc6Names};
pub use germeval::{GermEvalLayer, GermEvalReader, GermEvalTypes};
pub use muc::{MUC_TYPES, MUC6_FORMAT, MucReader, MucSamples, muc_files};
pub use tokens::{NewlineSentences, RuleTokenizer, SentenceSpans, TokenSpans};

use crate::error::FormatError;
use crate::ner::brat::BratCause;
use crate::sample::{Name, NameSample};
use crate::source::{Field, Line};

/// The registry name of the GermEval 2014 format.
pub const GERMEVAL_FORMAT: &str = "germeval2014";

/// The registry name of the BioNLP 2004 format.
pub const BIONLP_FORMAT: &str = "bionlp2004";

/// The registry name of the Evalita format.
pub const EVALITA_FORMAT: &str = "evalita";

/// The line Evalita documents start with (`EvalitaNameSampleStream.DOCSTART`).
pub const DOCSTART: &str = "-DOCSTART-";

/// What a reader of a named entity corpus refuses. `Display` is the
/// message of the exception OpenNLP throws, and [`NerError::java_class`]
/// its class.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum NerError {
    /// A GermEval line with fewer than four tab-separated fields (after
    /// `StringUtil.split` drops the empty fields at its end). `count` is
    /// the number of fields and `text` the line.
    GermEvalFields { line: u32, count: usize, text: String },
    /// A BioNLP line without exactly two tab-separated fields.
    BioNlpFields { line: u32, count: usize, text: String },
    /// An Evalita line without exactly four space-separated fields.
    EvalitaFields { line: u32, text: String },
    /// A name tag that is not `O` and does not start with `B-` or `I-`.
    InvalidTag { line: u32, tag: String },
    /// A name tag whose type the stream has no name for, found when its
    /// name ends: `kind` is the tag after its prefix.
    UnknownType { line: u32, kind: String },
    /// An Evalita `-DOCSTART-` line followed by a line that is not empty;
    /// `line` and `text` are that line.
    DocstartNotFollowedByEmpty { line: u32, text: String },
    /// An Evalita `-DOCSTART-` line at the end of the text: the stream
    /// asks `StringUtil.isEmpty` about the null line after it.
    DocstartAtEnd { line: u32 },
    /// A census line that starts with a space: the name before it is
    /// empty and `substring(0, 1)` throws.
    EmptyCensusName { line: u32 },
    /// A `<` inside a MUC tag.
    MucUnexpectedOpen { line: u32 },
    /// A `>` outside a MUC tag.
    MucUnexpectedClose { line: u32 },
    /// A MUC tag still open at the end of the file; `line` is where it
    /// starts.
    MucUnclosedTag { line: u32 },
    /// A MUC name element without a `TYPE` value: the stream asks the set
    /// of types whether it holds null, which throws.
    MucMissingType { line: u32 },
    /// A MUC name element whose `TYPE` is not one of [`MUC_TYPES`].
    MucUnknownType { line: u32, kind: String },
    /// The end tag of a MUC name element with no name open: the stream
    /// pops an empty stack.
    MucUnopenedName { line: u32 },
    /// A line of a brat annotation file that a parser refuses, which the
    /// stream wraps in an `IOException` naming the document `file` (its
    /// path without `.ann`); `cause` is what the parser threw.
    BratAnnotation { file: String, line: u32, cause: BratCause },
    /// An array of a brat annotation line read at `index`, past its end
    /// (`len` values): `ArrayIndexOutOfBoundsException`.
    BratIndex { file: String, line: u32, index: i64, len: usize },
    /// Brat offsets `Span`'s constructor refuses: a negative one, or a
    /// start past the end.
    BratSpan { file: String, line: u32, start: i32, end: i32 },
    /// Two names of a brat sentence that overlap, which `NameSample`
    /// refuses naming the document `file`.
    BratOverlappingNames { file: String, first: String, second: String, sentence: String },
    /// A type in a brat annotation configuration before any section.
    BratConfigNoSection { line: u32 },
    /// A line of a brat annotation configuration with no field: only
    /// whitespace that `String.trim` keeps.
    BratConfigNoType { line: u32 },
    /// An empty set of brat name types, which `BratDocumentParser`
    /// refuses.
    BratNoNameTypes,
}

impl NerError {
    /// The class of the exception OpenNLP throws in the same place.
    pub fn java_class(&self) -> &'static str {
        use NerError::*;
        match self {
            GermEvalFields { .. }
            | BioNlpFields { .. }
            | EvalitaFields { .. }
            | InvalidTag { .. }
            | DocstartNotFollowedByEmpty { .. } => "java.io.IOException",
            UnknownType { .. } => "opennlp.tools.util.InvalidFormatException",
            DocstartAtEnd { .. } => "java.lang.NullPointerException",
            EmptyCensusName { .. } => "java.lang.StringIndexOutOfBoundsException",
            MucUnexpectedOpen { .. } | MucUnexpectedClose { .. } | MucUnclosedTag { .. } | MucUnknownType { .. } => {
                "opennlp.tools.util.InvalidFormatException"
            }
            MucMissingType { .. } => "java.lang.NullPointerException",
            MucUnopenedName { .. } => "java.util.EmptyStackException",
            BratAnnotation { .. } => "java.io.IOException",
            BratIndex { .. } | BratConfigNoType { .. } => "java.lang.ArrayIndexOutOfBoundsException",
            BratSpan { .. } | BratNoNameTypes => "java.lang.IllegalArgumentException",
            BratOverlappingNames { .. } => "java.lang.RuntimeException",
            BratConfigNoSection { .. } => "java.lang.NullPointerException",
        }
    }

    /// The source line the error is on.
    pub fn line(&self) -> Option<u32> {
        use NerError::*;
        match *self {
            GermEvalFields { line, .. }
            | BioNlpFields { line, .. }
            | EvalitaFields { line, .. }
            | InvalidTag { line, .. }
            | UnknownType { line, .. }
            | DocstartNotFollowedByEmpty { line, .. }
            | DocstartAtEnd { line }
            | EmptyCensusName { line }
            | MucUnexpectedOpen { line }
            | MucUnexpectedClose { line }
            | MucUnclosedTag { line }
            | MucMissingType { line }
            | MucUnknownType { line, .. }
            | MucUnopenedName { line }
            | BratAnnotation { line, .. }
            | BratIndex { line, .. }
            | BratSpan { line, .. }
            | BratConfigNoSection { line }
            | BratConfigNoType { line } => Some(line),
            BratOverlappingNames { .. } | BratNoNameTypes => None,
        }
    }
}

impl fmt::Display for NerError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        use NerError::*;
        match self {
            GermEvalFields { count, text, .. } => write!(
                f,
                "Expected at least four tab-separated fields per line in GermEval 2014 data, got {count} for line '{text}'!"
            ),
            BioNlpFields { count, text, .. } => {
                write!(f, "Expected two fields per line in training data, got {count} for line '{text}'!")
            }
            EvalitaFields { text, .. } => write!(f, "Incorrect number of fields per line for language: '{text}'!"),
            InvalidTag { tag, .. } => write!(f, "Invalid tag: {tag}"),
            UnknownType { kind, .. } => write!(f, "Unknown type: {kind}"),
            DocstartNotFollowedByEmpty { text, .. } => write!(f, "Empty line after -DOCSTART- not empty: '{text}'!"),
            DocstartAtEnd { .. } => {
                f.write_str("Cannot invoke \"java.lang.CharSequence.length()\" because \"theString\" is null")
            }
            EmptyCensusName { .. } => f.write_str("Range [0, 1) out of bounds for length 0"),
            MucUnexpectedOpen { .. } => f.write_str("Did not expect < char!"),
            MucUnexpectedClose { .. } => f.write_str("Did not expect > char!"),
            MucUnclosedTag { .. } => f.write_str("Did not find matching > char!"),
            MucMissingType { .. } | MucUnopenedName { .. } => Ok(()),
            MucUnknownType { kind, .. } => {
                write!(f, "Unknown timex, numex or namex type: {kind}, expected one of [{}]", MUC_TYPES.join(", "))
            }
            BratAnnotation { file, .. } => write!(f, "Failed to parse ann document with id [{file}.ann]"),
            BratIndex { index, len, .. } => write!(f, "Index {index} out of bounds for length {len}"),
            BratSpan { start, end, .. } => {
                if *start < 0 {
                    write!(f, "start index must be zero or greater: {start}")
                } else if *end < 0 {
                    write!(f, "end index must be zero or greater: {end}")
                } else {
                    write!(f, "start index must not be larger than end index: start={start}, end={end}")
                }
            }
            BratOverlappingNames { file, first, second, sentence } => {
                write!(f, "name spans {first} and {second} are overlapped in file: {file}, sentence: {sentence}")
            }
            BratConfigNoSection { .. } => {
                f.write_str("Cannot invoke \"String.hashCode()\" because \"<local6>\" is null")
            }
            BratConfigNoType { .. } => f.write_str("Index 0 out of bounds for length 0"),
            BratNoNameTypes => f.write_str("nameTypes should be null or have one or more elements"),
        }
    }
}

impl std::error::Error for NerError {}

impl From<NerError> for FormatError {
    fn from(e: NerError) -> FormatError {
        FormatError::Ner(e)
    }
}

/// A name tag and the line it is on.
#[derive(Copy, Clone, Debug)]
pub(crate) struct Tag<'a> {
    pub(crate) text: &'a str,
    pub(crate) line: u32,
}

/// The lines of a sentence read so far.
#[derive(Default)]
pub(crate) struct Sentence<'a> {
    pub(crate) tokens: Vec<Field<'a>>,
    pub(crate) tags: Vec<Tag<'a>>,
    pub(crate) clear_adaptive_data: bool,
    /// From the start of the first line read to the end of the last.
    source: Option<Span>,
    first_line: u32,
}

impl<'a> Sentence<'a> {
    /// Takes `line` into the sentence's source.
    pub(crate) fn cover(&mut self, line: &Line<'a>) {
        let span = line.span();
        match self.source {
            None => {
                self.source = Some(span);
                self.first_line = line.number;
            }
            Some(s) => self.source = Some(Span::new(s.start, span.end)),
        }
    }

    /// The sample of the sentence's tokens with `names`.
    pub(crate) fn sample(
        self,
        names: Result<Vec<Name<'a>>, FormatError>,
    ) -> Option<Result<NameSample<'a>, FormatError>> {
        let source = self.source.expect("a sentence with tokens has lines");
        Some(names.and_then(|n| NameSample::new(self.tokens, n, self.clear_adaptive_data, source, self.first_line)))
    }
}

/// The names of tagged tokens as the BioNLP, Evalita and GermEval streams
/// find them: a name starts at `B-` and grows with each `I-`, whatever
/// its type, until `O` or the next `B-`; any other tag is
/// [`NerError::InvalidTag`]. `filter` gives the tag the stream reads in
/// place of each (`O` for a type left out), and `kind` the type of a
/// name from its first tag as written, after the two-character prefix.
pub(crate) fn bio_names<'a>(
    tags: &[Tag<'a>],
    filter: impl Fn(&'a str) -> &'a str,
    kind: impl Fn(&'a str, u32) -> Result<&'a str, NerError>,
) -> Result<Vec<Name<'a>>, FormatError> {
    let extract = |begin: usize, end: i64| -> Result<Name<'a>, FormatError> {
        let tag = tags[begin];
        let kind = kind(&tag.text[2..], tag.line)?;
        let end = u32::try_from(end).expect("an open name ends after its first token");
        Name::new(begin as u32, end, Some(kind))
    };
    let mut names = Vec::new();
    let mut begin: Option<usize> = None;
    // Java's endIndex, which an `I-` with no name open moves on from -1;
    // the next `B-` sets it again, so that does not show
    let mut end: i64 = -1;
    for (i, t) in tags.iter().enumerate() {
        let tag = filter(t.text);
        if tag.starts_with("B-") {
            if let Some(b) = begin {
                names.push(extract(b, end)?);
            }
            begin = Some(i);
            end = i as i64 + 1;
        } else if tag.starts_with("I-") {
            end += 1;
        } else if tag == "O" {
            if let Some(b) = begin.take() {
                names.push(extract(b, end)?);
                end = -1;
            }
        } else {
            return Err(NerError::InvalidTag { line: t.line, tag: tag.to_owned() }.into());
        }
    }
    if let Some(b) = begin {
        names.push(extract(b, end)?);
    }
    Ok(names)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tags(texts: &[&'static str]) -> Vec<Tag<'static>> {
        texts.iter().enumerate().map(|(i, t)| Tag { text: t, line: i as u32 + 1 }).collect()
    }

    fn names(texts: &[&'static str]) -> Result<Vec<String>, String> {
        bio_names(&tags(texts), |t| t, |k, _| Ok(k))
            .map(|n| n.iter().map(Name::java_string).collect())
            .map_err(|e| e.to_string())
    }

    #[test]
    fn a_name_starts_at_b_and_takes_every_i() {
        assert_eq!(names(&["B-A", "I-B", "O", "B-C", "B-D", "I-D"]).unwrap(), ["[0..2) A", "[3..4) C", "[4..6) D"]);
    }

    #[test]
    fn an_i_with_no_name_open_is_read_as_o() {
        assert_eq!(names(&["I-A", "I-A", "O", "I-B", "B-C"]).unwrap(), ["[4..5) C"]);
    }

    #[test]
    fn another_tag_is_invalid() {
        assert_eq!(names(&["B-A", "X"]).unwrap_err(), "Invalid tag: X");
        assert_eq!(names(&["o"]).unwrap_err(), "Invalid tag: o");
    }

    #[test]
    fn errors_name_their_exceptions() {
        let e = NerError::UnknownType { line: 3, kind: "X".into() };
        assert_eq!(e.java_class(), "opennlp.tools.util.InvalidFormatException");
        assert_eq!(e.line(), Some(3));
        let e = FormatError::from(e);
        assert_eq!(e.java_class(), Some("opennlp.tools.util.InvalidFormatException"));
        assert_eq!(e.line(), Some(3));
        assert_eq!(e.to_string(), "Unknown type: X");
    }
}
