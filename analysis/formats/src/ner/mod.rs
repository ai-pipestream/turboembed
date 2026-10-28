//! Named-entity corpus readers: GermEval 2014, BioNLP 2004, Evalita 2007,
//! brat, WNUT-17, MultiCoNER v2, LitBank and the 1990 US Census lists.
//!
//! | Corpus | Reader |
//! |---|---|
//! | GermEval 2014 | [`GermEvalReader`] |
//! | BioNLP 2004 | [`BioNlpReader`] |
//! | Evalita 2007 | [`EvalitaReader`] |
//! | 1990 US Census surnames | [`CensusReader`]: it reads names for a dictionary, not samples |
//! | brat standoff | [`brat::BratParser`], [`brat::BratSamples`] for a directory |
//!
//! Three kept corpora (#1533) have readers here with no OpenNLP stream
//! behind them: WNUT-17 ([`WnutReader`]), MultiCoNER v2
//! ([`MulticonerReader`]) and LitBank ([`LitBankReader`]). Their formats
//! are pinned in each reader's documentation, and their errors carry no
//! JVM message.
//!
//! The readers take decoded text ([`crate::decode`]); OpenNLP reads each of
//! these corpora as UTF-8, and the census lists in the encoding the
//! dictionary tool is given (UTF-8 unless told otherwise). Tokens are
//! fields of the text with their byte spans. brat text is split into
//! tokens by a tokenizer ([`TokenSpans`], [`RuleTokenizer`]) and sentences
//! ([`SentenceSpans`], [`NewlineSentences`]).
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
//!   ([`FormatError::DocstartNotFollowedByEmpty`]), and at the end of the
//!   text it throws `NullPointerException` ([`FormatError::DocstartAtEnd`]),
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
mod germeval;
mod litbank;
mod multiconer2;
mod tokens;
mod wnut17;

use std::fmt;

use turbo_analysis::Span;

pub use bionlp::{BioNlpReader, BioNlpTypes};
pub use census::CensusReader;
pub use evalita::{EvalitaLanguage, EvalitaReader, EvalitaTypes};
pub use germeval::{GermEvalLayer, GermEvalReader, GermEvalTypes};
pub use litbank::{LITBANK_TYPES, LitBankError, LitBankReader};
pub use multiconer2::{MULTICONER_TYPES, MulticonerReader};
pub use tokens::{NewlineSentences, RuleTokenizer, SentenceSpans, TokenSpans};
pub use wnut17::{WNUT17_TYPES, WnutError, WnutReader};

use crate::error::FormatError;
use crate::ner::brat::BratCause;
use crate::sample::{Name, NameSample};
use crate::source::{Field, Line};

/// What a reader of a named entity corpus refuses.
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
    /// A census line that starts with a space: the name before it is
    /// empty and `substring(0, 1)` throws.
    EmptyCensusName { line: u32 },
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
    /// The source line the error is on.
    pub fn line(&self) -> Option<u32> {
        use NerError::*;
        match *self {
            GermEvalFields { line, .. }
            | BioNlpFields { line, .. }
            | EvalitaFields { line, .. }
            | EmptyCensusName { line }
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
            EmptyCensusName { .. } => f.write_str("Range [0, 1) out of bounds for length 0"),
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

/// Decodes strict IOB tags. Every continuation must follow a beginning and
/// retain its type. `filter` maps a source tag to the tag to decode, and
/// `kind` validates the beginning tag's type.
pub(crate) fn bio_names<'a>(
    tags: &[Tag<'a>],
    filter: impl Fn(&'a str) -> &'a str,
    kind: impl Fn(&'a str, u32) -> Result<&'a str, FormatError>,
) -> Result<Vec<Name<'a>>, FormatError> {
    let extract = |begin: usize, end: usize| -> Result<Name<'a>, FormatError> {
        let tag = tags[begin];
        let kind = kind(&tag.text[2..], tag.line)?;
        Name::new(begin as u32, end as u32, Some(kind))
    };
    let mut names = Vec::new();
    let mut begin: Option<usize> = None;
    for (i, t) in tags.iter().enumerate() {
        let tag = filter(t.text);
        if tag.starts_with("B-") {
            if let Some(b) = begin {
                names.push(extract(b, i)?);
            }
            begin = Some(i);
        } else if let Some(found) = tag.strip_prefix("I-") {
            let Some(b) = begin else {
                return Err(FormatError::UnexpectedNameContinuation { line: t.line, tag: tag.to_owned() });
            };
            let expected = &tags[b].text[2..];
            if found != expected {
                return Err(FormatError::NameTypeMismatch {
                    line: t.line,
                    expected: expected.to_owned(),
                    found: found.to_owned(),
                });
            }
        } else if tag == "O" {
            if let Some(b) = begin.take() {
                names.push(extract(b, i)?);
            }
        } else {
            return Err(FormatError::InvalidNameTag { line: t.line, tag: tag.to_owned() });
        }
    }
    if let Some(b) = begin {
        names.push(extract(b, tags.len())?);
    }
    Ok(names)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tags(texts: &[&'static str]) -> Vec<Tag<'static>> {
        texts.iter().enumerate().map(|(i, t)| Tag { text: t, line: i as u32 + 1 }).collect()
    }

    type Shape = (u32, u32, Option<&'static str>);

    fn names(texts: &[&'static str]) -> Result<Vec<Shape>, String> {
        bio_names(&tags(texts), |t| t, |k, _| Ok(k))
            .map(|n| n.iter().map(|name| (name.start, name.end, name.kind)).collect())
            .map_err(|e| e.to_string())
    }

    #[test]
    fn a_name_starts_at_b_and_takes_matching_i_tags() {
        assert_eq!(
            names(&["B-A", "I-A", "O", "B-C", "B-D", "I-D"]).unwrap(),
            [(0, 2, Some("A")), (3, 4, Some("C")), (4, 6, Some("D"))]
        );
    }

    #[test]
    fn an_i_with_no_name_open_is_an_error() {
        assert!(matches!(
            bio_names(&tags(&["I-A"]), |t| t, |k, _| Ok(k)),
            Err(FormatError::UnexpectedNameContinuation { line: 1, .. })
        ));
    }

    #[test]
    fn an_i_type_must_match_its_beginning() {
        assert!(matches!(
            bio_names(&tags(&["B-A", "I-B"]), |t| t, |k, _| Ok(k)),
            Err(FormatError::NameTypeMismatch { line: 2, .. })
        ));
    }

    #[test]
    fn another_tag_is_invalid() {
        assert_eq!(names(&["B-A", "X"]).unwrap_err(), "Invalid tag: X");
        assert_eq!(names(&["o"]).unwrap_err(), "Invalid tag: o");
    }

    #[test]
    fn errors_name_their_line_and_message() {
        let e = NerError::BioNlpFields { line: 3, count: 2, text: "x".into() };
        assert_eq!(e.line(), Some(3));
        let e = FormatError::from(e);
        assert_eq!(e.line(), Some(3));
        assert_eq!(e.to_string(), "Expected two fields per line in training data, got 2 for line 'x'!");
    }
}
