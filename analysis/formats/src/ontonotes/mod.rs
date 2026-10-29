//! The OntoNotes 4.0 corpus: `opennlp.tools.formats.ontonotes`.
//!
//! OpenNLP reads two kinds of OntoNotes file, each a document:
//!
//! - `.name` files, lines of tokens with the names marked by inline
//!   `ENAMEX` tags, which [`NameDocument`] reads into name samples
//!   (`OntoNotesNameSampleStream`);
//! - `.parse` files, Penn Treebank trees over several lines separated by
//!   empty lines, which [`document_lines`] cuts into lines
//!   (`DocumentToLineStream`) and [`ParseStream`] reads into parses
//!   (`OntoNotesParseSampleStream`). [`PosStream`] makes part-of-speech
//!   samples of them, as `OntoNotesPOSSampleStreamFactory` does with a
//!   `ParseToPOSSampleStream`.
//!
//! [`OntoNotesFormat`] holds the factories' one parameter, the corpus
//! directory (`OntoNotesFormatParameters`), and lists its files in the
//! order the factories read them. The readers take the documents as text:
//! the factories decode every file as UTF-8 and replace bytes that are not
//! valid UTF-8 with U+FFFD. Decoding is the caller's; `String::from_utf8_lossy`
//! also replaces them, though it has not been checked to replace the same
//! runs of bytes as Java's decoder.
//!
//! Every token and tag keeps the byte span it was read from in its
//! document. A name token whose tags are cut out of the middle and a token
//! OpenNLP replaces (`-LRB-` becomes `(`) have a text of their own; a
//! parse holds its own sentence text, as OpenNLP's does, and
//! [`OntoNotesParse::tokens`] maps each of its tokens back to the
//! document.
//!
//! # Where OpenNLP's behaviour is surprising
//!
//! These are reproduced:
//!
//! - Names: every line of a document is a sample, empty lines included,
//!   except the lines starting with `<DOC`; a line that is exactly
//!   `</DOC>` ends the document, and the lines after it are not read.
//! - A token starting with `<ENAMEX` opens a name and is dropped, with
//!   every token after it up to one holding `>`; that one is kept, with
//!   the start tag cut off, so `TYPE="PERSON">` alone is an empty token.
//! - The type is read from a token starting with `TYPE="` inside the
//!   start tag, lower-cased code point by code point. A line's type is
//!   not reset between its names, so a name whose start tag has no type
//!   takes the type of the name before it on the line, or none.
//! - A token ending with `</ENAMEX>` closes the open name. One with no
//!   open name throws (`Span` refuses the start -1), as does a `TYPE="`
//!   with no closing quote. A name still open at the end of the line is
//!   dropped.
//! - Tags are cut out of tokens by their first `<` and first `>`, so a
//!   token with a `>` before its first `<` throws
//!   `StringIndexOutOfBoundsException`, unless the two are next to each
//!   other, as in `a><b`, which is kept whole.
//! - A document that gives no sample ends the name stream, and the
//!   documents after it are not read. A document that throws on a line
//!   gives the error first and then the samples of the lines before it.
//! - Parses: a sentence ends at an empty or blank line, and an empty line
//!   with no sentence before it ends the parse stream. Two empty lines in
//!   a row, an empty first line or an empty document stop the reading.
//! - `DocumentToLineStream` splits a document at line feeds only (a
//!   carriage return stays at the end of its line, where trimming removes
//!   it), drops the empty lines at its end and adds one empty line when
//!   the last line left is not blank. A document of nothing but line
//!   feeds has no line left and throws `ArrayIndexOutOfBoundsException`.
//!
//! Where this port differs, deliberately:
//!
//! - The files of the corpus directory are listed as
//!   [`directory_files`](crate::directory::directory_files) lists them,
//!   which refuses a directory inside itself, where Java loops.
//! - Names, tokens and types are kept as their source bytes where they
//!   are the source's bytes, and each sample keeps its line's span; Java
//!   keeps strings.

mod lines;
mod names;
mod parse;

use std::fmt;
use std::path::{Path, PathBuf};

use turbo_analysis::Span;
use turbo_parser::ParseError;

use crate::directory::{DirectoryError, directory_files};

pub use lines::document_lines;
pub use names::{NameDocument, NameStream};
pub use parse::{OntoNotesParse, ParseStream, PosStream};

/// The name of the format in OpenNLP's registry, for name, part-of-speech
/// and parse samples.
pub const FORMAT_NAME: &str = "ontonotes";

/// An error of the OntoNotes readers.
///
/// [`OntoNotesError::java_class`] is the class of the exception OpenNLP
/// throws in the same place, and `Display` is its message, except for a
/// parse error, whose message is this crate's ([`ParseError`]).
#[derive(Debug)]
pub enum OntoNotesError {
    /// The corpus directory is not a directory: the factories'
    /// `TerminateToolException`.
    InvalidDirectory { path: PathBuf },
    /// The corpus directory could not be listed.
    Directory(DirectoryError),
    /// A `TYPE="` token of a name's start tag with no closing quote:
    /// `substring(6, -1)` throws. `len` is the token's length in UTF-16
    /// code units.
    UnclosedType { line: u32, token: Span, len: usize },
    /// A token holding a `<` after the character after its first `>`, once
    /// a start tag is cut off: `StringBuilder.delete` throws. `start` and
    /// `end` are what it was given and `len` the length of what is left
    /// of the token, in UTF-16 code units.
    TagOrder { line: u32, token: Span, start: usize, end: usize, len: usize },
    /// A token ending with `</ENAMEX>` when no name is open: `Span` refuses
    /// the start -1.
    EndWithoutStart { line: u32, token: Span },
    /// A document of nothing but line feeds, which has no line left once
    /// the empty lines at its end are dropped: `List.get(-1)` throws.
    NoLines,
    /// A sentence's tree could not be read. `sentence` is the bytes of its
    /// lines in the document.
    Parse { document: usize, sentence: Span, error: ParseError },
    /// A document longer than `u32::MAX` bytes, which spans cannot index.
    TooLong { len: usize },
}

impl OntoNotesError {
    /// The class of the exception OpenNLP throws in the same place, for
    /// the errors that have one.
    pub fn java_class(&self) -> Option<&'static str> {
        use OntoNotesError::*;
        match self {
            InvalidDirectory { .. } => Some("opennlp.tools.cmdline.TerminateToolException"),
            Directory(e) => e.java_class(),
            UnclosedType { .. } | TagOrder { .. } => Some("java.lang.StringIndexOutOfBoundsException"),
            EndWithoutStart { .. } => Some("java.lang.IllegalArgumentException"),
            NoLines => Some("java.lang.ArrayIndexOutOfBoundsException"),
            Parse { error, .. } => error.java_class(),
            TooLong { .. } => None,
        }
    }
}

impl fmt::Display for OntoNotesError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        use OntoNotesError::*;
        match self {
            InvalidDirectory { .. } => f.write_str("The specified OntoNotes directory is not valid!"),
            Directory(e) => e.fmt(f),
            UnclosedType { len, .. } => write!(f, "Range [6, -1) out of bounds for length {len}"),
            TagOrder { start, end, len, .. } => write!(f, "Range [{start}, {end}) out of bounds for length {len}"),
            EndWithoutStart { .. } => f.write_str("start index must be zero or greater: -1"),
            NoLines => f.write_str("Index -1 out of bounds for length 0"),
            Parse { error, .. } => error.fmt(f),
            TooLong { len } => write!(f, "a document of {len} bytes is longer than spans can index"),
        }
    }
}

impl std::error::Error for OntoNotesError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            OntoNotesError::Directory(e) => Some(e),
            OntoNotesError::Parse { error, .. } => Some(error),
            _ => None,
        }
    }
}

/// The parameters of the OntoNotes format factories
/// (`OntoNotesFormatParameters`): the corpus directory.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct OntoNotesFormat {
    /// The OntoNotes 4.0 corpus directory (`ontoNotesDir`).
    pub dir: PathBuf,
}

impl OntoNotesFormat {
    pub fn new(dir: impl Into<PathBuf>) -> OntoNotesFormat {
        OntoNotesFormat { dir: dir.into() }
    }

    /// The `.name` files the name factory reads, in its order.
    ///
    /// # Errors
    ///
    /// See [`OntoNotesFormat::files`].
    pub fn name_files(&self) -> Result<Vec<PathBuf>, OntoNotesError> {
        self.files(".name")
    }

    /// The `.parse` files the parse and part-of-speech factories read, in
    /// their order.
    ///
    /// # Errors
    ///
    /// See [`OntoNotesFormat::files`].
    pub fn parse_files(&self) -> Result<Vec<PathBuf>, OntoNotesError> {
        self.files(".parse")
    }

    /// The files under the corpus directory, in its subdirectories too,
    /// whose names end with `suffix`, in the order the factories'
    /// `DirectorySampleStream` gives them
    /// ([`directory_files`](crate::directory::directory_files)).
    ///
    /// # Errors
    ///
    /// [`OntoNotesError::InvalidDirectory`] when the directory is not a
    /// directory, and [`OntoNotesError::Directory`] when it cannot be
    /// listed.
    pub fn files(&self, suffix: &str) -> Result<Vec<PathBuf>, OntoNotesError> {
        if !std::fs::metadata(&self.dir).is_ok_and(|m| m.is_dir()) {
            return Err(OntoNotesError::InvalidDirectory { path: self.dir.clone() });
        }
        let accept = |path: &Path, meta: &std::fs::Metadata| {
            if meta.is_file() {
                return path.file_name().is_some_and(|n| n.to_string_lossy().ends_with(suffix));
            }
            meta.is_dir()
        };
        directory_files(&self.dir, true, Some(&accept)).map_err(OntoNotesError::Directory)
    }
}
