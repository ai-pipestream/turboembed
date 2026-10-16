//! The French Treebank: `opennlp.tools.formats.frenchtreebank`.
//!
//! A French Treebank file is an XML document of `SENT` elements, each a
//! sentence whose words are `w` elements inside phrase elements.
//! [`ConstitDocument`] reads one into parses as OpenNLP's
//! `ConstitDocumentHandler` does under a SAX parser, and [`ConstitStream`]
//! reads documents one after another as `ConstitParseSampleStream` does
//! over the files [`french_treebank_files`] lists, the files of the
//! `frenchtreebank` factory's data directory.
//!
//! A parse's text is its words, each followed by a space, and each node's
//! type is the name of its element; a `SENT` is an `S` node and a `w` one
//! of its category. Every parse keeps the bytes of its `SENT` element in
//! the document and the bytes each word was read from
//! ([`ConstitParse::tokens`]).
//!
//! # Where OpenNLP's behaviour is surprising
//!
//! These are reproduced:
//!
//! - A word's type is its `cat` attribute followed by its `subcat`, but an
//!   empty or missing attribute keeps the value of the last word that had
//!   one, across sentences, for the whole document. Before any word has a
//!   `cat`, the type is `null` followed by the word's `catint`, or by its
//!   `subcat` (`nullnull` when there is none).
//! - A word is the text after the last tag inside its `w` element, trimmed.
//!   A `w` holding other `w` elements (a compound) is a node only when
//!   text follows its last inner word; otherwise its words are nodes of
//!   the phrase around it. A `w` with no text is no node.
//! - An element with no word inside it is no node, and an element named
//!   `TOP` is left out of the tree. One named `TK` counts as a token in
//!   the head indexes.
//! - A `SENT` inside a `SENT` starts the sentence again; the outer one
//!   ends with the inner one, and what comes after the inner one is not
//!   read.
//! - A document with no sentence ends the stream, and the documents after
//!   it are not read. A document that is not well-formed XML is an error
//!   and gives no parse, not even of the sentences before the error.
//!
//! Where this port differs, deliberately:
//!
//! - The document is decoded as `turbo_maxent::xml::decode` decodes it:
//!   UTF-8, US-ASCII or ISO-8859-1. Xerces also reads UTF-16 and other
//!   encodings. A namespace prefix with no declaration is not refused,
//!   which the JDK's namespace aware parser does.
//! - The message of an XML error is this crate's, not Xerces's; the class
//!   is the same (`IOException`).
//! - Spans are UTF-8 byte offsets where Java's are UTF-16 indices.
//! - The files of the data directory are listed as
//!   [`directory_files`](crate::directory::directory_files) lists them.

mod handler;

use std::borrow::Cow;
use std::fmt;
use std::path::{Path, PathBuf};

use turbo_analysis::Span;
use turbo_parser::{Parse, ParseError};

use crate::directory::{DirectoryError, directory_files};

/// The name of the format in OpenNLP's registry, for parse samples.
pub const FORMAT_NAME: &str = "frenchtreebank";

/// An error of the French Treebank reader.
///
/// [`FtbError::java_class`] is the class of the exception OpenNLP throws
/// in the same place.
#[derive(Debug)]
pub enum FtbError {
    /// The data directory could not be listed, or is not a directory.
    Directory(DirectoryError),
    /// A document is not well-formed XML, or not in an encoding the reader
    /// decodes: the SAX parser's exception, which the stream throws as an
    /// `IOException`.
    Xml(turbo_maxent::Error),
    /// A node could not be put in its parse. The handler's nodes always
    /// fit; this stands for the exception `Parse.insert` would throw.
    Parse { sentence: Span, error: ParseError },
    /// A document longer than `u32::MAX` bytes, which spans cannot index.
    TooLong { len: usize },
}

impl FtbError {
    /// The class of the exception OpenNLP throws in the same place, for
    /// the errors that have one.
    pub fn java_class(&self) -> Option<&'static str> {
        match self {
            FtbError::Directory(e) => e.java_class(),
            FtbError::Xml(_) => Some("java.io.IOException"),
            FtbError::Parse { error, .. } => error.java_class(),
            FtbError::TooLong { .. } => None,
        }
    }
}

impl fmt::Display for FtbError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            FtbError::Directory(e) => e.fmt(f),
            FtbError::Xml(e) => e.fmt(f),
            FtbError::Parse { error, .. } => error.fmt(f),
            FtbError::TooLong { len } => write!(f, "a document of {len} bytes is longer than spans can index"),
        }
    }
}

impl std::error::Error for FtbError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            FtbError::Directory(e) => Some(e),
            FtbError::Xml(e) => Some(e),
            FtbError::Parse { error, .. } => Some(error),
            FtbError::TooLong { .. } => None,
        }
    }
}

/// The files the `frenchtreebank` factory reads from its data directory,
/// in its order: every file of the directory (not of its
/// subdirectories), in the order of
/// [`directory_files`](crate::directory::directory_files).
///
/// # Errors
///
/// [`FtbError::Directory`] when `dir` is not a directory or cannot be
/// listed.
pub fn french_treebank_files(dir: &Path) -> Result<Vec<PathBuf>, FtbError> {
    directory_files(dir, false, None).map_err(FtbError::Directory)
}

/// Decodes a document's bytes as the XML reader does: UTF-8, with or
/// without a byte order mark, unless the XML declaration names US-ASCII or
/// ISO-8859-1.
///
/// # Errors
///
/// [`FtbError::Xml`] for another encoding or bytes the encoding does not
/// allow.
pub fn decode(bytes: &[u8]) -> Result<Cow<'_, str>, FtbError> {
    turbo_maxent::xml::decode(bytes).map_err(FtbError::Xml)
}

/// A parse read from a French Treebank document, with where it came from.
#[derive(Clone, Debug)]
pub struct ConstitParse {
    parse: Parse,
    source: Span,
    tokens: Vec<Span>,
}

impl ConstitParse {
    pub fn parse(&self) -> &Parse {
        &self.parse
    }

    pub fn into_parse(self) -> Parse {
        self.parse
    }

    /// The bytes of the sentence's `SENT` element in the document, from
    /// its start tag to the end of its end tag.
    pub fn source(&self) -> Span {
        self.source
    }

    /// For each token of the parse, in order, the bytes it was read from:
    /// the token itself when the text of its `w` element is written as
    /// it is read (no reference, CDATA section or carriage return), and
    /// otherwise the text from the first character read into it to the
    /// last, untrimmed.
    pub fn tokens(&self) -> &[Span] {
        &self.tokens
    }
}

/// The parses of one French Treebank document, as
/// `ConstitDocumentHandler` makes them.
pub struct ConstitDocument;

impl ConstitDocument {
    /// The parses of the decoded document `xml` ([`decode`]), in order.
    ///
    /// # Errors
    ///
    /// [`FtbError::Xml`] when the document is not well-formed XML, in which
    /// case OpenNLP keeps none of its parses, and [`FtbError::TooLong`]
    /// when it is too long for spans.
    pub fn read(xml: &str) -> Result<Vec<ConstitParse>, FtbError> {
        if u32::try_from(xml.len()).is_err() {
            return Err(FtbError::TooLong { len: xml.len() });
        }
        handler::read(xml)
    }
}

/// The parses of French Treebank documents read one after another, as
/// `ConstitParseSampleStream` reads them, with the index of the document
/// each came from. Each document is decoded with [`decode`], and the
/// spans of its parses are offsets into the decoded text, which is the
/// document's bytes when it is UTF-8 without a byte order mark.
///
/// A document that is not well-formed gives an error and the stream goes
/// on with the next. A document with no sentence ends the stream: OpenNLP
/// returns null for it, and the documents after it are not read.
pub struct ConstitStream<'a, I> {
    documents: I,
    index: usize,
    parses: std::vec::IntoIter<ConstitParse>,
    current: usize,
    done: bool,
    _bytes: std::marker::PhantomData<&'a [u8]>,
}

impl<'a, I: Iterator<Item = &'a [u8]>> ConstitStream<'a, I> {
    pub fn new(documents: impl IntoIterator<IntoIter = I>) -> ConstitStream<'a, I> {
        ConstitStream {
            documents: documents.into_iter(),
            index: 0,
            parses: Vec::new().into_iter(),
            current: 0,
            done: false,
            _bytes: std::marker::PhantomData,
        }
    }
}

impl<'a, I: Iterator<Item = &'a [u8]>> Iterator for ConstitStream<'a, I> {
    /// A parse and the index of its document, counting from 0.
    type Item = Result<(usize, ConstitParse), FtbError>;

    fn next(&mut self) -> Option<Self::Item> {
        if let Some(p) = self.parses.next() {
            return Some(Ok((self.current, p)));
        }
        if self.done {
            return None;
        }
        let Some(bytes) = self.documents.next() else {
            self.done = true;
            return None;
        };
        self.current = self.index;
        self.index += 1;
        let parses = match decode(bytes).and_then(|xml| ConstitDocument::read(&xml)) {
            Ok(parses) => parses,
            Err(e) => return Some(Err(e)),
        };
        self.parses = parses.into_iter();
        match self.parses.next() {
            Some(p) => Some(Ok((self.current, p))),
            None => {
                self.done = true;
                None
            }
        }
    }
}
