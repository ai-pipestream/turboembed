//! XML surface for `turbo-formats` backed by [`turbo_maxent::xml`], with a
//! `Clone + Eq` [`Error`] carried by [`MascError`](crate::masc::MascError).
//!
//! The parser is `turbo_maxent`'s, unchanged, so its own callers (the
//! dictionary, corpora, sequence and recipe readers) compile it exactly as
//! before. A handler here returns this crate's [`Error`]; [`parse_with`]
//! keeps the handler's own failure as it is and converts a parser failure
//! once, where the parse returns.

use std::borrow::Cow;
use std::fmt;

pub use turbo_maxent::xml::{Attribute, Names};

/// XML parsing or decoding failed at a byte offset or named encoding.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Error {
    Xml { offset: usize, message: &'static str },
    XmlEncoding(String),
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Error::Xml { offset, message } => write!(f, "XML at byte {offset}: {message}"),
            Error::XmlEncoding(name) => write!(f, "XML encoding {name:?} is not UTF-8, US-ASCII or ISO-8859-1"),
        }
    }
}

impl std::error::Error for Error {}

impl From<turbo_maxent::Error> for Error {
    /// The parser and the decoder fail only with these two variants; any
    /// other comes from a handler, which [`parse_with`] keeps apart.
    fn from(e: turbo_maxent::Error) -> Self {
        match e {
            turbo_maxent::Error::Xml { offset, message } => Error::Xml { offset, message },
            turbo_maxent::Error::XmlEncoding(name) => Error::XmlEncoding(name.into()),
            _ => Error::Xml { offset: 0, message: "the XML reader failed" },
        }
    }
}

pub type Result<T> = std::result::Result<T, Error>;

/// Receives the document's content in order, as
/// [`turbo_maxent::xml::Handler`] does, failing with this crate's [`Error`].
pub trait Handler {
    /// An element starts. `name` is its name as [`Names`] asked for.
    fn start_element(&mut self, name: &str, attributes: &[Attribute<'_>], offset: usize) -> Result<()>;
    /// An element ends (also right after `start_element` for an
    /// empty-element tag).
    fn end_element(&mut self, name: &str, offset: usize) -> Result<()>;
    /// Character data inside the root element.
    fn characters(&mut self, text: &str);
    /// See [`turbo_maxent::xml::Handler::element_closed`].
    fn element_closed(&mut self, end: usize) {
        let _ = end;
    }
    /// See [`turbo_maxent::xml::Handler::text_run`].
    fn text_run(&mut self, start: usize, end: usize) {
        let _ = (start, end);
    }
    /// See [`turbo_maxent::xml::Handler::cdata_section`].
    fn cdata_section(&mut self, start: usize, end: usize) {
        let _ = (start, end);
    }
    /// See [`turbo_maxent::xml::Handler::comment`].
    fn comment(&mut self, start: usize, end: usize) {
        let _ = (start, end);
    }
    /// See [`turbo_maxent::xml::Handler::processing_instruction`].
    fn processing_instruction(&mut self, target: &str, start: usize, end: usize) {
        let _ = (target, start, end);
    }
}

/// Hands the parser's calls to a [`Handler`], holding the first failure
/// the handler gives so it comes back as it was.
struct Adapter<'h, H> {
    handler: &'h mut H,
    failed: Option<Error>,
}

impl<H> Adapter<'_, H> {
    fn keep(&mut self, result: Result<()>) -> turbo_maxent::Result<()> {
        result.map_err(|e| {
            self.failed = Some(e);
            turbo_maxent::Error::Xml { offset: 0, message: "the handler failed" }
        })
    }
}

impl<H: Handler> turbo_maxent::xml::Handler for Adapter<'_, H> {
    fn start_element(&mut self, name: &str, attributes: &[Attribute<'_>], offset: usize) -> turbo_maxent::Result<()> {
        let result = self.handler.start_element(name, attributes, offset);
        self.keep(result)
    }
    fn end_element(&mut self, name: &str, offset: usize) -> turbo_maxent::Result<()> {
        let result = self.handler.end_element(name, offset);
        self.keep(result)
    }
    fn characters(&mut self, text: &str) {
        self.handler.characters(text);
    }
    fn element_closed(&mut self, end: usize) {
        self.handler.element_closed(end);
    }
    fn text_run(&mut self, start: usize, end: usize) {
        self.handler.text_run(start, end);
    }
    fn cdata_section(&mut self, start: usize, end: usize) {
        self.handler.cdata_section(start, end);
    }
    fn comment(&mut self, start: usize, end: usize) {
        self.handler.comment(start, end);
    }
    fn processing_instruction(&mut self, target: &str, start: usize, end: usize) {
        self.handler.processing_instruction(target, start, end);
    }
}

/// Parses a decoded document with [`turbo_maxent::xml::parse_with`],
/// calling `handler` with the `names` asked for.
pub fn parse_with(text: &str, names: Names, handler: &mut impl Handler) -> Result<()> {
    let mut adapter = Adapter { handler, failed: None };
    turbo_maxent::xml::parse_with(text, names, &mut adapter)
        .map_err(|e| adapter.failed.take().unwrap_or_else(|| Error::from(e)))
}

/// Parses a decoded document, calling `handler` with local names.
pub fn parse(text: &str, handler: &mut impl Handler) -> Result<()> {
    parse_with(text, Names::Local, handler)
}

/// Decodes the document's bytes to text: UTF-8 (with or without a byte
/// order mark) unless the XML declaration names US-ASCII or ISO-8859-1.
pub fn decode(bytes: &[u8]) -> Result<Cow<'_, str>> {
    turbo_maxent::xml::decode(bytes).map_err(Error::from)
}
