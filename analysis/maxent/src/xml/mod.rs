//! The XML reader the dictionary format and the feature generator
//! descriptors (`turbo-sequence`) need, in the style of a SAX parser: it
//! walks the document once and calls a [`Handler`] for each element
//! start, element end and run of character data.
//!
//! It supports elements, attributes in single or double quotes, the five
//! predefined entities, decimal and hexadecimal character references,
//! CDATA sections, comments, processing instructions and the XML
//! declaration (the last three are skipped). Line ends are normalized to
//! `\n` and attribute values have their whitespace normalized, as XML 1.0
//! requires. A namespace prefix is dropped from element and attribute
//! names, and `xmlns` attributes are not reported, which is what a
//! namespace-aware SAX parser hands OpenNLP's dictionary reader as local
//! names. [`parse_with`] can report the qualified names instead, which is
//! what a handler that reads SAX's `qName` sees (OpenNLP's MASC readers,
//! in `turbo-formats`).
//!
//! A document type declaration is an error, as it is for OpenNLP, which
//! sets `disallow-doctype-decl`. Other well-formedness errors (mismatched
//! tags, a second root, text outside the root, a character XML does not
//! allow, an undefined entity) are errors naming the byte offset in the
//! document's UTF-8 text.
//!
//! A handler that builds a tree, as a DOM parser does, can also take the
//! byte ranges of elements, text runs, CDATA sections, comments and
//! processing instructions: [`Handler`]'s provided methods, which do
//! nothing unless a handler overrides them.

use std::borrow::Cow;
use std::collections::HashSet;

use crate::error::{Error, Result};

/// One attribute of an element: its local name and its value with
/// references resolved.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Attribute<'a> {
    pub name: &'a str,
    pub value: Cow<'a, str>,
}

/// Receives the document's content in order.
pub trait Handler {
    /// An element starts. `name` is its local name.
    fn start_element(&mut self, name: &str, attributes: &[Attribute<'_>], offset: usize) -> Result<()>;
    /// An element ends (also called right after `start_element` for an
    /// empty-element tag).
    fn end_element(&mut self, name: &str, offset: usize) -> Result<()>;
    /// Character data inside the root element. One run of text may arrive
    /// in several calls.
    fn characters(&mut self, text: &str);

    /// The element whose end tag (or empty-element tag) was just reported
    /// to [`Handler::end_element`] ends before byte `end`.
    fn element_closed(&mut self, end: usize) {
        let _ = end;
    }

    /// A run of text inside the root element ends: its characters came in
    /// the [`Handler::characters`] calls since the last markup, and its
    /// bytes, references included, are `start..end`. A DOM builder makes
    /// one text node of each run.
    fn text_run(&mut self, start: usize, end: usize) {
        let _ = (start, end);
    }

    /// A CDATA section ends: its characters came in the
    /// [`Handler::characters`] calls since the last markup, and the
    /// section, `<![CDATA[` and `]]>` included, is `start..end`.
    fn cdata_section(&mut self, start: usize, end: usize) {
        let _ = (start, end);
    }

    /// A comment, inside or outside the root element, at `start..end`.
    fn comment(&mut self, start: usize, end: usize) {
        let _ = (start, end);
    }

    /// A processing instruction other than the XML declaration, inside or
    /// outside the root element, at `start..end`; `target` is its name.
    fn processing_instruction(&mut self, target: &str, start: usize, end: usize) {
        let _ = (target, start, end);
    }
}

/// The names a [`Handler`] is given for elements and attributes.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum Names {
    /// The local name, without a namespace prefix: SAX's `localName`.
    Local,
    /// The name as written, prefix and all: SAX's `qName`.
    Qualified,
}

/// Parses a decoded document, calling `handler` for its content with
/// local names.
pub fn parse(text: &str, handler: &mut impl Handler) -> Result<()> {
    parse_with(text, Names::Local, handler)
}

/// How many attributes a start tag holds before the names given twice are
/// looked for in a hash set rather than by scanning those before.
const SCANNED_ATTRIBUTES: usize = 16;

mod decode;
mod parser;
mod text;
pub use decode::decode;
pub use parser::parse_with;
use text::*;
#[cfg(test)]
mod tests;

#[cfg(test)]
mod boundary_tests;
