//! XML surface for `turbo-formats` backed by [`turbo_maxent::xml`], with a
//! `Clone + Eq` [`Error`] carried by [`MascError`](crate::masc::MascError).

use std::borrow::Cow;
use std::fmt;

pub use turbo_maxent::xml::{Attribute, Handler, Names, parse, parse_with};

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

impl turbo_maxent::xml::XmlError for Error {
    fn xml(offset: usize, message: &'static str) -> Self {
        Error::Xml { offset, message }
    }

    fn xml_encoding(name: String) -> Self {
        Error::XmlEncoding(name)
    }
}

pub type Result<T> = std::result::Result<T, Error>;

/// Decodes the document's bytes to text: UTF-8 (with or without a byte
/// order mark) unless the XML declaration names US-ASCII or ISO-8859-1.
pub fn decode(bytes: &[u8]) -> Result<Cow<'_, str>> {
    turbo_maxent::xml::decode_as(bytes)
}
