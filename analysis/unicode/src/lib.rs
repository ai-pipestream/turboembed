//! Unicode 17.0 character data and the operations the analysis crates
//! need on it: general category, the properties the normalizers test,
//! scripts, case mapping and folding, the normalization forms, confusable
//! prototypes and emoji sequences. The tables are generated from the
//! Unicode Character Database files in `ucd/` by `turbo-unicode-gen` and
//! committed; nothing is read at run time.
//!
//! The modules the segmentation ports use:
//!
//! - [`java`]: `java.lang.Character`'s classes (`isLetter`, `isDigit`,
//!   `isUpperCase`, `isAlphabetic`, `isWhitespace`, `toLowerCase` and the
//!   rest), so features and splits match OpenNLP's.
//! - [`whitespace`]: OpenNLP's two whitespace definitions, chosen by a
//!   [`WhitespaceMode`] value.
//! - [`word_break`]: the UAX #29 `Word_Break` and `Extended_Pictographic`
//!   properties.
//! - [`utf16`]: a cursor that steps over UTF-8 text one UTF-16 code unit
//!   at a time, the unit Java's algorithms work in.
//! - [`modified_utf8`]: Java's modified UTF-8, the encoding OpenNLP model
//!   files store their feature strings in.

pub mod java;
pub mod modified_utf8;
pub mod utf16;
pub mod whitespace;
pub mod word_break;

mod trie;

/// The generated tables; `analysis/unicode/ucd/README.md` has the command
/// that rebuilds them.
#[rustfmt::skip]
mod tables {
    pub(crate) mod java;
    pub(crate) mod word_break;
}

pub use java::{GeneralCategory, Script};
pub use utf16::Utf16Cursor;
pub use whitespace::WhitespaceMode;
pub use word_break::WordBreak;

/// The Unicode version the tables were generated from.
pub const UNICODE_VERSION: (u8, u8, u8) = (17, 0, 0);
