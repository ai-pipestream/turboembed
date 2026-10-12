//! Unicode 18.0 character data and the operations the analysis crates
//! need on it: general category, the properties the normalizers test,
//! scripts, case mapping and folding, the normalization forms, confusable
//! prototypes and emoji sequences.
//!
//! The tables in `tables/` are generated from the Unicode Character
//! Database files in `ucd/` by `turbo-unicode-gen` and committed; nothing
//! is read at run time and the crate depends on the standard library only.
//! A value stored per code point (general category, script, combining
//! class, property bits) is read from a two-stage table in two memory
//! loads. A mapping (case mappings, decompositions, compositions,
//! confusable prototypes, emoji names) is a sorted key array searched by
//! binary search, and a bit in the per-code-point properties says whether
//! a code point has an entry, so the search runs only for code points that
//! have one.
//!
//! [`java`] reproduces the `java.lang.Character` and `String.toLowerCase`
//! behaviour the OpenNLP normalizers rely on, on this crate's Unicode data
//! rather than the JDK's. [`java::code_point`] has the same classes on a
//! code point given as `u32`, for the segmentation ports, which walk text
//! in UTF-16 code units as OpenNLP does and so meet lone surrogates.
//!
//! The modules the segmentation ports use besides [`java`]:
//!
//! - [`whitespace`]: OpenNLP's two whitespace definitions, chosen by a
//!   [`WhitespaceMode`] value.
//! - [`word_break`]: the UAX #29 `Word_Break` and `Extended_Pictographic`
//!   properties.
//! - [`utf16`]: a cursor that steps over UTF-8 text one UTF-16 code unit
//!   at a time, the unit Java's algorithms work in.
//! - [`modified_utf8`]: Java's modified UTF-8, the encoding OpenNLP model
//!   files store their feature strings in.

pub mod casefold;
pub mod category;
pub mod chars;
pub mod confusables;
pub mod emoji;
pub mod java;
pub mod modified_utf8;
pub mod normalize;
pub mod props;
pub mod script;
pub mod utf16;
pub mod whitespace;
pub mod word_break;

mod ascii;
mod tables;
mod trie;

pub use category::GeneralCategory;
pub use script::Script;
pub use utf16::Utf16Cursor;
pub use whitespace::WhitespaceMode;
pub use word_break::WordBreak;

/// The Unicode version the tables were generated from.
pub const UNICODE_VERSION: (u8, u8, u8) = (18, 0, 0);

const _: () = assert!(
    UNICODE_VERSION.0 == tables::UNICODE_VERSION.0
        && UNICODE_VERSION.1 == tables::UNICODE_VERSION.1
        && UNICODE_VERSION.2 == tables::UNICODE_VERSION.2,
    "UNICODE_VERSION differs from the version of the generated tables"
);
