//! Numbers in text, ported from the OpenNLP numeric add-on onto the
//! analysis API.
//!
//! - [`scan`] reads one number at a position, as OpenNLP's `NumberScan`
//!   does: ASCII digits, strict grouping in a [`NumberNotation`], an
//!   optional fraction and an optional scale marker, with the exact
//!   decimal value.
//! - [`NumberScanner`] walks a whole text and writes every number it
//!   finds as the `turbo:numbers` layer ([`Numbers`]).
//!
//! Every offset is a UTF-8 byte offset into the text, on a character
//! boundary. docs/numbers.md describes the rules.

pub mod annotator;
pub mod scan;

pub use annotator::NumberScanner;
pub use scan::{ScanResult, parse};
pub use turbo_analysis_layers::{DecimalRef, Multiplier, NumberNotation, Numbers, ScaleMarker};
