//! Personal and sensitive identifiers in text, ported from OpenNLP's
//! `opennlp.tools.pii` package onto the analysis API.
//!
//! - [`PiiDetector`] runs [`Scanner`]s, one per OpenNLP extractor class,
//!   or a [`Pack`] of them, and writes the `turbo:pii` layer ([`Pii`]):
//!   one item per mention with its kind, its normalized form and a
//!   [`PiiConfidence`].
//! - [`checksum`] has the checks the detectors validate with, written
//!   here: Luhn, mod 97, SHA-256, HMAC-SHA-256, Keccak-256, base58check
//!   and bech32.
//! - [`rewrite`] masks, pseudonymizes and tokenizes the mentions of a
//!   text, and reports on them without their values.
//!
//! Every offset is a UTF-8 byte offset into the text, on a character
//! boundary. docs/pii.md describes the rules.

pub mod checksum;
mod detector;
mod hits;
pub mod rewrite;
mod scan;
mod tables;
pub mod text;

pub use detector::{Pack, PiiDetector};
pub use hits::{PiiSink, SinkError};
pub use scan::{CustomExtractor, Extractor, Scanner, ScannerError};
pub use turbo_analysis_layers::{Pii, PiiConfidence, PiiKind, PiiKinds, PiiMention};
