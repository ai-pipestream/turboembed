//! Personal and sensitive identifiers in text, ported from OpenNLP's
//! `opennlp.tools.pii` package onto the analysis API.
//!
//! - [`PiiDetector`] runs [`Scanner`]s, one per OpenNLP extractor class,
//!   or a [`Pack`] of them, together with optional [`AllowList`] and
//!   [`DenyList`] rules, and writes the `turbo:pii` layer ([`Pii`]):
//!   one item per mention with its kind, its detector, its typed
//!   [`PiiEvidence`] facts (the [`PiiConfidence`] tier derives from
//!   them), an optional [`PiiScore`] and its normalized form. Every
//!   candidate it considered, kept or not, is the `turbo:pii-candidates`
//!   layer ([`PiiCandidates`]) with its [`PiiDisposition`].
//! - [`lists`] holds caller allow lists (terms, files, and regex-free
//!   shape/glob patterns, scoped globally or per [`PiiKind`]) and deny
//!   lists (Aho-Corasick glossary matching and bounded fuzzy matching).
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
pub mod lists;
pub mod rewrite;
mod scan;
mod tables;
pub mod text;
pub mod window;

pub use detector::{DENY_LIST_DETECTOR, Pack, PiiDetector};
pub use hits::{PiiSink, SinkError};
pub use lists::{AllowList, Case, DenyList, ListError, Pattern, PatternKind};
pub use scan::{BUILTIN_VERSION, CustomExtractor, Extractor, Scanner, ScannerError};
pub use turbo_analysis_layers::{
    AllowRule, AllowScope, ChecksumAlgorithm, CustomKindId, CustomKinds, DetectorError, EvidenceList, KindError, Pii,
    PiiCandidate, PiiCandidates, PiiConfidence, PiiDetectorRef, PiiDetectors, PiiDisposition, PiiEvidence, PiiKind,
    PiiKinds, PiiMention, PiiScore, PiiScoreKind, RejectReason,
};
pub use window::LabelPolicy;
