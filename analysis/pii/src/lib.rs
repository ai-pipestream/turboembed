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
//! - [`PiiRiskScorer`] writes the `turbo:pii-risk` layer ([`PiiRisk`]):
//!   one score and level per document from the `turbo:pii` layer alone.
//! - [`rewrite`] masks, pseudonymizes and tokenizes the mentions of a
//!   text, and reports on them without their values.
//!
//! Every offset is a UTF-8 byte offset into the text, on a character
//! boundary. docs/pii.md describes the rules.

pub mod checksum;
mod detector;
mod hits;
pub mod lists;
pub mod nemotron;
pub mod rewrite;
mod risk;
mod scan;
mod tables;
pub mod text;
pub mod window;

pub use detector::{DENY_LIST_DETECTOR, Pack, PiiDetector};
pub use hits::{PiiSink, SinkError};
pub use lists::{AllowList, Case, DenyList, ListError, Pattern, PatternKind};
pub use nemotron::{
    ModelCandidate, ModelProvenance, ModelResolution, NEMOTRON_TYPES, nemotron_custom_kinds, nemotron_wire_name,
    normalize_model_value, register_nemotron_kinds, resolve_model_with_rules, resolve_nemotron_kind,
};
pub use risk::PiiRiskScorer;
pub use scan::{BUILTIN_VERSION, CustomExtractor, Extractor, Scanner, ScannerError};
pub use turbo_analysis_layers::{
    AllowRule, AllowScope, ChecksumAlgorithm, CustomKindId, CustomKinds, DetectorError, EvidenceList, KindError, Pii,
    PiiCandidate, PiiCandidates, PiiConfidence, PiiDetectorRef, PiiDetectors, PiiDisposition, PiiEdit, PiiEdits,
    PiiEvidence, PiiKind, PiiKinds, PiiMention, PiiPlan, PiiRule, PiiScore, PiiScoreKind, PiiStrategyKind,
    RejectReason,
};
pub use turbo_analysis_layers::{
    PiiRisk, PiiRiskKind, PiiRiskLevel, PiiSeverity, PiiSeverityTable, PiiSuggestedRewrite, confidence_weight,
};
pub use window::LabelPolicy;
