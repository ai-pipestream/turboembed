//! What a caller does with the mentions it found, OpenNLP's `Masker`,
//! `MaskPolicy`, `MaskPolicies`, `Pseudonymizer`, `HmacTokenizer`,
//! `PiiRewrite` and `PiiAuditReport`, plus format-string redaction
//! ([`FormatRedactor`]), fixed static replacement ([`StaticReplacer`]),
//! unkeyed SHA-256 digests ([`Sha256Tokenizer`]), caller lookup tables
//! ([`MapReplacer`]), partial character retention ([`Truncate`]), initials
//! abbreviation ([`Abbreviator`]), and composable per-kind policies
//! ([`RewritePolicy`], [`RewriteStrategy`]).
//!
//! - [`mask`] overwrites the characters of each span in place, so the
//!   text keeps its length and every offset stays valid.
//! - [`MaskedView`] writes the masked text as the `masked` view, whose
//!   alignment maps each masked character to its source character.
//! - [`Pseudonymizer`], [`HmacTokenizer`], [`FormatRedactor`],
//!   [`StaticReplacer`], [`Sha256Tokenizer`], [`MapReplacer`],
//!   [`Truncate`], [`Abbreviator`] and [`RewritePolicy`] replace each
//!   mention and return a [`PiiRewrite`] holding the new text, its label
//!   layer, and `map_offset` / `map_span` into the new text.
//! - [`PiiAuditReport`] counts mentions by kind with keyed tokens in
//!   place of values.

mod abbreviate;
mod audit;
mod format;
mod hmac;
mod map;
mod mask;
mod policy;
mod pseudonym;
mod replace;
mod sha256_replace;
mod truncate;
mod view;

pub use abbreviate::Abbreviator;
pub use audit::{KindReport, PiiAuditReport};
pub use format::{FormatError, FormatRedactor, StaticReplacer};
pub use hmac::{HmacTokenizer, TokenizerError};
pub use map::{MapError, MapReplacer, MapTarget, TSV_DELIMITER};
pub use mask::{MaskError, MaskPolicy, mask, mask_all, mask_by_kind};
pub use policy::{RewritePolicy, RewriteStrategy};
pub use pseudonym::Pseudonymizer;
pub use replace::{PiiRewrite, RewriteError};
pub use sha256_replace::{Sha256Replacer, Sha256Tokenizer};
pub use truncate::{KeepSide, Truncate};
pub use view::{MASKED, MaskedView, mask_into, mask_view};
