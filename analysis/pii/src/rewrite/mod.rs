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
//!   mention and return a [`PiiRewrite`]: a view derived from the text
//!   with an alignment both ways, its label layer, and its
//!   `turbo:pii-edits` layer.
//! - [`RedactedView`] is the pipeline step writing that view as
//!   `redacted` with its edits.
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
mod redacted;
mod replace;
mod sha256_replace;
mod state;
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
pub use redacted::{REDACTED, RedactedView};
pub use replace::{PiiRewrite, RewriteError};
pub use sha256_replace::{Sha256Replacer, Sha256Tokenizer};
pub use truncate::{KeepSide, Truncate};
pub use view::{MASKED, MaskedView, mask_into, mask_view};

use turbo_analysis::{Document, Span, View};

/// `text` as a source view, for rewriting a text that is not in a
/// document: the view has [`turbo_analysis::ViewId::SOURCE`] for its id,
/// so a pii layer over the source applies to it.
pub fn source_view(text: &str) -> View {
    Document::new(text).get::<View>().expect("a document has its source view").clone()
}
use turbo_analysis_layers::{CustomKinds, PiiKind};

/// The offset `rel` bytes into `span`, which holds it: `rel` is less than
/// the span's length, a `u32`, so the sum is a `u32` too.
pub(crate) fn absolute(span: Span, rel: usize) -> u32 {
    span.start + u32::try_from(rel).expect("an offset inside a u32 span")
}

/// The name of `kind` in `kinds`, the table of the layer it came from. Every
/// rewrite checks its mentions' kinds against that table first, so an
/// unknown kind never reaches a label writer; it writes no name there.
pub(crate) fn kind_name(kinds: &CustomKinds, kind: PiiKind) -> &str {
    kinds.name(kind).unwrap_or("")
}
