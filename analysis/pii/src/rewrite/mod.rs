//! What a caller does with the mentions it found, OpenNLP's `Masker`,
//! `MaskPolicy`, `MaskPolicies`, `Pseudonymizer`, `HmacTokenizer`,
//! `PiiRewrite` and `PiiAuditReport`.
//!
//! - [`mask`] overwrites the characters of each span in place, so the
//!   text keeps its length and every offset stays valid.
//! - [`MaskedView`] writes the masked text as the `masked` view, whose
//!   alignment maps each masked character to its source character.
//! - [`Pseudonymizer`] and [`HmacTokenizer`] replace each mention with a
//!   label, numbered per text or keyed and stable across texts;
//!   [`PiiRewrite`] holds the new text and maps offsets into it.
//! - [`PiiAuditReport`] counts mentions by kind with keyed tokens in
//!   place of values.

mod audit;
mod hmac;
mod mask;
mod pseudonym;
mod replace;
mod view;

pub use audit::{KindReport, PiiAuditReport};
pub use hmac::{HmacTokenizer, TokenizerError};
pub use mask::{MaskError, MaskPolicy, mask, mask_all, mask_by_kind};
pub use pseudonym::Pseudonymizer;
pub use replace::{PiiRewrite, RewriteError};
pub use view::{MASKED, MaskedView, mask_into, mask_view};
