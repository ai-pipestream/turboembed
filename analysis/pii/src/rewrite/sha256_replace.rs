//! Unkeyed SHA-256 hex digest replacement ([`Sha256Tokenizer`]).
//!
//! By default, each mention is replaced with the 64 lowercase hexadecimal
//! digits of the SHA-256 digest ([`crate::checksum::sha256::digest`]) of its
//! normalized UTF-8 value. Callers may also request a shorter prefix (4 to
//! 64 hex digits) or prepend the uppercase kind tag (`EMAIL-<hex>`) or a
//! custom prefix.
//!
//! # Security warning
//!
//! Plain SHA-256 uses no secret key: any party hashing the same normalized
//! value obtains the exact same digest. As a consequence, unkeyed SHA-256
//! digests are linkable across documents and datasets, and low-entropy or
//! bounded-domain identifiers (such as 9-digit US SSNs, 10-digit phone
//! numbers, or known e-mail lists) can be recovered by brute-force search or
//! precomputed rainbow tables. Prefer [`super::HmacTokenizer`] with a secret
//! key of at least 32 bytes whenever unkeyed interoperability is not
//! explicitly required.

use turbo_analysis_layers::Pii;

use super::{PiiRewrite, RewriteError, TokenizerError, kind_name};
use crate::checksum::sha256;

const HEX: &[u8; 16] = b"0123456789abcdef";

/// Replaces each mention with the lowercase hexadecimal SHA-256 digest of
/// its normalized value.
///
/// See the [module documentation](self) for the linkability and brute-force
/// risks of unkeyed SHA-256 over low-entropy identifiers; prefer
/// [`super::HmacTokenizer`] when a secret key can be held.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Sha256Tokenizer {
    length: usize,
    kind_prefix: bool,
    prefix: String,
    refuse_overlaps: bool,
}

/// Type alias for [`Sha256Tokenizer`].
pub type Sha256Replacer = Sha256Tokenizer;

impl Default for Sha256Tokenizer {
    fn default() -> Sha256Tokenizer {
        Sha256Tokenizer::new()
    }
}

impl Sha256Tokenizer {
    /// The full SHA-256 digest length in hexadecimal digits (256 bits).
    pub const DEFAULT_LENGTH: usize = 64;

    /// A tokenizer writing all 64 lowercase hexadecimal digits of the
    /// SHA-256 digest of each mention's normalized value.
    pub const fn new() -> Sha256Tokenizer {
        Sha256Tokenizer {
            length: Sha256Tokenizer::DEFAULT_LENGTH,
            kind_prefix: false,
            prefix: String::new(),
            refuse_overlaps: false,
        }
    }

    /// A tokenizer writing the first `length` lowercase hexadecimal digits
    /// of the SHA-256 digest, between 4 and 64 inclusive.
    pub fn with_length(length: usize) -> Result<Sha256Tokenizer, TokenizerError> {
        if !(4..=64).contains(&length) {
            return Err(TokenizerError::Length(length));
        }
        Ok(Sha256Tokenizer { length, kind_prefix: false, prefix: String::new(), refuse_overlaps: false })
    }

    /// Prepends the uppercase kind name and `-` (such as `EMAIL-`) before
    /// the hexadecimal digest.
    pub fn with_kind_prefix(mut self) -> Sha256Tokenizer {
        self.kind_prefix = true;
        self
    }

    /// Prepends `prefix` before the output (and before the kind prefix if
    /// [`Sha256Tokenizer::with_kind_prefix`] is also enabled).
    pub fn with_prefix(mut self, prefix: impl Into<String>) -> Sha256Tokenizer {
        self.prefix = prefix.into();
        self
    }

    /// Number of hexadecimal digits written per digest (4 to 64).
    pub const fn length(&self) -> usize {
        self.length
    }

    /// Whether the uppercase kind name and `-` are prepended.
    pub const fn has_kind_prefix(&self) -> bool {
        self.kind_prefix
    }

    /// The custom prefix prepended to each output.
    pub fn prefix(&self) -> &str {
        &self.prefix
    }

    /// This tokenizer, refusing overlapping mentions with
    /// [`RewriteError::Overlap`].
    pub fn refusing_overlaps(mut self) -> Sha256Tokenizer {
        self.refuse_overlaps = true;
        self
    }

    /// Whether overlapping mentions are refused rather than coalesced.
    pub const fn refuses_overlaps(&self) -> bool {
        self.refuse_overlaps
    }

    /// Appends the formatted SHA-256 digest of `value` of the kind named
    /// `kind` to `out`.
    pub fn write_digest(&self, kind: &str, value: &str, out: &mut String) -> Result<(), TokenizerError> {
        if value.is_empty() {
            return Err(TokenizerError::EmptyValue);
        }
        let hash = sha256::digest(value.as_bytes());
        out.push_str(&self.prefix);
        if self.kind_prefix {
            out.extend(kind.chars().map(|c| c.to_ascii_uppercase()));
            out.push('-');
        }
        out.extend(
            hash.iter()
                .flat_map(|b| [HEX[usize::from(b >> 4)], HEX[usize::from(b & 15)]])
                .take(self.length)
                .map(char::from),
        );
        Ok(())
    }

    /// The formatted SHA-256 digest of `value` of the kind named `kind`.
    pub fn digest(&self, kind: &str, value: &str) -> Result<String, TokenizerError> {
        let mut out = String::new();
        self.write_digest(kind, value, &mut out)?;
        Ok(out)
    }

    /// Replaces every mention of `pii`, a layer over `text`, with the
    /// SHA-256 hex digest of its normalized value.
    pub fn rewrite(&self, text: &str, pii: &Pii) -> Result<PiiRewrite, RewriteError> {
        if let Some(m) = pii.mentions().find(|m| m.normalized.is_empty()) {
            return Err(RewriteError::EmptyValue(m.span));
        }
        let mentions: Vec<_> = if self.refuse_overlaps {
            pii.mentions().collect()
        } else {
            PiiRewrite::coalesce_clusters(text, &pii.custom_kinds, pii.mentions())?
        };
        PiiRewrite::try_replace(text, pii, mentions, |m, out| {
            self.write_digest(kind_name(&pii.custom_kinds, m.kind), m.normalized, out)
                .map_err(|_| RewriteError::EmptyValue(m.span))
        })
    }
}
