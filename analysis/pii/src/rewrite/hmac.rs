//! Keyed labels, OpenNLP's `HmacTokenizer`: the kind in capitals, `-`,
//! and the first hexadecimal digits of the HMAC-SHA-256 of the UTF-8 kind
//! name, a zero byte and the normalized value. The same key gives the
//! same label for a value in any text, so labels made with one key are
//! linkable and need the access control of the data they stand for.

use std::fmt;

use turbo_analysis::View;
use turbo_analysis_layers::{Pii, PiiStrategyKind};

use super::replace::EditOrigin;
use super::{PiiRewrite, RewriteError, kind_name};

/// A key or length the tokenizer cannot take.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum TokenizerError {
    /// Fewer than 32 key bytes.
    ShortKey(usize),
    /// A token length outside 4 to 64 hexadecimal digits.
    Length(usize),
    /// An empty value.
    EmptyValue,
}

impl fmt::Display for TokenizerError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            TokenizerError::ShortKey(n) => write!(f, "the key has {n} bytes, at least 32 are needed"),
            TokenizerError::Length(n) => write!(f, "the length must be between 4 and 64: {n}"),
            TokenizerError::EmptyValue => f.write_str("the value must not be empty"),
        }
    }
}

impl std::error::Error for TokenizerError {}

#[derive(Clone, PartialEq, Eq)]
pub struct HmacTokenizer {
    key: Vec<u8>,
    length: usize,
    refuse_overlaps: bool,
}

impl fmt::Debug for HmacTokenizer {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        // The key stays out of logs.
        f.debug_struct("HmacTokenizer")
            .field("length", &self.length)
            .field("refuse_overlaps", &self.refuse_overlaps)
            .finish_non_exhaustive()
    }
}

impl HmacTokenizer {
    /// Folds every parameter that changes what this strategy writes into
    /// `h`, for the step's config hash.
    pub(super) fn fold_config(&self, h: &mut turbo_analysis::Fnv1a) {
        // The key enters only as an HMAC under a fixed label, never as itself.
        h.bytes(&turbo_sha256::hmac_sha256(&self.key, b"turbo-pii:redacted config key").0);
        h.usize(self.length).u8(u8::from(self.refuse_overlaps));
    }

    pub const DEFAULT_LENGTH: usize = 16;
    pub const MIN_KEY_BYTES: usize = 32;

    /// A tokenizer writing 16 hexadecimal digits (64 bits).
    pub fn new(key: &[u8]) -> Result<HmacTokenizer, TokenizerError> {
        HmacTokenizer::with_length(key, HmacTokenizer::DEFAULT_LENGTH)
    }

    /// A tokenizer writing `length` hexadecimal digits, 4 to 64. Shorter
    /// tokens collide more often.
    pub fn with_length(key: &[u8], length: usize) -> Result<HmacTokenizer, TokenizerError> {
        if key.len() < HmacTokenizer::MIN_KEY_BYTES {
            return Err(TokenizerError::ShortKey(key.len()));
        }
        if !(4..=64).contains(&length) {
            return Err(TokenizerError::Length(length));
        }
        Ok(HmacTokenizer { key: key.to_vec(), length, refuse_overlaps: false })
    }

    /// This tokenizer, refusing overlapping mentions with
    /// [`RewriteError::Overlap`] as OpenNLP does.
    pub fn refusing_overlaps(mut self) -> HmacTokenizer {
        self.refuse_overlaps = true;
        self
    }

    /// Whether overlapping mentions are refused rather than coalesced.
    pub const fn refuses_overlaps(&self) -> bool {
        self.refuse_overlaps
    }

    /// Appends the token of `value` of the kind named `kind` to `out`.
    pub fn write_token(&self, kind: &str, value: &str, out: &mut String) -> Result<(), TokenizerError> {
        if value.is_empty() {
            return Err(TokenizerError::EmptyValue);
        }
        let mut message = Vec::with_capacity(kind.len() + 1 + value.len());
        message.extend_from_slice(kind.as_bytes());
        message.push(0);
        message.extend_from_slice(value.as_bytes());
        let mac = turbo_sha256::hmac_sha256(&self.key, &message).0;
        out.extend(kind.chars().map(|c| c.to_ascii_uppercase()));
        out.push('-');
        const HEX: &[u8; 16] = b"0123456789abcdef";
        out.extend(
            mac.iter()
                .flat_map(|b| [HEX[usize::from(b >> 4)], HEX[usize::from(b & 15)]])
                .take(self.length)
                .map(char::from),
        );
        Ok(())
    }

    /// The token of `value` of the kind named `kind`, such as
    /// `EMAIL-3f2a1c9d7e4b6a20`.
    pub fn token(&self, kind: &str, value: &str) -> Result<String, TokenizerError> {
        let mut out = String::new();
        self.write_token(kind, value, &mut out)?;
        Ok(out)
    }

    /// Replaces every mention of `pii`, a layer over `src`, with its
    /// token, as a view derived from `src`.
    pub fn rewrite(&self, src: &View, pii: &Pii) -> Result<PiiRewrite, RewriteError> {
        if let Some(m) = pii.mentions().find(|m| m.normalized.is_empty()) {
            return Err(RewriteError::EmptyValue(m.span));
        }
        PiiRewrite::derive(
            src,
            pii,
            |_| self.refuse_overlaps,
            0,
            |c, out| {
                let m = c.merged;
                self.write_token(kind_name(&pii.custom_kinds, m.kind), m.normalized, out)
                    .map_err(|_| RewriteError::EmptyValue(m.span))?;
                Ok(EditOrigin::only(PiiStrategyKind::Hmac))
            },
        )
    }
}

#[cfg(test)]
mod tests {
    use turbo_analysis::{Span, ViewId};
    use turbo_analysis_layers::{Pii, PiiEvidence, PiiKind, PiiMention};

    use super::*;
    use crate::rewrite::{PiiAuditReport, source_view};

    #[test]
    fn empty_normalized_mention_is_refused_in_hmac_rewrite_and_skipped_in_audit_samples() {
        let mut pii = Pii::new(ViewId::SOURCE);
        let detector = pii.detector("test", "1").unwrap();
        pii.push(PiiMention {
            span: Span::new(0, 5),
            kind: PiiKind::Email,
            detector,
            evidence: &[PiiEvidence::Shape],
            score: None,
            normalized: "",
        });
        let tokenizer = HmacTokenizer::new(&[0x42; 32]).unwrap();
        assert_eq!(
            tokenizer.rewrite(&source_view("hello world"), &pii).map(|r| r.into_text()),
            Err(RewriteError::EmptyValue(Span::new(0, 5)))
        );
        let report = PiiAuditReport::of(&pii, &tokenizer, 3);
        assert_eq!(report.total, 1);
        let email = report.kind(PiiKind::Email).unwrap();
        assert_eq!((email.mentions, email.distinct), (1, 1));
        assert!(email.samples.is_empty());
    }
}
