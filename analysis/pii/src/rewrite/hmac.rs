//! Keyed labels, OpenNLP's `HmacTokenizer`: the kind in capitals, `-`,
//! and the first hexadecimal digits of the HMAC-SHA-256 of the UTF-8 kind
//! name, a zero byte and the normalized value. The same key gives the
//! same label for a value in any text, so labels made with one key are
//! linkable and need the access control of the data they stand for.

use std::fmt;

use turbo_analysis_layers::{Pii, PiiKind};

use super::{PiiRewrite, RewriteError};
use crate::checksum::sha256;

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

#[derive(Clone)]
pub struct HmacTokenizer {
    key: Vec<u8>,
    length: usize,
}

impl fmt::Debug for HmacTokenizer {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        // The key stays out of logs.
        f.debug_struct("HmacTokenizer").field("length", &self.length).finish_non_exhaustive()
    }
}

impl HmacTokenizer {
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
        Ok(HmacTokenizer { key: key.to_vec(), length })
    }

    /// Appends the token of `value` of `kind` to `out`.
    pub fn write_token(&self, kind: PiiKind, value: &str, out: &mut String) -> Result<(), TokenizerError> {
        if value.is_empty() {
            return Err(TokenizerError::EmptyValue);
        }
        let mut message = Vec::with_capacity(kind.name().len() + 1 + value.len());
        message.extend_from_slice(kind.name().as_bytes());
        message.push(0);
        message.extend_from_slice(value.as_bytes());
        let mac = sha256::hmac(&self.key, &message);
        out.extend(kind.name().chars().map(|c| c.to_ascii_uppercase()));
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

    /// The token of `value` of `kind`, such as `EMAIL-3f2a1c9d7e4b6a20`.
    pub fn token(&self, kind: PiiKind, value: &str) -> Result<String, TokenizerError> {
        let mut out = String::new();
        self.write_token(kind, value, &mut out)?;
        Ok(out)
    }

    /// Replaces every mention of `pii`, a layer over `text`, with its
    /// token. The mentions must not overlap or have empty normalized forms.
    pub fn rewrite(&self, text: &str, pii: &Pii) -> Result<PiiRewrite, RewriteError> {
        if let Some(m) = pii.mentions().find(|m| m.normalized.is_empty()) {
            return Err(RewriteError::EmptyValue(m.span));
        }
        PiiRewrite::replace(text, pii.mentions(), |m, out| {
            self.write_token(m.kind, m.normalized, out).expect("checked non-empty normalized values");
        })
    }
}
