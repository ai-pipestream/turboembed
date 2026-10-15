//! Masking in place, OpenNLP's `Masker`, `MaskPolicy` and
//! `MaskPolicies`.
//!
//! OpenNLP keeps the UTF-16 length and writes one mask character per
//! UTF-16 unit. Here the text is UTF-8 and the mask an ASCII byte, so a
//! masked character becomes as many mask bytes as it has UTF-8 bytes: the
//! byte length, and so every byte offset, stays the same. For ASCII spans,
//! which every detected mention is, both write the same text.

use std::fmt;

use turbo_analysis::Span;
use turbo_analysis_layers::{Pii, PiiKind, PiiMention};

use crate::text::letter_or_digit;

/// A mask the policy cannot take.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum MaskError {
    /// The mask is not one ASCII byte, so writing it would change byte
    /// offsets.
    NotAscii(char),
    /// A span is past the end of the text or not on character boundaries.
    Span(Span),
}

impl fmt::Display for MaskError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            MaskError::NotAscii(c) => write!(f, "the mask {c:?} is not an ASCII character"),
            MaskError::Span(s) => write!(f, "the span {}..{} is not a range of the text", s.start, s.end),
        }
    }
}

impl std::error::Error for MaskError {}

/// How the characters of a span are replaced: all of them, or the
/// letters and digits only (`keeping_format`), leaving the last
/// `keep_trailing` letters or digits readable.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub struct MaskPolicy {
    mask: u8,
    keep_format: bool,
    keep_trailing: usize,
}

impl MaskPolicy {
    /// The character the defaults mask with.
    pub const DEFAULT_MASK: char = '*';

    /// Masks every character of a span with `mask`, an ASCII character.
    pub fn new(mask: char) -> Result<MaskPolicy, MaskError> {
        if !mask.is_ascii() {
            return Err(MaskError::NotAscii(mask));
        }
        Ok(MaskPolicy { mask: mask as u8, keep_format: false, keep_trailing: 0 })
    }

    /// This policy, leaving what is not a letter or digit readable.
    pub const fn keeping_format(self) -> MaskPolicy {
        MaskPolicy { keep_format: true, ..self }
    }

    /// This policy, leaving the last `count` letters or digits of each
    /// span readable.
    pub const fn keeping_trailing(self, count: usize) -> MaskPolicy {
        MaskPolicy { keep_trailing: count, ..self }
    }

    pub const fn mask(self) -> char {
        self.mask as char
    }

    /// `MaskPolicy.apply`: `value` with this policy over all of it.
    pub fn masked(self, value: &str) -> String {
        assert!(u32::try_from(value.len()).is_ok(), "the value is longer than u32 offsets reach");
        let mut out = value.as_bytes().to_vec();
        self.apply(value, Span::new(0, value.len() as u32), &mut out);
        String::from_utf8(out).expect("masking writes whole characters")
    }

    /// `MaskPolicies.forType`: cards and IBANs keep separators and their
    /// last four letters or digits; contact, network, national and device
    /// identifiers and routing numbers keep separators; credentials and
    /// wallet addresses are masked whole.
    pub fn for_kind(kind: PiiKind, mask: char) -> Result<MaskPolicy, MaskError> {
        let whole = MaskPolicy::new(mask)?;
        Ok(match kind {
            PiiKind::Card | PiiKind::Iban => whole.keeping_format().keeping_trailing(4),
            PiiKind::Email
            | PiiKind::Phone
            | PiiKind::Ipv4
            | PiiKind::Ipv6
            | PiiKind::Mac
            | PiiKind::UsSsn
            | PiiKind::UsItin
            | PiiKind::UkNhs
            | PiiKind::DeSteuerId
            | PiiKind::Imei
            | PiiKind::CaSin
            | PiiKind::AbaRouting => whole.keeping_format(),
            PiiKind::Jwt
            | PiiKind::AwsAccessKey
            | PiiKind::GithubToken
            | PiiKind::UrlCredential
            | PiiKind::BtcAddress
            | PiiKind::EthAddress => whole,
        })
    }

    /// Masks the characters of `original[span]` this policy selects in
    /// `out`, a copy of `original` with any earlier masks applied. What it
    /// keeps readable does not undo an earlier mask.
    fn apply(self, original: &str, span: Span, out: &mut [u8]) {
        let (start, end) = (span.start as usize, span.end as usize);
        let piece = &original[start..end];
        let alphanumeric = piece.chars().filter(|&c| letter_or_digit(c)).count();
        let first_kept = alphanumeric.saturating_sub(self.keep_trailing);
        let mut seen = 0;
        for (at, c) in piece.char_indices() {
            let redact = if letter_or_digit(c) {
                seen += 1;
                seen <= first_kept
            } else {
                !self.keep_format
            };
            if redact {
                let from = start + at;
                out[from..from + c.len_utf8()].fill(self.mask);
            }
        }
    }
}

fn check(text: &str, span: Span) -> Result<(), MaskError> {
    let (start, end) = (span.start as usize, span.end as usize);
    if start > end || !text.is_char_boundary(start) || !text.is_char_boundary(end) {
        return Err(MaskError::Span(span));
    }
    Ok(())
}

/// `Masker.mask` over spans of `text`, each with the policy `policy`
/// gives for it, in order. The result has the byte length of `text`.
pub fn mask(text: &str, spans: impl IntoIterator<Item = (Span, MaskPolicy)>) -> Result<String, MaskError> {
    let mut out = text.as_bytes().to_vec();
    for (span, policy) in spans {
        check(text, span)?;
        policy.apply(text, span, &mut out);
    }
    // Only whole characters were overwritten, each with ASCII bytes.
    Ok(String::from_utf8(out).expect("masking writes whole characters"))
}

/// Masks every character of every mention of `pii` (a layer over
/// `text`) with `mask`: `Masker.mask(document, layer, mask)`.
pub fn mask_all(text: &str, pii: &Pii, mask_char: char) -> Result<String, MaskError> {
    let policy = MaskPolicy::new(mask_char)?;
    mask(text, pii.mentions().map(|m| (m.span, policy)))
}

/// Masks every mention of `pii` (a layer over `text`) with the default
/// policy of its kind and `mask`: `Masker.mask` with
/// `MaskPolicies.byType`.
pub fn mask_by_kind(text: &str, pii: &Pii, mask_char: char) -> Result<String, MaskError> {
    let mut policies = [None; PiiKind::ALL.len()];
    for k in PiiKind::ALL {
        policies[k.rank() as usize] = Some(MaskPolicy::for_kind(k, mask_char)?);
    }
    mask(text, pii.mentions().map(|m: PiiMention<'_>| (m.span, policies[m.kind.rank() as usize].expect("every kind"))))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn s(start: u32, end: u32) -> Span {
        Span::new(start, end)
    }

    #[test]
    fn policies_select_characters() {
        let whole = MaskPolicy::new('*').unwrap();
        let text = "card 4111-1111-1111-1111 ok";
        let span = s(5, 24);
        assert_eq!(mask(text, [(span, whole)]).unwrap(), "card ******************* ok");
        assert_eq!(mask(text, [(span, whole.keeping_format())]).unwrap(), "card ****-****-****-**** ok");
        assert_eq!(
            mask(text, [(span, whole.keeping_format().keeping_trailing(4))]).unwrap(),
            "card ****-****-****-1111 ok"
        );
        assert_eq!(mask(text, [(span, whole.keeping_trailing(4))]).unwrap(), "card ***************1111 ok");
        assert_eq!(mask(text, [(span, whole.keeping_trailing(99))]).unwrap(), "card 4111*1111*1111*1111 ok");
    }

    #[test]
    fn non_ascii_characters_keep_their_byte_length() {
        let whole = MaskPolicy::new('#').unwrap();
        let text = "é\u{1D400}x";
        let out = mask(text, [(s(0, text.len() as u32), whole)]).unwrap();
        assert_eq!(out, "#######");
        assert_eq!(out.len(), text.len());
    }

    #[test]
    fn a_kept_character_does_not_undo_an_earlier_mask() {
        let whole = MaskPolicy::new('*').unwrap();
        let text = "ab-cd";
        let out = mask(text, [(s(0, 5), whole), (s(0, 5), whole.keeping_format().keeping_trailing(2))]).unwrap();
        assert_eq!(out, "*****");
    }

    #[test]
    fn bad_masks_and_spans_are_refused() {
        assert_eq!(MaskPolicy::new('é'), Err(MaskError::NotAscii('é')));
        let whole = MaskPolicy::new('*').unwrap();
        assert_eq!(mask("é", [(s(0, 1), whole)]), Err(MaskError::Span(s(0, 1))));
        assert_eq!(mask("ab", [(s(0, 3), whole)]), Err(MaskError::Span(s(0, 3))));
        assert_eq!(whole.mask(), '*');
    }

    #[test]
    fn kinds_have_the_java_defaults() {
        let whole = MaskPolicy::new('*').unwrap();
        assert_eq!(MaskPolicy::for_kind(PiiKind::Card, '*').unwrap(), whole.keeping_format().keeping_trailing(4));
        assert_eq!(MaskPolicy::for_kind(PiiKind::Iban, '*').unwrap(), whole.keeping_format().keeping_trailing(4));
        assert_eq!(MaskPolicy::for_kind(PiiKind::Phone, '*').unwrap(), whole.keeping_format());
        assert_eq!(MaskPolicy::for_kind(PiiKind::Jwt, '*').unwrap(), whole);
        assert_eq!(MaskPolicy::for_kind(PiiKind::EthAddress, '*').unwrap(), whole);
        assert!(MaskPolicy::for_kind(PiiKind::Email, 'é').is_err());
    }
}
