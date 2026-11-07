//! Masking in place, OpenNLP's `Masker`, `MaskPolicy` and
//! `MaskPolicies`.
//!
//! OpenNLP keeps the UTF-16 length and writes one mask character per
//! UTF-16 unit. Here the text is UTF-8 and the mask an ASCII byte, so a
//! masked character becomes as many mask bytes as it has UTF-8 bytes: the
//! byte length, and so every byte offset, stays the same. For ASCII spans,
//! which every detected mention is, both write the same text.

use std::fmt;

use turbo_analysis::{Span, View, ViewId};
use turbo_analysis_layers::{KindError, Pii, PiiKind, PiiMention, PiiStrategyKind};

use super::replace::EditOrigin;
use super::{PiiRewrite, RewriteError, absolute};
use crate::text::letter_or_digit;

/// Why a mask cannot be written: a mask the policy cannot take, a span
/// the text does not have, or a layer over another view.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum MaskError {
    /// The mask is not one ASCII byte, so writing it would change byte
    /// offsets.
    NotAscii(char),
    /// A span is past the end of the text or not on character boundaries.
    Span(Span),
    /// A mention's custom kind is outside its layer's table.
    Kind(KindError),
    /// The pii layer is over view `layer`, not over `view`, the view being
    /// masked, so its spans are offsets of another text.
    ForeignView { layer: ViewId, view: ViewId },
}

impl fmt::Display for MaskError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            MaskError::NotAscii(c) => write!(f, "the mask {c:?} is not an ASCII character"),
            MaskError::Span(s) => write!(f, "the span {}..{} is not a range of the text", s.start, s.end),
            MaskError::Kind(e) => e.fmt(f),
            MaskError::ForeignView { layer, view } => {
                write!(f, "the pii layer is over view {}, not view {}", layer.0, view.0)
            }
        }
    }
}

impl std::error::Error for MaskError {}

/// Precomputed byte-offset retention bounds for linear-time per-character
/// overlap checking across [`MaskPolicy`] and [`super::Truncate`].
#[derive(Copy, Clone)]
pub(super) struct CharRetention {
    pub(super) keep_until: usize,
    pub(super) keep_from: usize,
    pub(super) keep_format: bool,
    pub(super) only_alnum_bounds: bool,
}

impl CharRetention {
    pub(super) const NONE: CharRetention =
        CharRetention { keep_until: 0, keep_from: usize::MAX, keep_format: false, only_alnum_bounds: true };

    #[inline]
    pub(super) fn keeps(self, rel_at: usize, c: char) -> bool {
        if letter_or_digit(c) || !self.only_alnum_bounds {
            rel_at < self.keep_until || rel_at >= self.keep_from
        } else {
            self.keep_format
        }
    }
}

/// How the characters of a span are replaced: all of them, or the
/// letters and digits only (`keeping_format`), leaving the first
/// `keep_leading` and/or last `keep_trailing` letters or digits readable.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub struct MaskPolicy {
    mask: u8,
    keep_format: bool,
    keep_leading: usize,
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
        Ok(MaskPolicy { mask: mask as u8, keep_format: false, keep_leading: 0, keep_trailing: 0 })
    }

    /// A policy keeping the first `count` letters or digits readable and
    /// masking the rest with `mask`.
    ///
    /// A span with `<= count` letters or digits stays unmasked (matching
    /// [`MaskPolicy::keeping_trailing`]); use [`super::Truncate::keep_start`]
    /// when spans of length `<= count` must be masked whole.
    pub fn first_n(count: usize, mask: char) -> Result<MaskPolicy, MaskError> {
        Ok(MaskPolicy::new(mask)?.keeping_leading(count))
    }

    /// A policy keeping the last `count` letters or digits readable and
    /// masking the rest with `mask`.
    ///
    /// A span with `<= count` letters or digits stays unmasked (matching
    /// OpenNLP's `lastFour`); use [`super::Truncate::last_n`] when spans of
    /// length `<= count` must be masked whole.
    pub fn last_n(count: usize, mask: char) -> Result<MaskPolicy, MaskError> {
        Ok(MaskPolicy::new(mask)?.keeping_trailing(count))
    }

    /// This policy, leaving what is not a letter or digit readable.
    pub const fn keeping_format(self) -> MaskPolicy {
        MaskPolicy { keep_format: true, ..self }
    }

    /// This policy, leaving the first `count` letters or digits of each
    /// span readable. Spans with `<= count` letters or digits stay
    /// unmasked; see [`super::Truncate::keep_start`] to mask short spans.
    pub const fn keeping_leading(self, count: usize) -> MaskPolicy {
        MaskPolicy { keep_leading: count, ..self }
    }

    /// This policy, leaving the last `count` letters or digits of each
    /// span readable. Spans with `<= count` letters or digits stay
    /// unmasked; see [`super::Truncate::last_n`] to mask short spans.
    pub const fn keeping_trailing(self, count: usize) -> MaskPolicy {
        MaskPolicy { keep_trailing: count, ..self }
    }

    pub const fn mask(self) -> char {
        self.mask as char
    }

    /// Whether what is not a letter or digit stays readable.
    pub const fn keeps_format(self) -> bool {
        self.keep_format
    }

    /// How many leading letters or digits stay readable.
    pub const fn leading(self) -> usize {
        self.keep_leading
    }

    /// How many trailing letters or digits stay readable.
    pub const fn trailing(self) -> usize {
        self.keep_trailing
    }

    /// `MaskPolicy.apply`: `value` with this policy over all of it.
    pub fn masked(self, value: &str) -> String {
        let len = u32::try_from(value.len()).expect("the value is longer than u32 offsets reach");
        let mut out = value.as_bytes().to_vec();
        self.apply(value, Span::new(0, len), &mut out);
        String::from_utf8(out).expect("masking writes whole characters")
    }

    /// Replaces every mention of `pii`, a layer over `src`, with its
    /// masked surface text, coalescing overlapping mentions, as a view
    /// derived from `src`.
    pub fn rewrite(&self, src: &View, pii: &Pii) -> Result<PiiRewrite, RewriteError> {
        let text = src.text();
        PiiRewrite::derive(
            src,
            pii,
            |_| false,
            0,
            |cluster, out| {
                let m = cluster.merged;
                let piece = &text[m.span.range()];
                if cluster.members.len() == 1 {
                    out.push_str(&self.masked(piece));
                    return Ok(EditOrigin::only(PiiStrategyKind::Mask));
                }
                let winner_ret = self.retention(piece);
                let member_rets: Vec<(Span, CharRetention)> =
                    cluster.members.iter().map(|mem| (mem.span, self.retention(&text[mem.span.range()]))).collect();
                for (rel_at, c) in piece.char_indices() {
                    let abs_at = absolute(m.span, rel_at);
                    let keep = winner_ret.keeps(rel_at, c)
                        && member_rets
                            .iter()
                            .filter(|(span, _)| span.start <= abs_at && abs_at < span.end)
                            .all(|(span, ret)| ret.keeps((abs_at - span.start) as usize, c));
                    if keep {
                        out.push(c);
                    } else {
                        for _ in 0..c.len_utf8() {
                            out.push(self.mask());
                        }
                    }
                }
                Ok(EditOrigin::only(PiiStrategyKind::Mask))
            },
        )
    }

    /// Precomputes byte-offset retention bounds for `piece` in `O(|piece|)`.
    pub(super) fn retention(self, piece: &str) -> CharRetention {
        let mut keep_until = 0;
        if self.keep_leading > 0 {
            keep_until = piece.len();
            let mut kept = 0;
            for (at, c) in piece.char_indices() {
                if letter_or_digit(c) {
                    kept += 1;
                    if kept == self.keep_leading {
                        keep_until = at + c.len_utf8();
                        break;
                    }
                }
            }
        }
        let mut keep_from = piece.len();
        if self.keep_trailing > 0 {
            keep_from = 0;
            let mut kept = 0;
            for (at, c) in piece.char_indices().rev() {
                if letter_or_digit(c) {
                    kept += 1;
                    if kept == self.keep_trailing {
                        keep_from = at;
                        break;
                    }
                }
            }
        }
        CharRetention { keep_until, keep_from, keep_format: self.keep_format, only_alnum_bounds: true }
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
            | PiiKind::EsDni
            | PiiKind::EsNie
            | PiiKind::EsCif
            | PiiKind::DePersonalausweis
            | PiiKind::FrNir
            | PiiKind::FrSiren
            | PiiKind::FrSiret
            | PiiKind::BrCpf
            | PiiKind::BrCnpj
            | PiiKind::Bic
            | PiiKind::Imei
            | PiiKind::CaSin
            | PiiKind::AbaRouting => whole.keeping_format(),
            PiiKind::Jwt
            | PiiKind::AwsAccessKey
            | PiiKind::GithubToken
            | PiiKind::UrlCredential
            | PiiKind::BtcAddress
            | PiiKind::EthAddress
            | PiiKind::Custom(_) => whole,
        })
    }

    /// Masks the characters of `original[span]` this policy selects in
    /// `out`, a copy of `original` with any earlier masks applied. What it
    /// keeps readable does not undo an earlier mask.
    fn apply(self, original: &str, span: Span, out: &mut [u8]) {
        let (start, end) = (span.start as usize, span.end as usize);
        let piece = &original[start..end];
        let ret = self.retention(piece);
        for (at, c) in piece.char_indices() {
            if !ret.keeps(at, c) {
                let from = start + at;
                out[from..from + c.len_utf8()].fill(self.mask);
            }
        }
    }
}

pub(super) fn check(text: &str, span: Span) -> Result<(), MaskError> {
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
    check_kinds(pii)?;
    mask(text, pii.mentions().map(|m| (m.span, policy)))
}

/// Masks every mention of `pii` (a layer over `text`) with the default
/// policy of its kind and `mask`: `Masker.mask` with
/// `MaskPolicies.byType`.
pub fn mask_by_kind(text: &str, pii: &Pii, mask_char: char) -> Result<String, MaskError> {
    let _ = MaskPolicy::new(mask_char)?;
    check_kinds(pii)?;
    mask(
        text,
        pii.mentions()
            .map(|m: PiiMention<'_>| (m.span, MaskPolicy::for_kind(m.kind, mask_char).expect("validated ASCII mask"))),
    )
}

/// Refuses a mention whose custom kind is outside its layer's table.
fn check_kinds(pii: &Pii) -> Result<(), MaskError> {
    pii.mentions().try_for_each(|m| pii.custom_kinds.check(m.kind).map_err(MaskError::Kind))
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
