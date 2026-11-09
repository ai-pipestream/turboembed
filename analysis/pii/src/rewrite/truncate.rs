//! Partial character retention ([`Truncate`], [`KeepSide`]): keeping `N`
//! characters (`char`, Unicode scalar values) at the start or end of a
//! mention and masking the rest.
//!
//! When a mention has `<= N` characters (or `<= N` letters or digits when
//! [`Truncate::keeping_format`] is enabled), every character (or every
//! letter and digit) is masked so that a short identifier or secret is
//! never left in the clear.

use turbo_analysis::View;
use turbo_analysis_layers::{Pii, PiiStrategyKind};

use super::mask::CharRetention;
use super::replace::EditOrigin;
use super::{PiiRewrite, RewriteError, absolute};
use crate::text::letter_or_digit;

/// Which end of a mention [`Truncate`] keeps readable when the mention is
/// strictly longer than `count` characters.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub enum KeepSide {
    /// Keep the first `N` characters and mask the rest (`KeepStart`).
    Start,
    /// Keep the last `N` characters and mask the rest (`KeepEnd` / `LastN`).
    End,
}

/// Keeps `count` Unicode scalar values (`char`) at the start or end of a
/// mention and replaces the remaining characters with `mask` (default
/// `'*'`).
///
/// If a mention has `<= count` characters (or `<= count` letters or digits
/// when [`Truncate::keeping_format`] is enabled), all of its characters are
/// masked so a short value is never left unmasked.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub struct Truncate {
    side: KeepSide,
    count: usize,
    mask: char,
    keep_format: bool,
    refuse_overlaps: bool,
}

impl Truncate {
    /// Folds every parameter that changes what this strategy writes into
    /// `h`, for the step's config hash.
    pub(super) fn fold_config(&self, h: &mut turbo_analysis::Fnv1a) {
        h.u8(self.side as u8).usize(self.count).u32(u32::from(self.mask));
        h.u8(u8::from(self.keep_format)).u8(u8::from(self.refuse_overlaps));
    }

    /// The default mask character (`'*'`).
    pub const DEFAULT_MASK: char = '*';

    /// Keeps `count` characters on `side` and masks the rest with `'*'`.
    pub const fn new(side: KeepSide, count: usize) -> Truncate {
        Truncate { side, count, mask: Truncate::DEFAULT_MASK, keep_format: false, refuse_overlaps: false }
    }

    /// Keeps the first `count` characters and masks the trailing characters
    /// with `'*'` (or masks all characters when the mention has `<= count`
    /// characters).
    pub const fn keep_start(count: usize) -> Truncate {
        Truncate::new(KeepSide::Start, count)
    }

    /// Keeps the last `count` characters and masks the leading characters
    /// with `'*'` (or masks all characters when the mention has `<= count`
    /// characters).
    pub const fn keep_end(count: usize) -> Truncate {
        Truncate::new(KeepSide::End, count)
    }

    /// Alias for [`Truncate::keep_end`]: keeps the last `count` characters
    /// (`LastN`).
    pub const fn last_n(count: usize) -> Truncate {
        Truncate::keep_end(count)
    }

    /// Sets the replacement character written for each masked `char`.
    pub const fn with_mask(mut self, mask: char) -> Truncate {
        self.mask = mask;
        self
    }

    /// Preserves non-alphanumeric separator characters and counts `count`
    /// over letters and digits only.
    pub const fn keeping_format(mut self) -> Truncate {
        self.keep_format = true;
        self
    }

    /// Which side of the mention is kept readable.
    pub const fn side(&self) -> KeepSide {
        self.side
    }

    /// Number of characters kept readable when the mention is longer than
    /// `count`.
    pub const fn count(&self) -> usize {
        self.count
    }

    /// The mask character written for each masked `char`.
    pub const fn mask(&self) -> char {
        self.mask
    }

    /// Whether non-alphanumeric separators are left readable.
    pub const fn keeps_format(&self) -> bool {
        self.keep_format
    }

    /// This strategy, refusing overlapping mentions with
    /// [`RewriteError::Overlap`].
    pub const fn refusing_overlaps(mut self) -> Truncate {
        self.refuse_overlaps = true;
        self
    }

    /// Whether overlapping mentions are refused rather than coalesced.
    pub const fn refuses_overlaps(&self) -> bool {
        self.refuse_overlaps
    }

    /// Appends `value` with this truncation/masking policy applied to `out`.
    pub fn write_truncated(&self, value: &str, out: &mut String) {
        let ret = self.retention(value);
        for (at, c) in value.char_indices() {
            out.push(if ret.keeps(at, c) { c } else { self.mask });
        }
    }

    /// Returns `value` with this truncation/masking policy applied.
    pub fn apply(&self, value: &str) -> String {
        let mut out = String::with_capacity(value.len());
        self.write_truncated(value, &mut out);
        out
    }

    /// Precomputes byte-offset retention bounds for `value` in `O(|value|)`.
    pub(super) fn retention(&self, value: &str) -> CharRetention {
        #[cfg(test)]
        super::mask::RETENTION_WORK.with(|n| n.set(n.get() + value.len().max(1)));
        if self.keep_format {
            let alpha = value.chars().filter(|&c| letter_or_digit(c)).count();
            if self.count == 0 || alpha <= self.count {
                return CharRetention {
                    keep_until: 0,
                    keep_from: value.len(),
                    keep_format: true,
                    only_alnum_bounds: true,
                };
            }
            match self.side {
                KeepSide::Start => {
                    let mut keep_until = value.len();
                    let mut seen = 0;
                    for (at, c) in value.char_indices() {
                        if letter_or_digit(c) {
                            seen += 1;
                            if seen == self.count {
                                keep_until = at + c.len_utf8();
                                break;
                            }
                        }
                    }
                    CharRetention { keep_until, keep_from: value.len(), keep_format: true, only_alnum_bounds: true }
                }
                KeepSide::End => {
                    let mut keep_from = 0;
                    let mut seen = 0;
                    for (at, c) in value.char_indices().rev() {
                        if letter_or_digit(c) {
                            seen += 1;
                            if seen == self.count {
                                keep_from = at;
                                break;
                            }
                        }
                    }
                    CharRetention { keep_until: 0, keep_from, keep_format: true, only_alnum_bounds: true }
                }
            }
        } else {
            let total = value.chars().count();
            if self.count == 0 || total <= self.count {
                return CharRetention::NONE;
            }
            match self.side {
                KeepSide::Start => {
                    let keep_until = value.char_indices().nth(self.count).map_or(value.len(), |(at, _)| at);
                    CharRetention { keep_until, keep_from: value.len(), keep_format: false, only_alnum_bounds: false }
                }
                KeepSide::End => {
                    let keep_from = value.char_indices().rev().nth(self.count - 1).map_or(0, |(at, _)| at);
                    CharRetention { keep_until: 0, keep_from, keep_format: false, only_alnum_bounds: false }
                }
            }
        }
    }

    /// Replaces every mention of `pii`, a layer over `src`, with its
    /// truncated/masked surface form, as a view derived from `src`.
    pub fn rewrite(&self, src: &View, pii: &Pii) -> Result<PiiRewrite, RewriteError> {
        let text = src.text();
        PiiRewrite::derive(
            src,
            pii,
            |_| self.refuse_overlaps,
            0,
            |cluster, out| {
                let m = cluster.merged;
                let piece = &text[m.span.range()];
                if cluster.members.len() == 1 {
                    self.write_truncated(piece, out);
                    return Ok(EditOrigin::only(PiiStrategyKind::Truncate));
                }
                let winner_ret = self.retention(piece);
                let member_rets: Vec<_> =
                    cluster.members.iter().map(|mem| (mem.span, self.retention(&text[mem.span.range()]))).collect();
                for (rel_at, c) in piece.char_indices() {
                    let abs_at = absolute(m.span, rel_at);
                    let keep = winner_ret.keeps(rel_at, c)
                        && member_rets
                            .iter()
                            .filter(|(span, _)| span.start <= abs_at && abs_at < span.end)
                            .all(|(span, ret)| ret.keeps((abs_at - span.start) as usize, c));
                    out.push(if keep { c } else { self.mask });
                }
                Ok(EditOrigin::only(PiiStrategyKind::Truncate))
            },
        )
    }
}
