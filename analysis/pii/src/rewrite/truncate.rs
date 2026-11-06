//! Partial character retention ([`Truncate`], [`KeepSide`]): keeping `N`
//! characters (`char`, Unicode scalar values) at the start or end of a
//! mention and masking the rest.
//!
//! When a mention has `<= N` characters (or `<= N` letters or digits when
//! [`Truncate::keeping_format`] is enabled), every character (or every
//! letter and digit) is masked so that a short identifier or secret is
//! never left in the clear.

use turbo_analysis_layers::Pii;

use super::{PiiRewrite, RewriteError};
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
        if self.keep_format {
            let alpha = value.chars().filter(|&c| letter_or_digit(c)).count();
            if self.count == 0 || alpha <= self.count {
                for c in value.chars() {
                    out.push(if letter_or_digit(c) { self.mask } else { c });
                }
                return;
            }
            let tail_start = alpha - self.count;
            let mut seen = 0;
            for c in value.chars() {
                if letter_or_digit(c) {
                    let keep = match self.side {
                        KeepSide::Start => seen < self.count,
                        KeepSide::End => seen >= tail_start,
                    };
                    seen += 1;
                    out.push(if keep { c } else { self.mask });
                } else {
                    out.push(c);
                }
            }
        } else {
            let total = value.chars().count();
            if self.count == 0 || total <= self.count {
                for _ in 0..total {
                    out.push(self.mask);
                }
                return;
            }
            let tail_start = total - self.count;
            for (i, c) in value.chars().enumerate() {
                let keep = match self.side {
                    KeepSide::Start => i < self.count,
                    KeepSide::End => i >= tail_start,
                };
                out.push(if keep { c } else { self.mask });
            }
        }
    }

    /// Returns `value` with this truncation/masking policy applied.
    pub fn apply(&self, value: &str) -> String {
        let mut out = String::with_capacity(value.len());
        self.write_truncated(value, &mut out);
        out
    }

    /// Replaces every mention of `pii`, a layer over `text`, with its
    /// truncated/masked surface form.
    pub fn rewrite(&self, text: &str, pii: &Pii) -> Result<PiiRewrite, RewriteError> {
        let mentions: Vec<_> = if self.refuse_overlaps {
            pii.mentions().collect()
        } else {
            PiiRewrite::coalesce_clusters(text, pii.mentions())?
        };
        PiiRewrite::replace(text, mentions, |m, out| {
            self.write_truncated(&text[m.span.range()], out);
        })
    }
}
