//! Reducing mentions to initials ([`Abbreviator`]), such as
//! `"John Fitzgerald Kennedy"` to `"J.F.K."` or `"JFK"`.

use turbo_analysis::View;
use turbo_analysis_layers::{Pii, PiiStrategyKind};

use super::replace::EditOrigin;
use super::{PiiRewrite, RewriteError};
use crate::text::letter_or_digit;

fn is_hyphen(c: char) -> bool {
    matches!(c, '-' | '\u{2010}' | '\u{2011}' | '\u{2013}')
}

/// Reduces each mention to the initials of its words (contiguous runs of
/// Unicode letters or digits), with configurable punctuation and hyphen
/// handling.
///
/// `Abbreviator` is intended for name or multi-word phrase kinds and should
/// not be used as a default strategy on numeric kinds (for example,
/// `123-45-6789` would abbreviate to `1.4.6.`).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Abbreviator {
    separator: String,
    trailing: String,
    preserve_hyphens: bool,
    uppercase: bool,
    refuse_overlaps: bool,
}

impl Default for Abbreviator {
    fn default() -> Abbreviator {
        Abbreviator::dotted()
    }
}

impl Abbreviator {
    /// Dotted initials (`"John Fitzgerald Kennedy"` -> `"J.F.K."`).
    pub fn new() -> Abbreviator {
        Abbreviator::dotted()
    }

    /// Dotted initials with `.` between and after initials
    /// (`"John Fitzgerald Kennedy"` -> `"J.F.K."`).
    pub fn dotted() -> Abbreviator {
        Abbreviator {
            separator: ".".to_owned(),
            trailing: ".".to_owned(),
            preserve_hyphens: false,
            uppercase: true,
            refuse_overlaps: false,
        }
    }

    /// Compact initials with no punctuation
    /// (`"John Fitzgerald Kennedy"` -> `"JFK"`).
    pub fn compact() -> Abbreviator {
        Abbreviator {
            separator: String::new(),
            trailing: String::new(),
            preserve_hyphens: false,
            uppercase: true,
            refuse_overlaps: false,
        }
    }

    /// Initials joined by `separator` and followed by `trailing` (for
    /// example, `Abbreviator::with_punctuation(". ", ".")` writes
    /// `"J. F. K."`).
    pub fn with_punctuation(separator: impl Into<String>, trailing: impl Into<String>) -> Abbreviator {
        Abbreviator {
            separator: separator.into(),
            trailing: trailing.into(),
            preserve_hyphens: false,
            uppercase: true,
            refuse_overlaps: false,
        }
    }

    /// Preserves hyphens between hyphenated word parts (for example,
    /// `"Jean-Luc Picard"` becomes `"J.-L.P."` when dotted or `"J-LP"` when
    /// compact).
    pub fn preserving_hyphens(mut self) -> Abbreviator {
        self.preserve_hyphens = true;
        self
    }

    /// Keeps the original case of each initial instead of uppercasing it.
    pub fn preserving_case(mut self) -> Abbreviator {
        self.uppercase = false;
        self
    }

    /// Whether hyphens between word parts are preserved.
    pub const fn preserves_hyphens(&self) -> bool {
        self.preserve_hyphens
    }

    /// Whether initials are uppercased.
    pub const fn uppercases(&self) -> bool {
        self.uppercase
    }

    /// This abbreviator, refusing overlapping mentions with
    /// [`RewriteError::Overlap`].
    pub fn refusing_overlaps(mut self) -> Abbreviator {
        self.refuse_overlaps = true;
        self
    }

    /// Whether overlapping mentions are refused rather than coalesced.
    pub const fn refuses_overlaps(&self) -> bool {
        self.refuse_overlaps
    }

    /// Appends the abbreviated form of `value` to `out`.
    pub fn write_abbreviated(&self, value: &str, out: &mut String) {
        self.write_abbreviated_filtered(value, |_, _| true, out);
    }

    /// Whether `target_offset` is the start byte of a word in `value`.
    pub(crate) fn is_initial_at(value: &str, target_offset: usize) -> bool {
        value.get(target_offset..).and_then(|s| s.chars().next()).is_some_and(letter_or_digit)
            && !crate::text::before(value, target_offset).is_some_and(letter_or_digit)
    }

    /// Appends the abbreviated form of `value` to `out`, masking any
    /// initial at relative byte offset `rel_at` where `allow_at(rel_at, c)` is
    /// `false` with `'*'`.
    pub(crate) fn write_abbreviated_filtered(
        &self,
        value: &str,
        mut allow_at: impl FnMut(usize, char) -> bool,
        out: &mut String,
    ) {
        let mut in_word = false;
        let mut words = 0usize;
        let mut saw_hyphen = false;
        let mut saw_other = false;
        for (rel_at, c) in value.char_indices() {
            if letter_or_digit(c) {
                if !in_word {
                    if words > 0 {
                        if self.preserve_hyphens && saw_hyphen && !saw_other {
                            out.push_str(&self.trailing);
                            out.push('-');
                        } else {
                            out.push_str(&self.separator);
                        }
                    }
                    if !allow_at(rel_at, c) {
                        out.push('*');
                    } else if self.uppercase {
                        out.extend(c.to_uppercase());
                    } else {
                        out.push(c);
                    }
                    words += 1;
                    in_word = true;
                    saw_hyphen = false;
                    saw_other = false;
                }
            } else {
                in_word = false;
                if words > 0 {
                    if is_hyphen(c) {
                        saw_hyphen = true;
                    } else {
                        saw_other = true;
                    }
                }
            }
        }
        if words > 0 {
            out.push_str(&self.trailing);
        } else {
            out.push('*');
        }
    }

    /// Returns the abbreviated form of `value`.
    pub fn abbreviate(&self, value: &str) -> String {
        let mut out = String::new();
        self.write_abbreviated(value, &mut out);
        out
    }

    /// Replaces every mention of `pii`, a layer over `src`, with the
    /// abbreviated form of its surface text, as a view derived from `src`.
    pub fn rewrite(&self, src: &View, pii: &Pii) -> Result<PiiRewrite, RewriteError> {
        let text = src.text();
        PiiRewrite::derive(
            src,
            pii,
            |_| self.refuse_overlaps,
            0,
            |c, out| {
                self.write_abbreviated(&text[c.merged.span.range()], out);
                Ok(EditOrigin::only(PiiStrategyKind::Abbreviate))
            },
        )
    }
}
