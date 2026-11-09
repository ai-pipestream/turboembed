//! HIPAA Safe Harbor generalization of a United States ZIP code
//! ([`ZipSafeHarbor`]), 45 CFR 164.514(b)(2)(i)(B).
//!
//! Safe Harbor lets a de-identified record keep the first three digits of
//! a ZIP code only when the geographic unit formed by all ZIP codes with
//! those three digits holds more than 20,000 people; for every other
//! three-digit area the digits become `000`. The strategy keeps those
//! three digits (or writes `000`) and masks every other digit, leaving the
//! ZIP+4 hyphen readable, so `62704-1234` becomes `627**-****` and
//! `03601` becomes `000**`.
//!
//! The restricted areas come from a table: by default the 17 prefixes the
//! HHS Office for Civil Rights published from the 2000 Census in its
//! de-identification guidance ([`HHS_2000_RESTRICTED_PREFIXES`], public
//! domain, see `docs/licences.md`), or a caller's own list built from a
//! later Census.
//!
//! A mention that is not a ZIP code (five ASCII digits, optionally
//! followed by `-` and four more) is masked whole, so a policy that sends
//! another kind here never leaves its value readable. A cluster of
//! overlapping mentions is masked whole too: another mention's strategy
//! may not allow any of its characters to stay.

use std::fmt;

use turbo_analysis::View;
use turbo_analysis_layers::{Pii, PiiStrategyKind};

use super::replace::EditOrigin;
use super::{PiiRewrite, RewriteError};

/// The three-digit ZIP areas of 20,000 people or fewer that the HHS
/// Office for Civil Rights lists from the 2000 Census in its guidance on
/// the HIPAA Privacy Rule's de-identification standard (section 3.6).
pub const HHS_2000_RESTRICTED_PREFIXES: [&str; 17] = [
    "036", "059", "063", "102", "203", "556", "692", "790", "821", "823", "830", "831", "878", "879", "884", "890",
    "893",
];

/// Why a restricted-prefix table cannot be taken.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ZipTableError {
    /// The table lists no prefix. Safe Harbor always has restricted areas,
    /// so an empty table is a mistake, not a choice.
    Empty,
    /// An entry is not exactly three ASCII digits. Holds the entry's
    /// position, never its text.
    NotThreeDigits(usize),
}

impl fmt::Display for ZipTableError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ZipTableError::Empty => f.write_str("the restricted ZIP prefix table is empty"),
            ZipTableError::NotThreeDigits(at) => {
                write!(f, "restricted ZIP prefix {at} is not exactly three ASCII digits")
            }
        }
    }
}

impl std::error::Error for ZipTableError {}

/// Keeps a ZIP code's first three digits, or writes `000` for a
/// restricted three-digit area, and masks the other digits.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct ZipSafeHarbor {
    /// Sorted and without duplicates.
    restricted: Vec<[u8; 3]>,
    mask: char,
    refuse_overlaps: bool,
}

impl Default for ZipSafeHarbor {
    fn default() -> ZipSafeHarbor {
        ZipSafeHarbor::hhs_2000()
    }
}

impl ZipSafeHarbor {
    /// The default mask character (`'*'`).
    pub const DEFAULT_MASK: char = '*';

    /// The strategy over [`HHS_2000_RESTRICTED_PREFIXES`] with `'*'`.
    pub fn hhs_2000() -> ZipSafeHarbor {
        ZipSafeHarbor::with_restricted(HHS_2000_RESTRICTED_PREFIXES).expect("the bundled table is valid")
    }

    /// The strategy over a caller's restricted-prefix table, each entry
    /// exactly three ASCII digits; duplicates are folded.
    pub fn with_restricted<S: AsRef<str>>(
        prefixes: impl IntoIterator<Item = S>,
    ) -> Result<ZipSafeHarbor, ZipTableError> {
        let mut restricted = Vec::new();
        for (at, p) in prefixes.into_iter().enumerate() {
            match p.as_ref().as_bytes() {
                &[a, b, c] if [a, b, c].iter().all(u8::is_ascii_digit) => restricted.push([a, b, c]),
                _ => return Err(ZipTableError::NotThreeDigits(at)),
            }
        }
        if restricted.is_empty() {
            return Err(ZipTableError::Empty);
        }
        restricted.sort_unstable();
        restricted.dedup();
        Ok(ZipSafeHarbor { restricted, mask: ZipSafeHarbor::DEFAULT_MASK, refuse_overlaps: false })
    }

    /// Sets the character written for each masked digit or character.
    pub fn with_mask(mut self, mask: char) -> ZipSafeHarbor {
        self.mask = mask;
        self
    }

    /// This strategy, refusing overlapping mentions with
    /// [`RewriteError::Overlap`].
    pub fn refusing_overlaps(mut self) -> ZipSafeHarbor {
        self.refuse_overlaps = true;
        self
    }

    /// The mask character.
    pub fn mask(&self) -> char {
        self.mask
    }

    /// Whether overlapping mentions are refused rather than coalesced.
    pub fn refuses_overlaps(&self) -> bool {
        self.refuse_overlaps
    }

    /// The restricted prefixes, sorted.
    pub fn restricted(&self) -> impl Iterator<Item = &str> {
        self.restricted.iter().map(|p| std::str::from_utf8(p).expect("ASCII digits"))
    }

    /// Whether the table is exactly [`HHS_2000_RESTRICTED_PREFIXES`].
    pub fn is_hhs_2000(&self) -> bool {
        self.restricted().eq(HHS_2000_RESTRICTED_PREFIXES)
    }

    /// Whether three-digit area `prefix` is restricted.
    pub fn is_restricted(&self, prefix: [u8; 3]) -> bool {
        self.restricted.binary_search(&prefix).is_ok()
    }

    pub(super) fn fold_config(&self, h: &mut turbo_analysis::Fnv1a) {
        h.u32(u32::from(self.mask)).u8(u8::from(self.refuse_overlaps)).usize(self.restricted.len());
        for p in &self.restricted {
            h.u8(p[0]).u8(p[1]).u8(p[2]);
        }
    }

    /// Appends `value` generalized to `out`: a ZIP code keeps its area
    /// digits (or `000`) and masks the rest; anything else is masked whole.
    pub fn write_generalized(&self, value: &str, out: &mut String) {
        let b = value.as_bytes();
        let zip = (b.len() == 5 || (b.len() == 10 && b[5] == b'-'))
            && b.iter().enumerate().all(|(i, c)| i == 5 || c.is_ascii_digit());
        if !zip {
            self.write_masked(value, out);
            return;
        }
        let area = [b[0], b[1], b[2]];
        if self.is_restricted(area) {
            out.push_str("000");
        } else {
            out.push_str(&value[..3]);
        }
        for &c in &b[3..] {
            out.push(if c == b'-' { '-' } else { self.mask });
        }
    }

    /// Returns `value` generalized.
    pub fn apply(&self, value: &str) -> String {
        let mut out = String::with_capacity(value.len());
        self.write_generalized(value, &mut out);
        out
    }

    /// Appends one mask character per character of `value`.
    pub(super) fn write_masked(&self, value: &str, out: &mut String) {
        out.extend(value.chars().map(|_| self.mask));
    }

    /// Replaces every mention of `pii`, a layer over `src`, with its
    /// generalized form, as a view derived from `src`.
    pub fn rewrite(&self, src: &View, pii: &Pii) -> Result<PiiRewrite, RewriteError> {
        let text = src.text();
        PiiRewrite::derive(
            src,
            pii,
            |_| self.refuse_overlaps,
            0,
            |cluster, out| {
                let piece = &text[cluster.merged.span.range()];
                if cluster.members.len() == 1 {
                    self.write_generalized(piece, out);
                } else {
                    self.write_masked(piece, out);
                }
                Ok(EditOrigin::only(PiiStrategyKind::ZipSafeHarbor))
            },
        )
    }
}
