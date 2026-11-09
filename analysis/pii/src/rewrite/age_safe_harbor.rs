//! HIPAA Safe Harbor aggregation of a person's age ([`AgeSafeHarbor`]),
//! 45 CFR 164.514(b)(2)(i)(C).
//!
//! Safe Harbor lets a de-identified record keep an age of 89 or under,
//! and requires every age over 89 to be aggregated into a single category
//! of 90 or older. The strategy keeps an age of 89 or under as written and
//! writes every older age as `90+`, so `aged 93` becomes `aged 90+` and
//! `34-year-old` stays `34-year-old`. The `age` kind's mention is the
//! number alone, so the words around it stay.
//!
//! A mention of another kind than `age`, or whose text is not one or more
//! ASCII digits, is masked whole, so a policy that sends another kind here
//! never leaves its value readable, digits or not. A cluster of overlapping mentions is masked whole too:
//! another mention's strategy may not allow any of its characters to stay.

use turbo_analysis::View;
use turbo_analysis_layers::{Pii, PiiKind, PiiStrategyKind};

use super::replace::EditOrigin;
use super::{PiiRewrite, RewriteError};

/// Keeps an age of 89 or under and writes every older age as
/// [`AgeSafeHarbor::OVER_89`].
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct AgeSafeHarbor {
    mask: char,
    refuse_overlaps: bool,
}

impl Default for AgeSafeHarbor {
    fn default() -> AgeSafeHarbor {
        AgeSafeHarbor::new()
    }
}

impl AgeSafeHarbor {
    /// The default mask character (`'*'`).
    pub const DEFAULT_MASK: char = '*';
    /// The oldest age Safe Harbor lets a record keep.
    pub const MAX_KEPT: u32 = 89;
    /// What every age over [`AgeSafeHarbor::MAX_KEPT`] becomes.
    pub const OVER_89: &'static str = "90+";

    /// The strategy with `'*'` for what is not an age.
    pub fn new() -> AgeSafeHarbor {
        AgeSafeHarbor { mask: AgeSafeHarbor::DEFAULT_MASK, refuse_overlaps: false }
    }

    /// Sets the character written for each character of a mention that is
    /// not an age, or of an overlap cluster.
    pub fn with_mask(mut self, mask: char) -> AgeSafeHarbor {
        self.mask = mask;
        self
    }

    /// This strategy, refusing overlapping mentions with
    /// [`RewriteError::Overlap`].
    pub fn refusing_overlaps(mut self) -> AgeSafeHarbor {
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

    pub(super) fn fold_config(&self, h: &mut turbo_analysis::Fnv1a) {
        h.u32(u32::from(self.mask)).u8(u8::from(self.refuse_overlaps));
    }

    /// Appends `value` aggregated to `out`: an age of 89 or under as
    /// written, an older age as `90+`, anything else masked whole.
    pub fn write_aggregated(&self, value: &str, out: &mut String) {
        if value.is_empty() || !value.bytes().all(|c| c.is_ascii_digit()) {
            self.write_masked(value, out);
            return;
        }
        // Leading zeros do not make an age younger, and a run too long to
        // hold in two digits is over 89 whatever it says.
        let digits = value.trim_start_matches('0');
        let over = digits.len() > 2 || digits.parse::<u32>().is_ok_and(|n| n > AgeSafeHarbor::MAX_KEPT);
        out.push_str(if over { AgeSafeHarbor::OVER_89 } else { value });
    }

    /// Appends a mention of `kind` reading `value` to `out`: aggregated
    /// when `kind` is [`PiiKind::Age`], masked whole otherwise.
    pub fn write_mention(&self, kind: PiiKind, value: &str, out: &mut String) {
        if kind == PiiKind::Age {
            self.write_aggregated(value, out);
        } else {
            self.write_masked(value, out);
        }
    }

    /// Returns `value` aggregated.
    pub fn apply(&self, value: &str) -> String {
        let mut out = String::with_capacity(value.len());
        self.write_aggregated(value, &mut out);
        out
    }

    /// Appends one mask character per character of `value`.
    pub(super) fn write_masked(&self, value: &str, out: &mut String) {
        out.extend(value.chars().map(|_| self.mask));
    }

    /// Replaces every mention of `pii`, a layer over `src`, with its
    /// aggregated form, as a view derived from `src`.
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
                    self.write_mention(cluster.merged.kind, piece, out);
                } else {
                    self.write_masked(piece, out);
                }
                Ok(EditOrigin::only(PiiStrategyKind::AgeSafeHarbor))
            },
        )
    }
}
