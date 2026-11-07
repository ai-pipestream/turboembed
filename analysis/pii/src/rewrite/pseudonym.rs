//! Numbered labels, OpenNLP's `Pseudonymizer`: `EMAIL-1`, `EMAIL-2`, the
//! same label for the same kind and normalized value within one text.
//! Numbering starts again with each text, so equal labels in two texts
//! say nothing about the values; [`super::HmacTokenizer`] gives labels
//! that hold across texts.

use std::collections::HashMap;
use std::fmt::Write;

use turbo_analysis::View;
use turbo_analysis_layers::{Pii, PiiKind, PiiStrategyKind};

use super::replace::EditOrigin;
use super::{PiiRewrite, RewriteError, kind_name};

/// Writes `prefix`, the kind's name in capitals, `-`, the number and
/// `suffix` for each mention. By default, overlapping mentions are
/// coalesced into their union span under the top-ranked kind;
/// [`Pseudonymizer::refusing_overlaps`] refuses them for OpenNLP parity.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Pseudonymizer {
    pub prefix: String,
    pub suffix: String,
    refuse_overlaps: bool,
}

impl Pseudonymizer {
    /// Labels such as `[EMAIL-1]` with `prefix` `[` and `suffix` `]`.
    pub fn new(prefix: impl Into<String>, suffix: impl Into<String>) -> Pseudonymizer {
        Pseudonymizer { prefix: prefix.into(), suffix: suffix.into(), refuse_overlaps: false }
    }

    /// This pseudonymizer, refusing overlapping mentions with
    /// [`RewriteError::Overlap`] as OpenNLP does.
    pub fn refusing_overlaps(mut self) -> Pseudonymizer {
        self.refuse_overlaps = true;
        self
    }

    /// Whether overlapping mentions are refused rather than coalesced.
    pub const fn refuses_overlaps(&self) -> bool {
        self.refuse_overlaps
    }

    /// Appends the pseudonym label for the kind named `kind` and 1-based
    /// `number` to `out`.
    pub(crate) fn write_label(&self, kind: &str, number: u32, out: &mut String) {
        out.push_str(&self.prefix);
        out.extend(kind.chars().map(|c| c.to_ascii_uppercase()));
        let _ = write!(out, "-{number}");
        out.push_str(&self.suffix);
    }

    /// Replaces every mention of `pii`, a layer over `src`, with its
    /// label, as a view derived from `src`.
    pub fn rewrite(&self, src: &View, pii: &Pii) -> Result<PiiRewrite, RewriteError> {
        let mut assigned: HashMap<(PiiKind, &str), String> = HashMap::new();
        let mut counters: HashMap<PiiKind, u32> = HashMap::new();
        PiiRewrite::derive(
            src,
            pii,
            |_| self.refuse_overlaps,
            0,
            |c, out| {
                let m = c.merged;
                let label = assigned.entry((m.kind, m.normalized)).or_insert_with(|| {
                    let n = counters.entry(m.kind).or_insert(0);
                    *n += 1;
                    let mut l = String::new();
                    self.write_label(kind_name(&pii.custom_kinds, m.kind), *n, &mut l);
                    l
                });
                out.push_str(label);
                Ok(EditOrigin::only(PiiStrategyKind::Pseudonym))
            },
        )
    }
}
