//! Numbered labels, OpenNLP's `Pseudonymizer`: `EMAIL-1`, `EMAIL-2`, the
//! same label for the same kind and normalized value within one text.
//! Numbering starts again with each text, so equal labels in two texts
//! say nothing about the values; [`super::HmacTokenizer`] gives labels
//! that hold across texts.

use std::collections::HashMap;
use std::fmt::Write;

use turbo_analysis_layers::{Pii, PiiKind};

use super::{PiiRewrite, RewriteError};

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

    /// Replaces every mention of `pii`, a layer over `text`, with its
    /// label.
    pub fn rewrite(&self, text: &str, pii: &Pii) -> Result<PiiRewrite, RewriteError> {
        let mut assigned: HashMap<(PiiKind, &str), String> = HashMap::new();
        let mut counters = [0u32; PiiKind::ALL.len()];
        let mentions: Vec<_> = if self.refuse_overlaps {
            pii.mentions().collect()
        } else {
            PiiRewrite::coalesce_clusters(text, pii.mentions())?
        };
        PiiRewrite::replace(text, mentions, |m, out| {
            let label = assigned.entry((m.kind, m.normalized)).or_insert_with(|| {
                let n = &mut counters[m.kind.rank() as usize];
                *n += 1;
                let mut l = String::new();
                l.push_str(&self.prefix);
                l.extend(m.kind.name().chars().map(|c| c.to_ascii_uppercase()));
                let _ = write!(l, "-{n}");
                l.push_str(&self.suffix);
                l
            });
            out.push_str(label);
        })
    }
}
