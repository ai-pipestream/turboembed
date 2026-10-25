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
/// `suffix` for each mention.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Pseudonymizer {
    pub prefix: String,
    pub suffix: String,
}

impl Pseudonymizer {
    /// Labels such as `[EMAIL-1]` with `prefix` `[` and `suffix` `]`.
    pub fn new(prefix: impl Into<String>, suffix: impl Into<String>) -> Pseudonymizer {
        Pseudonymizer { prefix: prefix.into(), suffix: suffix.into() }
    }

    /// Replaces every mention of `pii`, a layer over `text`, with its
    /// label. The mentions must not overlap.
    pub fn rewrite(&self, text: &str, pii: &Pii) -> Result<PiiRewrite, RewriteError> {
        let mut assigned: HashMap<(PiiKind, &str), String> = HashMap::new();
        let mut counters: HashMap<PiiKind, u32> = HashMap::new();
        PiiRewrite::replace(text, pii.mentions(), |m, out| {
            let label = assigned.entry((m.kind, m.normalized)).or_insert_with(|| {
                let n = counters.entry(m.kind).or_insert(0);
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
