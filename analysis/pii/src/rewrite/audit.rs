//! A report of what was found without the values, OpenNLP's
//! `PiiAuditReport`: per kind the mentions, the distinct normalized
//! values, and keyed tokens of the first few distinct values.

use std::collections::HashSet;
use std::fmt;

use turbo_analysis_layers::{Pii, PiiKind};

use super::HmacTokenizer;

/// One kind's line of the report.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct KindReport {
    pub kind: PiiKind,
    pub mentions: usize,
    pub distinct: usize,
    /// Tokens of the first distinct values, in the order first seen.
    pub samples: Vec<String>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PiiAuditReport {
    /// The kinds found, ordered by name as the Java's `TreeMap` orders
    /// its type strings.
    pub kinds: Vec<KindReport>,
    pub total: usize,
}

impl PiiAuditReport {
    pub const DEFAULT_SAMPLES: usize = 3;

    /// Reports on every mention of `pii`, overlaps included, with tokens
    /// of up to `samples` distinct values per kind.
    pub fn of(pii: &Pii, tokenizer: &HmacTokenizer, samples: usize) -> PiiAuditReport {
        let mut kinds: Vec<KindReport> = Vec::new();
        let mut seen: HashSet<(PiiKind, &str)> = HashSet::new();
        for m in pii.mentions() {
            let at = match kinds.iter().position(|k| k.kind == m.kind) {
                Some(at) => at,
                None => {
                    kinds.push(KindReport { kind: m.kind, mentions: 0, distinct: 0, samples: Vec::new() });
                    kinds.len() - 1
                }
            };
            let k = &mut kinds[at];
            k.mentions += 1;
            if seen.insert((m.kind, m.normalized)) {
                k.distinct += 1;
                if k.samples.len() < samples {
                    k.samples.push(tokenizer.token(m.kind, m.normalized).expect("a layer's values are not empty"));
                }
            }
        }
        kinds.sort_by_key(|k| k.kind.name());
        PiiAuditReport { total: kinds.iter().map(|k| k.mentions).sum(), kinds }
    }

    /// The report of one kind, when it was found.
    pub fn kind(&self, kind: PiiKind) -> Option<&KindReport> {
        self.kinds.iter().find(|k| k.kind == kind)
    }
}

/// One line per kind, `email: 2 mentions, 1 distinct [EMAIL-…]`, lines
/// split by `\n`; `no pii found` for none. Kind names and tokens hold
/// nothing the Java escapes.
impl fmt::Display for PiiAuditReport {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.total == 0 {
            return f.write_str("no pii found");
        }
        for (i, k) in self.kinds.iter().enumerate() {
            if i > 0 {
                f.write_str("\n")?;
            }
            write!(f, "{}: {} mentions, {} distinct", k.kind, k.mentions, k.distinct)?;
            if !k.samples.is_empty() {
                write!(f, " [{}]", k.samples.join(", "))?;
            }
        }
        Ok(())
    }
}
