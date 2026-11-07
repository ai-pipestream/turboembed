//! A report of what was found without the values, OpenNLP's
//! `PiiAuditReport`: per kind the mentions, the distinct normalized
//! values, the mentions suppressed by allow rules, and keyed tokens of
//! the first few distinct values.

use std::collections::HashSet;
use std::fmt;

use turbo_analysis_layers::{Pii, PiiKind};

use super::HmacTokenizer;

/// One kind's line of the report.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct KindReport {
    pub kind: PiiKind,
    /// The kind's name in the reported layer.
    pub name: Box<str>,
    pub mentions: usize,
    pub distinct: usize,
    /// Mentions of this kind suppressed by an allow rule (counted without
    /// storing or tokenizing their values).
    pub suppressed: usize,
    /// Tokens of the first distinct values, in the order first seen.
    pub samples: Vec<String>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PiiAuditReport {
    /// The kinds found or suppressed, ordered by name as the Java's `TreeMap`
    /// orders its type strings.
    pub kinds: Vec<KindReport>,
    pub total: usize,
    /// Total mentions suppressed across all kinds.
    pub total_suppressed: usize,
}

impl PiiAuditReport {
    pub const DEFAULT_SAMPLES: usize = 3;

    /// Reports on every mention of `pii`, overlaps included, with tokens
    /// of up to `samples` distinct values per kind.
    pub fn of(pii: &Pii, tokenizer: &HmacTokenizer, samples: usize) -> PiiAuditReport {
        Self::with_suppressed(pii, std::iter::empty(), tokenizer, samples)
    }

    /// Reports on every mention of `pii` together with per-kind counts of
    /// mentions suppressed by allow rules. Original values of suppressed
    /// mentions are never stored or tokenized.
    pub fn with_suppressed(
        pii: &Pii,
        suppressed: impl IntoIterator<Item = (PiiKind, usize)>,
        tokenizer: &HmacTokenizer,
        samples: usize,
    ) -> PiiAuditReport {
        let mut kinds: Vec<KindReport> = Vec::new();
        let mut seen: HashSet<(PiiKind, &str)> = HashSet::new();
        for m in pii.mentions() {
            let at = match kinds.iter().position(|k| k.kind == m.kind) {
                Some(at) => at,
                None => {
                    kinds.push(KindReport {
                        kind: m.kind,
                        name: pii.kind_name(m.kind).unwrap_or("").into(),
                        mentions: 0,
                        distinct: 0,
                        suppressed: 0,
                        samples: Vec::new(),
                    });
                    kinds.len() - 1
                }
            };
            let k = &mut kinds[at];
            k.mentions += 1;
            if seen.insert((m.kind, m.normalized)) {
                k.distinct += 1;
                if k.samples.len() < samples
                    && let Ok(token) = tokenizer.token(pii.kind_name(m.kind).unwrap_or(""), m.normalized)
                {
                    k.samples.push(token);
                }
            }
        }
        for (kind, count) in suppressed {
            if count == 0 {
                continue;
            }
            match kinds.iter_mut().find(|k| k.kind == kind) {
                Some(k) => k.suppressed += count,
                None => kinds.push(KindReport {
                    kind,
                    name: pii.kind_name(kind).unwrap_or("").into(),
                    mentions: 0,
                    distinct: 0,
                    suppressed: count,
                    samples: Vec::new(),
                }),
            }
        }
        kinds.sort_by(|a, b| a.name.cmp(&b.name));
        PiiAuditReport {
            total: kinds.iter().map(|k| k.mentions).sum(),
            total_suppressed: kinds.iter().map(|k| k.suppressed).sum(),
            kinds,
        }
    }

    /// The report of one kind, when it was found or suppressed.
    pub fn kind(&self, kind: PiiKind) -> Option<&KindReport> {
        self.kinds.iter().find(|k| k.kind == kind)
    }

    /// How many mentions of `kind` were suppressed by allow rules.
    pub fn suppressed(&self, kind: PiiKind) -> usize {
        self.kind(kind).map_or(0, |k| k.suppressed)
    }
}

/// One line per kind, `email: 2 mentions, 1 distinct [EMAIL-...]`, lines
/// split by `\n`; `no pii found` for none. When mentions of a kind were
/// suppressed, `, N suppressed` is appended before any sample list.
impl fmt::Display for PiiAuditReport {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.total == 0 && self.total_suppressed == 0 {
            return f.write_str("no pii found");
        }
        for (i, k) in self.kinds.iter().enumerate() {
            if i > 0 {
                f.write_str("\n")?;
            }
            write!(f, "{}: {} mentions, {} distinct", k.name, k.mentions, k.distinct)?;
            if k.suppressed > 0 {
                write!(f, ", {} suppressed", k.suppressed)?;
            }
            if !k.samples.is_empty() {
                write!(f, " [{}]", k.samples.join(", "))?;
            }
        }
        Ok(())
    }
}
