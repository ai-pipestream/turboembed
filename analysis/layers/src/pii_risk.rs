//! The per-document PII risk (`turbo:pii-risk`, #1889): one score and
//! level for routing a document (index as is, redact, quarantine),
//! computed from the `turbo:pii` layer alone. It holds kinds, counts and
//! weights, never a value.
//!
//! The score is a sum over kinds, in kind code order, of
//! `severity weight x log2(1 + c)`, where `c` is the kind's
//! confidence-weighted count: the sum of [`confidence_weight`] over its
//! mentions. The logarithm keeps one kind from dominating by volume: one
//! fully supported mention scores its weight, three score twice it, seven
//! three times. Every term is positive and grows with its count, so adding
//! a mention never lowers the score. The level comes from the score by
//! fixed cut points ([`PiiRiskLevel::of`]). docs/pii.md has the tables.

mod severity;

use std::collections::BTreeMap;

use turbo_analysis::{Anchor, Layer};

pub use severity::{PiiSeverity, PiiSeverityTable, PiiSuggestedRewrite};

use crate::pii::{CustomKinds, Pii, PiiConfidence, PiiEvidence, PiiKind};

/// A document's PII risk level. [`PiiRiskLevel::id`] is the stable number
/// of the proto `PiiRiskLevel` enum; 0 is its `UNSPECIFIED`.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum PiiRiskLevel {
    /// No mention.
    None = 1,
    Low = 2,
    Medium = 3,
    High = 4,
    Critical = 5,
}

impl PiiRiskLevel {
    pub const ALL: [PiiRiskLevel; 5] =
        [PiiRiskLevel::None, PiiRiskLevel::Low, PiiRiskLevel::Medium, PiiRiskLevel::High, PiiRiskLevel::Critical];

    /// The lowest score of each level above `None`: low above 0, medium
    /// from 3, high from 6, critical from 9.
    pub const CUTS: [(PiiRiskLevel, f32); 3] =
        [(PiiRiskLevel::Medium, 3.0), (PiiRiskLevel::High, 6.0), (PiiRiskLevel::Critical, 9.0)];

    pub const fn id(self) -> u8 {
        self as u8
    }

    pub fn from_id(id: u32) -> Option<PiiRiskLevel> {
        PiiRiskLevel::ALL.into_iter().find(|l| u32::from(l.id()) == id)
    }

    pub const fn name(self) -> &'static str {
        match self {
            PiiRiskLevel::None => "none",
            PiiRiskLevel::Low => "low",
            PiiRiskLevel::Medium => "medium",
            PiiRiskLevel::High => "high",
            PiiRiskLevel::Critical => "critical",
        }
    }

    /// The level of `score`: none at 0 (or below, or NaN), then by
    /// [`PiiRiskLevel::CUTS`].
    pub fn of(score: f32) -> PiiRiskLevel {
        if score.is_nan() || score <= 0.0 {
            return PiiRiskLevel::None;
        }
        PiiRiskLevel::CUTS.iter().rev().find(|&&(_, min)| score >= min).map_or(PiiRiskLevel::Low, |&(l, _)| l)
    }
}

/// The lowest [`confidence_weight`]: the shape alone.
const MIN_WEIGHT: f64 = 0.75;

/// How much one mention's recorded facts support it, from 0.75 to 1: a
/// documented heuristic over its evidence, never a probability. A deny
/// list hit (the caller named the term) and a label with a passed checksum
/// are 1; a passed checksum alone 0.9; a label alone 0.85; the shape alone,
/// or a rule, 0.75.
pub fn confidence_weight(evidence: &[PiiEvidence]) -> f64 {
    if evidence.iter().any(|e| matches!(e, PiiEvidence::DenyHit { .. })) {
        return 1.0;
    }
    match PiiConfidence::derive(evidence) {
        PiiConfidence::Format => MIN_WEIGHT,
        PiiConfidence::Labelled => 0.85,
        PiiConfidence::Checksum => 0.9,
        PiiConfidence::LabelledChecksum => 1.0,
    }
}

/// One kind found in the document: how often, how severe, and what it
/// adds to the score.
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct PiiRiskKind {
    /// A built-in kind, or a custom kind indexing [`PiiRisk::custom_kinds`].
    pub kind: PiiKind,
    pub mentions: u32,
    pub severity: PiiSeverity,
    /// The sum of [`confidence_weight`] over the kind's mentions.
    pub weighted_count: f32,
    /// `severity weight x log2(1 + weighted_count)`.
    pub contribution: f32,
    /// What a suggested redaction policy rewrites the kind with.
    pub suggested: PiiSuggestedRewrite,
}

/// The `turbo:pii-risk` layer: one item for the whole document.
#[derive(Clone, Debug, PartialEq)]
pub struct PiiRisk {
    pub score: f32,
    pub level: PiiRiskLevel,
    pub mentions: u32,
    /// The length of the scanned view in Unicode scalar values.
    pub chars: u32,
    /// Mentions per 1,000 characters; 0 for an empty view.
    pub density_per_1000_chars: f32,
    /// One entry per kind found, by kind code.
    pub kinds: Vec<PiiRiskKind>,
    /// The custom kind names of the `turbo:pii` layer the risk was
    /// computed from; custom [`PiiRiskKind::kind`]s index it.
    pub custom_kinds: CustomKinds,
    /// [`PiiSeverityTable::digest`] of the table used.
    pub severity_table_digest: u64,
}

impl PiiRisk {
    /// The risk of `pii`, found in a view of `text`, under `table`. The
    /// same layer, text and table always give the same risk.
    pub fn compute(pii: &Pii, text: &str, table: &PiiSeverityTable) -> PiiRisk {
        let mut by_code: BTreeMap<u32, (PiiKind, u32, f64)> = BTreeMap::new();
        for m in pii.mentions() {
            let e = by_code.entry(m.kind.code()).or_insert((m.kind, 0, 0.0));
            e.1 += 1;
            e.2 += confidence_weight(m.evidence);
        }
        let mut score = 0.0f64;
        let mut mentions = 0u32;
        let kinds = by_code
            .into_values()
            .map(|(kind, n, weighted)| {
                let severity = table.severity(kind, &pii.custom_kinds);
                let contribution = severity.weight() * (1.0 + weighted).log2();
                score += contribution;
                mentions += n;
                PiiRiskKind {
                    kind,
                    mentions: n,
                    severity,
                    weighted_count: weighted as f32,
                    contribution: contribution as f32,
                    suggested: PiiSuggestedRewrite::default_for(kind),
                }
            })
            .collect();
        let chars = u32::try_from(text.chars().count()).unwrap_or(u32::MAX);
        let density = if chars == 0 { 0.0 } else { f64::from(mentions) * 1000.0 / f64::from(chars) };
        let score = score as f32;
        PiiRisk {
            score,
            level: PiiRiskLevel::of(score),
            mentions,
            chars,
            density_per_1000_chars: density as f32,
            kinds,
            custom_kinds: pii.custom_kinds.clone(),
            severity_table_digest: table.digest(),
        }
    }

    /// The name of `kind` here: built-in, or from [`PiiRisk::custom_kinds`].
    pub fn kind_name(&self, kind: PiiKind) -> Option<&str> {
        self.custom_kinds.name(kind)
    }
}

/// Whether a stored `f32` is the `f64` it was rounded from, give or take
/// the rounding of `compute` and of a sum of rounded terms.
fn close(stored: f64, expected: f64) -> bool {
    (stored - expected).abs() <= 1e-5 * expected.abs().max(1.0)
}

impl PiiRiskKind {
    /// The kind's numbers hold together: each mention weighs from the
    /// lowest to the highest [`confidence_weight`], and the contribution
    /// is the severity weight times `log2(1 + weighted_count)`.
    fn check_weights(&self) -> Result<(), String> {
        let (n, w) = (f64::from(self.mentions), f64::from(self.weighted_count));
        if w < n * MIN_WEIGHT * (1.0 - 1e-6) || w > n * (1.0 + 1e-6) {
            return Err("weighted count is outside what its mentions can weigh".into());
        }
        if !close(f64::from(self.contribution), self.severity.weight() * (1.0 + w).log2()) {
            return Err("contribution is not its severity weight times log2(1 + weighted count)".into());
        }
        Ok(())
    }
}

fn finite(x: f32, what: &str) -> Result<(), String> {
    if x.is_finite() && x >= 0.0 { Ok(()) } else { Err(format!("pii risk {what} must be finite and not negative")) }
}

impl Layer for PiiRisk {
    const ID: &'static str = "turbo:pii-risk";

    fn anchor() -> Anchor {
        Anchor::Document
    }

    fn len(&self) -> usize {
        1
    }

    fn check(&self) -> Result<(), String> {
        finite(self.score, "score")?;
        finite(self.density_per_1000_chars, "density")?;
        if self.level != PiiRiskLevel::of(self.score) {
            return Err(format!("pii risk level {} does not match score {}", self.level.name(), self.score));
        }
        let mut total = 0u64;
        let mut sum = 0.0f64;
        let mut previous: Option<u32> = None;
        // Every relation compute writes is held: counts sum, each kind's
        // weight and contribution follow from its mentions and severity,
        // the score is the sum of the contributions, and the density
        // follows from the counts, within f32 rounding.
        for (i, k) in self.kinds.iter().enumerate() {
            self.custom_kinds.check(k.kind).map_err(|e| format!("pii risk kind {i}: {e}"))?;
            if previous.is_some_and(|p| p >= k.kind.code()) {
                return Err(format!("pii risk kind {i} is out of order or repeated"));
            }
            if k.mentions == 0 {
                return Err(format!("pii risk kind {i} has no mention"));
            }
            finite(k.weighted_count, "weighted count")?;
            finite(k.contribution, "contribution")?;
            k.check_weights().map_err(|e| format!("pii risk kind {i}: {e}"))?;
            previous = Some(k.kind.code());
            total += u64::from(k.mentions);
            sum += f64::from(k.contribution);
        }
        if total != u64::from(self.mentions) {
            return Err("pii risk mentions are not the sum over its kinds".into());
        }
        if !close(f64::from(self.score), sum) {
            return Err(format!("pii risk score {} is not the sum of its contributions", self.score));
        }
        if self.mentions > 0 && self.chars == 0 {
            return Err("pii risk has mentions in an empty view".into());
        }
        let density = if self.chars == 0 { 0.0 } else { f64::from(self.mentions) * 1000.0 / f64::from(self.chars) };
        if !close(f64::from(self.density_per_1000_chars), density) {
            return Err("pii risk density is not its mentions per 1,000 characters".into());
        }
        Ok(())
    }

    fn content_hash(&self) -> Option<u64> {
        let mut h = turbo_analysis::Fnv1a::new();
        h.f32(self.score).u8(self.level.id()).u32(self.mentions).u32(self.chars).f32(self.density_per_1000_chars);
        h.u64(self.severity_table_digest).usize(self.custom_kinds.len());
        for name in self.custom_kinds.names() {
            h.str(name);
        }
        h.usize(self.kinds.len());
        for k in &self.kinds {
            h.u32(k.kind.code()).u32(k.mentions).u8(k.severity.id()).f32(k.weighted_count).f32(k.contribution);
            h.u8(k.suggested.id());
        }
        Some(h.finish())
    }
}

#[cfg(test)]
#[path = "pii_risk_tests.rs"]
mod tests;
