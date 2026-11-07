//! Every candidate a PII detector considered, with the decision it took:
//! the `turbo:pii-candidates` layer behind `turbo:pii`. One item per
//! candidate in mention order, each with its kind, detector, evidence and
//! optional score, and a [`PiiDisposition`] saying whether it became a
//! mention and if not, why. The layer carries no values: a kept candidate
//! points at its mention in `turbo:pii`, and a dropped one has nothing to
//! point at. Candidate locations and evidence are still sensitive and are
//! released under the same disclosure decision as mentions.

use turbo_analysis::{Fnv1a, Ix, Layer, Overlap, Span, Spans, ViewId};

use crate::pii::{
    CustomKinds, EvidenceList, Pii, PiiDetectorRef, PiiDetectors, PiiEvidence, PiiKind, PiiScore, mention_order,
};

/// Which allow list suppressed a candidate.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub enum AllowScope {
    /// The detector's global allow list.
    Global,
    /// The allow list scoped to the candidate's kind.
    Kind,
}

/// The allow rule that matched: an exact term or a pattern, by its index
/// in its list's registration order.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub enum AllowRule {
    Term(u32),
    Pattern(u32),
}

/// Why a candidate was rejected by a detector rule.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub enum RejectReason {
    /// The kind requires a context label and none was in the window.
    LabelRequired,
    /// The score is below the detector's threshold.
    Threshold,
}

impl RejectReason {
    /// The stable id: label required 1, threshold 2.
    pub const fn id(self) -> u8 {
        match self {
            RejectReason::LabelRequired => 1,
            RejectReason::Threshold => 2,
        }
    }

    pub const fn from_id(id: u32) -> Option<RejectReason> {
        match id {
            1 => Some(RejectReason::LabelRequired),
            2 => Some(RejectReason::Threshold),
            _ => None,
        }
    }

    pub const fn name(self) -> &'static str {
        match self {
            RejectReason::LabelRequired => "label-required",
            RejectReason::Threshold => "threshold",
        }
    }
}

/// What the detector did with a candidate.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub enum PiiDisposition {
    /// Became this mention of `turbo:pii`.
    Kept { mention: Ix<Pii> },
    /// Suppressed by an allow rule.
    AllowedByRule { scope: AllowScope, rule: AllowRule },
    /// Dropped as strictly nested inside this mention of `turbo:pii`.
    NestedIn { mention: Ix<Pii> },
    /// Rejected by a detector rule.
    RejectedByRule { reason: RejectReason },
    /// Dropped as a repeat (same span, kind and value) of this mention.
    Repeats { mention: Ix<Pii> },
}

impl PiiDisposition {
    /// The stable tag: kept 1, allowed 2, nested 3, rejected 4, repeats 5.
    pub const fn tag(self) -> u8 {
        match self {
            PiiDisposition::Kept { .. } => 1,
            PiiDisposition::AllowedByRule { .. } => 2,
            PiiDisposition::NestedIn { .. } => 3,
            PiiDisposition::RejectedByRule { .. } => 4,
            PiiDisposition::Repeats { .. } => 5,
        }
    }

    pub const fn name(self) -> &'static str {
        match self {
            PiiDisposition::Kept { .. } => "kept",
            PiiDisposition::AllowedByRule { .. } => "allowed-by-rule",
            PiiDisposition::NestedIn { .. } => "nested-in",
            PiiDisposition::RejectedByRule { .. } => "rejected-by-rule",
            PiiDisposition::Repeats { .. } => "repeats",
        }
    }

    /// Whether the candidate was dropped by an allow rule.
    pub const fn is_suppressed(self) -> bool {
        matches!(self, PiiDisposition::AllowedByRule { .. })
    }

    /// The mention this decision refers to, when it refers to one.
    pub const fn mention(self) -> Option<Ix<Pii>> {
        match self {
            PiiDisposition::Kept { mention }
            | PiiDisposition::NestedIn { mention }
            | PiiDisposition::Repeats { mention } => Some(mention),
            PiiDisposition::AllowedByRule { .. } | PiiDisposition::RejectedByRule { .. } => None,
        }
    }

    pub fn fold_hash(self, h: &mut Fnv1a) {
        h.u8(self.tag());
        match self {
            PiiDisposition::Kept { mention }
            | PiiDisposition::NestedIn { mention }
            | PiiDisposition::Repeats { mention } => {
                h.usize(mention.get());
            }
            PiiDisposition::AllowedByRule { scope, rule } => {
                h.u8(match scope {
                    AllowScope::Global => 1,
                    AllowScope::Kind => 2,
                });
                match rule {
                    AllowRule::Term(i) => h.u8(1).u32(i),
                    AllowRule::Pattern(i) => h.u8(2).u32(i),
                };
            }
            PiiDisposition::RejectedByRule { reason } => {
                h.u8(reason.id());
            }
        }
    }
}

/// One candidate, as handed to [`PiiCandidates::push`].
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct PiiCandidate<'a> {
    pub span: Span,
    pub kind: PiiKind,
    pub detector: PiiDetectorRef,
    pub evidence: &'a [PiiEvidence],
    pub score: Option<PiiScore>,
    pub disposition: PiiDisposition,
}

/// Every candidate of one detection run, in mention order.
pub struct PiiCandidates {
    pub spans: Spans<PiiCandidates>,
    /// The same tables as the `turbo:pii` layer of the run.
    pub custom_kinds: CustomKinds,
    pub detectors: PiiDetectors,
    pub kind: Vec<PiiKind>,
    pub detector: Vec<PiiDetectorRef>,
    pub evidence: EvidenceList,
    pub score: Vec<Option<PiiScore>>,
    pub disposition: Vec<PiiDisposition>,
}

impl PiiCandidates {
    pub fn new(view: ViewId) -> PiiCandidates {
        PiiCandidates::with_tables(view, CustomKinds::new(), PiiDetectors::new())
    }

    pub fn with_tables(view: ViewId, custom_kinds: CustomKinds, detectors: PiiDetectors) -> PiiCandidates {
        PiiCandidates {
            spans: Spans::new(view),
            custom_kinds,
            detectors,
            kind: Vec::new(),
            detector: Vec::new(),
            evidence: EvidenceList::default(),
            score: Vec::new(),
            disposition: Vec::new(),
        }
    }

    pub fn kind_name(&self, kind: PiiKind) -> Option<&str> {
        self.custom_kinds.name(kind)
    }

    pub fn push(&mut self, c: PiiCandidate<'_>) -> Ix<PiiCandidates> {
        self.kind.push(c.kind);
        self.detector.push(c.detector);
        self.evidence.push(c.evidence);
        self.score.push(c.score);
        self.disposition.push(c.disposition);
        self.spans.push(c.span)
    }

    pub fn get(&self, i: Ix<PiiCandidates>) -> PiiCandidate<'_> {
        let at = i.get();
        PiiCandidate {
            span: self.spans.get(i),
            kind: self.kind[at],
            detector: self.detector[at],
            evidence: self.evidence.get(at),
            score: self.score[at],
            disposition: self.disposition[at],
        }
    }

    pub fn candidates(&self) -> impl Iterator<Item = PiiCandidate<'_>> {
        self.spans.iter().map(|(i, _)| self.get(i))
    }

    /// How many candidates of each kind an allow rule suppressed, ordered
    /// by kind name as the audit report orders its kinds.
    pub fn suppressed_by_kind(&self) -> Vec<(PiiKind, usize)> {
        let mut counts: Vec<(PiiKind, usize)> = Vec::new();
        for c in self.candidates().filter(|c| c.disposition.is_suppressed()) {
            match counts.iter_mut().find(|(k, _)| *k == c.kind) {
                Some((_, n)) => *n += 1,
                None => counts.push((c.kind, 1)),
            }
        }
        counts.sort_by(|(a, _), (b, _)| self.kind_name(*a).cmp(&self.kind_name(*b)));
        counts
    }

    /// Checks every mention reference against `pii`, the layer of the same
    /// run.
    pub fn check_against(&self, pii: &Pii) -> Result<(), String> {
        for (i, d) in self.disposition.iter().enumerate() {
            if let Some(m) = d.mention()
                && m.get() >= pii.len()
            {
                return Err(format!("pii candidate {i} refers to mention {} of {} mentions", m.get(), pii.len()));
            }
        }
        Ok(())
    }
}

impl Layer for PiiCandidates {
    const ID: &'static str = "turbo:pii-candidates";
    const OVERLAP: Overlap = Overlap::Any;
    positional!(PiiCandidates);

    fn content_hash(&self) -> Option<u64> {
        let mut h = Fnv1a::new();
        h.usize(self.kind.len()).usize(self.custom_kinds.len());
        for name in self.custom_kinds.names() {
            h.str(name);
        }
        self.detectors.fold_hash(&mut h);
        self.evidence.fold_hash(&mut h);
        for ((&k, &d), &disp) in self.kind.iter().zip(&self.detector).zip(&self.disposition) {
            h.u32(k.code()).u16(d.index());
            disp.fold_hash(&mut h);
        }
        for s in &self.score {
            match s {
                None => {
                    h.u8(0);
                }
                Some(s) => {
                    h.u8(1);
                    s.fold_hash(&mut h);
                }
            }
        }
        Some(h.finish())
    }

    fn check(&self) -> Result<(), String> {
        let n = self.spans.len();
        if self.kind.len() != n
            || self.detector.len() != n
            || self.evidence.len() != n
            || self.score.len() != n
            || self.disposition.len() != n
        {
            return Err("every column as long as the spans".into());
        }
        self.spans.check(Self::OVERLAP)?;
        self.evidence.check()?;
        let mut previous = None;
        for (i, c) in self.candidates().enumerate() {
            if c.span.is_empty() {
                return Err(format!("pii candidate {i} is empty at {}", c.span.start));
            }
            self.custom_kinds.check(c.kind).map_err(|e| format!("pii candidate {i}: {e}"))?;
            self.detectors.check(c.detector).map_err(|e| format!("pii candidate {i}: {e}"))?;
            if let Some(s) = c.score {
                s.check().map_err(|e| format!("pii candidate {i}: {e}"))?;
                self.detectors.check(s.model).map_err(|e| format!("pii candidate {i} score model: {e}"))?;
            }
            let key = mention_order(c.span.start, c.span.end, c.kind);
            if previous.is_some_and(|p| p > key) {
                return Err(format!(
                    "pii candidate {i} is out of order: by start, longer first, then by kind priority"
                ));
            }
            previous = Some(key);
        }
        Ok(())
    }
}
