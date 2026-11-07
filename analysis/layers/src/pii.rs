//! Personal and sensitive identifiers written in a text view: e-mail
//! addresses, phone numbers, payment and bank account numbers, network
//! addresses, credentials, wallet addresses, national and device
//! identifiers.
//!
//! One item per mention with its kind, the detector that found it, the
//! typed facts ([`PiiEvidence`]) that support it, an optional score, and
//! the normalized form. The [`PiiConfidence`] tier is derived from the
//! facts. Mentions of different kinds may overlap, so the layer allows any
//! overlap. The decisions behind the layer (what was found and dropped,
//! and why) are the `turbo:pii-candidates` layer, [`crate::PiiCandidates`].

mod confidence;
mod custom;
mod detectors;
mod evidence;
mod kind;
mod score;

pub use confidence::PiiConfidence;
pub use custom::{CustomKindId, CustomKinds, KindError};
pub use detectors::{DetectorError, PiiDetectorRef, PiiDetectors};
pub use evidence::{ChecksumAlgorithm, EvidenceList, PiiEvidence};
pub use kind::{CUSTOM_CODE_BASE, KindRow, PiiKind, PiiKinds, mention_order};
pub use score::{PiiScore, PiiScoreKind};

use turbo_analysis::{Document, Ix, KeyId, Layer, Overlap, Span, Spans, ViewId};

use crate::Strs;

/// Every PII mention in a view, one item per mention, ordered by start,
/// then longer first, then by kind priority ([`mention_order`]). Equal
/// mentions (span, kind and normalized form) appear once; mentions of
/// different kinds, or of one kind with different normalized forms, may
/// share a span or overlap.
pub struct Pii {
    pub spans: Spans<Pii>,
    /// The table every [`PiiKind::Custom`] of [`Pii::kind`] indexes,
    /// exported with the layer.
    pub custom_kinds: CustomKinds,
    /// The table every [`Pii::detector`] and every [`PiiScore::model`]
    /// indexes, exported with the layer.
    pub detectors: PiiDetectors,
    pub kind: Vec<PiiKind>,
    /// The detector that found each mention.
    pub detector: Vec<PiiDetectorRef>,
    /// The facts behind each mention, at least one.
    pub evidence: EvidenceList,
    /// A numeric score where a model gave one; `None` is not zero.
    pub score: Vec<Option<PiiScore>>,
    /// The normalized value: digits without separators, a lowercased
    /// domain, an RFC 5952 IPv6 address, an EIP-55 address.
    pub normalized: Strs,
}

/// One mention, as handed to [`Pii::push`].
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct PiiMention<'a> {
    pub span: Span,
    pub kind: PiiKind,
    pub detector: PiiDetectorRef,
    pub evidence: &'a [PiiEvidence],
    pub score: Option<PiiScore>,
    pub normalized: &'a str,
}

impl PiiMention<'_> {
    /// The tier the facts derive to ([`PiiConfidence::derive`]).
    pub fn confidence(&self) -> PiiConfidence {
        PiiConfidence::derive(self.evidence)
    }
}

impl Pii {
    /// An empty layer over `view`.
    pub fn new(view: ViewId) -> Pii {
        Pii::with_tables(view, CustomKinds::new(), PiiDetectors::new())
    }

    /// An empty layer over `view` whose custom kinds index `custom_kinds`.
    pub fn with_custom_kinds(view: ViewId, custom_kinds: CustomKinds) -> Pii {
        Pii::with_tables(view, custom_kinds, PiiDetectors::new())
    }

    /// An empty layer over `view` with both tables.
    pub fn with_tables(view: ViewId, custom_kinds: CustomKinds, detectors: PiiDetectors) -> Pii {
        Pii {
            spans: Spans::new(view),
            custom_kinds,
            detectors,
            kind: Vec::new(),
            detector: Vec::new(),
            evidence: EvidenceList::default(),
            score: Vec::new(),
            normalized: Strs::default(),
        }
    }

    /// The name of `kind` in this layer: the built-in name or the entry of
    /// [`Pii::custom_kinds`].
    pub fn kind_name(&self, kind: PiiKind) -> Option<&str> {
        self.custom_kinds.name(kind)
    }

    /// The entry of [`Pii::detectors`] for `id` at `version`, added when
    /// new.
    pub fn detector(&mut self, id: &str, version: &str) -> Result<PiiDetectorRef, DetectorError> {
        self.detectors.add(id, version)
    }

    /// Appends one mention to every column. [`Layer::check`] refuses a
    /// custom kind outside [`Pii::custom_kinds`], a detector outside
    /// [`Pii::detectors`] and an empty evidence list.
    pub fn push(&mut self, m: PiiMention<'_>) -> Ix<Pii> {
        self.kind.push(m.kind);
        self.detector.push(m.detector);
        self.evidence.push(m.evidence);
        self.score.push(m.score);
        self.normalized.push(m.normalized);
        self.spans.push(m.span)
    }

    /// Mention `i`, column by column.
    pub fn get(&self, i: Ix<Pii>) -> PiiMention<'_> {
        let at = i.get();
        PiiMention {
            span: self.spans.get(i),
            kind: self.kind[at],
            detector: self.detector[at],
            evidence: self.evidence.get(at),
            score: self.score[at],
            normalized: self.normalized.get(at),
        }
    }

    /// Every mention, in layer order.
    pub fn mentions(&self) -> impl Iterator<Item = PiiMention<'_>> {
        self.spans.iter().map(|(i, _)| self.get(i))
    }

    fn check_tables(&self, i: usize, m: &PiiMention<'_>) -> Result<(), String> {
        self.custom_kinds.check(m.kind).map_err(|e| format!("pii mention {i}: {e}"))?;
        self.detectors.check(m.detector).map_err(|e| format!("pii mention {i}: {e}"))?;
        if let Some(s) = m.score {
            s.check().map_err(|e| format!("pii mention {i}: {e}"))?;
            self.detectors.check(s.model).map_err(|e| format!("pii mention {i} score model: {e}"))?;
        }
        Ok(())
    }
}

impl Layer for Pii {
    const ID: &'static str = "turbo:pii";
    const OVERLAP: Overlap = Overlap::Any;
    positional!(Pii);

    fn content_hash(&self) -> Option<u64> {
        let mut h = turbo_analysis::Fnv1a::new();
        h.usize(self.kind.len()).u64(self.normalized.content_hash());
        h.usize(self.custom_kinds.len());
        for name in self.custom_kinds.names() {
            h.str(name);
        }
        self.detectors.fold_hash(&mut h);
        self.evidence.fold_hash(&mut h);
        for (&k, &d) in self.kind.iter().zip(&self.detector) {
            h.u32(k.code()).u16(d.index());
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
            || self.normalized.len() != n
        {
            return Err("every column as long as the spans".into());
        }
        self.spans.check(Self::OVERLAP)?;
        self.evidence.check()?;
        let mut previous: Option<PiiMention<'_>> = None;
        for (i, m) in self.mentions().enumerate() {
            if m.span.is_empty() {
                return Err(format!("pii mention {i} is empty at {}", m.span.start));
            }
            self.check_tables(i, &m)?;
            if m.normalized.is_empty() {
                return Err(format!("pii mention {i} has no normalized form"));
            }
            let key = |x: PiiMention<'_>| mention_order(x.span.start, x.span.end, x.kind);
            if let Some(p) = previous
                && key(p) > key(m)
            {
                return Err(format!("pii mention {i} is out of order: by start, longer first, then by kind priority"));
            }
            // Mentions with one span and kind are adjacent, so a duplicate
            // is among the run of those just before.
            let duplicate =
                (0..i).rev().map(|j| self.get(Ix::new(j as u32))).take_while(|p| p.span == m.span && p.kind == m.kind);
            if duplicate.into_iter().any(|p| p.normalized == m.normalized) {
                return Err(format!("pii mention {i} repeats an earlier one"));
            }
            previous = Some(m);
        }
        Ok(())
    }

    fn check_document(&self, doc: &Document, _on: &[KeyId]) -> Result<(), String> {
        self.check_text(doc)
    }
}

impl Pii {
    /// Label spans lie in the view text; the document checks this on
    /// [`Layer::check_document`].
    fn check_text(&self, doc: &Document) -> Result<(), String> {
        self.evidence.check_text(doc.text(self.spans.view()), "pii mention")
    }
}

#[cfg(test)]
#[path = "pii_tests.rs"]
mod tests;
