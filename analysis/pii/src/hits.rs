//! Candidates and their order, OpenNLP's `Hits`: by start, longer first,
//! then by kind priority, then in the order they were found; equal mentions
//! once, the first kept. By default, a candidate strictly nested inside a
//! longer candidate is dropped; OpenNLP parity mode keeps it. Every
//! candidate ends up in the `turbo:pii-candidates` layer with the decision
//! taken on it; the kept ones also become `turbo:pii` mentions.

use std::fmt;
use std::ops::Range;

use turbo_analysis::{Ix, Span};
use turbo_analysis_layers::{
    AllowRule, AllowScope, KindError, Pii, PiiCandidate, PiiCandidates, PiiDetectorRef, PiiDetectors, PiiDisposition,
    PiiEvidence, PiiKind, PiiMention, PiiScore, RejectReason, mention_order,
};

use crate::window::{LabelPolicy, WindowConfig};

mod sink;

pub use sink::PiiSink;

/// Why a candidate handed to [`PiiSink`] was refused.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum SinkError {
    /// The span is empty, past the end of the text, or not on UTF-8
    /// character boundaries.
    Span(Span),
    /// A [`PiiKind::Custom`] that is not in the detector's custom-kind table.
    Kind(KindError),
    /// The normalized form is empty.
    EmptyValue(Span),
    /// No evidence fact was given.
    NoEvidence(Span),
    /// A fact only the detector records: a context label or a deny list
    /// hit. The detector's own label window decides labels, so an
    /// extractor cannot mark a candidate as labelled.
    ForeignEvidence(Span, &'static str),
    /// A score that is not finite or outside its kind's range.
    Score(Span, String),
}

impl fmt::Display for SinkError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            SinkError::Span(s) => write!(f, "the span {}..{} is not a non-empty range of the text", s.start, s.end),
            SinkError::Kind(e) => e.fmt(f),
            SinkError::EmptyValue(s) => {
                write!(f, "candidate at {}..{} has an empty normalized form", s.start, s.end)
            }
            SinkError::NoEvidence(s) => write!(f, "candidate at {}..{} has no evidence", s.start, s.end),
            SinkError::ForeignEvidence(s, kind) => {
                write!(
                    f,
                    "candidate at {}..{} records {kind} evidence, which only the detector records",
                    s.start, s.end
                )
            }
            SinkError::Score(s, detail) => write!(f, "candidate at {}..{}: {detail}", s.start, s.end),
        }
    }
}

impl std::error::Error for SinkError {}

/// One candidate: its facts as a range of the shared fact list, the label
/// the window found (added by the policy pass), its normalized form as a
/// range of the shared buffer, and the rule that rejected it, if one did.
#[derive(Clone, Debug)]
struct Hit {
    start: u32,
    end: u32,
    kind: PiiKind,
    detector: PiiDetectorRef,
    facts: Range<u32>,
    label: Option<PiiEvidence>,
    score: Option<PiiScore>,
    normalized: Range<u32>,
    rejected: Option<RejectReason>,
}

/// The candidates of one text: the scanners append, [`Hits::resolve`]
/// orders them and decides each one.
#[derive(Debug)]
pub(crate) struct Hits {
    hits: Vec<Hit>,
    buf: String,
    facts: Vec<PiiEvidence>,
    detectors: PiiDetectors,
    current: PiiDetectorRef,
}

impl Default for Hits {
    fn default() -> Hits {
        Hits {
            hits: Vec::new(),
            buf: String::new(),
            facts: Vec::new(),
            detectors: PiiDetectors::new(),
            current: PiiDetectorRef::new(0),
        }
    }
}

/// What an allow rule decided for a resolved mention, or nothing.
pub(crate) type AllowDecision = Option<(AllowScope, AllowRule)>;

impl Hits {
    /// Names the detector every candidate added from now on belongs to.
    pub(crate) fn begin_detector(&mut self, id: &str, version: &str) {
        self.current = self.detectors.add(id, version).expect("a valid detector id");
    }

    /// The detector table so far, for a sink registering a scoring model.
    pub(crate) fn detectors_mut(&mut self) -> &mut PiiDetectors {
        &mut self.detectors
    }

    /// Adds a candidate with `facts` whose normalized form `write` appends
    /// to the buffer it is given, so no string is made per candidate.
    pub(crate) fn add_candidate(
        &mut self,
        start: usize,
        end: usize,
        kind: PiiKind,
        facts: &[PiiEvidence],
        score: Option<PiiScore>,
        write: impl FnOnce(&mut String),
    ) -> bool {
        let from = self.buf.len() as u32;
        write(&mut self.buf);
        let to = self.buf.len() as u32;
        if from == to {
            return false;
        }
        let f0 = self.facts.len() as u32;
        self.facts.extend_from_slice(facts);
        let f1 = self.facts.len() as u32;
        self.hits.push(Hit {
            start: start as u32,
            end: end as u32,
            kind,
            detector: self.current,
            facts: f0..f1,
            label: None,
            score,
            normalized: from..to,
            rejected: None,
        });
        true
    }

    /// Adds a candidate with one fact.
    pub(crate) fn add(
        &mut self,
        start: usize,
        end: usize,
        kind: PiiKind,
        fact: PiiEvidence,
        write: impl FnOnce(&mut String),
    ) {
        self.add_candidate(start, end, kind, &[fact], None, write);
    }

    /// Adds a candidate with one fact whose normalized form is the text it
    /// covers.
    pub(crate) fn add_text(&mut self, text: &str, start: usize, end: usize, kind: PiiKind, fact: PiiEvidence) {
        self.add(start, end, kind, fact, |s| s.push_str(&text[start..end]));
    }

    fn normalized(&self, h: &Hit) -> &str {
        &self.buf[h.normalized.start as usize..h.normalized.end as usize]
    }

    /// Applies the context label policies: every candidate of a kind with
    /// a `Required` or `Boosts` policy is checked against the window, a
    /// found label is recorded as a fact, and under `Required` a candidate
    /// without one is rejected. Nothing here reads a tier or a flag an
    /// extractor set.
    pub(crate) fn apply_label_policy(&mut self, text: &str, window: &WindowConfig) {
        for h in &mut self.hits {
            let policy = window.policy(h.kind);
            if policy == LabelPolicy::None {
                continue;
            }
            match window.find_label(text, Span::new(h.start, h.end), h.kind) {
                Some(found) => h.label = Some(PiiEvidence::Label { span: found.span, phrase: found.phrase }),
                None if policy == LabelPolicy::Required => h.rejected = Some(RejectReason::LabelRequired),
                None => {}
            }
        }
    }

    /// Rejects every scored candidate whose score is below `min`. A
    /// candidate without a score is not subject to the threshold.
    pub(crate) fn apply_threshold(&mut self, min: f32) {
        for h in &mut self.hits {
            if h.rejected.is_none() && h.score.is_some_and(|s| s.value < min) {
                h.rejected = Some(RejectReason::Threshold);
            }
        }
    }

    fn evidence(&self, h: &Hit, into: &mut Vec<PiiEvidence>) {
        into.clear();
        into.extend_from_slice(&self.facts[h.facts.start as usize..h.facts.end as usize]);
        into.extend(h.label);
    }

    /// Orders the candidates (a stable sort keeps the finding order of
    /// ties), writes each distinct, admitted mention to `out` and every
    /// candidate with its decision to `candidates`. Equal mentions share a
    /// span, so they sit in one run of that span. When `keep_nested` is
    /// `false`, any candidate whose span lies strictly inside an earlier,
    /// longer mention's span is dropped. `allow(kind, start, end,
    /// normalized)` names the allow rule that suppresses a mention. Both
    /// layers take this run's tables.
    pub(crate) fn resolve(
        mut self,
        keep_nested: bool,
        mut allow: impl FnMut(PiiKind, u32, u32, &str) -> AllowDecision,
        out: &mut Pii,
        candidates: &mut PiiCandidates,
    ) {
        self.hits.sort_by_key(|h| mention_order(h.start, h.end, h.kind));
        out.detectors = self.detectors.clone();
        candidates.detectors = self.detectors.clone();
        let mut run_start = 0;
        let mut outer: Option<(u32, u32, Ix<Pii>)> = None;
        let mut facts = Vec::new();
        // The mention each admitted hit became, by hit index, for repeats.
        let mut became: Vec<Option<Ix<Pii>>> = vec![None; self.hits.len()];
        for i in 0..self.hits.len() {
            let h = &self.hits[i];
            if i > 0 && (self.hits[i - 1].start, self.hits[i - 1].end) != (h.start, h.end) {
                run_start = i;
            }
            self.evidence(h, &mut facts);
            let disposition = if let Some(reason) = h.rejected {
                PiiDisposition::RejectedByRule { reason }
            } else if let Some(mention) = outer
                .filter(|&(os, oe, _)| !keep_nested && os <= h.start && h.end <= oe && (os < h.start || h.end < oe))
            {
                PiiDisposition::NestedIn { mention: mention.2 }
            } else {
                let value = self.normalized(h);
                // The admitted twin, if any: a rejected or suppressed hit
                // with the same kind and value became no mention, so the
                // search goes on past it.
                let repeat = (run_start..i)
                    .filter(|&j| self.hits[j].kind == h.kind && self.normalized(&self.hits[j]) == value)
                    .find_map(|j| became[j]);
                if let Some(mention) = repeat {
                    PiiDisposition::Repeats { mention }
                } else if let Some((scope, rule)) = allow(h.kind, h.start, h.end, value) {
                    PiiDisposition::AllowedByRule { scope, rule }
                } else {
                    let mention = out.push(PiiMention {
                        span: Span::new(h.start, h.end),
                        kind: h.kind,
                        detector: h.detector,
                        evidence: &facts,
                        score: h.score,
                        normalized: value,
                    });
                    became[i] = Some(mention);
                    if outer.is_none_or(|(_, oe, _)| h.end > oe) {
                        outer = Some((h.start, h.end, mention));
                    }
                    PiiDisposition::Kept { mention }
                }
            };
            candidates.push(PiiCandidate {
                span: Span::new(h.start, h.end),
                kind: h.kind,
                detector: h.detector,
                evidence: &facts,
                score: h.score,
                disposition,
            });
        }
    }

    /// The resolved mentions as a layer over `view`, every candidate
    /// admitted.
    #[cfg(test)]
    pub(crate) fn into_layer(self, keep_nested: bool, view: turbo_analysis::ViewId) -> Pii {
        let mut out = Pii::new(view);
        let mut candidates = PiiCandidates::new(view);
        self.resolve(keep_nested, |_, _, _, _| None, &mut out, &mut candidates);
        out
    }
}

#[cfg(test)]
mod tests;
