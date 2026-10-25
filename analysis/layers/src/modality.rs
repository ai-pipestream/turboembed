//! Modality (`turbo:modality`): clause or predicate scopes tagged with
//! polarity, epistemic status and deontic force, so a consumer of
//! relations or events can tell an asserted fact from a negated, hedged,
//! reported, conditional or obligated one.
//!
//! [`Polarity`] is re-exported at the crate root as `ModalityPolarity`,
//! since the root `Polarity` is the sentiment one.

use turbo_analysis::{Ix, Layer, Overlap, Span, Spans, TextSpans, ViewId};

/// Whether a scope is affirmed or negated.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
#[repr(u8)]
pub enum Polarity {
    #[default]
    Affirmed = 0,
    Negated = 1,
}

/// How firmly a scope is put forward.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
#[repr(u8)]
pub enum EpistemicStatus {
    /// An unhedged statement.
    #[default]
    Asserted = 0,
    /// Uncertain or speculative (`might`, `possibly`, `estimated`).
    Hedged = 1,
    /// Second-hand (`allegedly`, `reportedly`, `according to`).
    Reported = 2,
    /// A hypothetical antecedent (`if`, `unless`, `assuming`).
    Conditional = 3,
    /// A factive complement (`realized that`, `confirmed that`).
    Presupposed = 4,
}

/// The deontic force of a scope.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
#[repr(u8)]
pub enum DeonticKind {
    #[default]
    None = 0,
    /// `must`, `shall`, `required to`.
    Obligation = 1,
    /// `may` (of a party), `permitted to`, `allowed to`.
    Permission = 2,
    /// `must not`, `shall not`, `prohibited from`.
    Prohibition = 3,
}

impl Polarity {
    pub fn from_byte(b: u8) -> Option<Polarity> {
        [Polarity::Affirmed, Polarity::Negated].get(b as usize).copied()
    }
}

impl EpistemicStatus {
    pub fn from_byte(b: u8) -> Option<EpistemicStatus> {
        use EpistemicStatus::*;
        [Asserted, Hedged, Reported, Conditional, Presupposed].get(b as usize).copied()
    }
}

impl DeonticKind {
    pub fn from_byte(b: u8) -> Option<DeonticKind> {
        use DeonticKind::*;
        [None, Obligation, Permission, Prohibition].get(b as usize).copied()
    }
}

/// The modality of one scope.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct ModalityTag {
    pub polarity: Polarity,
    pub epistemic: EpistemicStatus,
    pub deontic: DeonticKind,
    /// The primary cue (`not`, `allegedly`, `shall`), when there is one.
    pub cue: Option<Span>,
    /// In `[0, 1]`.
    pub confidence: f32,
}

/// Tagged scopes in start order, by column. Scopes may nest; the
/// innermost one covering a span is its modality.
#[derive(Debug, Clone)]
pub struct Modality {
    pub spans: Spans<Modality>,
    pub polarity: Vec<Polarity>,
    pub epistemic: Vec<EpistemicStatus>,
    pub deontic: Vec<DeonticKind>,
    pub cue: Vec<Option<Span>>,
    pub confidence: Vec<f32>,
    /// The largest scope end among rows `0..=i`, for [`Self::tag_for_span`].
    reach: Vec<u32>,
}

impl Default for Modality {
    fn default() -> Self {
        Modality::new(ViewId(0))
    }
}

impl PartialEq for Modality {
    fn eq(&self, other: &Self) -> bool {
        self.spans.view() == other.spans.view() && self.iter().eq(other.iter())
    }
}

impl Modality {
    pub fn new(view: ViewId) -> Modality {
        Modality {
            spans: Spans::new(view),
            polarity: Vec::new(),
            epistemic: Vec::new(),
            deontic: Vec::new(),
            cue: Vec::new(),
            confidence: Vec::new(),
            reach: Vec::new(),
        }
    }

    /// Appends a scope; scopes must be pushed in start order for
    /// [`Self::tag_for_span`] and `check`.
    pub fn push(&mut self, scope: Span, tag: ModalityTag) -> Ix<Modality> {
        self.polarity.push(tag.polarity);
        self.epistemic.push(tag.epistemic);
        self.deontic.push(tag.deontic);
        self.cue.push(tag.cue);
        self.confidence.push(tag.confidence);
        self.reach.push(self.reach.last().map_or(scope.end, |&r| r.max(scope.end)));
        self.spans.push(scope)
    }

    /// The number of scopes.
    pub fn len(&self) -> usize {
        self.polarity.len()
    }

    pub fn is_empty(&self) -> bool {
        self.polarity.is_empty()
    }

    /// The tag of row `i`.
    ///
    /// # Panics
    ///
    /// When `i` is out of bounds.
    pub fn tag(&self, i: usize) -> ModalityTag {
        ModalityTag {
            polarity: self.polarity[i],
            epistemic: self.epistemic[i],
            deontic: self.deontic[i],
            cue: self.cue[i],
            confidence: self.confidence[i],
        }
    }

    /// Every row as `(scope, tag)`.
    pub fn iter(&self) -> impl Iterator<Item = (Span, ModalityTag)> + '_ {
        (0..self.spans.len()).map(|i| (self.spans.get(Ix::new(i as u32)), self.tag(i)))
    }

    /// The tag of the innermost scope covering `target`, if any. A binary
    /// search finds the last scope starting at or before `target`; the
    /// walk back from there stops once no earlier scope reaches
    /// `target`'s end, so it costs `O(log N + k)` for `k` scopes that
    /// start before `target` and end after it.
    pub fn tag_for_span(&self, target: Span) -> Option<ModalityTag> {
        let n = self.spans.len();
        let mut lo = 0;
        let mut hi = n;
        while lo < hi {
            let mid = (lo + hi) / 2;
            if self.spans.get(Ix::new(mid as u32)).start <= target.start {
                lo = mid + 1;
            } else {
                hi = mid;
            }
        }
        let mut best: Option<(u32, usize)> = None;
        for i in (0..lo).rev() {
            if self.reach[i] < target.end {
                break;
            }
            let s = self.spans.get(Ix::new(i as u32));
            if s.end >= target.end && best.is_none_or(|(len, _)| s.end - s.start < len) {
                best = Some((s.end - s.start, i));
            }
        }
        best.map(|(_, i)| self.tag(i))
    }
}

impl Layer for Modality {
    const ID: &'static str = "turbo:modality";
    const OVERLAP: Overlap = Overlap::Any;

    fn len(&self) -> usize {
        self.spans.len()
    }

    fn text_spans(&self) -> Option<TextSpans<'_>> {
        Some(self.spans.text_spans())
    }

    fn check(&self) -> Result<(), String> {
        let n = self.spans.len();
        let lens = [
            self.polarity.len(),
            self.epistemic.len(),
            self.deontic.len(),
            self.cue.len(),
            self.confidence.len(),
            self.reach.len(),
        ];
        if lens.iter().any(|&l| l != n) {
            return Err(format!("columns of lengths {lens:?} for {n} scopes"));
        }
        self.spans.check(Self::OVERLAP)?;
        if !self.spans.is_start_ordered() {
            return Err("modality scopes must be in start order".into());
        }
        let mut reach = 0;
        for (i, (scope, tag)) in self.iter().enumerate() {
            reach = if i == 0 { scope.end } else { reach.max(scope.end) };
            if self.reach[i] != reach {
                return Err(format!("stale reach {} at scope {i}, expected {reach}", self.reach[i]));
            }
            if scope.is_empty() {
                return Err(format!("an empty modality scope at {}", scope.start));
            }
            if tag.cue.is_some_and(|c| c.is_empty() || c.start > c.end) {
                return Err(format!("an empty cue on the scope at {}", scope.start));
            }
            if !(0.0..=1.0).contains(&tag.confidence) {
                return Err(format!("modality confidence {} outside [0, 1]", tag.confidence));
            }
        }
        Ok(())
    }
}
