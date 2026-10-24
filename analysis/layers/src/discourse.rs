//! Reported speech (`turbo:quotations`): each quoted or reported
//! statement with its speaker, its reporting verb and the coreference
//! chain the speaker belongs to.

use turbo_analysis::{Layer, Overlap, Span, Spans, TextSpans, ViewId};

/// How a statement is reported.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum QuoteKind {
    /// Inside quotation marks: `"We grew," she said.`
    Direct = 0,
    /// A clausal complement of a reporting verb: `She said that they grew.`
    Indirect = 1,
    /// A reported clause that quotes part of itself: `She said the year
    /// was "a turning point".`
    Mixed = 2,
}

impl QuoteKind {
    pub const fn to_byte(self) -> u8 {
        self as u8
    }

    pub const fn from_byte(b: u8) -> Option<QuoteKind> {
        match b {
            0 => Some(QuoteKind::Direct),
            1 => Some(QuoteKind::Indirect),
            2 => Some(QuoteKind::Mixed),
            _ => None,
        }
    }
}

/// One row of [`Quotations`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Quotation {
    /// The bytes of the statement, quotation marks excluded.
    pub content: Span,
    /// The speaker mention (`the CFO`, `she`), when found.
    pub speaker: Option<Span>,
    /// The reporting verb (`said`, `warned`), when found.
    pub cue_verb: Option<Span>,
    /// The row of the default `Groups<Mentions>` (coreference chains) the
    /// speaker belongs to, when resolved.
    pub speaker_chain: Option<u32>,
    pub kind: QuoteKind,
    /// In `[0, 1]`.
    pub confidence: f32,
}

/// Quoted and reported statements, in content start order. Speaker and
/// cue spans are on the content spans' view.
#[derive(Debug, Clone)]
pub struct Quotations {
    pub spans: Spans<Quotations>,
    pub speaker: Vec<Option<Span>>,
    pub cue_verb: Vec<Option<Span>>,
    pub speaker_chain: Vec<Option<u32>>,
    pub kind: Vec<QuoteKind>,
    pub confidence: Vec<f32>,
}

impl PartialEq for Quotations {
    fn eq(&self, other: &Self) -> bool {
        self.spans.view() == other.spans.view() && self.iter().eq(other.iter())
    }
}

impl Default for Quotations {
    fn default() -> Self {
        Quotations::new(ViewId(0))
    }
}

impl Quotations {
    pub fn new(view: ViewId) -> Quotations {
        Quotations {
            spans: Spans::new(view),
            speaker: Vec::new(),
            cue_verb: Vec::new(),
            speaker_chain: Vec::new(),
            kind: Vec::new(),
            confidence: Vec::new(),
        }
    }

    pub fn push(&mut self, q: Quotation) {
        self.spans.push(q.content);
        self.speaker.push(q.speaker);
        self.cue_verb.push(q.cue_verb);
        self.speaker_chain.push(q.speaker_chain);
        self.kind.push(q.kind);
        self.confidence.push(q.confidence);
    }

    /// Row `i`, when there is one.
    pub fn get(&self, i: usize) -> Option<Quotation> {
        (i < self.spans.len()).then(|| Quotation {
            content: self.spans.get(turbo_analysis::Ix::new(i as u32)),
            speaker: self.speaker[i],
            cue_verb: self.cue_verb[i],
            speaker_chain: self.speaker_chain[i],
            kind: self.kind[i],
            confidence: self.confidence[i],
        })
    }

    pub fn iter(&self) -> impl Iterator<Item = Quotation> + '_ {
        (0..self.spans.len()).filter_map(|i| self.get(i))
    }
}

impl Layer for Quotations {
    const ID: &'static str = "turbo:quotations";
    const OVERLAP: Overlap = Overlap::Any;

    fn len(&self) -> usize {
        self.spans.len()
    }

    fn text_spans(&self) -> Option<TextSpans<'_>> {
        Some(self.spans.text_spans())
    }

    fn check(&self) -> Result<(), String> {
        let n = self.spans.len();
        let lens =
            [self.speaker.len(), self.cue_verb.len(), self.speaker_chain.len(), self.kind.len(), self.confidence.len()];
        if lens.iter().any(|&l| l != n) {
            return Err(format!("columns of lengths {lens:?} for {n} quotations"));
        }
        self.spans.check(Self::OVERLAP)?;
        if !self.spans.is_start_ordered() {
            return Err("quotations must be in start order".into());
        }
        for q in self.iter() {
            if q.content.is_empty() {
                return Err(format!("an empty quotation at {}", q.content.start));
            }
            if [q.speaker, q.cue_verb].iter().flatten().any(|s| s.is_empty()) {
                return Err(format!("an empty speaker or cue span on the quotation at {}", q.content.start));
            }
            if !(0.0..=1.0).contains(&q.confidence) {
                return Err(format!("quotation confidence {} outside [0, 1]", q.confidence));
            }
        }
        Ok(())
    }
}
