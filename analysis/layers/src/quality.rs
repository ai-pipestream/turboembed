//! View-specific, deterministic text quality signals. Scores are heuristics,
//! not probabilities or language proficiency assessments.

use std::ops::{BitOr, BitOrAssign};

use turbo_analysis::{Document, FilterableLayer, Ix, KeyId, Layer, Span, Spans, TextSpans, ViewId};

mod gates;
pub use gates::{lexical_quality_gate, lexical_quality_gate_on, quality_gate, quality_gate_on};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub struct QualityFlags(pub u16);

impl QualityFlags {
    pub const CLEAN: Self = Self(0);
    pub const MOJIBAKE_SUSPECT: Self = Self(1 << 0);
    pub const OCR_BROKEN_HYPHENS: Self = Self(1 << 1);
    pub const HIGH_SYMBOL_NOISE: Self = Self(1 << 2);
    pub const HIGH_DIGIT_DENSITY: Self = Self(1 << 3);
    pub const REPETITIVE_NGRAM: Self = Self(1 << 4);
    pub const NAV_BOILERPLATE: Self = Self(1 << 5);
    pub const TOO_SHORT: Self = Self(1 << 6);
    pub const ALL: Self = Self((1 << 7) - 1);

    pub const fn contains(self, other: Self) -> bool {
        self.0 & other.0 == other.0
    }

    pub const fn intersects(self, other: Self) -> bool {
        self.0 & other.0 != 0
    }
}

impl BitOr for QualityFlags {
    type Output = Self;
    fn bitor(self, rhs: Self) -> Self {
        Self(self.0 | rhs.0)
    }
}

impl BitOrAssign for QualityFlags {
    fn bitor_assign(&mut self, rhs: Self) {
        self.0 |= rhs.0;
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct QualityMetrics {
    pub composite_score: f32,
    /// Forward/reverse mean MTLD, with a TTR threshold of 0.72.
    /// Zero for no words; all-unique remainders use their observed length.
    pub mtld_lexical_diversity: f32,
    pub flesch_kincaid_grade: f32,
    pub automated_readability_index: f32,
    pub coleman_liau_index: f32,
    /// Non-stopword alphabetic tokens divided by all input tokens.
    pub information_density: f32,
    /// Character ratios use Unicode scalar counts, including whitespace.
    pub alpha_ratio: f32,
    pub digit_ratio: f32,
    /// Everything except letters, combining marks, digits and whitespace.
    /// Includes replacement characters; `replacement_ratio` is a subset.
    pub symbol_ratio: f32,
    pub whitespace_ratio: f32,
    pub replacement_ratio: f32,
    pub flags: QualityFlags,
}

impl QualityMetrics {
    pub fn check(&self) -> Result<(), String> {
        let ratios = [
            self.composite_score,
            self.information_density,
            self.alpha_ratio,
            self.digit_ratio,
            self.symbol_ratio,
            self.whitespace_ratio,
            self.replacement_ratio,
        ];
        if ratios.iter().any(|x| !x.is_finite() || !(0.0..=1.0).contains(x)) {
            return Err("quality ratios must be finite and in [0, 1]".into());
        }
        if !self.mtld_lexical_diversity.is_finite() || self.mtld_lexical_diversity < 0.0 {
            return Err("MTLD must be finite and nonnegative".into());
        }
        if [self.flesch_kincaid_grade, self.automated_readability_index, self.coleman_liau_index]
            .iter()
            .any(|x| !x.is_finite())
        {
            return Err("readability indices must be finite".into());
        }
        if self.flags.0 & !QualityFlags::ALL.0 != 0 {
            return Err("unknown quality flag bits".into());
        }
        Ok(())
    }
}

/// One summary covering an entire text view. The span preserves view identity
/// through the graph wire format, including named and normalized views.
#[derive(Debug, Clone)]
pub struct TextQuality {
    pub spans: Spans<TextQuality>,
    pub metrics: QualityMetrics,
}

impl TextQuality {
    pub fn new(view: ViewId, text_len: u32, metrics: QualityMetrics) -> Self {
        let mut spans = Spans::with_capacity(view, 1);
        spans.push(Span::new(0, text_len));
        Self { spans, metrics }
    }

    pub fn view(&self) -> ViewId {
        self.spans.view()
    }
}

impl Layer for TextQuality {
    const ID: &'static str = "turbo:text_quality";
    fn len(&self) -> usize {
        1
    }
    fn text_spans(&self) -> Option<TextSpans<'_>> {
        Some(self.spans.text_spans())
    }
    fn check(&self) -> Result<(), String> {
        if self.spans.len() != 1 || self.spans.is_discontinuous(Ix::new(0)) {
            return Err("one continuous span covering the quality view".into());
        }
        self.metrics.check()
    }
    fn check_document(&self, doc: &Document, _on: &[KeyId]) -> Result<(), String> {
        if self.spans.get(Ix::new(0)) != Span::new(0, doc.text(self.view()).len() as u32) {
            return Err("text quality must cover its entire view".into());
        }
        Ok(())
    }
}

/// Nonoverlapping paragraph or leaf-block summaries in text order.
#[derive(Debug, Clone)]
pub struct SpanQuality {
    pub spans: Spans<SpanQuality>,
    pub metrics: Vec<QualityMetrics>,
}

impl SpanQuality {
    pub fn new(view: ViewId) -> Self {
        Self { spans: Spans::new(view), metrics: Vec::new() }
    }
    pub fn push(&mut self, span: Span, metrics: QualityMetrics) -> Ix<Self> {
        self.metrics.push(metrics);
        self.spans.push(span)
    }
}

impl Layer for SpanQuality {
    const ID: &'static str = "turbo:span_quality";
    fn len(&self) -> usize {
        self.spans.len()
    }
    fn text_spans(&self) -> Option<TextSpans<'_>> {
        Some(self.spans.text_spans())
    }
    fn check(&self) -> Result<(), String> {
        self.spans.check(Self::OVERLAP)?;
        if self.metrics.len() != self.spans.len() || !self.spans.is_ordered() {
            return Err("one quality row per span, in nonoverlapping text order".into());
        }
        for (i, span) in self.spans.iter() {
            if span.is_empty() || self.spans.is_discontinuous(i) {
                return Err("quality regions must be nonempty continuous spans".into());
            }
            self.metrics[i.get()].check()?;
        }
        Ok(())
    }
}

impl FilterableLayer for SpanQuality {
    fn retain_indices(&self, kept: &[Ix<Self>]) -> Self {
        Self { spans: self.spans.retain_indices(kept), metrics: kept.iter().map(|i| self.metrics[i.get()]).collect() }
    }
}
