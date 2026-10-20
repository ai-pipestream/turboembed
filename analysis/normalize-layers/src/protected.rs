//! Caller-supplied spans that normalizers and spelling correctors must leave
//! untouched.

use turbo_analysis::{FilterableLayer, Ix, Layer, Span, Spans, TextSpans, ViewId};

/// Stretches of a text view that downstream normalizers and spelling
/// correctors must preserve verbatim.
///
/// The spans are stored in text order with overlapping intervals merged into
/// disjoint ranges.
#[derive(Clone, Debug)]
pub struct ProtectedSpans {
    pub spans: Spans<ProtectedSpans>,
}

impl ProtectedSpans {
    /// An empty `ProtectedSpans` layer anchored to `view`.
    pub fn new(view: ViewId) -> ProtectedSpans {
        ProtectedSpans { spans: Spans::new(view) }
    }

    /// Builds a `ProtectedSpans` layer from arbitrary spans in `view`,
    /// dropping empty spans and merging any overlapping intervals in text
    /// order.
    pub fn from_spans(view: ViewId, raw: impl IntoIterator<Item = Span>) -> ProtectedSpans {
        let mut sorted: Vec<Span> = raw.into_iter().filter(|s| !s.is_empty()).collect();
        sorted.sort_unstable();
        let mut out = ProtectedSpans::new(view);
        let mut merged: Vec<Span> = Vec::with_capacity(sorted.len());
        for span in sorted {
            match merged.last_mut() {
                Some(last) if span.start < last.end => last.end = last.end.max(span.end),
                _ => merged.push(span),
            }
        }
        for span in merged {
            out.spans.push(span);
        }
        out
    }

    /// Appends a single non-empty span that follows every previously pushed
    /// span in text order.
    pub fn push(&mut self, span: Span) -> Ix<ProtectedSpans> {
        self.spans.push(span)
    }
}

impl Layer for ProtectedSpans {
    const ID: &'static str = "turbo:protected_spans";

    fn len(&self) -> usize {
        self.spans.len()
    }

    fn text_spans(&self) -> Option<TextSpans<'_>> {
        Some(self.spans.text_spans())
    }

    fn check(&self) -> Result<(), String> {
        if !self.spans.is_ordered() {
            return Err("protected spans in text order, not overlapping".into());
        }
        self.spans.check(Self::OVERLAP)?;
        if self.spans.iter().any(|(_, s)| s.is_empty()) {
            return Err("no empty protected span".into());
        }
        Ok(())
    }
}

impl FilterableLayer for ProtectedSpans {
    fn retain_indices(&self, kept: &[Ix<Self>]) -> Self {
        ProtectedSpans { spans: self.spans.retain_indices(kept) }
    }
}
