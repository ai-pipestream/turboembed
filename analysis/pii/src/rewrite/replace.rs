//! Replacing mentions with labels, OpenNLP's `PiiRewrite`: the new text,
//! the labels as mentions of it, and a map from offsets in the original
//! to offsets in the new text.

use std::fmt;

use turbo_analysis::{Span, ViewId};
use turbo_analysis_layers::{Pii, PiiMention};

/// Mentions a rewrite cannot replace.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum RewriteError {
    /// A span past the end of the text or not on character boundaries.
    Span(Span),
    /// Two mentions overlap, so one replacement would cut the other.
    Overlap(Span, Span),
    /// A mention has an empty normalized form that cannot be tokenized.
    EmptyValue(Span),
}

impl fmt::Display for RewriteError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            RewriteError::Span(s) => write!(f, "the span {}..{} is not a range of the text", s.start, s.end),
            RewriteError::Overlap(a, b) => {
                write!(f, "mentions must not overlap: {}..{} and {}..{}", a.start, a.end, b.start, b.end)
            }
            RewriteError::EmptyValue(s) => {
                write!(f, "mention at {}..{} has an empty normalized form", s.start, s.end)
            }
        }
    }
}

impl std::error::Error for RewriteError {}

/// A text with its mentions replaced by labels.
pub struct PiiRewrite {
    text: String,
    labels: Pii,
    original: Vec<Span>,
    shifts: Vec<i64>,
    original_len: u32,
}

impl PiiRewrite {
    /// Validates every mention's span in `text` and coalesces overlapping
    /// mentions into disjoint union spans, keeping the top-ranked mention's
    /// kind, confidence, and normalized form so every covered byte is
    /// replaced without leaking trailing characters.
    pub(crate) fn coalesce_clusters<'a>(
        text: &str,
        mentions: impl IntoIterator<Item = PiiMention<'a>>,
    ) -> Result<Vec<PiiMention<'a>>, RewriteError> {
        let mut ordered: Vec<PiiMention<'a>> = mentions.into_iter().collect();
        for m in &ordered {
            let (start, end) = (m.span.start as usize, m.span.end as usize);
            if start >= end || end > text.len() || !text.is_char_boundary(start) || !text.is_char_boundary(end) {
                return Err(RewriteError::Span(m.span));
            }
        }
        ordered.sort_by_key(|m| (m.span.start, std::cmp::Reverse(m.span.end), m.kind));
        let mut out: Vec<PiiMention<'a>> = Vec::with_capacity(ordered.len());
        let mut best_key = (turbo_analysis_layers::PiiKind::Phone, 0u32, std::cmp::Reverse(0u32));
        for m in ordered {
            let key = (m.kind, m.span.start, std::cmp::Reverse(m.span.end - m.span.start));
            if let Some(last) = out.last_mut()
                && m.span.start < last.span.end
            {
                last.span.end = last.span.end.max(m.span.end);
                if key < best_key {
                    last.kind = m.kind;
                    last.confidence = m.confidence;
                    last.normalized = m.normalized;
                    best_key = key;
                }
            } else {
                best_key = key;
                out.push(m);
            }
        }
        Ok(out)
    }

    /// Replaces each mention with the label `label` writes for it, in
    /// text order. The mentions must be non-empty, lie in `text`, and not
    /// overlap; their order does not matter.
    pub(crate) fn replace<'a>(
        text: &str,
        mentions: impl IntoIterator<Item = PiiMention<'a>>,
        mut label: impl FnMut(&PiiMention<'a>, &mut String),
    ) -> Result<PiiRewrite, RewriteError> {
        let mut ordered: Vec<PiiMention<'a>> = mentions.into_iter().collect();
        for m in &ordered {
            let (start, end) = (m.span.start as usize, m.span.end as usize);
            if start >= end || end > text.len() || !text.is_char_boundary(start) || !text.is_char_boundary(end) {
                return Err(RewriteError::Span(m.span));
            }
        }
        ordered.sort_by_key(|m| m.span.start);
        if let Some(w) = ordered.windows(2).find(|w| w[1].span.start < w[0].span.end) {
            return Err(RewriteError::Overlap(w[0].span, w[1].span));
        }
        let mut out = String::with_capacity(text.len());
        let mut labels = Pii::new(ViewId::SOURCE);
        let mut original = Vec::with_capacity(ordered.len());
        let mut shifts = Vec::with_capacity(ordered.len());
        let mut buf = String::new();
        let mut copied = 0;
        for m in &ordered {
            let (start, end) = (m.span.start as usize, m.span.end as usize);
            out.push_str(&text[copied..start]);
            buf.clear();
            label(m, &mut buf);
            if buf.is_empty() {
                return Err(RewriteError::EmptyValue(m.span));
            }
            let from = out.len() as u32;
            out.push_str(&buf);
            let to = out.len() as u32;
            labels.push(PiiMention {
                span: Span::new(from, to),
                kind: m.kind,
                confidence: m.confidence,
                normalized: &buf,
            });
            original.push(m.span);
            shifts.push(i64::from(to) - i64::from(from) - (end - start) as i64 + shifts.last().copied().unwrap_or(0));
            copied = end;
        }
        out.push_str(&text[copied..]);
        Ok(PiiRewrite { text: out, labels, original, shifts, original_len: text.len() as u32 })
    }

    /// The new text.
    pub fn text(&self) -> &str {
        &self.text
    }

    pub fn into_text(self) -> String {
        self.text
    }

    /// The labels as mentions of the new text, in text order, each with
    /// its label as the normalized form and the kind and confidence of the
    /// mention it replaced.
    pub fn labels(&self) -> &Pii {
        &self.labels
    }

    /// `mapOffset`: an offset of the original text in the new text. An
    /// offset inside a replaced mention maps to the start of its label.
    ///
    /// # Panics
    ///
    /// When `offset` is past the end of the original text.
    pub fn map_offset(&self, offset: u32) -> u32 {
        self.map(offset, true)
    }

    /// `mapSpan`: a span of the original text in the new text. A
    /// non-empty span that touches a replaced mention takes in its whole
    /// label; an empty span stays empty, at [`PiiRewrite::map_offset`].
    ///
    /// # Panics
    ///
    /// When the span is past the end of the original text.
    pub fn map_span(&self, span: Span) -> Span {
        let start = self.map(span.start, true);
        let end = if span.is_empty() { start } else { start.max(self.map(span.end, false)) };
        Span::new(start, end)
    }

    fn map(&self, offset: u32, towards_start: bool) -> u32 {
        assert!(offset <= self.original_len, "offset {offset} is past the end of the original text");
        let preceding = self.original.partition_point(|s| s.start < offset);
        let Some(p) = preceding.checked_sub(1) else { return offset };
        if offset < self.original[p].end {
            let label = self.labels.spans.get(turbo_analysis::Ix::new(p as u32));
            return if towards_start { label.start } else { label.end };
        }
        (i64::from(offset) + self.shifts[p]) as u32
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use turbo_analysis::Layer;
    use turbo_analysis_layers::{PiiConfidence, PiiKind};

    #[test]
    fn replace_rejects_empty_spans_and_empty_labels_and_keeps_labels_valid() {
        let empty_span = PiiMention {
            span: Span::new(3, 3),
            kind: PiiKind::Email,
            confidence: PiiConfidence::Format,
            normalized: "a@b.com",
        };
        assert_eq!(
            PiiRewrite::replace("0123456789", [empty_span], |_, buf| buf.push_str("EMAIL-1")).err(),
            Some(RewriteError::Span(Span::new(3, 3)))
        );

        let valid = PiiMention {
            span: Span::new(3, 6),
            kind: PiiKind::Email,
            confidence: PiiConfidence::Format,
            normalized: "a@b.com",
        };
        assert_eq!(
            PiiRewrite::replace("0123456789", [valid], |_, _| {}).err(),
            Some(RewriteError::EmptyValue(Span::new(3, 6)))
        );

        let ok = PiiRewrite::replace("0123456789", [valid], |_, buf| buf.push_str("EMAIL-1")).unwrap();
        assert_eq!(ok.labels().check(), Ok(()));
    }
}
