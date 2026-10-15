//! The pattern matcher's layer, so another add-on can read its matches
//! without depending on the matcher that makes them.

use std::sync::Arc;

use turbo_analysis::{IxRange, Key, KeyId, Layer, Spans, Tag, TagSet, ViewId};
use turbo_analysis_layers::Tokens;

/// Runs of tokens that a rule matched, each with its rule's label: what
/// spaCy's `SpanRuler` writes to a span group. Matches may overlap and
/// nest unless the step that wrote them was told to keep only
/// non-overlapping ones. They are in order of their first token, then of
/// their last, then of label.
///
/// A gazetteer, a glossary or a custom entity type each write their own
/// instance under a `Key::named` name, so several rule sets sit side by
/// side on one document.
pub struct Matches {
    /// Each match's text: from the start of its first token to the end of
    /// its last, in the view the tokens are in.
    pub spans: Spans<Matches>,
    pub tokens: Vec<IxRange<Tokens>>,
    pub label: Vec<Tag>,
    pub labels: Arc<TagSet>,
}

impl Matches {
    pub fn new(view: ViewId, labels: Arc<TagSet>) -> Matches {
        Matches { spans: Spans::new(view), tokens: Vec::new(), label: Vec::new(), labels }
    }

    /// The label of match `i`.
    pub fn label_of(&self, i: usize) -> &str {
        self.labels.label(self.label[i])
    }
}

impl Layer for Matches {
    const ID: &'static str = "turbo:matches";

    fn len(&self) -> usize {
        self.spans.len()
    }

    fn text_spans(&self) -> Option<(ViewId, &[u32], &[u32])> {
        let (start, end) = self.spans.columns();
        Some((self.spans.view(), start, end))
    }

    fn indexes(&self) -> Vec<(KeyId, usize)> {
        vec![(Key::<Tokens>::DEFAULT.id(), self.tokens.iter().map(|r| r.end.get()).max().unwrap_or(0))]
    }

    fn check(&self) -> Result<(), String> {
        let n = self.spans.len();
        if self.tokens.len() != n || self.label.len() != n {
            return Err("tokens and a label for every match".into());
        }
        if self.tokens.iter().any(|r| r.start >= r.end) {
            return Err("a match has at least one token".into());
        }
        if self.label.iter().any(|l| l.0 as usize >= self.labels.len()) {
            return Err(format!("a label outside {}", self.labels.name));
        }
        let order = |i: usize| (self.tokens[i].start, self.tokens[i].end, self.label[i]);
        if (1..n).any(|i| order(i - 1) >= order(i)) {
            return Err("matches in order of first token, last token and label, each once".into());
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use turbo_analysis::Span;

    fn layer(ranges: &[(u32, u32, u16)]) -> Matches {
        let mut m = Matches::new(ViewId::SOURCE, TagSet::new("rules", &["A", "B"]));
        for &(s, e, l) in ranges {
            m.spans.push(Span::new(s, e));
            m.tokens.push(IxRange::new(s, e));
            m.label.push(Tag(l));
        }
        m
    }

    #[test]
    fn overlapping_matches_in_order_are_valid() {
        assert_eq!(layer(&[(0, 2, 0), (0, 2, 1), (1, 3, 0)]).check(), Ok(()));
    }

    #[test]
    fn out_of_order_repeated_and_empty_matches_are_refused() {
        assert!(layer(&[(1, 3, 0), (0, 2, 0)]).check().is_err());
        assert!(layer(&[(0, 2, 0), (0, 2, 0)]).check().is_err());
        assert!(layer(&[(1, 1, 0)]).check().is_err());
        assert!(layer(&[(0, 1, 2)]).check().is_err());
    }
}
