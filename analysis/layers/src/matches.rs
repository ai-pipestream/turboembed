//! Rule matches: the token runs a pattern or phrase rule matched, with the
//! rule's label. A span ruler writes them under a named instance
//! (`Key::named`), so the matches of several rule sets live side by side
//! on one document.

use std::sync::Arc;

use turbo_analysis::{Ix, IxRange, Key, KeyId, Layer, Overlap, Span, Spans, Tag, TagSet, ViewId};

use crate::Tokens;

/// One span per match, in token order (by first token, then by last),
/// with the token range it covers, its label and the rule that made it.
/// Matches may overlap and nest unless the writer resolved them: a ruler
/// that keeps every match writes overlapping items, one that filters them
/// writes items that do not overlap.
pub struct Matches {
    pub spans: Spans<Matches>,
    /// The tokens of each match; its span runs from the earliest start
    /// of those tokens to the latest end.
    pub tokens: Vec<IxRange<Tokens>>,
    pub label: Vec<Tag>,
    pub labels: Arc<TagSet>,
    /// The number of the rule that matched, as the writer numbers its
    /// rules (a span ruler numbers them in the order they were added).
    pub rule: Vec<u32>,
}

impl Matches {
    pub fn new(view: ViewId, labels: Arc<TagSet>) -> Matches {
        Matches { spans: Spans::new(view), tokens: Vec::new(), label: Vec::new(), labels, rule: Vec::new() }
    }

    pub fn push(&mut self, span: Span, tokens: IxRange<Tokens>, label: Tag, rule: u32) -> Ix<Matches> {
        self.tokens.push(tokens);
        self.label.push(label);
        self.rule.push(rule);
        self.spans.push(span)
    }

    /// The label of match `i`, as text.
    pub fn label_of(&self, i: Ix<Matches>) -> &str {
        self.labels.label(self.label[i.get()])
    }
}

impl Layer for Matches {
    const ID: &'static str = "turbo:matches";
    const OVERLAP: Overlap = Overlap::Any;
    positional!(Matches);

    fn indexes(&self) -> Vec<(KeyId, usize)> {
        vec![(Key::<Tokens>::DEFAULT.id(), self.tokens.iter().map(|r| r.end.get()).max().unwrap_or(0))]
    }

    fn check(&self) -> Result<(), String> {
        let n = self.spans.len();
        if self.tokens.len() != n || self.label.len() != n || self.rule.len() != n {
            return Err("tokens, a label and a rule for every match".into());
        }
        if self.label.iter().any(|t| t.0 as usize >= self.labels.len()) {
            return Err(format!("a label outside {}", self.labels.name));
        }
        if self.tokens.iter().any(|r| r.is_empty()) {
            return Err("a match with no tokens".into());
        }
        if !self.tokens.windows(2).all(|w| (w[0].start, w[0].end) <= (w[1].start, w[1].end)) {
            return Err("matches not in token order".into());
        }
        self.spans.check(Self::OVERLAP)?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn check_wants_columns_labels_and_token_order() {
        let labels = TagSet::new("rules", &["ORG", "PERSON"]);
        let mut m = Matches::new(ViewId::SOURCE, labels.clone());
        m.push(Span::new(0, 6), IxRange::new(0, 1), Tag(0), 0);
        m.push(Span::new(0, 12), IxRange::new(0, 2), Tag(1), 1);
        m.push(Span::new(7, 12), IxRange::new(1, 2), Tag(0), 0);
        assert_eq!(m.check(), Ok(()));
        assert_eq!(m.label_of(Ix::new(1)), "PERSON");
        assert_eq!(m.indexes()[0].1, 2);
        m.push(Span::new(0, 6), IxRange::new(0, 1), Tag(0), 0);
        assert_eq!(m.check().unwrap_err(), "matches not in token order");

        let mut m = Matches::new(ViewId::SOURCE, labels.clone());
        m.push(Span::new(0, 6), IxRange::new(0, 1), Tag(2), 0);
        assert!(m.check().unwrap_err().starts_with("a label outside"));
        let mut m = Matches::new(ViewId::SOURCE, labels.clone());
        m.push(Span::new(0, 0), IxRange::new(1, 1), Tag(0), 0);
        assert_eq!(m.check().unwrap_err(), "a match with no tokens");
        let mut m = Matches::new(ViewId::SOURCE, labels.clone());
        m.push(Span::new(0, 6), IxRange::new(0, 1), Tag(0), 0);
        m.rule.pop();
        assert_eq!(m.check().unwrap_err(), "tokens, a label and a rule for every match");

        let mut bad_span = Matches::new(ViewId::SOURCE, labels);
        bad_span.push(Span { start: 8, end: 3 }, IxRange::new(0, 1), Tag(0), 0);
        assert!(bad_span.check().unwrap_err().contains("ends before it starts"));
        assert_eq!(Matches::ID, "turbo:matches");
    }
}
