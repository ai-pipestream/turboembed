//! The matchers as one type, and the composite of several by priority.

use turbo_analysis_layers::{Strs, Tokens};

use crate::boundaries::Boundaries;
use crate::error::{BuildError, MatchError};
use crate::select::{GlossaryMatch, merge};
use crate::text::TextMatcher;
use crate::token::{TokenMatcher, TokenValue, check_tokens};

/// What a matcher reads of one text: the text, its tokens, the offsets
/// a character match may start and end at, and a value per token for a
/// token matcher that reads one.
#[derive(Clone, Copy)]
pub struct MatchInput<'a> {
    pub text: &'a str,
    pub tokens: &'a Tokens,
    pub boundaries: &'a Boundaries,
    pub values: Option<&'a Strs>,
}

/// A glossary matcher: OpenNLP's `GlossaryMatcher` and its three
/// implementations. Every variant gives its matches in text order without
/// overlaps.
pub enum GlossaryMatcher {
    /// By characters ([`TextMatcher`]).
    Text(TextMatcher),
    /// By token values ([`TokenMatcher`]).
    Tokens(TokenMatcher),
    /// Several matchers in priority order, highest first: a match is kept
    /// when it intersects no match of a matcher before it and no match
    /// of its own matcher kept before it. OpenNLP's
    /// `CompositeGlossaryMatcher`; put a character matcher before a token
    /// matcher to prefer exact matches to inflected ones.
    Composite(Vec<GlossaryMatcher>),
}

impl GlossaryMatcher {
    /// A composite of `matchers`, highest priority first; at least one.
    pub fn composite(matchers: Vec<GlossaryMatcher>) -> Result<GlossaryMatcher, BuildError> {
        if matchers.is_empty() {
            return Err(BuildError::NoMatchers);
        }
        Ok(GlossaryMatcher::Composite(matchers))
    }

    /// Whether some token matcher in it reads a value per token.
    pub fn reads_values(&self) -> bool {
        match self {
            GlossaryMatcher::Text(_) => false,
            GlossaryMatcher::Tokens(t) => t.value() == TokenValue::Values,
            GlossaryMatcher::Composite(ms) => ms.iter().any(GlossaryMatcher::reads_values),
        }
    }

    /// Whether some character matcher in it reads `boundaries`.
    pub(crate) fn has_text(&self) -> bool {
        match self {
            GlossaryMatcher::Text(_) => true,
            GlossaryMatcher::Tokens(_) => false,
            GlossaryMatcher::Composite(ms) => ms.iter().any(GlossaryMatcher::has_text),
        }
    }

    /// Whether some token matcher in it reads `tokens`.
    fn has_tokens(&self) -> bool {
        match self {
            GlossaryMatcher::Text(_) => false,
            GlossaryMatcher::Tokens(_) => true,
            GlossaryMatcher::Composite(ms) => ms.iter().any(GlossaryMatcher::has_tokens),
        }
    }

    /// Validates `values` alignment for any token matcher in `self`.
    pub(crate) fn check_values(&self, tokens: &Tokens, values: Option<&Strs>) -> Result<(), MatchError> {
        match self {
            GlossaryMatcher::Text(_) => Ok(()),
            GlossaryMatcher::Tokens(t) => t.check_values(tokens, values).map(|_| ()),
            GlossaryMatcher::Composite(ms) => ms.iter().try_for_each(|m| m.check_values(tokens, values)),
        }
    }

    /// Appends the matches in `input` to `out`, in text order.
    pub fn find(&self, input: &MatchInput<'_>, out: &mut Vec<GlossaryMatch>) -> Result<(), MatchError> {
        if self.has_tokens() {
            self.check_values(input.tokens, input.values)?;
            check_tokens(input.text, input.tokens)?;
        }
        self.find_checked(input, out)
    }

    /// Runs matching assuming token spans and `values` alignment have
    /// already been validated.
    pub(crate) fn find_checked(&self, input: &MatchInput<'_>, out: &mut Vec<GlossaryMatch>) -> Result<(), MatchError> {
        match self {
            GlossaryMatcher::Text(t) => {
                t.find(input.text, input.boundaries, out);
                Ok(())
            }
            GlossaryMatcher::Tokens(t) => {
                let values = (t.value() == TokenValue::Values).then_some(input.values).flatten();
                t.find_checked(input.text, input.tokens, values, out)
            }
            GlossaryMatcher::Composite(ms) => {
                let mut accepted = Vec::new();
                let mut next = Vec::new();
                for m in ms {
                    next.clear();
                    m.find_checked(input, &mut next)?;
                    merge(&mut accepted, &next);
                }
                out.append(&mut accepted);
                Ok(())
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::text::Case;
    use turbo_analysis::{Span, Spans, ViewId};
    use turbo_glossary_layers::Glossary;

    #[test]
    fn exact_matches_win_over_token_matches() {
        let mut g = Glossary::new();
        g.push("exact", "dogs and").unwrap();
        g.push("token", "and cats").unwrap();
        g.push("token", "cats").unwrap();
        let text = "dogs and cats";
        let mut spans = Spans::new(ViewId::SOURCE);
        for s in [(0, 4), (5, 8), (9, 13)] {
            spans.push(Span::new(s.0, s.1));
        }
        let tokens = Tokens { spans, sentence: None };
        let boundaries = Boundaries::of_tokens(text.len() as u32, &tokens);
        let exact = GlossaryMatcher::Text(TextMatcher::new(&g, Case::Sensitive).unwrap());
        let words = |t: &str, out: &mut Vec<String>| out.extend(t.split(' ').map(str::to_string));
        let token = GlossaryMatcher::Tokens(TokenMatcher::new(&g, TokenValue::Text, words).unwrap());
        assert!(!token.reads_values());
        let both = GlossaryMatcher::composite(vec![exact, token]).unwrap();
        let mut out = Vec::new();
        both.find(&MatchInput { text, tokens: &tokens, boundaries: &boundaries, values: None }, &mut out).unwrap();
        let got: Vec<(&str, u32)> = out.iter().map(|m| (&text[m.span.range()], m.entry.0)).collect();
        // The character matcher finds "dogs and" and "cats" first; the
        // token matcher's "and cats" intersects both.
        assert_eq!(got, [("dogs and", 0), ("cats", 2)]);
        assert_eq!(GlossaryMatcher::composite(Vec::new()).err(), Some(BuildError::NoMatchers));
    }
}
