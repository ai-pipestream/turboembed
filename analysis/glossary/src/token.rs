//! Glossary terms found by their tokens' normalized values, OpenNLP's
//! `TermAnalyzingGlossaryMatcher`, on the phrase matcher of turbo-matcher.

use std::hash::{DefaultHasher, Hash, Hasher};

use turbo_analysis::{Span, Spans};
use turbo_analysis_layers::{Strs, Tokens};
use turbo_glossary_layers::{EntryIx, Glossary, is_blank};
use turbo_matcher::{Attr, Field, Input, PatternError, PhraseMatch, PhraseMatcher};

use crate::error::{BuildError, MatchError};
use crate::select::{Candidate, GlossaryMatch, select};

/// What a token is matched by.
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq, Hash)]
pub enum TokenValue {
    /// The token's text in its view, so normalization is the view's.
    #[default]
    Text,
    /// A string per token the caller aligns to the tokens (stems, for
    /// the step): OpenNLP's `Term.normalized` after stemming.
    Values,
}

/// The label every phrase is added under: one label, so a phrase added
/// twice keeps its first number and reports the first entry.
const LABEL: &str = "glossary";

/// Finds glossary terms as runs of tokens whose values equal the analyzed
/// tokens of a term, ignoring what lies between the tokens: with stemmed
/// values, "hot dog" matches "hot-dogs" and "hot, dogs". The match runs
/// from the first token's start to the last token's end. Tokens whose
/// value is blank are skipped, so a term can match across them.
/// Overlaps are resolved by earliest start, then longest span, then most
/// tokens, then registration order.
///
/// The phrases are a trie over token value ids ([`PhraseMatcher`]); see
/// the crate documentation for why this is not an Aho-Corasick automaton.
pub struct TokenMatcher {
    phrases: PhraseMatcher,
    value: TokenValue,
    /// Per phrase, the first entry that gave it and its length in tokens.
    entry_of_phrase: Vec<EntryIx>,
    len_of_phrase: Vec<u32>,
    /// A hash of the phrases and the entry each one reports.
    fingerprint: u64,
}

impl TokenMatcher {
    /// The matcher over the terms of `glossary`, each analyzed by
    /// `analyze` into the values its tokens must have: the tokenizer and
    /// normalizers the documents' tokens and values come from. A term with
    /// no token, or with a blank one, is refused. Two terms with the same
    /// values are one phrase, which reports the first one registered.
    pub fn new(
        glossary: &Glossary,
        value: TokenValue,
        analyze: impl FnMut(&str, &mut Vec<String>),
    ) -> Result<TokenMatcher, BuildError> {
        let all: Vec<EntryIx> = glossary.iter().map(|(ix, _)| ix).collect();
        TokenMatcher::over(glossary, &all, value, analyze)
    }

    /// The matcher over the terms of the entries `entries` of `glossary`,
    /// in that order, as [`TokenMatcher::new`] builds it; its matches name
    /// the entries by their place in `glossary`. A composite can then hold
    /// a character matcher over the whole glossary and a token matcher
    /// over the entries whose terms analyze to tokens.
    pub fn over(
        glossary: &Glossary,
        entries: &[EntryIx],
        value: TokenValue,
        mut analyze: impl FnMut(&str, &mut Vec<String>),
    ) -> Result<TokenMatcher, BuildError> {
        if entries.is_empty() {
            return Err(BuildError::Empty);
        }
        if let Some(&entry) = entries.iter().find(|e| e.get() >= glossary.len()) {
            return Err(BuildError::NoEntry { entry });
        }
        let field = match value {
            TokenValue::Text => Field::Attr(Attr::Text),
            TokenValue::Values => Field::Attr(Attr::Norm),
        };
        let mut m = TokenMatcher {
            phrases: PhraseMatcher::new(field),
            value,
            entry_of_phrase: Vec::new(),
            len_of_phrase: Vec::new(),
            fingerprint: 0,
        };
        let mut h = DefaultHasher::new();
        value.hash(&mut h);
        let mut tokens = Vec::new();
        for &ix in entries {
            tokens.clear();
            analyze(glossary.term(ix), &mut tokens);
            if tokens.is_empty() {
                return Err(BuildError::NoTokens { entry: ix });
            }
            if let Some(token) = tokens.iter().position(|t| is_blank(t)) {
                return Err(BuildError::BlankToken { entry: ix, token });
            }
            let refs: Vec<&str> = tokens.iter().map(String::as_str).collect();
            let id = m.phrases.add(LABEL, &refs).map_err(|e| match e {
                PatternError::Empty => BuildError::NoTokens { entry: ix },
                _ => unreachable!("a phrase over a text attribute is refused only when empty"),
            })?;
            if id.0 as usize == m.entry_of_phrase.len() {
                m.entry_of_phrase.push(ix);
                m.len_of_phrase.push(refs.len() as u32);
                ix.hash(&mut h);
                refs.hash(&mut h);
            }
        }
        m.fingerprint = h.finish();
        Ok(m)
    }

    pub fn value(&self) -> TokenValue {
        self.value
    }

    /// A hash of the phrases the matcher holds and the entry each one
    /// reports, which differs between matchers built with different
    /// analyzers.
    pub(crate) fn fingerprint(&self) -> u64 {
        self.fingerprint
    }

    /// The bytes the matcher holds on the heap.
    pub fn heap_bytes(&self) -> usize {
        self.phrases.heap_bytes() + self.entry_of_phrase.capacity() * 4 + self.len_of_phrase.capacity() * 4
    }

    /// Appends to `out` the matches over `tokens`, whose spans index
    /// `text`. `values` is one string per token, needed with
    /// [`TokenValue::Values`] and ignored otherwise.
    pub fn find(
        &self,
        text: &str,
        tokens: &Tokens,
        values: Option<&Strs>,
        out: &mut Vec<GlossaryMatch>,
    ) -> Result<(), MatchError> {
        let n = tokens.spans.len();
        let values = match self.value {
            TokenValue::Text => None,
            TokenValue::Values => {
                let v = values.ok_or(MatchError::NoValues)?;
                if v.len() != n {
                    return Err(MatchError::Misaligned { values: v.len(), tokens: n });
                }
                Some(v)
            }
        };
        check_tokens(text, tokens)?;
        let (starts, ends) = tokens.spans.columns();
        let value = |i: usize| match values {
            Some(v) => v.get(i),
            None => &text[starts[i] as usize..ends[i] as usize],
        };
        let blank: Vec<bool> = (0..n).map(|i| is_blank(value(i))).collect();
        let found = if blank.iter().any(|&b| b) {
            // Match over the tokens that are not blank, then take the
            // matches back to the tokens' own numbers.
            let kept: Vec<u32> = (0..n as u32).filter(|&i| !blank[i as usize]).collect();
            let mut spans = Spans::new(tokens.spans.view());
            let mut kept_values = Strs::default();
            for &i in &kept {
                spans.push(Span::new(starts[i as usize], ends[i as usize]));
                if values.is_some() {
                    kept_values.push(value(i as usize));
                }
            }
            let sub = Tokens { spans, sentence: None };
            let mut found = self.phrases_in(text, &sub, values.map(|_| &kept_values))?;
            for m in &mut found {
                let (s, e) = (kept[m.tokens.start.get()], kept[m.tokens.end.get() - 1] + 1);
                m.tokens = turbo_analysis::IxRange::new(s, e);
            }
            found
        } else {
            self.phrases_in(text, tokens, values)?
        };
        let mut candidates: Vec<Candidate> = found
            .iter()
            .map(|m| {
                let first = m.tokens.start.get();
                let last = m.tokens.end.get() - 1;
                Candidate {
                    span: Span::new(starts[first], ends[last]),
                    tokens: self.len_of_phrase[m.phrase.0 as usize],
                    entry: self.entry_of_phrase[m.phrase.0 as usize],
                }
            })
            .collect();
        select(&mut candidates, out);
        Ok(())
    }

    fn phrases_in(&self, text: &str, tokens: &Tokens, values: Option<&Strs>) -> Result<Vec<PhraseMatch>, MatchError> {
        let convert = |e: turbo_matcher::MatchError| match e {
            turbo_matcher::MatchError::Token { index, start, end } => MatchError::Token { index, start, end },
            other => unreachable!("the phrase matcher refused checked tokens: {other}"),
        };
        let mut input = Input::new(text, tokens).map_err(convert)?;
        if let Some(v) = values {
            input = input.with_norms(v).map_err(convert)?;
        }
        self.phrases.find(&input).map_err(convert)
    }
}

/// Validates that `tokens.spans` are in bounds of `text`, lie on UTF-8
/// character boundaries, and are in non-overlapping text order.
pub(crate) fn check_tokens(text: &str, tokens: &Tokens) -> Result<(), MatchError> {
    let (starts, ends) = tokens.spans.columns();
    for (index, (&start, &end)) in starts.iter().zip(ends).enumerate() {
        if start > end
            || end as usize > text.len()
            || !text.is_char_boundary(start as usize)
            || !text.is_char_boundary(end as usize)
        {
            return Err(MatchError::Token { index, start, end });
        }
        if index > 0 && start < ends[index - 1] {
            return Err(MatchError::Unordered { index });
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use turbo_analysis::ViewId;

    fn glossary(entries: &[(&str, &str)]) -> Glossary {
        let mut g = Glossary::new();
        for (id, term) in entries {
            g.push(id, term).unwrap();
        }
        g
    }

    /// Words split at spaces, hyphens and commas; a lone "," or "-" is a token.
    fn tokens(text: &str) -> Tokens {
        let mut spans = Spans::new(ViewId::SOURCE);
        let mut start = None;
        for (i, c) in text.char_indices() {
            let sep = c == ' ' || c == '-' || c == ',';
            match (sep, start) {
                (true, Some(s)) => {
                    spans.push(Span::new(s, i as u32));
                    start = None;
                }
                (false, None) => start = Some(i as u32),
                _ => {}
            }
            if c == ',' || c == '-' {
                spans.push(Span::new(i as u32, i as u32 + 1));
            }
        }
        if let Some(s) = start {
            spans.push(Span::new(s, text.len() as u32));
        }
        Tokens { spans, sentence: None }
    }

    /// Lowercase, a crude plural strip, and punctuation becomes blank.
    fn norm(token: &str) -> String {
        let lower = token.to_lowercase();
        if lower == "," || lower == "-" {
            return " ".into();
        }
        lower.strip_suffix('s').map(str::to_string).unwrap_or(lower)
    }

    fn analyze(term: &str, out: &mut Vec<String>) {
        out.extend(term.split(' ').map(norm));
    }

    #[test]
    fn values_skip_blank_tokens_and_separators() {
        let g = glossary(&[("c1", "hot dog"), ("c2", "dog"), ("c3", "Hot Dogs")]);
        let m = TokenMatcher::new(&g, TokenValue::Values, analyze).unwrap();
        let text = "Hot-dogs, hot , dogs and dog";
        let toks = tokens(text);
        let mut values = Strs::default();
        for (_, s) in toks.spans.iter() {
            values.push(&norm(&text[s.range()]));
        }
        let mut out = Vec::new();
        m.find(text, &toks, Some(&values), &mut out).unwrap();
        let got: Vec<(&str, u32)> = out.iter().map(|x| (&text[x.span.range()], x.entry.0)).collect();
        assert_eq!(got, [("Hot-dogs", 0), ("hot , dogs", 0), ("dog", 1)]);
    }

    #[test]
    fn text_values_and_refusals() {
        let g = glossary(&[("a", "New York"), ("b", "York")]);
        let m = TokenMatcher::new(&g, TokenValue::Text, |t, out| out.extend(t.split(' ').map(str::to_string))).unwrap();
        let text = "in New York";
        let mut out = Vec::new();
        m.find(text, &tokens(text), None, &mut out).unwrap();
        assert_eq!(out.len(), 1);
        assert_eq!(&text[out[0].span.range()], "New York");
        let mv = TokenMatcher::new(&g, TokenValue::Values, analyze).unwrap();
        assert_eq!(mv.find(text, &tokens(text), None, &mut out), Err(MatchError::NoValues));
        assert_eq!(
            TokenMatcher::new(&g, TokenValue::Text, |_, _| {}).err(),
            Some(BuildError::NoTokens { entry: EntryIx(0) })
        );
        let g = glossary(&[("a", "x , y")]);
        assert_eq!(
            TokenMatcher::new(&g, TokenValue::Values, analyze).err(),
            Some(BuildError::BlankToken { entry: EntryIx(0), token: 1 })
        );
    }

    #[test]
    fn refuses_bad_tokens_and_entries() {
        let g = glossary(&[("a", "x"), ("b", "y")]);
        let split = |t: &str, out: &mut Vec<String>| out.push(t.to_string());
        let m = TokenMatcher::new(&g, TokenValue::Text, split).unwrap();
        let mut spans = Spans::new(ViewId::SOURCE);
        spans.push(Span::new(2, 3));
        spans.push(Span::new(0, 1));
        let unordered = Tokens { spans, sentence: None };
        let mut out = Vec::new();
        assert_eq!(m.find("x y", &unordered, None, &mut out), Err(MatchError::Unordered { index: 1 }));
        let mut spans = Spans::new(ViewId::SOURCE);
        spans.push(Span::new(0, 1));
        let cut = Tokens { spans, sentence: None };
        assert_eq!(m.find("é", &cut, None, &mut out), Err(MatchError::Token { index: 0, start: 0, end: 1 }));
        // A subset keeps the entries' numbers in the whole glossary.
        let only_b = TokenMatcher::over(&g, &[EntryIx(1)], TokenValue::Text, split).unwrap();
        only_b.find("x y", &tokens("x y"), None, &mut out).unwrap();
        assert_eq!(out, [GlossaryMatch { span: Span::new(2, 3), entry: EntryIx(1) }]);
        assert_eq!(TokenMatcher::over(&g, &[], TokenValue::Text, split).err(), Some(BuildError::Empty));
        assert_eq!(
            TokenMatcher::over(&g, &[EntryIx(2)], TokenValue::Text, split).err(),
            Some(BuildError::NoEntry { entry: EntryIx(2) })
        );
    }
}
