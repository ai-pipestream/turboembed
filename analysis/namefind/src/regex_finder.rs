//! `RegexNameFinder` and `RegexNameFinderFactory`.
//!
//! The finder joins the tokens with single spaces and runs each pattern
//! over them. The factory's built-in patterns scan the joined text with
//! [`builtin`] and keep a match only when it starts and ends where a
//! token starts or ends; a match that starts with the space after a
//! token starts at the next token, and one that ends with the space
//! before a token ends after the one before it. Matches that start or end
//! inside a token are dropped, and so are the positions they cover:
//! `find` resumes after a dropped match, as it does after a kept one.
//!
//! A token pattern ([`Pattern::Tokens`]) matches whole tokens with
//! `turbo_matcher`: every match starts and ends at token boundaries by
//! construction, so nothing it finds is dropped.
//!
//! Patterns run type by type, in the order the finder's map iterates, and
//! within a type in the order given; the names come out in that order,
//! not in text order. The factory keeps the types in the order they were
//! first given; a configuration entry replaces the patterns of a type the
//! defaults already have, in place, and appends a type they do not.

use std::cell::OnceCell;
use std::fmt;
use std::sync::Arc;

use turbo_analysis::{Span, Spans};
use turbo_analysis_layers::{Tag, TagSet, Tokens};
use turbo_matcher::{Input, Matcher};

use crate::builtin::{self, BuiltIn};
use crate::error::Error;
use crate::name::{Name, TokenNameFinder, kinds};

pub use turbo_matcher::{Test, Tok};

/// One pattern of a regex name finder.
#[derive(Clone)]
pub enum Pattern {
    /// One of the factory's patterns, matched as [`builtin`] documents.
    BuiltIn(BuiltIn),
    /// One token pattern, matched over the tokens by `turbo_matcher`.
    Tokens(TokensPattern),
}

impl fmt::Debug for Pattern {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Pattern::BuiltIn(b) => write!(f, "BuiltIn({b:?})"),
            Pattern::Tokens(p) => write!(f, "Tokens({p:?})"),
        }
    }
}

impl Pattern {
    /// One token pattern: each `Tok` is the tests one token must pass, so
    /// a match is a run of whole tokens. Refused with the matcher's
    /// [`PatternError`](turbo_matcher::PatternError) when the pattern is
    /// empty or too large.
    pub fn tokens(pattern: Vec<Tok>) -> Result<Pattern, Error> {
        TokensPattern::new(pattern).map(Pattern::Tokens)
    }
}

impl From<BuiltIn> for Pattern {
    fn from(b: BuiltIn) -> Pattern {
        Pattern::BuiltIn(b)
    }
}

impl From<TokensPattern> for Pattern {
    fn from(p: TokensPattern) -> Pattern {
        Pattern::Tokens(p)
    }
}

/// A compiled token pattern ([`Pattern::Tokens`]).
pub struct TokensPattern {
    toks: Box<[Tok]>,
    matcher: Arc<Matcher>,
}

impl TokensPattern {
    /// Compiles `pattern` as one `turbo_matcher` pattern.
    pub fn new(pattern: Vec<Tok>) -> Result<TokensPattern, Error> {
        let mut matcher = Matcher::new();
        matcher.add("pattern", &pattern)?;
        Ok(TokensPattern { toks: pattern.into(), matcher: Arc::new(matcher) })
    }

    /// The pattern's tokens, in order.
    pub fn toks(&self) -> &[Tok] {
        &self.toks
    }

    fn matches(&self, input: &Input<'_>) -> Result<Vec<(u32, u32)>, Error> {
        let found = self.matcher.find(input).map_err(Error::Match)?;
        Ok(found.into_iter().map(|m| (m.tokens.start.get() as u32, m.tokens.end.get() as u32)).collect())
    }
}

impl Clone for TokensPattern {
    fn clone(&self) -> TokensPattern {
        TokensPattern { toks: self.toks.clone(), matcher: Arc::clone(&self.matcher) }
    }
}

impl fmt::Debug for TokensPattern {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "TokensPattern({:?})", self.toks)
    }
}

/// `opennlp.tools.namefind.RegexNameFinder`.
#[derive(Clone, Debug)]
pub struct RegexNameFinder {
    kinds: Arc<TagSet>,
    /// The patterns of each type, in the map's order.
    groups: Vec<(Tag, Box<[Pattern]>)>,
}

impl RegexNameFinder {
    /// `new RegexNameFinder(Pattern[] patterns, String type)`.
    pub fn new(patterns: Vec<Pattern>, kind: &str) -> Result<RegexNameFinder, Error> {
        if patterns.is_empty() {
            return Err(Error::NoPatterns);
        }
        RegexNameFinder::from_map(vec![(kind.into(), patterns)])
    }

    /// `new RegexNameFinder(Map<String, Pattern[]> regexMap)`, with the
    /// entries in the order the map iterates them.
    pub fn from_map(entries: Vec<(Box<str>, Vec<Pattern>)>) -> Result<RegexNameFinder, Error> {
        let kinds = kinds(entries.iter().map(|(k, _)| &**k))?;
        let groups =
            entries.into_iter().enumerate().map(|(i, (_, patterns))| (Tag(i as u16), patterns.into())).collect();
        Ok(RegexNameFinder { kinds, groups })
    }

    /// `RegexNameFinderFactory.getDefaultRegexNameFinders(defaults...)`.
    pub fn defaults(defaults: &[BuiltIn]) -> RegexNameFinder {
        RegexNameFinder::from_map(defaults_map(defaults)).expect("a map has no repeated keys")
    }

    /// `RegexNameFinderFactory.getDefaultRegexNameFinders(config,
    /// defaults...)`: the defaults in order, then `config`, which
    /// replaces the patterns of a type the defaults already have, in
    /// place, and appends a type they do not. `None` for `defaults` is a
    /// null array.
    pub fn with_config(
        config: Vec<(Box<str>, Vec<Pattern>)>,
        defaults: Option<&[BuiltIn]>,
    ) -> Result<RegexNameFinder, Error> {
        kinds(config.iter().map(|(k, _)| &**k))?;
        let mut entries: Vec<(Box<str>, Vec<Pattern>)> =
            defaults.map(|d| defaults_map(d).into_iter().collect()).unwrap_or_default();
        for (kind, patterns) in config {
            match entries.iter_mut().find(|(k, _)| **k == *kind) {
                Some((_, existing)) => *existing = patterns,
                None => entries.push((kind, patterns)),
            }
        }
        RegexNameFinder::from_map(entries)
    }

    /// The entity types in the order their patterns run.
    pub fn types(&self) -> impl Iterator<Item = &str> {
        self.groups.iter().map(|(t, _)| self.kinds.label(*t))
    }

    /// The patterns of each type, in the order they run.
    pub fn patterns(&self) -> impl Iterator<Item = (&str, &[Pattern])> {
        self.groups.iter().map(|(t, p)| (self.kinds.label(*t), &**p))
    }
}

/// The defaults as map entries, in order.
fn defaults_map(defaults: &[BuiltIn]) -> Vec<(Box<str>, Vec<Pattern>)> {
    defaults.iter().map(|d| (Box::from(d.kind()), vec![Pattern::BuiltIn(*d)])).collect()
}

impl TokenNameFinder for RegexNameFinder {
    fn kinds(&self) -> &Arc<TagSet> {
        &self.kinds
    }

    fn find(&self, tokens: &[&str], out: &mut Vec<Name>) -> Result<(), Error> {
        if tokens.is_empty() {
            return Ok(());
        }
        let joined = Joined::new(tokens);
        let layer: OnceCell<Tokens> = OnceCell::new();
        for (kind, patterns) in &self.groups {
            for pattern in patterns {
                match pattern {
                    Pattern::BuiltIn(b) => {
                        joined.each_match(*kind, out, |from| builtin::search(*b, &joined.text, from));
                    }
                    Pattern::Tokens(p) => {
                        let tokens = layer.get_or_init(|| joined.tokens());
                        let input = Input::new(&joined.text, tokens).map_err(Error::Match)?;
                        for (start, end) in p.matches(&input)? {
                            out.push(Name { start, end, kind: *kind });
                        }
                    }
                }
            }
        }
        Ok(())
    }
}

/// The tokens joined by spaces, with the token each boundary position
/// maps to.
struct Joined {
    text: String,
    /// Positions in increasing order and the token index each gives.
    bounds: Vec<(u32, u32)>,
}

impl Joined {
    fn new(tokens: &[&str]) -> Joined {
        let len = tokens.iter().map(|t| t.len() + 1).sum();
        let mut joined = Joined { text: String::with_capacity(len), bounds: Vec::with_capacity(tokens.len() * 2) };
        for (i, token) in tokens.iter().enumerate() {
            joined.bound(i as u32);
            joined.text.push_str(token);
            joined.bound(i as u32 + 1);
            if i + 1 < tokens.len() {
                joined.text.push(' ');
            }
        }
        joined
    }

    /// Maps the current end of the text to `token`; a later entry for the
    /// same position replaces an earlier one, as `HashMap.put` does.
    fn bound(&mut self, token: u32) {
        let at = self.text.len() as u32;
        match self.bounds.last_mut() {
            Some(last) if last.0 == at => last.1 = token,
            _ => self.bounds.push((at, token)),
        }
    }

    fn token_at(&self, at: usize) -> Option<u32> {
        let i = self.bounds.binary_search_by_key(&(at as u32), |b| b.0).ok()?;
        Some(self.bounds[i].1)
    }

    /// The tokens as a token layer over the joined text, for token
    /// patterns: the boundary pairs are (start of token i, end of token i).
    fn tokens(&self) -> Tokens {
        let mut spans = Spans::new(turbo_analysis::ViewId::SOURCE);
        let mut i = 0;
        while i + 1 < self.bounds.len() {
            spans.push(Span::new(self.bounds[i].0, self.bounds[i + 1].0));
            i += 2;
        }
        Tokens { spans, sentence: None }
    }

    /// Runs `Matcher.find` to the end: `search(from)` gives the leftmost
    /// match at or after `from`. After an empty match the next search
    /// starts one character on.
    fn each_match(&self, kind: Tag, out: &mut Vec<Name>, mut search: impl FnMut(usize) -> Option<(usize, usize)>) {
        let mut from = 0;
        while let Some((start, end)) = search(from) {
            if let (Some(a), Some(b)) = (self.token_at(start), self.token_at(end)) {
                out.push(Name { start: a, end: b, kind });
            }
            from = if end > start {
                end
            } else {
                match self.text[end..].chars().next() {
                    Some(c) => end + c.len_utf8(),
                    None => break,
                }
            };
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn names(f: &RegexNameFinder, tokens: &[&str]) -> Vec<(u32, u32, String)> {
        let mut out = Vec::new();
        f.find(tokens, &mut out).unwrap();
        out.iter().map(|n| (n.start, n.end, f.kinds().label(n.kind).to_string())).collect()
    }

    fn tok(text: &str) -> Tok {
        Tok::text(text)
    }

    #[test]
    fn token_patterns_match_whole_tokens() {
        let f = RegexNameFinder::new(vec![Pattern::tokens(vec![tok("b"), tok("c")]).unwrap()], "X").unwrap();
        assert_eq!(names(&f, &["a", "b", "c", "d"]), [(1, 3, "X".into())]);
        // A repeated token matches a run of it, at every start and end.
        let f = RegexNameFinder::new(vec![Pattern::tokens(vec![tok("x").plus()]).unwrap()], "X").unwrap();
        assert_eq!(
            names(&f, &["x", "a", "x", "x"]),
            [(0, 1, "X".into()), (2, 3, "X".into()), (2, 4, "X".into()), (3, 4, "X".into())]
        );
    }

    #[test]
    fn empty_and_missing_patterns_are_refused() {
        assert_eq!(Pattern::tokens(Vec::new()).unwrap_err(), Error::Pattern(turbo_matcher::PatternError::Empty));
        assert_eq!(RegexNameFinder::new(Vec::new(), "X").unwrap_err(), Error::NoPatterns);
        let twice = vec![("A".into(), Vec::new()), ("A".into(), Vec::new())];
        assert_eq!(RegexNameFinder::from_map(twice).unwrap_err(), Error::DuplicateType("A".into()));
    }

    #[test]
    fn factory_order_is_first_given_with_in_place_replacement() {
        let f = RegexNameFinder::defaults(&BuiltIn::ALL);
        let order: Vec<&str> = f.types().collect();
        assert_eq!(order, ["PHONE_NUM", "EMAIL", "URL", "MGRS", "DEGREES_MIN_SEC_LAT_LON"]);
        let config = vec![("EMAIL".into(), vec![Pattern::tokens(vec![tok("x")]).unwrap()])];
        let f = RegexNameFinder::with_config(config, Some(&[BuiltIn::Email, BuiltIn::Url])).unwrap();
        assert_eq!(f.types().collect::<Vec<_>>(), ["EMAIL", "URL"]);
        let (_, p) = f.patterns().find(|(t, _)| *t == "EMAIL").unwrap();
        assert!(matches!(p, [Pattern::Tokens(_)]));
    }
}
