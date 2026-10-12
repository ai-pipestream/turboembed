//! `RegexNameFinder` and `RegexNameFinderFactory`.
//!
//! The finder joins the tokens with single spaces, runs each pattern over
//! the joined text with `Matcher.find`, and keeps a match only when it
//! starts and ends where a token starts or ends. A match is mapped to
//! tokens through one table of positions: the start of token i gives i
//! and its end gives i + 1. So a match that starts with the space after a
//! token starts at the next token, and a match that ends with the space
//! before a token ends after the one before it. Matches that start or end
//! inside a token are dropped, and so are the positions they cover:
//! `find` resumes after a dropped match, as it does after a kept one.
//!
//! Patterns run type by type, in the order the finder's map iterates, and
//! within a type in the order given; the names come out in that order,
//! not in text order. The factory builds its map as a `HashMap`, whose
//! order this reproduces (`hash_order`).

use std::cell::OnceCell;
use std::sync::Arc;

use turbo_analysis_layers::{Tag, TagSet};

use crate::builtin::{self, BuiltIn, push_stand_in};
use crate::error::Error;
use crate::hash_order::JavaHashMap;
use crate::java_regex::{JavaRegex, TextFacts};
use crate::name::{Name, TokenNameFinder, kinds};

/// One pattern of a regex name finder.
#[derive(Clone, Debug)]
pub enum Pattern {
    /// One of the factory's patterns, matched exactly as Java does.
    BuiltIn(BuiltIn),
    /// A Java pattern, translated (`crate::java_regex`).
    Java(JavaRegex),
}

impl Pattern {
    /// `Pattern.compile(pattern)`.
    pub fn java(pattern: &str) -> Result<Pattern, Error> {
        JavaRegex::compile(pattern).map(Pattern::Java)
    }
}

impl From<BuiltIn> for Pattern {
    fn from(b: BuiltIn) -> Pattern {
        Pattern::BuiltIn(b)
    }
}

impl From<JavaRegex> for Pattern {
    fn from(r: JavaRegex) -> Pattern {
        Pattern::Java(r)
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
    /// entries in the order the Java map iterates them.
    pub fn from_map(entries: Vec<(Box<str>, Vec<Pattern>)>) -> Result<RegexNameFinder, Error> {
        let kinds = kinds(entries.iter().map(|(k, _)| &**k))?;
        let groups =
            entries.into_iter().enumerate().map(|(i, (_, patterns))| (Tag(i as u16), patterns.into())).collect();
        Ok(RegexNameFinder { kinds, groups })
    }

    /// `RegexNameFinderFactory.getDefaultRegexNameFinders(defaults...)`.
    pub fn defaults(defaults: &[BuiltIn]) -> RegexNameFinder {
        RegexNameFinder::from_map(defaults_map(defaults).into_ordered()).expect("a map has no repeated keys")
    }

    /// `RegexNameFinderFactory.getDefaultRegexNameFinders(config,
    /// defaults...)`: the defaults' map, then `config` put over it, so a
    /// type in both gets the config's patterns. `config` is in the order
    /// the Java map iterates; `None` for `defaults` is a null array.
    pub fn with_config(
        config: Vec<(Box<str>, Vec<Pattern>)>,
        defaults: Option<&[BuiltIn]>,
    ) -> Result<RegexNameFinder, Error> {
        kinds(config.iter().map(|(k, _)| &**k))?;
        let mut map = defaults.map_or_else(JavaHashMap::new, defaults_map);
        map.put_all(config.into_iter());
        RegexNameFinder::from_map(map.into_ordered())
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

/// `defaultsToMap`: each default's one-entry map put into a new `HashMap`.
fn defaults_map(defaults: &[BuiltIn]) -> JavaHashMap<Vec<Pattern>> {
    let mut map = JavaHashMap::new();
    for &d in defaults {
        map.put_all(std::iter::once((Box::from(d.kind()), vec![Pattern::BuiltIn(d)])));
    }
    map
}

impl TokenNameFinder for RegexNameFinder {
    fn kinds(&self) -> &Arc<TagSet> {
        &self.kinds
    }

    fn find(&self, tokens: &[&str], out: &mut Vec<Name>) -> Result<(), Error> {
        let text = OnceCell::new();
        let facts = OnceCell::new();
        let stand_in = OnceCell::new();
        for (kind, patterns) in &self.groups {
            for pattern in patterns {
                match pattern {
                    Pattern::BuiltIn(b) => {
                        let joined = stand_in.get_or_init(|| Joined::new(tokens, true));
                        joined.each_match(*kind, out, |from| builtin::search(*b, &joined.text, from));
                    }
                    Pattern::Java(r) => {
                        let joined = text.get_or_init(|| Joined::new(tokens, false));
                        r.check(facts.get_or_init(|| TextFacts::of(&joined.text)))?;
                        let regex = r.regex();
                        joined.each_match(*kind, out, |from| {
                            regex.find_at(&joined.text, from).map(|m| (m.start(), m.end()))
                        });
                    }
                }
            }
        }
        Ok(())
    }
}

/// The tokens joined by spaces, or their stand-in, with the token each
/// boundary position maps to.
struct Joined {
    text: String,
    /// Positions in increasing order and the token index each gives.
    bounds: Vec<(u32, u32)>,
}

impl Joined {
    fn new(tokens: &[&str], stand_in: bool) -> Joined {
        let len = tokens.iter().map(|t| t.len() + 1).sum();
        let mut joined = Joined { text: String::with_capacity(len), bounds: Vec::with_capacity(tokens.len() * 2) };
        for (i, token) in tokens.iter().enumerate() {
            joined.bound(i as u32);
            if stand_in {
                push_stand_in(&mut joined.text, token);
            } else {
                joined.text.push_str(token);
            }
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

    #[test]
    fn matches_map_to_token_boundaries() {
        let f = RegexNameFinder::new(vec![Pattern::java("b c").unwrap(), Pattern::java("c d").unwrap()], "X").unwrap();
        assert_eq!(names(&f, &["a", "b", "c", "d"]), [(1, 3, "X".into()), (2, 4, "X".into())]);
        // A match inside a token is dropped.
        let f = RegexNameFinder::new(vec![Pattern::java("b").unwrap()], "X").unwrap();
        assert_eq!(names(&f, &["ab", "b"]), [(1, 2, "X".into())]);
        // A leading space starts at the next token.
        let f = RegexNameFinder::new(vec![Pattern::java(" b").unwrap()], "X").unwrap();
        assert_eq!(names(&f, &["a", "b"]), [(1, 2, "X".into())]);
    }

    #[test]
    fn empty_matches_step_one_character() {
        let f = RegexNameFinder::new(vec![Pattern::java("x*").unwrap()], "E").unwrap();
        let got = names(&f, &["é", "x"]);
        assert_eq!(got, [(0, 0, "E".into()), (1, 1, "E".into()), (1, 2, "E".into()), (2, 2, "E".into())]);
    }

    #[test]
    fn no_patterns_and_repeated_types_are_refused() {
        assert_eq!(RegexNameFinder::new(Vec::new(), "X").unwrap_err(), Error::NoPatterns);
        let twice = vec![("A".into(), Vec::new()), ("A".into(), Vec::new())];
        assert_eq!(RegexNameFinder::from_map(twice).unwrap_err(), Error::DuplicateType("A".into()));
    }

    #[test]
    fn factory_order_is_the_hash_map_order() {
        let f = RegexNameFinder::defaults(&BuiltIn::ALL);
        let order: Vec<&str> = f.types().collect();
        assert_eq!(order.len(), 5);
        let config = vec![("EMAIL".into(), vec![Pattern::java("x").unwrap()])];
        let f = RegexNameFinder::with_config(config, Some(&[BuiltIn::Email, BuiltIn::Url])).unwrap();
        let (_, p) = f.patterns().find(|(t, _)| *t == "EMAIL").unwrap();
        assert!(matches!(p, [Pattern::Java(_)]));
    }
}
