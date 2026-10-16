//! `COMPOUNDRULE`: a sequence of flags, each optionally followed by `*`
//! (any number of parts) or `?` (at most one). OpenNLP's
//! `HunspellCompoundRule`.

use crate::error::LoadError;
use crate::flag::{Flag, FlagMode, FlagSet};

/// How often a position of a rule may be taken.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
enum Repeat {
    Once,
    /// `*`: zero or more parts.
    Any,
    /// `?`: zero or one part.
    Optional,
}

/// The most positions one rule may have.
const MAX_ELEMENTS: usize = 4096;

/// One compound rule, matched as a small automaton over the flags of the
/// entries the parts select.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct CompoundRule {
    flags: Box<[Flag]>,
    repeat: Box<[Repeat]>,
}

impl CompoundRule {
    /// Parses a rule. A flag is one character, or any text in parentheses
    /// (for the `long` and `num` modes).
    pub(crate) fn parse(pattern: &str, line: usize, mode: FlagMode) -> Result<CompoundRule, LoadError> {
        let mut flags = Vec::new();
        let mut repeat = Vec::new();
        let mut rest = pattern;
        while let Some(c) = rest.chars().next() {
            let text;
            if c == '(' {
                let close = rest[1..].find(')').map(|i| i + 1);
                match close {
                    Some(end) if end > 1 => {
                        text = &rest[1..end];
                        rest = &rest[end + 1..];
                    }
                    _ => return Err(LoadError::InvalidCompoundRule { line }),
                }
            } else {
                if c == '*' || c == '?' || c == ')' {
                    return Err(LoadError::InvalidCompoundRule { line });
                }
                text = &rest[..c.len_utf8()];
                rest = &rest[c.len_utf8()..];
            }
            flags.push(mode.parse_one(text, line)?);
            if flags.len() > MAX_ELEMENTS {
                return Err(LoadError::CompoundRuleTooLong { line });
            }
            repeat.push(match rest.as_bytes().first() {
                Some(b'*') => {
                    rest = &rest[1..];
                    Repeat::Any
                }
                Some(b'?') => {
                    rest = &rest[1..];
                    Repeat::Optional
                }
                _ => Repeat::Once,
            });
        }
        if flags.is_empty() {
            return Err(LoadError::EmptyCompoundRule { line });
        }
        Ok(CompoundRule { flags: flags.into(), repeat: repeat.into() })
    }

    /// Whether the parts, one entry's flags each, follow the rule: to its end
    /// when `complete`, or as far as they go otherwise.
    pub(crate) fn matches<'a>(&self, parts: impl IntoIterator<Item = &'a FlagSet>, complete: bool) -> bool {
        let n = self.flags.len();
        let mut states = vec![false; n + 1];
        states[0] = true;
        self.skip_optional(&mut states);
        let mut next = vec![false; n + 1];
        for part in parts {
            next.fill(false);
            for i in 0..n {
                if states[i] && part.contains(self.flags[i]) {
                    next[if self.repeat[i] == Repeat::Any { i } else { i + 1 }] = true;
                }
            }
            self.skip_optional(&mut next);
            std::mem::swap(&mut states, &mut next);
        }
        if complete { states[n] } else { states.iter().any(|&s| s) }
    }

    /// Lets the states pass over the positions that may be skipped.
    fn skip_optional(&self, states: &mut [bool]) {
        for i in 0..self.flags.len() {
            if states[i] && self.repeat[i] != Repeat::Once {
                states[i + 1] = true;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn set(s: &str) -> FlagSet {
        FlagSet::new(s.chars().map(Flag::of).collect())
    }

    #[test]
    fn rules_match_like_patterns() {
        let r = CompoundRule::parse("n*1t", 1, FlagMode::Char).unwrap();
        let (n, one, t) = (set("n"), set("1"), set("t"));
        assert!(r.matches([&one, &t], true));
        assert!(r.matches([&n, &n, &one, &t], true));
        assert!(!r.matches([&n, &t], true));
        assert!(r.matches([&n, &one], false));
        let o = CompoundRule::parse("a?b", 1, FlagMode::Char).unwrap();
        assert!(o.matches([&set("b")], true));
        assert!(o.matches([&set("a"), &set("b")], true));
        let long = CompoundRule::parse("(aa)(bb)*", 1, FlagMode::Long).unwrap();
        assert!(long.matches([&FlagSet::new(vec![Flag(0x61_0061)])], true));
    }

    #[test]
    fn malformed_rules_fail() {
        for bad in ["*a", "()", "(a", "a)"] {
            assert!(matches!(
                CompoundRule::parse(bad, 4, FlagMode::Char),
                Err(LoadError::InvalidCompoundRule { line: 4 })
            ));
        }
        assert!(matches!(CompoundRule::parse("(ab)", 4, FlagMode::Char), Err(LoadError::ExpectedOneFlag { line: 4 })));
    }
}
