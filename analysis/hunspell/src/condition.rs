//! An affix rule's condition: OpenNLP's `AffixCondition`.

use crate::error::LoadError;

/// One position of a condition.
#[derive(Clone, Debug, PartialEq, Eq)]
enum Position {
    /// `.`: any character.
    Any,
    /// A literal character or a bracketed class; a class starting with `^`
    /// matches every character outside it.
    Class { members: Box<[char]>, negated: bool },
}

/// A fixed number of positions, each a character, `.` or a bracketed class,
/// matched at the end of a candidate stem for a suffix rule and at its
/// start for a prefix rule. The condition `.` accepts every stem. Positions
/// are code points.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Condition {
    positions: Box<[Position]>,
    at_end: bool,
}

impl Condition {
    /// Parses a condition field. A `[` without a later `]` is an error; a
    /// `]` or `^` elsewhere is a literal.
    pub(crate) fn parse(pattern: &str, at_end: bool, line: usize) -> Result<Condition, LoadError> {
        if pattern == "." {
            return Ok(Condition { positions: Box::new([]), at_end });
        }
        let mut positions = Vec::new();
        let mut rest = pattern;
        while let Some(c) = rest.chars().next() {
            match c {
                '[' => {
                    let body = &rest[1..];
                    let end = body.find(']').ok_or(LoadError::UnterminatedClass { line })?;
                    let (members, negated) = match body[..end].strip_prefix('^') {
                        Some(m) => (m, true),
                        None => (&body[..end], false),
                    };
                    positions.push(Position::Class { members: members.chars().collect(), negated });
                    rest = &body[end + 1..];
                }
                '.' => {
                    positions.push(Position::Any);
                    rest = &rest[1..];
                }
                _ => {
                    positions.push(Position::Class { members: Box::new([c]), negated: false });
                    rest = &rest[c.len_utf8()..];
                }
            }
        }
        Ok(Condition { positions: positions.into_boxed_slice(), at_end })
    }

    /// Whether `stem` satisfies the condition. A stem with fewer code
    /// points than the condition has positions never does.
    pub fn matches(&self, stem: &str) -> bool {
        if self.positions.is_empty() {
            return true;
        }
        if self.at_end {
            let mut chars = stem.chars().rev();
            for p in self.positions.iter().rev() {
                let Some(c) = chars.next() else { return false };
                if !p.accepts(c) {
                    // A longer stem still fails here, and a shorter one
                    // fails on length; both answer false.
                    return false;
                }
            }
            true
        } else {
            let mut chars = stem.chars();
            for p in self.positions.iter() {
                let Some(c) = chars.next() else { return false };
                if !p.accepts(c) {
                    return false;
                }
            }
            true
        }
    }
}

impl Position {
    #[inline]
    fn accepts(&self, c: char) -> bool {
        match self {
            Position::Any => true,
            Position::Class { members, negated } => members.contains(&c) != *negated,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn suffix_conditions_anchor_at_the_end() {
        let c = Condition::parse("[^aeiou]y", true, 1).unwrap();
        assert!(!c.matches("boy"));
        assert!(c.matches("pony"));
        assert!(c.matches("fly"));
        assert!(!c.matches("y"));
        let any = Condition::parse(".", true, 1).unwrap();
        assert!(any.matches(""));
        let dot = Condition::parse("..", true, 1).unwrap();
        assert!(!dot.matches("a"));
        assert!(dot.matches("ab"));
    }

    #[test]
    fn prefix_conditions_anchor_at_the_start() {
        let c = Condition::parse("[\u{1F600}x]b", false, 1).unwrap();
        assert!(c.matches("\u{1F600}bc"));
        assert!(c.matches("xb"));
        assert!(!c.matches("bb"));
        assert!(matches!(Condition::parse("[ab", false, 7), Err(LoadError::UnterminatedClass { line: 7 })));
        let empty = Condition::parse("[]a", false, 1).unwrap();
        assert!(!empty.matches("xa"));
        let all = Condition::parse("[^]a", false, 1).unwrap();
        assert!(all.matches("xa"));
    }
}
