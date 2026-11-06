//! Deterministic, regex-free shape and glob patterns for allow-list matching.

use turbo_analysis::Fnv1a;
use turbo_glossary::Case;
use turbo_unicode::java::to_lower_char;

use super::{ListError, case_tag};

/// Which regex-free pattern syntax a [`Pattern`] uses.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub enum PatternKind {
    /// Character-class shape syntax:
    /// - `d`: any ASCII decimal digit (`0..=9`)
    /// - `a`: any ASCII letter (`A..=Z`, `a..=z`)
    /// - `w`: any ASCII letter or digit (`A..=Z`, `a..=z`, `0..=9`)
    /// - `?`: any single Unicode scalar (`char`)
    /// - `*`: any sequence of zero or more characters
    /// - `\x`: literal character `x` (such as `\d`, `\a`, `\w`, `\?`, `\*`, `\\`)
    /// - any other character: literal match
    Shape,
    /// Wildcard glob syntax:
    /// - `*`: any sequence of zero or more characters
    /// - `?`: any single Unicode scalar (`char`)
    /// - `\x`: literal character `x` (such as `\*`, `\?`, `\\`)
    /// - any other character: literal match
    Glob,
}

#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
enum PatToken {
    Literal(char),
    Digit,
    Alpha,
    Alnum,
    AnyOne,
    AnyMany,
}

impl PatToken {
    #[inline]
    fn matches_char(self, ch: char, case: Case) -> bool {
        match self {
            PatToken::Literal(want) => match case {
                Case::Sensitive => ch == want,
                Case::Insensitive => to_lower_char(ch) == to_lower_char(want),
            },
            PatToken::Digit => ch.is_ascii_digit(),
            PatToken::Alpha => ch.is_ascii_alphabetic(),
            PatToken::Alnum => ch.is_ascii_alphanumeric(),
            PatToken::AnyOne => true,
            PatToken::AnyMany => true,
        }
    }

    fn fold_hash(self, case: Case, h: &mut Fnv1a) {
        match self {
            PatToken::Literal(c) => {
                let folded = match case {
                    Case::Sensitive => c,
                    Case::Insensitive => to_lower_char(c),
                };
                h.u8(1).u32(folded as u32);
            }
            PatToken::Digit => {
                h.u8(2);
            }
            PatToken::Alpha => {
                h.u8(3);
            }
            PatToken::Alnum => {
                h.u8(4);
            }
            PatToken::AnyOne => {
                h.u8(5);
            }
            PatToken::AnyMany => {
                h.u8(6);
            }
        }
    }
}

/// A compiled, regex-free pattern for allow-list matching.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct Pattern {
    kind: PatternKind,
    raw: String,
    case: Case,
    tokens: Vec<PatToken>,
    has_star: bool,
}

impl Pattern {
    /// Compiles a shape pattern (`d` digit, `a` letter, `w` alphanumeric,
    /// `?` any char, `*` any sequence, `\x` escaped literal).
    pub fn shape(spec: &str) -> Result<Pattern, ListError> {
        Self::compile(PatternKind::Shape, spec, Case::Sensitive)
    }

    /// Compiles a glob pattern (`*` any sequence, `?` any char, `\x` escaped literal).
    pub fn glob(spec: &str) -> Result<Pattern, ListError> {
        Self::compile(PatternKind::Glob, spec, Case::Sensitive)
    }

    fn compile(kind: PatternKind, spec: &str, case: Case) -> Result<Pattern, ListError> {
        if spec.is_empty() {
            return Err(ListError::EmptyPattern);
        }
        let mut tokens = Vec::with_capacity(spec.len());
        let mut chars = spec.chars();
        let mut has_star = false;
        while let Some(c) = chars.next() {
            let tok = match c {
                '\\' => {
                    let Some(escaped) = chars.next() else {
                        return Err(ListError::TrailingEscape(spec.to_owned()));
                    };
                    PatToken::Literal(escaped)
                }
                '*' => PatToken::AnyMany,
                '?' => PatToken::AnyOne,
                'd' if kind == PatternKind::Shape => PatToken::Digit,
                'a' if kind == PatternKind::Shape => PatToken::Alpha,
                'w' if kind == PatternKind::Shape => PatToken::Alnum,
                other => PatToken::Literal(other),
            };
            if tok == PatToken::AnyMany {
                has_star = true;
                if tokens.last() == Some(&PatToken::AnyMany) {
                    continue;
                }
            }
            tokens.push(tok);
        }
        Ok(Pattern { kind, raw: spec.to_owned(), case, tokens, has_star })
    }

    /// Sets how literal characters in this pattern compare letters.
    pub fn with_case(mut self, case: Case) -> Pattern {
        self.case = case;
        self
    }

    /// Compares literal characters case-insensitively (`Character.toLowerCase`).
    pub fn case_insensitive(mut self) -> Pattern {
        self.case = Case::Insensitive;
        self
    }

    /// Compares literal characters case-sensitively.
    pub fn case_sensitive(mut self) -> Pattern {
        self.case = Case::Sensitive;
        self
    }

    /// The syntax kind (`Shape` or `Glob`).
    pub const fn kind(&self) -> PatternKind {
        self.kind
    }

    /// The pattern string as given.
    pub fn as_str(&self) -> &str {
        &self.raw
    }

    /// How literal characters in this pattern compare letters.
    pub const fn case(&self) -> Case {
        self.case
    }

    /// Whether `candidate` matches the whole pattern from start to end.
    pub fn matches(&self, candidate: &str) -> bool {
        if !self.has_star {
            let mut chars = candidate.chars();
            for &tok in &self.tokens {
                let Some(ch) = chars.next() else {
                    return false;
                };
                if !tok.matches_char(ch, self.case) {
                    return false;
                }
            }
            return chars.next().is_none();
        }
        let text: Vec<char> = candidate.chars().collect();
        let pat = &self.tokens;
        let (mut ti, mut pi) = (0usize, 0usize);
        let mut star_pi: Option<usize> = None;
        let mut match_ti = 0usize;
        while ti < text.len() {
            if pi < pat.len() && pat[pi] == PatToken::AnyMany {
                star_pi = Some(pi);
                pi += 1;
                match_ti = ti;
            } else if pi < pat.len() && pat[pi].matches_char(text[ti], self.case) {
                ti += 1;
                pi += 1;
            } else if let Some(sp) = star_pi {
                pi = sp + 1;
                match_ti += 1;
                ti = match_ti;
            } else {
                return false;
            }
        }
        while pi < pat.len() && pat[pi] == PatToken::AnyMany {
            pi += 1;
        }
        pi == pat.len()
    }

    pub(crate) fn fold_hash(&self, h: &mut Fnv1a) {
        let kind_tag = match self.kind {
            PatternKind::Shape => 0u8,
            PatternKind::Glob => 1u8,
        };
        h.u8(kind_tag).u8(case_tag(self.case)).str(&self.raw).usize(self.tokens.len());
        for &tok in &self.tokens {
            tok.fold_hash(self.case, h);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn shape_and_glob_matching() {
        let ssn = Pattern::shape("ddd-dd-dddd").unwrap();
        assert!(ssn.matches("123-45-6789"));
        assert!(!ssn.matches("123-4a-6789"));
        assert!(!ssn.matches("123456789"));

        let ticket = Pattern::shape(r"TICKET-\d-aaaa-wwww").unwrap();
        assert!(ticket.matches("TICKET-d-AbCd-12aB"));
        assert!(!ticket.matches("TICKET-9-AbCd-12aB"));
        assert!(!ticket.matches("ticket-d-AbCd-12aB"));
        assert!(ticket.clone().case_insensitive().matches("ticket-D-AbCd-12aB"));

        let glob = Pattern::glob("*@example.com").unwrap();
        assert!(glob.matches("desk@example.com"));
        assert!(glob.matches("@example.com"));
        assert!(!glob.matches("desk@EXAMPLE.COM"));
        assert!(glob.case_insensitive().matches("DESK@EXAMPLE.COM"));

        let wild = Pattern::shape("4111*dddd").unwrap();
        assert!(wild.matches("4111 1111 1111 1234"));
        assert!(wild.matches("41111234"));
        assert!(!wild.matches("4111 1111 1111 123a"));

        assert_eq!(Pattern::shape(""), Err(ListError::EmptyPattern));
        assert_eq!(Pattern::glob("abc\\"), Err(ListError::TrailingEscape("abc\\".to_owned())));
    }
}
