//! Token patterns, built in Rust: a pattern is a sequence of [`Tok`]s,
//! each a set of tests one token must pass, with an operator saying how
//! many tokens in a row it takes.
//!
//! ```
//! use turbo_matcher::{Attr, Flag, Tok};
//! // "hello", then an optional punctuation token, then "world" in any case.
//! let pattern = vec![
//!     Tok::lower("hello"),
//!     Tok::flag(Flag::Punct, true).opt(),
//!     Tok::lower("world"),
//! ];
//! // A noun phrase-ish run: an optional determiner, adjectives, a noun.
//! let np = vec![
//!     Tok::attr(Attr::Pos, "DT").opt(),
//!     Tok::attr(Attr::Pos, "JJ").star(),
//!     Tok::attr_in(Attr::Pos, ["NN", "NNS"]).plus(),
//! ];
//! # let _ = (pattern, np);
//! ```

use std::fmt;
use std::sync::Arc;

use crate::attr::{Attr, Cmp, Flag, Iob};

/// A test of one token's text by the caller's own function: what spaCy's
/// custom extension attributes and `REGEX` do.
pub type TextTest = Arc<dyn Fn(&str) -> bool + Send + Sync>;

/// One test a token must pass.
#[derive(Clone)]
pub enum Test {
    /// The attribute equals the string.
    Is(Attr, Box<str>),
    /// The attribute is one of the strings: spaCy's `IN`.
    In(Attr, Box<[Box<str>]>),
    /// The attribute is none of the strings: spaCy's `NOT_IN`. A token
    /// no entity covers has the empty string as its [`Attr::Entity`], so
    /// it passes unless `""` is one of them.
    NotIn(Attr, Box<[Box<str>]>),
    /// The flag is set (`true`) or not (`false`).
    Flag(Flag, bool),
    /// The token's place in the entities.
    Iob(Iob),
    /// The token's length in characters compared with a number: spaCy's
    /// `LENGTH` with `==`, `!=`, `<`, `<=`, `>`, `>=`.
    Length(Cmp, u32),
    /// The caller's own test of the token's text.
    Text(TextTest),
}

impl fmt::Debug for Test {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Test::Is(a, s) => write!(f, "{a:?} == {s:?}"),
            Test::In(a, s) => write!(f, "{a:?} in {s:?}"),
            Test::NotIn(a, s) => write!(f, "{a:?} not in {s:?}"),
            Test::Flag(flag, on) => write!(f, "{flag:?} == {on}"),
            Test::Iob(i) => write!(f, "iob == {i:?}"),
            Test::Length(c, n) => write!(f, "length {c:?} {n}"),
            Test::Text(_) => f.write_str("text test"),
        }
    }
}

/// One token of a pattern: the tests it must pass (all of them; none is
/// any token), whether the result is negated, and how many tokens in a
/// row it takes, `min..=max` (`max` `None` for no limit).
#[derive(Clone, Debug)]
pub struct Tok {
    pub(crate) tests: Vec<Test>,
    pub(crate) negated: bool,
    pub(crate) min: u32,
    pub(crate) max: Option<u32>,
}

impl Tok {
    /// Any one token: spaCy's `{}`.
    pub fn any() -> Tok {
        Tok { tests: Vec::new(), negated: false, min: 1, max: Some(1) }
    }

    /// A token passing one test.
    pub fn test(t: Test) -> Tok {
        Tok { tests: vec![t], ..Tok::any() }
    }

    /// A token whose `attr` is `value`.
    pub fn attr(attr: Attr, value: &str) -> Tok {
        Tok::test(Test::Is(attr, value.into()))
    }

    /// A token whose `attr` is one of `values`.
    pub fn attr_in<'v>(attr: Attr, values: impl IntoIterator<Item = &'v str>) -> Tok {
        Tok::test(Test::In(attr, values.into_iter().map(Into::into).collect()))
    }

    /// A token whose `attr` is none of `values`.
    pub fn attr_not_in<'v>(attr: Attr, values: impl IntoIterator<Item = &'v str>) -> Tok {
        Tok::test(Test::NotIn(attr, values.into_iter().map(Into::into).collect()))
    }

    /// A token whose text is `text`: spaCy's `ORTH`.
    pub fn text(text: &str) -> Tok {
        Tok::attr(Attr::Text, text)
    }

    /// A token whose lowercase is `lower`: spaCy's `LOWER`.
    pub fn lower(lower: &str) -> Tok {
        Tok::attr(Attr::Lower, lower)
    }

    /// A token whose flag is `on`.
    pub fn flag(flag: Flag, on: bool) -> Tok {
        Tok::test(Test::Flag(flag, on))
    }

    /// A token at this place in the entities.
    pub fn iob(iob: Iob) -> Tok {
        Tok::test(Test::Iob(iob))
    }

    /// A token whose length in characters compares so with `n`.
    pub fn length(cmp: Cmp, n: u32) -> Tok {
        Tok::test(Test::Length(cmp, n))
    }

    /// A token whose text passes `f`.
    pub fn text_where(f: impl Fn(&str) -> bool + Send + Sync + 'static) -> Tok {
        Tok::test(Test::Text(Arc::new(f)))
    }

    /// This token's tests and `other`'s, all of which must pass. The
    /// negation (`!`) and repeat of `self` are kept.
    pub fn and(mut self, other: Tok) -> Tok {
        self.tests.extend(other.tests);
        self
    }

    /// Zero or one token: `?`.
    pub fn opt(self) -> Tok {
        self.repeat(0, Some(1))
    }

    /// Zero or more tokens: `*`.
    pub fn star(self) -> Tok {
        self.repeat(0, None)
    }

    /// One or more tokens: `+`.
    pub fn plus(self) -> Tok {
        self.repeat(1, None)
    }

    /// `min` to `max` tokens (no limit when `max` is `None`): spaCy's
    /// `{n}`, `{n,m}`, `{n,}` and `{,m}`.
    pub fn repeat(mut self, min: u32, max: Option<u32>) -> Tok {
        self.min = min;
        self.max = max;
        self
    }
}

impl std::ops::Not for Tok {
    type Output = Tok;

    /// A token that fails this token's tests: spaCy's `!`. It still takes
    /// one token (or as many as its repeat says).
    fn not(mut self) -> Tok {
        self.negated = !self.negated;
        self
    }
}
