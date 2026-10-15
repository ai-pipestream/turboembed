//! Token patterns as Rust values: a pattern is a list of `Tok`, each a set
//! of tests one token must pass and an operator saying how many tokens it
//! takes.

use std::fmt;

use crate::attr::{Attr, Flag};

/// How many tokens a `Tok` takes, as spaCy's `OP`.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub enum Op {
    /// Exactly one token that passes.
    One,
    /// `?`: zero or one.
    Optional,
    /// `*`: zero or more.
    ZeroOrMore,
    /// `+`: one or more.
    OneOrMore,
    /// `!`: exactly one token that does not pass.
    Not,
    /// `{min,max}`: from `min` to `max` tokens that pass, any number from
    /// `min` up when `max` is `None`.
    Repeat { min: u32, max: Option<u32> },
}

/// A comparison of a token's length in characters.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub enum Cmp {
    Eq,
    Ne,
    Lt,
    Le,
    Gt,
    Ge,
}

impl Cmp {
    pub(crate) fn holds(self, a: u32, b: u32) -> bool {
        match self {
            Cmp::Eq => a == b,
            Cmp::Ne => a != b,
            Cmp::Lt => a < b,
            Cmp::Le => a <= b,
            Cmp::Gt => a > b,
            Cmp::Ge => a >= b,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub(crate) enum Test {
    Eq(Attr, String),
    In(Attr, Vec<String>),
    NotIn(Attr, Vec<String>),
    /// Matches one of the expressions; with `true`, none of them.
    Regex(Attr, Vec<String>, bool),
    /// Close to one of the values; with `true`, close to none of them.
    Fuzzy(Attr, Vec<String>, Option<u32>, bool),
    Length(Cmp, u32),
    Flag(Flag, bool),
}

/// One token of a pattern: the tests it must pass, all of them, and how
/// many tokens it takes. `Tok::any()` passes every token.
///
/// ```
/// use turbo_matcher::{Attr, Flag, Tok};
/// // "boston", then optionally punctuation, then "celtics", any case.
/// let pattern = vec![
///     Tok::lower("boston"),
///     Tok::any().flag(Flag::Punct, true).optional(),
///     Tok::lower("celtics"),
/// ];
/// # let _ = pattern;
/// ```
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct Tok {
    pub(crate) tests: Vec<Test>,
    pub(crate) op: Op,
}

/// A token pattern: tokens matched one after another.
pub type Pattern = Vec<Tok>;

impl Tok {
    pub fn any() -> Tok {
        Tok { tests: Vec::new(), op: Op::One }
    }

    /// A token whose text is `s`.
    pub fn text(s: &str) -> Tok {
        Tok::any().eq(Attr::Text, s)
    }

    /// A token whose lowercase form is `s`.
    pub fn lower(s: &str) -> Tok {
        Tok::any().eq(Attr::Lower, s)
    }

    /// A token whose lemma is `s`.
    pub fn lemma(s: &str) -> Tok {
        Tok::any().eq(Attr::Lemma, s)
    }

    /// A token tagged `s`.
    pub fn pos(s: &str) -> Tok {
        Tok::any().eq(Attr::Pos, s)
    }

    /// The value of `attr` is `s`.
    pub fn eq(mut self, attr: Attr, s: &str) -> Tok {
        self.tests.push(Test::Eq(attr, s.into()));
        self
    }

    /// The value of `attr` is one of `values` (spaCy's `IN`).
    pub fn is_in(mut self, attr: Attr, values: &[&str]) -> Tok {
        self.tests.push(Test::In(attr, values.iter().map(|&v| v.into()).collect()));
        self
    }

    /// The value of `attr` is none of `values` (spaCy's `NOT_IN`).
    pub fn not_in(mut self, attr: Attr, values: &[&str]) -> Tok {
        self.tests.push(Test::NotIn(attr, values.iter().map(|&v| v.into()).collect()));
        self
    }

    /// The value of `attr` contains a match of the regular expression, in
    /// the syntax of the `regex` crate. Like spaCy's `REGEX`, which uses
    /// Python's `re.search`, it is not anchored: write `^` and `$` to
    /// match the whole value.
    pub fn regex(self, attr: Attr, pattern: &str) -> Tok {
        self.regex_in(attr, &[pattern])
    }

    /// The value of `attr` contains a match of one of the expressions.
    pub fn regex_in(mut self, attr: Attr, patterns: &[&str]) -> Tok {
        self.tests.push(Test::Regex(attr, patterns.iter().map(|&p| p.into()).collect(), false));
        self
    }

    /// The value of `attr` contains a match of none of the expressions.
    pub fn regex_not_in(mut self, attr: Attr, patterns: &[&str]) -> Tok {
        self.tests.push(Test::Regex(attr, patterns.iter().map(|&p| p.into()).collect(), true));
        self
    }

    /// The value of `attr` is within `max_edits` Levenshtein edits of `s`,
    /// counted in characters. With `None`, the most edits allowed are 30
    /// percent of the length of `s`, rounded, and at least two, as spaCy's
    /// `FUZZY`.
    pub fn fuzzy(self, attr: Attr, s: &str, max_edits: Option<u32>) -> Tok {
        self.fuzzy_in(attr, &[s], max_edits)
    }

    /// The value of `attr` is close to one of `values`, by `fuzzy`'s rule.
    pub fn fuzzy_in(mut self, attr: Attr, values: &[&str], max_edits: Option<u32>) -> Tok {
        self.tests.push(Test::Fuzzy(attr, values.iter().map(|&v| v.into()).collect(), max_edits, false));
        self
    }

    /// The value of `attr` is close to none of `values`, by `fuzzy`'s rule.
    pub fn fuzzy_not_in(mut self, attr: Attr, values: &[&str], max_edits: Option<u32>) -> Tok {
        self.tests.push(Test::Fuzzy(attr, values.iter().map(|&v| v.into()).collect(), max_edits, true));
        self
    }

    /// The token's text has a length in characters that compares with `n`.
    pub fn length(mut self, cmp: Cmp, n: u32) -> Tok {
        self.tests.push(Test::Length(cmp, n));
        self
    }

    /// The token has the flag (`true`) or does not (`false`).
    pub fn flag(mut self, flag: Flag, value: bool) -> Tok {
        self.tests.push(Test::Flag(flag, value));
        self
    }

    pub fn op(mut self, op: Op) -> Tok {
        self.op = op;
        self
    }

    pub fn optional(self) -> Tok {
        self.op(Op::Optional)
    }

    pub fn zero_or_more(self) -> Tok {
        self.op(Op::ZeroOrMore)
    }

    pub fn one_or_more(self) -> Tok {
        self.op(Op::OneOrMore)
    }

    pub fn negated(self) -> Tok {
        self.op(Op::Not)
    }

    pub fn repeat(self, min: u32, max: Option<u32>) -> Tok {
        self.op(Op::Repeat { min, max })
    }
}

/// Why rules could not be added.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum PatternError {
    /// A pattern with no tokens, or whose tokens can only match nothing.
    Empty,
    /// A repeat whose maximum is zero or below its minimum.
    Repeat { min: u32, max: u32 },
    /// A regular expression that does not compile.
    Regex { pattern: String, error: String },
    /// A phrase with no tokens.
    EmptyPhrase,
    /// A test on a string attribute in a matcher over flags, or the reverse.
    PhraseValue,
    /// More labels than a layer can tag (65536).
    Labels,
    /// A label that is the empty string.
    EmptyLabel,
    /// A pattern that, with its repeats written out, has more than
    /// `MAX_ELEMENTS` steps.
    TooLong,
}

/// The most steps a compiled pattern may have, counting each step of a
/// repeat: `a{2,5}` is five.
pub const MAX_ELEMENTS: usize = 65536;

impl fmt::Display for PatternError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            PatternError::Empty => f.write_str("a pattern needs at least one token that can match"),
            PatternError::Repeat { min, max } => write!(f, "a repeat from {min} to {max} matches nothing"),
            PatternError::Regex { pattern, error } => write!(f, "regular expression {pattern:?}: {error}"),
            PatternError::EmptyPhrase => f.write_str("a phrase needs at least one token"),
            PatternError::PhraseValue => f.write_str("a phrase's values must be of the kind its matcher compares"),
            PatternError::Labels => f.write_str("more than 65536 labels"),
            PatternError::EmptyLabel => f.write_str("a label must not be empty"),
            PatternError::TooLong => {
                write!(f, "a pattern with its repeats written out has more than {MAX_ELEMENTS} steps")
            }
        }
    }
}

impl std::error::Error for PatternError {}
