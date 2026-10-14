//! Why a pattern or a finder was refused.

use std::fmt;

/// A Java regular expression feature this port does not translate. A
/// pattern that uses one is refused when it is compiled, so it never
/// matches differently from `java.util.regex` without saying so.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum Feature {
    /// `(?=X)` and `(?!X)`.
    Lookahead,
    /// `(?<=X)` and `(?<!X)`.
    Lookbehind,
    /// `(?>X)`.
    AtomicGroup,
    /// `\1` to `\9` and `\k<name>`.
    Backreference,
    /// `X*+`, `X++`, `X?+` and `X{n,m}+`.
    PossessiveQuantifier,
    /// A quantifier on a group that can match the empty string, such as
    /// `(a*)*` or `(a|)+`: backtracking engines end such loops by rules of
    /// their own, which this port does not reproduce.
    EmptyLoop,
    /// A quantifier on `^`, `$`, `\b` or another assertion.
    QuantifiedAssertion,
    /// A class inside a class, `[a[b]]`.
    NestedClass,
    /// `&&` inside a class.
    ClassIntersection,
    /// A surrogate code point written as an escape, such as `\uD800`.
    Surrogate,
    /// `\p{..}` and `\P{..}`: Java's property names and its Unicode version
    /// are not supported by this port.
    Property,
    /// `\G`, `\R`, `\X`, `\N{..}` and `\b{g}`.
    Escape(char),
    /// An inline flag other than `i` and `s`: `m`, `x`, `u`, `U`, `d` and
    /// `c`.
    Flag(char),
}

impl fmt::Display for Feature {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Feature::Lookahead => f.write_str("lookahead"),
            Feature::Lookbehind => f.write_str("lookbehind"),
            Feature::AtomicGroup => f.write_str("an atomic group"),
            Feature::Backreference => f.write_str("a backreference"),
            Feature::PossessiveQuantifier => f.write_str("a possessive quantifier"),
            Feature::EmptyLoop => f.write_str("a quantifier on a group that can match nothing"),
            Feature::QuantifiedAssertion => f.write_str("a quantifier on an assertion"),
            Feature::NestedClass => f.write_str("a class inside a class"),
            Feature::ClassIntersection => f.write_str("a class intersection"),
            Feature::Surrogate => f.write_str("a surrogate code point"),
            Feature::Property => f.write_str("a Unicode property class"),
            Feature::Escape(c) => write!(f, "the escape \\{c}"),
            Feature::Flag(c) => write!(f, "the flag {c}"),
        }
    }
}

/// A text on which a compiled pattern would match differently from Java.
/// Each is refused for the patterns that use the construct concerned; the
/// same pattern runs on other texts.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum TextLimit {
    /// `\b` or `\B` on a text with a nonspacing mark that follows a letter
    /// or digit: Java counts such a mark as a word character at a
    /// boundary, and ASCII word boundary checks do not.
    MarkAtBoundary,
    /// `$` or `\Z` on a text that ends with a line terminator, before which
    /// Java's `$` also matches.
    FinalLineTerminator,
    /// A pattern that can start a match on the second half of a surrogate
    /// pair in Java, on a text with a character outside the Basic
    /// Multilingual Plane: one with `\B`, with a class that holds a
    /// surrogate but not every character outside the BMP, or that can
    /// match the empty string and uses `\b`, `^` or `\A`.
    Supplementary,
}

impl fmt::Display for TextLimit {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            TextLimit::MarkAtBoundary => "\\b or \\B next to a nonspacing mark on a letter or digit",
            TextLimit::FinalLineTerminator => "$ or \\Z on a text that ends with a line terminator",
            TextLimit::Supplementary => "a match that Java can start inside a character outside the BMP",
        })
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Error {
    /// A pattern `java.util.regex.Pattern.compile` refuses. `at` is the
    /// offset in characters where the parse stopped.
    Syntax { pattern: Box<str>, at: usize, message: &'static str },
    /// A pattern Java accepts that uses a feature this port does not
    /// translate.
    Unsupported { pattern: Box<str>, at: usize, feature: Feature },
    /// A pattern whose compiled bytecode exceeds the instruction limit.
    Compile { pattern: Box<str>, message: String },
    /// A text this pattern cannot match as Java would.
    Text { pattern: Box<str>, limit: TextLimit },
    /// `RegexNameFinder(Pattern[], String)` with no patterns.
    NoPatterns,
    /// The same entity type twice in what stands for a Java `Map`.
    DuplicateType(Box<str>),
    /// More entity types than a `TagSet` holds.
    TooManyTypes,
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Error::Syntax { pattern, at, message } => write!(f, "pattern {pattern:?} at {at}: {message}"),
            Error::Unsupported { pattern, at, feature } => {
                write!(f, "pattern {pattern:?} at {at} uses {feature}, which is not supported")
            }
            Error::Compile { pattern, message } => write!(f, "pattern {pattern:?}: {message}"),
            Error::Text { pattern, limit } => {
                write!(f, "pattern {pattern:?} cannot match this text as Java would: {limit}")
            }
            Error::NoPatterns => f.write_str("patterns must not be empty"),
            Error::DuplicateType(t) => write!(f, "the entity type {t:?} is given twice"),
            Error::TooManyTypes => f.write_str("more than 65536 entity types"),
        }
    }
}

impl std::error::Error for Error {}
