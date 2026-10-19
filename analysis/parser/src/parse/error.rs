//! Errors raised while reading or changing a [`Parse`](super::Parse).

use std::fmt;

use turbo_analysis::Span;

/// An error raised while reading or changing a [`Parse`](super::Parse). Each variant stands for the exception
/// OpenNLP throws in the same place ([`ParseError::java_class`]), except
/// [`ParseError::NotCharBoundary`], which has none.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ParseError {
    /// A bracket without a type was closed around tokens, or a gap labeler read the label of one
    /// (`NullPointerException`). `byte_offset` is the input offset of the `-NONE-` bracket the
    /// gap labeler was called for, or the input's length for a closed bracket.
    MissingType { byte_offset: usize },
    /// A `)` with no open bracket (`EmptyStackException`), at `byte_offset` in the input.
    UnmatchedClose { byte_offset: usize },
    /// `Parse::from_tokens` was given an empty token slice (`IllegalArgumentException`).
    EmptyTokens,
    /// `Parse::insert` was given a constituent whose span is reversed or outside the receiving node's span
    /// (`IllegalArgumentException`).
    NotContained { parent: Span, child: Span },
    /// A span that does not start and end on character boundaries of the sentence text. Java
    /// indexes UTF-16 units and has no such case.
    NotCharBoundary { span: Span },
    /// `add` or `adjoin` got no head from the head rules, as when the first child is a token node
    /// (`NullPointerException` on the head's index).
    MissingHead,
    /// `prune_parse` would replace the root, which has no parent (`NullPointerException`).
    MissingParent,
    /// `add_names` was given a token index outside the tokens (`ArrayIndexOutOfBoundsException`).
    TokenIndex { index: i64, len: usize },
    /// `add_names` was given a name whose span would end before it starts
    /// (`IllegalArgumentException` from `Span`).
    ReversedName { name: Span },
    /// A child-mutating method was given an index outside the children, or `adjoin` was called on
    /// a node with none (`IndexOutOfBoundsException`).
    ChildIndex { index: usize, len: usize },
    /// The sentence text would grow past the u32 span offset limit. Java indexes UTF-16 `int`s and
    /// wraps instead, so it has no such case.
    TooLong { len: usize },
}

impl ParseError {
    /// The class of the exception OpenNLP throws in the same place, if it throws one.
    pub fn java_class(&self) -> Option<&'static str> {
        match self {
            ParseError::MissingType { .. } | ParseError::MissingHead | ParseError::MissingParent => {
                Some("java.lang.NullPointerException")
            }
            ParseError::UnmatchedClose { .. } => Some("java.util.EmptyStackException"),
            ParseError::EmptyTokens | ParseError::NotContained { .. } | ParseError::ReversedName { .. } => {
                Some("java.lang.IllegalArgumentException")
            }
            ParseError::TokenIndex { .. } => Some("java.lang.ArrayIndexOutOfBoundsException"),
            ParseError::ChildIndex { .. } => Some("java.lang.IndexOutOfBoundsException"),
            ParseError::NotCharBoundary { .. } | ParseError::TooLong { .. } => None,
        }
    }
}

impl fmt::Display for ParseError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ParseError::MissingType { byte_offset } => {
                write!(f, "constituent without a type (input byte {byte_offset})")
            }
            ParseError::UnmatchedClose { byte_offset } => {
                write!(f, "closing bracket without an open one (input byte {byte_offset})")
            }
            ParseError::EmptyTokens => f.write_str("tokens must not be empty"),
            ParseError::NotContained { .. } => f.write_str("Inserting constituent not contained in the sentence!"),
            ParseError::NotCharBoundary { span } => {
                write!(f, "span {}..{} is not on character boundaries", span.start, span.end)
            }
            ParseError::MissingHead => f.write_str("the head rules gave no head"),
            ParseError::MissingParent => f.write_str("the root has no parent to take its only child"),
            ParseError::TokenIndex { index, len } => write!(f, "Index {index} out of bounds for length {len}"),
            ParseError::ReversedName { name } => {
                write!(f, "name {}..{} ends before it starts", name.start, name.end)
            }
            ParseError::ChildIndex { index, len } => {
                write!(f, "child index {index} out of bounds for {len} children")
            }
            ParseError::TooLong { len } => {
                write!(f, "sentence text of {len} bytes exceeds the u32 span offset limit")
            }
        }
    }
}

impl std::error::Error for ParseError {}
