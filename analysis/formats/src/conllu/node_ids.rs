//! The Universal Dependencies rules for the ids of multiword tokens and
//! empty nodes, which `ConlluDependencySampleStream` does not check: it
//! passes over every line whose id holds a `-` or a `.`, whatever else the
//! id is. [`check_node_ids`] reports the lines that break the rules, so a
//! caller can tell a well-formed treebank from one that only reads.
//!
//! The rules, as the UD guidelines give them for the `ID` column:
//!
//! - A multiword token's id is `N-M`, two decimals without a leading zero
//!   and `N` < `M`. Its line comes before the line of word `N`, its range
//!   does not reach past the sentence's last word, and it does not
//!   overlap the range of the multiword token before it.
//! - An empty node's id is `N.M`: `N` is the id of the word before it, or
//!   `0` before the first word, and `M` counts the empty nodes after that
//!   word from 1.
//!
//! The words' own ids are not checked here; the reader skips a sentence
//! whose words are not numbered `1`, `2`, ... ([`super::dependency::SkipReason::WordId`]).

use turbo_analysis::Span;

use crate::source::Piece;

/// A multiword token or empty node line of a sentence, with the number of
/// words before it in the sentence.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct NodeLine<'a> {
    /// The id column.
    pub id: Piece<'a>,
    /// The line's number, counting from 1.
    pub line: u32,
    /// The words of the sentence before the line.
    pub words_before: u32,
}

/// How a multiword token or empty node line breaks the UD rules.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub enum NodeIdError {
    /// A multiword token id that is not `N-M` with `N` < `M`.
    MultiwordSyntax,
    /// A multiword token whose range does not start at the word after it.
    MultiwordStart { expected: u32 },
    /// A multiword token whose range reaches past the sentence's last
    /// word, `words`.
    MultiwordPastEnd { words: u32 },
    /// A multiword token whose range starts inside the range before it,
    /// which ends at `previous_end`.
    MultiwordOverlap { previous_end: u32 },
    /// An empty node id that is not `N.M`.
    EmptyNodeSyntax,
    /// An empty node whose `N` is not the id of the word before it.
    EmptyNodeWord { expected: u32 },
    /// An empty node whose `M` does not count on from the one before it.
    EmptyNodeIndex { expected: u32 },
}

/// One line that breaks the rules.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub struct NodeIdIssue {
    pub line: u32,
    /// The id's bytes in the source.
    pub id: Span,
    pub error: NodeIdError,
}

/// The lines of `nodes`, the multiword tokens and empty nodes of a
/// sentence of `words` words in the order read, that break the UD rules
/// for their ids. A line is a multiword token when its id holds a `-` and
/// otherwise an empty node, as the reader tells them apart.
pub fn check_node_ids(nodes: &[NodeLine<'_>], words: u32) -> Vec<NodeIdIssue> {
    let mut issues = Vec::new();
    let mut previous_end = 0u32;
    // The word the empty nodes being counted follow, and how many so far.
    let mut empty_after: Option<(u32, u32)> = None;
    for node in nodes {
        let text = node.id.text;
        let error = if text.contains('-') {
            multiword(text, node.words_before, words, &mut previous_end)
        } else {
            empty_node(text, node.words_before, &mut empty_after)
        };
        if let Some(error) = error {
            issues.push(NodeIdIssue { line: node.line, id: node.id.span, error });
        }
    }
    issues
}

fn multiword(text: &str, words_before: u32, words: u32, previous_end: &mut u32) -> Option<NodeIdError> {
    let Some((first, last)) = text.split_once('-').and_then(|(a, b)| Some((decimal(a)?, decimal(b)?))) else {
        return Some(NodeIdError::MultiwordSyntax);
    };
    if first == 0 || first >= last {
        return Some(NodeIdError::MultiwordSyntax);
    }
    let expected = words_before.saturating_add(1);
    let error = if first != expected {
        Some(NodeIdError::MultiwordStart { expected })
    } else if last > words {
        Some(NodeIdError::MultiwordPastEnd { words })
    } else if first <= *previous_end {
        Some(NodeIdError::MultiwordOverlap { previous_end: *previous_end })
    } else {
        None
    };
    *previous_end = (*previous_end).max(last);
    error
}

fn empty_node(text: &str, words_before: u32, empty_after: &mut Option<(u32, u32)>) -> Option<NodeIdError> {
    let Some((word, index)) = text.split_once('.').and_then(|(a, b)| Some((decimal(a)?, decimal(b)?))) else {
        return Some(NodeIdError::EmptyNodeSyntax);
    };
    if index == 0 {
        return Some(NodeIdError::EmptyNodeSyntax);
    }
    if word != words_before {
        return Some(NodeIdError::EmptyNodeWord { expected: words_before });
    }
    let expected = match *empty_after {
        Some((w, count)) if w == words_before => count + 1,
        _ => 1,
    };
    *empty_after = Some((words_before, expected));
    (index != expected).then_some(NodeIdError::EmptyNodeIndex { expected })
}

/// A decimal of ASCII digits without a leading zero (`0` itself allowed)
/// that fits a `u32`.
fn decimal(s: &str) -> Option<u32> {
    let b = s.as_bytes();
    if b.is_empty() || !b.iter().all(u8::is_ascii_digit) || (b.len() > 1 && b[0] == b'0') {
        return None;
    }
    s.parse().ok()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Checks ids given as `(id, words before)` in a sentence of `words`
    /// words, each id on its own line from 1.
    fn check(ids: &[(&'static str, u32)], words: u32) -> Vec<(u32, NodeIdError)> {
        let mut at = 0u32;
        let nodes: Vec<NodeLine<'static>> = (1u32..)
            .zip(ids)
            .map(|(line, &(id, words_before))| {
                let span = Span::new(at, at + id.len() as u32);
                at = span.end + 1;
                NodeLine { id: Piece { text: id, span }, line, words_before }
            })
            .collect();
        check_node_ids(&nodes, words).into_iter().map(|i| (i.line, i.error)).collect()
    }

    use NodeIdError::*;

    #[test]
    fn well_formed_ids_pass() {
        assert_eq!(check(&[("0.1", 0), ("1-2", 0), ("3-4", 2), ("4.1", 4), ("4.2", 4), ("5.1", 5)], 5), []);
    }

    #[test]
    fn multiword_tokens() {
        assert_eq!(check(&[("2-1", 0)], 3), [(1, MultiwordSyntax)]);
        assert_eq!(check(&[("1-1", 0)], 3), [(1, MultiwordSyntax)]);
        assert_eq!(check(&[("01-2", 0)], 3), [(1, MultiwordSyntax)]);
        assert_eq!(check(&[("1-2-3", 0)], 3), [(1, MultiwordSyntax)]);
        assert_eq!(check(&[("-", 0), ("1.5-2", 0)], 3), [(1, MultiwordSyntax), (2, MultiwordSyntax)]);
        assert_eq!(check(&[("2-3", 0)], 3), [(1, MultiwordStart { expected: 1 })]);
        assert_eq!(check(&[("2-4", 1)], 3), [(1, MultiwordPastEnd { words: 3 })]);
        // The second range's line comes before word 2, inside 1-3.
        assert_eq!(check(&[("1-3", 0), ("2-3", 1)], 3), [(2, MultiwordOverlap { previous_end: 3 })]);
        assert_eq!(check(&[("4294967296-4294967297", 0)], 3), [(1, MultiwordSyntax)]);
    }

    #[test]
    fn empty_nodes() {
        assert_eq!(check(&[("1.1", 0)], 2), [(1, EmptyNodeWord { expected: 0 })]);
        assert_eq!(check(&[("2.0", 2)], 2), [(1, EmptyNodeSyntax)]);
        assert_eq!(check(&[("2.", 2), ("x.1", 2)], 2), [(1, EmptyNodeSyntax), (2, EmptyNodeSyntax)]);
        assert_eq!(check(&[("2.2", 2)], 2), [(1, EmptyNodeIndex { expected: 1 })]);
        assert_eq!(check(&[("1.1", 1), ("1.3", 1)], 2), [(2, EmptyNodeIndex { expected: 2 })]);
        // The count starts again after the next word.
        assert_eq!(check(&[("1.1", 1), ("2.1", 2)], 2), []);
        assert_eq!(check(&[("1.1.1", 1)], 2), [(1, EmptyNodeSyntax)]);
    }

    #[test]
    fn spans_are_the_ids() {
        let src = "x 2.2";
        let node = NodeLine { id: Piece { text: &src[2..], span: Span::new(2, 5) }, line: 7, words_before: 2 };
        let issues = check_node_ids(&[node], 2);
        assert_eq!(issues, [NodeIdIssue { line: 7, id: Span::new(2, 5), error: EmptyNodeIndex { expected: 1 } }]);
        assert_eq!(&src[issues[0].id.range()], "2.2");
    }
}
