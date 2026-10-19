//! The lines of an Arvores Deitadas tree and the tree they make:
//! `ADSentenceStream.SentenceParser.getElement` and the node stack of
//! `SentenceParser.parse`.
//!
//! A tree line is a node (`==P<:np`), a leaf (`===H:n("dia" <np-def> M S)
//! dia`), or punctuation (`==,`). The run of `=` (and `-`) at its start
//! gives the level. Every index here is a byte offset into the line; the
//! characters the parser tests are ASCII or whitespace in the Basic
//! Multilingual Plane, so the scans find the same pieces Java's scans of
//! UTF-16 units find. The one level counted over arbitrary text (a leaf
//! after the last `=` of a line no other rule reads) is counted in UTF-16
//! units, as Java counts it.

use turbo_analysis::Span;
use turbo_unicode::WhitespaceMode;

use crate::source::{Line, Piece};

const LEVEL_MARK: u8 = b'=';
const LEVEL_HYPHEN: u8 = b'-';
const TAG_SEPARATOR: u8 = b':';
const BIZARRE_TAG_SEPARATOR: u8 = b'=';
const GROUP_OPEN: u8 = b'(';
const GROUP_CLOSE: u8 = b')';
const BRACKET_OPEN: u8 = b'<';
const BRACKET_CLOSE: u8 = b'>';

/// `StringUtil.isUnicodeWhitespace`, which the parser uses whatever the
/// whitespace mode.
pub(crate) fn is_ws(c: char) -> bool {
    WhitespaceMode::Unicode.is_whitespace(c as u32)
}

fn is_quote(b: u8) -> bool {
    b == b'"' || b == b'\''
}

/// A leaf: a word or punctuation and its tags. A tag Java leaves `null`
/// is `None`; a tag set to the empty string is an empty piece.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct Leaf<'a> {
    /// The syntactic function, such as `H` or `>N`.
    pub syntactic: Option<Piece<'a>>,
    /// The word class, such as `n` or `v-fin`.
    pub functional: Option<Piece<'a>>,
    pub lemma: Option<Piece<'a>>,
    /// The secondary tags, such as `<np-def> <NER:hum>`.
    pub secondary: Option<Piece<'a>>,
    /// The inflection, such as `M S`.
    pub morphological: Option<Piece<'a>>,
    /// The word as it occurs in the text.
    pub lexeme: Piece<'a>,
    /// The leaf's line.
    pub line: Span,
}

/// A node or a leaf of a tree.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) enum Element<'a> {
    /// A node, its tag (such as `P<:np`; `None` for the root) and its
    /// children, indexes into [`Tree::elements`].
    Node {
        tag: Option<Piece<'a>>,
        children: Vec<usize>,
    },
    Leaf(Leaf<'a>),
}

/// The tree of one sentence. Element 0 is the root; the others are in the
/// order of their lines, which is the order a depth-first walk visits
/// them.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct Tree<'a> {
    pub elements: Vec<Element<'a>>,
}

impl<'a> Tree<'a> {
    /// The leaves, in the order of their lines.
    pub fn leaves(&self) -> impl Iterator<Item = &Leaf<'a>> {
        self.elements.iter().filter_map(|e| match e {
            Element::Leaf(l) => Some(l),
            Element::Node { .. } => None,
        })
    }

    /// The children of the root.
    pub fn root_children(&self) -> &[usize] {
        match &self.elements[0] {
            Element::Node { children, .. } => children,
            Element::Leaf(_) => unreachable!("the root is a node"),
        }
    }
}

/// Builds a tree from its lines, as `SentenceParser.parse` does with its
/// stack: each element goes under the last node above it of a lower level.
pub(crate) struct TreeBuilder<'a> {
    elements: Vec<Element<'a>>,
    /// The open nodes, as their index and level; the root, of level 0, is
    /// never removed, since every element has a level of 1 or more.
    stack: Vec<(usize, u32)>,
}

impl<'a> TreeBuilder<'a> {
    pub fn new() -> TreeBuilder<'a> {
        TreeBuilder { elements: vec![Element::Node { tag: None, children: Vec::new() }], stack: vec![(0, 0)] }
    }

    /// Adds the element of one tree line, if the line has one.
    pub fn line(&mut self, line: Line<'a>) {
        let Some((level, element)) = parse_line(line.piece) else {
            return;
        };
        while self.stack.last().is_some_and(|&(_, l)| level <= l) {
            self.stack.pop();
        }
        let index = self.elements.len();
        let &(parent, _) = self.stack.last().expect("the root stays");
        if let Element::Node { children, .. } = &mut self.elements[parent] {
            children.push(index);
        }
        let is_node = matches!(element, Element::Node { .. });
        self.elements.push(element);
        if is_node {
            self.stack.push((index, level));
        }
    }

    pub fn finish(self) -> Tree<'a> {
        Tree { elements: self.elements }
    }
}

/// A tag at the start of a line: where it starts (the length of the
/// level prefix it follows), its separator and its end.
#[derive(Copy, Clone, Debug)]
struct Tag {
    start: usize,
    separator: usize,
    end: usize,
}

/// The end of a leaf line: the morphological tag, the closing
/// parenthesis that ends it and the start of the lexeme.
#[derive(Copy, Clone, Debug)]
struct LeafEnd {
    morphology_start: usize,
    close: usize,
    lexeme_start: usize,
}

/// One line and what its scans remember.
struct Scan<'a> {
    line: Piece<'a>,
    bytes: &'a [u8],
    /// For each index, whether no end of a leaf line was found there; made
    /// on the first leaf scan.
    no_end_at: Vec<bool>,
}

impl<'a> Scan<'a> {
    fn new(line: Piece<'a>) -> Scan<'a> {
        Scan { line, bytes: line.text.as_bytes(), no_end_at: Vec::new() }
    }

    fn len(&self) -> usize {
        self.bytes.len()
    }

    fn piece(&self, start: usize, end: usize) -> Piece<'a> {
        self.line.slice(start, end)
    }

    /// The index of the first character at or after `from` that is not
    /// whitespace, or the length.
    fn skip_ws(&self, from: usize) -> usize {
        let text = self.line.text;
        text[from..].char_indices().find(|&(_, c)| !is_ws(c)).map_or(text.len(), |(i, _)| from + i)
    }

    /// The end of the run of `=` and `-` at the start of the line.
    fn level_prefix_end(&self) -> usize {
        self.bytes.iter().take_while(|&&b| b == LEVEL_MARK || b == LEVEL_HYPHEN).count()
    }

    /// `scanLevelAndTag`: the tag after the longest level prefix whose
    /// end is at most `max_start` and that is the whole prefix or ends
    /// before a hyphen; the hyphens after it then start the tag.
    fn level_and_tag(&self, prefix: usize, separator: u8, max_start: Option<usize>) -> Option<Tag> {
        let mut start = prefix.min(max_start?);
        loop {
            if (start == prefix || self.bytes[start] == LEVEL_HYPHEN)
                && let Some(tag) = self.tag(start, separator)
            {
                return Some(tag);
            }
            start = start.checked_sub(1)?;
        }
    }

    /// `scanTag`: characters up to the first `:` or `=`, which must be
    /// `separator` and not first, then one or more characters that are
    /// neither `(` nor whitespace.
    fn tag(&self, from: usize, separator: u8) -> Option<Tag> {
        let b = self.bytes;
        let i = from + b[from..].iter().position(|&c| c == TAG_SEPARATOR || c == BIZARRE_TAG_SEPARATOR)?;
        if i == from || b[i] != separator {
            return None;
        }
        let rest = &self.line.text[i + 1..];
        let stop = rest.char_indices().find(|&(_, c)| c == '(' || is_ws(c)).map_or(rest.len(), |(o, _)| o);
        (stop > 0).then_some(Tag { start: from, separator: i, end: i + 1 + stop })
    }

    /// `isNodeTail`: after the tag, an optional part in parentheses, then
    /// tag groups.
    fn is_node_tail(&self, from: usize) -> bool {
        if self.bytes.get(from) == Some(&GROUP_OPEN)
            && let Some(rel) = self.bytes[from + 1..].iter().position(|&c| c == GROUP_CLOSE)
        {
            let close = from + 1 + rel;
            if close > from + 1 && self.is_tag_group_run(close + 1) {
                return true;
            }
        }
        self.is_tag_group_run(from)
    }

    /// `isTagGroupRun`: after optional whitespace, nothing, or `(<`, one
    /// or more characters and `>)`, then optional whitespace.
    fn is_tag_group_run(&self, from: usize) -> bool {
        let start = self.skip_ws(from);
        let text = self.line.text;
        let end = start + text[start..].trim_end_matches(is_ws).len();
        if start == end {
            return true;
        }
        end >= start + 5 && self.bytes[start..].starts_with(b"(<") && self.bytes[..end].ends_with(b">)")
    }

    /// `scanLeafEnd`: optional whitespace, an optional morphological tag up
    /// to the first `)`, whitespace, and a lexeme that is not empty.
    fn leaf_end(&self, from: usize) -> Option<LeafEnd> {
        let morphology_start = self.skip_ws(from);
        let close = morphology_start + self.bytes[morphology_start..].iter().position(|&c| c == GROUP_CLOSE)?;
        let lexeme_start = self.skip_ws(close + 1);
        if lexeme_start == close + 1 || lexeme_start == self.len() {
            return None;
        }
        Some(LeafEnd { morphology_start, close, lexeme_start })
    }

    /// `scanSecondaryTags`: secondary tags (`<`, one or more characters,
    /// `>`) from `from`, then the end of the leaf line. The longest tag,
    /// then one more tag, is tried first; a place where the end of the
    /// line was not found is remembered. Java recurses once per tag; this
    /// walks the same order with a stack of its own.
    fn secondary_tags(&mut self, from: usize) -> Option<(usize, LeafEnd)> {
        if self.no_end_at.is_empty() {
            self.no_end_at = vec![false; self.len() + 1];
        }
        // Each frame: where its tags start, and the next index to look for
        // a closing bracket at, going down (None once they are all tried).
        let mut stack: Vec<(usize, Option<usize>)> = Vec::new();
        let enter = |scan: &Scan<'_>, from: usize| -> Option<(usize, Option<usize>)> {
            if scan.no_end_at[from] {
                return None;
            }
            let cursor = (scan.bytes.get(from) == Some(&BRACKET_OPEN) && scan.len() >= 1).then(|| scan.len() - 1);
            Some((from, cursor))
        };
        stack.extend(enter(self, from));
        while let Some(frame) = stack.last_mut() {
            let (start, cursor) = *frame;
            if let Some(c) = cursor {
                if c > start + 1 {
                    frame.1 = Some(c - 1);
                    if self.bytes[c] == BRACKET_CLOSE
                        && let Some(child) = enter(self, c + 1)
                    {
                        stack.push(child);
                    }
                    continue;
                }
                frame.1 = None;
            }
            stack.pop();
            match self.leaf_end(start) {
                Some(end) => return Some((start, end)),
                None => self.no_end_at[start] = true,
            }
        }
        None
    }

    fn morphology(&self, end: LeafEnd) -> Option<Piece<'a>> {
        (end.morphology_start != end.close).then(|| self.piece(end.morphology_start, end.close))
    }
}

mod parse;
#[cfg(test)]
mod tests;

pub(crate) use self::parse::parse_line;
