//! Constituency parse trees and Penn Treebank bracket I/O (`opennlp.tools.parser.Parse`).

use std::cmp::Ordering;
use std::fmt;
use std::hash::{Hash, Hasher};
use std::sync::Arc;

use turbo_analysis::Span;
use turbo_jvm::strict_math;

mod edit;
mod error;
mod io;
#[cfg(test)]
mod tests;
mod walk;

pub use error::ParseError;
pub use io::{
    BRACKET_LCB, BRACKET_LRB, BRACKET_LSB, BRACKET_RCB, BRACKET_RRB, BRACKET_RSB, Constituent, TokenSource,
    decode_token, encode_token, space_untokenized_parens,
};
/// The constituent label of the root node (`Parser.TOP_NODE`).
pub const TOP_NODE: &str = "TOP";

/// The constituent label of an incomplete root node (`Parser.INC_NODE`).
pub const INC_NODE: &str = "INC";

/// The constituent label of a token leaf node (`Parser.TOK_NODE`).
pub const TOK_NODE: &str = "TK";

/// Prefix for outcomes starting a constituent (`AbstractBottomUpParser.START`).
pub const START: &str = "S-";

/// Prefix for outcomes continuing a constituent (`AbstractBottomUpParser.CONT`).
pub const CONT: &str = "C-";

/// Outcome for a token not contained in a basal constituent (`AbstractBottomUpParser.OTHER`).
pub const OTHER: &str = "O";

/// Outcome used when a constituent is complete (`AbstractBottomUpParser.COMPLETE`).
pub const COMPLETE: &str = "c";

/// Outcome used when a constituent is incomplete (`AbstractBottomUpParser.INCOMPLETE`).
pub const INCOMPLETE: &str = "i";

/// The node a parse takes as its head (`Parse.getHead`), recorded as the head node's span and
/// head token index.
///
/// OpenNLP keeps a reference to the head node. Its users read only the head's covered text and
/// head index, so those are kept instead, and they stay right when the tree is rearranged around
/// the head. They are what the head node had when it was chosen: a later change to that node's
/// span is not seen, where Java would see it.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Head {
    /// The head node's span in the sentence text.
    pub span: Span,
    /// The head node's head token index (`Parse.getHeadIndex`).
    pub head_index: i32,
}

/// Builds a tree by inserting constituents into its root as [`Parse::insert`] does, in less time
/// when they come in the order a bracketed tree closes them: each after the constituents inside it
/// and after those to its left.
///
/// `insert` looks at every child of the root for each constituent, so a flat sentence of `n`
/// tokens takes time in `n` squared. While the constituents come in that order, the root's
/// children are sorted, do not overlap and are not empty, and a new constituent that ends at or
/// after the last child and starts at or after the end of the child before the first one it
/// covers takes the children from that one on and goes last, which is what `insert` does with
/// it. The builder finds that child by binary search and checks the conditions; the first
/// constituent that does not meet them, and every one after it, goes through `insert`.
#[derive(Debug)]
pub struct TreeBuilder {
    root: Parse,
    ordered: bool,
}

impl TreeBuilder {
    /// A builder over `root`, which has no children.
    pub fn new(root: Parse) -> TreeBuilder {
        let ordered = root.parts.is_empty();
        TreeBuilder { root, ordered }
    }

    /// Inserts `constituent` into the tree, as [`Parse::insert`] does.
    pub fn insert(&mut self, mut constituent: Parse) -> Result<(), ParseError> {
        let ic = constituent.span;
        if self.ordered && ic.start < ic.end && self.root.span.contains(ic) {
            let parts = &mut self.root.parts;
            let first = parts.partition_point(|p| p.span.start < ic.start);
            let after_left = first == 0 || parts[first - 1].span.end <= ic.start;
            let covers_rest = parts.last().is_none_or(|p| p.span.end <= ic.end);
            if after_left && covers_rest {
                if !self.root.text.is_char_boundary(ic.start as usize)
                    || !self.root.text.is_char_boundary(ic.end as usize)
                {
                    return Err(ParseError::NotCharBoundary { span: ic });
                }
                constituent.parts.extend(parts.drain(first..));
                parts.push(constituent);
                return Ok(());
            }
        }
        self.ordered = false;
        self.root.insert(constituent)
    }

    /// The tree.
    pub fn finish(self) -> Parse {
        self.root
    }
}

/// A constituency parse node (`opennlp.tools.parser.Parse`).
///
/// All nodes in the same sentence share an `Arc<str>` holding the space-joined sentence text, and
/// `span` stores UTF-8 byte offsets into that text. A node is its own head until head rules give it
/// another ([`Head`]).
///
/// `Clone` copies the whole subtree. Java's `clone` copies the node and shares its children, and a
/// clone of a node that is its own head keeps the original as its head; here the clone is its own
/// head, and [`Parse::clone_keeping_head`] makes the copy the parsers make of a sentence's root.
#[derive(Clone, Debug)]
pub struct Parse {
    text: Arc<str>,
    span: Span,
    kind: String,
    parts: Vec<Parse>,
    /// `None` when the node is its own head (Java's `head == this`).
    head: Option<Head>,
    head_index: i32,
    label: Option<String>,
    prob: f64,
    is_chunk: bool,
    derivation: Option<String>,
    prev_punct: Vec<Parse>,
    next_punct: Vec<Parse>,
}

/// `Double.compare`: numeric order with `-0.0` before `0.0`, and every NaN equal to every other
/// and after everything else.
fn java_double_compare(a: f64, b: f64) -> Ordering {
    match a.partial_cmp(&b) {
        Some(Ordering::Equal) | None => {
            let bits = |x: f64| if x.is_nan() { 0x7ff8_0000_0000_0000_i64 } else { x.to_bits() as i64 };
            bits(a).cmp(&bits(b))
        }
        Some(o) => o,
    }
}

/// A sentence text length as a u32 span offset, or [`ParseError::TooLong`]. Kept as its own
/// function so its integer boundary needs no multi-gigabyte allocation to test.
pub(super) fn checked_text_len(len: usize) -> Result<u32, ParseError> {
    u32::try_from(len).map_err(|_| ParseError::TooLong { len })
}

/// `TreeSet.add` under `Parse.compareTo`: keeps `set` sorted and drops `punct` when an element
/// compares equal to it.
pub(crate) fn insert_by_prob(set: &mut Vec<Parse>, punct: Parse) {
    if let Err(pos) = set.binary_search_by(|p| p.cmp_prob(&punct)) {
        set.insert(pos, punct);
    }
}

/// `Parse.equals`: the label, span, text and children, but not the type, so `(NP x)` and `(VP x)`
/// over the same text are equal.
impl PartialEq for Parse {
    fn eq(&self, other: &Parse) -> bool {
        self.label == other.label && self.span == other.span && *self.text == *other.text && self.parts == other.parts
    }
}

impl Eq for Parse {}

/// `Parse.hashCode`: the span and text.
impl Hash for Parse {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.span.hash(state);
        (*self.text).hash(state);
    }
}

/// `Parse.toString`: the covered text.
/// Drops the nodes under a parse one by one rather than recursively: a tree read from a few
/// megabytes of nested brackets is deep enough to overflow the stack if each level's drop calls
/// the next.
impl Drop for Parse {
    fn drop(&mut self) {
        if self.parts.is_empty() && self.prev_punct.is_empty() && self.next_punct.is_empty() {
            return;
        }
        let mut pending = Vec::new();
        take_children(self, &mut pending);
        while let Some(mut node) = pending.pop() {
            // Once its children are taken, `node` drops without recursing.
            take_children(&mut node, &mut pending);
        }
    }
}

/// Moves the children and punctuation of `node` onto `into`.
fn take_children(node: &mut Parse, into: &mut Vec<Parse>) {
    into.append(&mut node.parts);
    into.append(&mut node.prev_punct);
    into.append(&mut node.next_punct);
}

impl fmt::Display for Parse {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.covered_text())
    }
}

impl Parse {
    /// Creates a parse node that is its own head (`new Parse(text, span, type, p, index)`).
    ///
    /// `span` must start and end on character boundaries of `text`; [`Parse::insert`] checks this,
    /// and [`Parse::show`] and [`Parse::covered_text`] panic on a node that breaks it.
    pub fn new(text: impl Into<Arc<str>>, span: Span, kind: impl Into<String>, prob: f64, head_index: i32) -> Parse {
        Parse {
            text: text.into(),
            span,
            kind: kind.into(),
            parts: Vec::new(),
            head: None,
            head_index,
            label: None,
            prob,
            is_chunk: false,
            derivation: None,
            prev_punct: Vec::new(),
            next_punct: Vec::new(),
        }
    }

    /// Creates a parse node with `head` as its head and head index, or its own head with index 0
    /// when `head` is `None` (`new Parse(text, span, type, p, h)`).
    pub fn with_head(
        text: impl Into<Arc<str>>,
        span: Span,
        kind: impl Into<String>,
        prob: f64,
        head: Option<Head>,
    ) -> Parse {
        let mut p = Parse::new(text, span, kind, prob, 0);
        if let Some(h) = head {
            p.head = Some(h);
            p.head_index = h.head_index;
        }
        p
    }

    /// Inserts `constituent` into this parse tree based on its text span (`Parse.insert`).
    ///
    /// The constituent goes into the deepest node with a child that contains it, and takes the
    /// children of that node that it contains, a child with the same span included.
    pub fn insert(&mut self, mut constituent: Parse) -> Result<(), ParseError> {
        let ic = constituent.span;
        if ic.start > ic.end || !self.span.contains(ic) {
            return Err(ParseError::NotContained { parent: self.span, child: ic });
        }
        if !self.text.is_char_boundary(ic.start as usize) || !self.text.is_char_boundary(ic.end as usize) {
            return Err(ParseError::NotCharBoundary { span: ic });
        }
        let mut node = self;
        'descend: loop {
            let mut pi = 0;
            while pi < node.parts.len() {
                let sp = node.parts[pi].span;
                if sp.start >= ic.end {
                    break;
                } else if ic.contains(sp) {
                    constituent.parts.push(node.parts.remove(pi));
                } else if sp.contains(ic) {
                    node = &mut node.parts[pi];
                    continue 'descend;
                } else {
                    pi += 1;
                }
            }
            node.parts.insert(pi, constituent);
            return Ok(());
        }
    }

    /// Appends the Penn Treebank bracketed representation of this parse to `out` (`Parse.show`).
    ///
    /// Text of a node that its children do not cover is written as a token, with brackets
    /// escaped; a parse read by [`Parse::parse`] has the space after its last token inside the
    /// root, so the root closes with ` )`.
    pub fn show(&self, out: &mut String) {
        let mut start = self.span.start as usize;
        if self.kind != TOK_NODE {
            out.push('(');
            out.push_str(&self.kind);
            out.push(' ');
        }
        for c in &self.parts {
            let s_start = c.span.start as usize;
            if start < s_start {
                out.push_str(encode_token(&self.text[start..s_start]));
            }
            c.show(out);
            start = c.span.end as usize;
        }
        let end = self.span.end as usize;
        if start < end {
            out.push_str(encode_token(&self.text[start..end]));
        }
        if self.kind != TOK_NODE {
            out.push(')');
        }
    }

    /// Returns the Penn Treebank bracketed representation of this parse (`Parse.toStringPennTreebank`).
    pub fn to_string_penn_treebank(&self) -> String {
        let mut out = String::with_capacity(self.text.len() * 4);
        self.show(&mut out);
        out
    }

    /// Returns the log probability sum of the POS-tag sequence under this parse
    /// (`Parse.getTagSequenceProb`), using `turbo_jvm::strict_math::log` (`StrictMath.log`).
    pub fn tag_sequence_prob(&self) -> f64 {
        if self.parts.len() == 1 && self.parts[0].kind == TOK_NODE {
            strict_math::log(self.prob)
        } else if self.parts.is_empty() {
            0.0
        } else {
            let mut sum = 0.0;
            for part in &self.parts {
                sum += part.tag_sequence_prob();
            }
            sum
        }
    }

    /// Returns `true` if this parse has a single top-level child (`Parse.complete`).
    pub fn complete(&self) -> bool {
        self.parts.len() == 1
    }

    /// Returns the substring of `text` covered by this node's `span` (`Parse.getCoveredText`).
    pub fn covered_text(&self) -> &str {
        &self.text[self.span.range()]
    }

    /// Returns the shared sentence text (`Parse.getText`).
    pub fn text(&self) -> &str {
        &self.text
    }

    /// Returns the shared `Arc<str>` holding the sentence text.
    pub fn text_arc(&self) -> &Arc<str> {
        &self.text
    }

    /// Returns the UTF-8 byte span of this constituent in `text` (`Parse.getSpan`).
    pub fn span(&self) -> Span {
        self.span
    }

    /// Returns the syntactic constituent type (`Parse.getType`).
    pub fn kind(&self) -> &str {
        &self.kind
    }

    /// Sets the syntactic constituent type (`Parse.setType`).
    pub fn set_kind(&mut self, kind: impl Into<String>) {
        self.kind = kind.into();
    }

    /// Returns the probability or log-probability score associated with this node (`Parse.getProb`).
    pub fn prob(&self) -> f64 {
        self.prob
    }

    /// Adds `log_prob` to this node's probability score (`Parse.addProb`).
    pub fn add_prob(&mut self, log_prob: f64) {
        self.prob += log_prob;
    }

    /// Returns the child constituents (`Parse.getChildren`).
    pub fn children(&self) -> &[Parse] {
        &self.parts
    }

    /// Returns a mutable slice of the child constituents.
    pub fn children_mut(&mut self) -> &mut [Parse] {
        &mut self.parts
    }

    /// Returns the number of child constituents (`Parse.getChildCount`).
    pub fn child_count(&self) -> usize {
        self.parts.len()
    }

    /// Returns the head of this node (`Parse.getHead`): the node itself until head rules give it
    /// another.
    pub fn head(&self) -> Head {
        self.head.unwrap_or(Head { span: self.span, head_index: self.head_index })
    }

    /// Returns this node as the head of another node: its own span and head index, as Java's head
    /// rules return a constituent itself.
    pub fn as_head(&self) -> Head {
        Head { span: self.span, head_index: self.head_index }
    }

    /// Copies this node as the bottom-up parsers' `(Parse) p.clone()` copies the sentence's root
    /// for each derivation. Java's copy of a node that is its own head keeps that node as its
    /// head, so when tree insertion later trims the copy's span by removing a first or last child,
    /// the head still has the span the node has now. The copy takes that span as its head here.
    /// [`Clone`] leaves a copy its own head.
    pub fn clone_keeping_head(&self) -> Parse {
        let mut copy = self.clone();
        if copy.head.is_none() {
            copy.head = Some(self.as_head());
        }
        copy
    }

    /// Returns `true` when this node is its own head (Java's `getHead() == this`).
    pub fn is_own_head(&self) -> bool {
        self.head.is_none()
    }

    /// Returns the text the head covers (`getHead().getCoveredText()`).
    pub fn head_text(&self) -> &str {
        &self.text[self.head().span.range()]
    }

    /// Returns the sentence token index of the head token (`Parse.getHeadIndex`).
    pub fn head_index(&self) -> i32 {
        self.head_index
    }

    /// Returns the parser stage label assigned to this node (`Parse.getLabel`).
    pub fn label(&self) -> Option<&str> {
        self.label.as_deref()
    }

    /// Assigns a parser stage label to this node (`Parse.setLabel`).
    pub fn set_label(&mut self, label: Option<String>) {
        self.label = label;
    }

    /// Returns `true` if this node is a POS-tag preterminal (`Parse.isPosTag`).
    pub fn is_pos_tag(&self) -> bool {
        self.parts.len() == 1 && self.parts[0].kind == TOK_NODE
    }

    /// Returns `true` if every child of this node is a POS-tag preterminal (`Parse.isFlat`).
    pub fn is_flat(&self) -> bool {
        self.parts.iter().all(Parse::is_pos_tag)
    }

    /// Marks whether this node was built during the chunking phase (`Parse.isChunk(boolean)`).
    pub fn set_is_chunk(&mut self, is_chunk: bool) {
        self.is_chunk = is_chunk;
    }

    /// Returns whether this node was built during the chunking phase (`Parse.isChunk()`).
    pub fn is_chunk(&self) -> bool {
        self.is_chunk
    }

    pub(crate) fn replace_text_and_span(&mut self, text: Arc<str>, span: Span) {
        self.text = text;
        self.span = span;
    }

    /// Compares two parses by descending probability (`Parse.compareTo`, which uses
    /// `Double.compare`).
    pub fn cmp_prob(&self, other: &Parse) -> Ordering {
        java_double_compare(other.prob, self.prob)
    }
}
