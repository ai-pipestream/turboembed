//! The chart parser: every reading of every stretch of lexemes the rules
//! allow, each with a score.
//!
//! Positions are filled from the last lexeme to the first. A node covers
//! lexemes `start..end`, and every child after its first starts later, so
//! when the parser works on position `s` all nodes starting after `s` are
//! final. At `s` it seeds the lexicon's nodes for the lexeme there and the
//! rules that open with a word, then takes each node made at `s` in turn
//! as the first child of the rules indexed under its category (the left
//! corner), matching their other items against the finished positions.
//!
//! Two nodes over the same lexemes with the same category and value are
//! one: the better is kept. At most [`CELL`] nodes share a span and a
//! category; a better one pushes out the worst. "Better" is a higher score
//! (rule weights plus lexemes covered), then fewer rule applications, then
//! the lower rule index. Nothing depends on hashing or timing, so a text
//! always gives the same chart.

use crate::value::{Num, NumKind, Value, flag};

use crate::engine::interned::Interned;
use crate::engine::lexer::{Kind, Lexed};
use crate::engine::rule::{Ctx, Dim, Grammar, Item, Output, Rule};

/// Nodes kept per span and category.
pub const CELL: usize = 4;

/// The rule index of a node seeded from a lexeme.
pub const TERMINAL: u16 = u16::MAX;

const NONE: u32 = u32::MAX;

/// One reading of lexemes `start..end`.
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct Node {
    pub start: u32,
    pub end: u32,
    pub dim: Dim,
    /// The rule that built it, or [`TERMINAL`].
    pub rule: u16,
    pub score: i32,
    /// Rule applications in the node's tree, itself included.
    pub apps: u32,
    pub value: Value,
    /// Its children's indexes, a range of [`Chart::kids`].
    pub kids: (u32, u32),
    /// The next node starting at the same lexeme.
    next: u32,
}

impl Node {
    fn better_than(&self, other: &Node) -> bool {
        (self.score, std::cmp::Reverse(self.apps), std::cmp::Reverse(self.rule))
            > (other.score, std::cmp::Reverse(other.apps), std::cmp::Reverse(other.rule))
    }
}

/// The nodes of one text.
#[derive(Clone, Debug, Default)]
pub struct Chart {
    pub nodes: Vec<Node>,
    pub kids: Vec<u32>,
    /// The first node starting at each lexeme.
    head: Vec<u32>,
}

impl Chart {
    /// The nodes starting at lexeme `pos`, newest first, the replaced ones
    /// included (their category is [`Dim::DEAD`]).
    pub fn starting_at(&self, pos: usize) -> impl Iterator<Item = (u32, &Node)> + '_ {
        let mut id = self.head.get(pos).copied().unwrap_or(NONE);
        std::iter::from_fn(move || {
            if id == NONE {
                return None;
            }
            let at = id;
            let n = &self.nodes[at as usize];
            id = n.next;
            Some((at, n))
        })
    }

    /// The children of node `id`.
    pub fn children(&self, id: u32) -> &[u32] {
        let (s, e) = self.nodes[id as usize].kids;
        &self.kids[s as usize..e as usize]
    }
}

/// A complete match of a rule's items, before its value is built.
struct Done {
    end: u32,
    kids_at: u32,
    kids_len: u32,
    words: u32,
}

/// A node chosen for an output: its byte span in the text, its value,
/// and its index in the chart.
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct Mention {
    pub start: u32,
    pub end: u32,
    pub value: Value,
    pub node: u32,
}

/// Parses texts with one grammar, keeping its buffers between texts.
pub struct Parser {
    grammar: &'static dyn Grammar,
    /// Rules by the category of their first item.
    by_dim: Vec<Vec<u16>>,
    /// Rules whose first item is a word, by that word's lowercase form,
    /// sorted.
    by_word: Vec<(String, u16)>,
    lexed: Lexed,
    chart: Chart,
    times: Interned,
    done: Vec<Done>,
    done_kids: Vec<u32>,
    stack: Vec<u32>,
    values: Vec<Value>,
    taken: Vec<bool>,
    cands: Vec<(u32, u32, u32)>,
    /// The ordinals chosen before numbers are, and per lexeme the index of
    /// the one covering it (or [`NONE`]).
    shadows: Vec<Mention>,
    shadowed: Vec<u32>,
}

impl Parser {
    /// # Panics
    ///
    /// When a rule has no items, opens with an optional word, or names a
    /// category at or above the grammar's `dims()`.
    pub fn new(grammar: &'static dyn Grammar) -> Parser {
        assert!(grammar.numeral().0 < grammar.dims(), "numeral uses an unknown category");
        assert!(grammar.lexicon().categories_below(grammar.dims()), "lexicon uses an unknown category");
        let rules = grammar.rules();
        assert!(rules.len() < usize::from(TERMINAL), "fewer rules than the terminal marker");
        let mut by_dim = vec![Vec::new(); usize::from(grammar.dims())];
        let mut by_word = Vec::new();
        for (i, r) in rules.iter().enumerate() {
            assert!(r.dim.0 < grammar.dims(), "rule {} builds an unknown category", r.name);
            for item in r.items {
                if let Item::Node { dim, .. } = item {
                    assert!(dim.0 < grammar.dims(), "rule {} reads an unknown category", r.name);
                }
            }
            match r.items.first() {
                Some(Item::Node { dim, .. }) => by_dim[usize::from(dim.0)].push(i as u16),
                Some(Item::Word { text, optional: false, .. }) => by_word.push((text.to_lowercase(), i as u16)),
                _ => panic!("rule {} opens with no item or an optional word", r.name),
            }
        }
        by_word.sort();
        Parser {
            grammar,
            by_dim,
            by_word,
            lexed: Lexed::default(),
            chart: Chart::default(),
            times: Interned::default(),
            done: Vec::new(),
            done_kids: Vec::new(),
            stack: Vec::new(),
            values: Vec::new(),
            taken: Vec::new(),
            cands: Vec::new(),
            shadows: Vec::new(),
            shadowed: Vec::new(),
        }
    }

    pub fn grammar(&self) -> &'static dyn Grammar {
        self.grammar
    }

    pub fn lexed(&self) -> &Lexed {
        &self.lexed
    }

    pub fn chart(&self) -> &Chart {
        &self.chart
    }

    /// The time values of the last parse.
    pub fn times(&self) -> &Interned {
        &self.times
    }
}

mod parse;
mod select;
mod storage;
use storage::{insert, repeats, walk};
#[cfg(test)]
mod tests;
