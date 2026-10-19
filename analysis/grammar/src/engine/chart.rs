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

use super::interned::Interned;
use super::lexer::{Kind, Lexed};
use super::rule::{Ctx, Dim, Grammar, Item, Output, Rule};

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
        let rules = grammar.rules();
        assert!(rules.len() < usize::from(TERMINAL), "fewer rules than the terminal marker");
        let mut by_dim = vec![Vec::new(); usize::from(grammar.dims())];
        let mut by_word = Vec::new();
        for (i, r) in rules.iter().enumerate() {
            assert!(r.dim.0 < grammar.dims(), "rule {} builds an unknown category", r.name);
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

    /// Lexes and parses `text`, replacing the chart.
    pub fn parse(&mut self, text: &str) {
        self.lexed.lex_words(text, &self.grammar.style(), self.grammar.words());
        let n = self.lexed.len();
        self.chart.nodes.clear();
        self.chart.kids.clear();
        self.chart.head.clear();
        self.chart.head.resize(n, NONE);
        self.times.clear();
        let rules = self.grammar.rules();
        for s in (0..n).rev() {
            let first = self.chart.nodes.len();
            self.seed(s);
            let w = self.lexed.word(s);
            let lo = self.by_word.partition_point(|(k, _)| k.as_str() < w);
            let hi = lo + self.by_word[lo..].partition_point(|(k, _)| k.as_str() == w);
            for k in lo..hi {
                self.complete(text, rules, self.by_word[k].1, s, None);
            }
            let mut cursor = first;
            while cursor < self.chart.nodes.len() {
                let id = cursor as u32;
                cursor += 1;
                let dim = self.chart.nodes[id as usize].dim;
                if dim == Dim::DEAD {
                    continue;
                }
                for k in 0..self.by_dim[usize::from(dim.0)].len() {
                    let r = self.by_dim[usize::from(dim.0)][k];
                    self.complete(text, rules, r, s, Some(id));
                }
            }
        }
    }

    /// The lexicon's nodes, or the numeral's, for lexeme `s`.
    fn seed(&mut self, s: usize) {
        let lexeme = self.lexed.lexemes[s];
        let add = |chart: &mut Chart, dim: Dim, value: Value| {
            insert(
                chart,
                Node {
                    start: s as u32,
                    end: s as u32 + 1,
                    dim,
                    rule: TERMINAL,
                    score: 1,
                    apps: 0,
                    value,
                    kids: (0, 0),
                    next: NONE,
                },
            );
        };
        match lexeme.kind {
            Kind::Numeral => {
                let v = self.lexed.numeral(s).expect("a numeral lexeme has a value");
                add(&mut self.chart, self.grammar.numeral(), Value::Num(Num::new(v, NumKind::of(v), flag::DIGITS)));
            }
            Kind::Word | Kind::Punct => {
                for e in self.grammar.lexicon().lookup(self.lexed.word(s)) {
                    add(&mut self.chart, e.dim, Value::Term { value: e.value, flags: e.flags });
                }
            }
        }
    }

    /// Matches rule `r` at lexeme `s`, with `first` as its first child when
    /// the rule opens with a node, and adds the nodes it builds.
    fn complete(&mut self, text: &str, rules: &'static [Rule], r: u16, s: usize, first: Option<u32>) {
        let rule = &rules[usize::from(r)];
        self.done.clear();
        self.done_kids.clear();
        self.stack.clear();
        match first {
            Some(id) => {
                let n = &self.chart.nodes[id as usize];
                let Item::Node { test, .. } = rule.items[0] else { unreachable!("indexed by its first node") };
                if test.is_some_and(|t| !t(&n.value)) {
                    return;
                }
                self.stack.push(id);
                walk(
                    &self.chart,
                    &self.lexed,
                    text,
                    &rule.items[1..],
                    n.end,
                    &mut self.stack,
                    0,
                    &mut self.done,
                    &mut self.done_kids,
                );
            }
            None => walk(
                &self.chart,
                &self.lexed,
                text,
                rule.items,
                s as u32,
                &mut self.stack,
                0,
                &mut self.done,
                &mut self.done_kids,
            ),
        }
        for d in 0..self.done.len() {
            let Done { end, kids_at, kids_len, words } = self.done[d];
            let kids = &self.done_kids[kids_at as usize..(kids_at + kids_len) as usize];
            self.values.clear();
            let (mut score, mut apps) = (i32::from(rule.weight) + words as i32, 1u32);
            for &k in kids {
                let child = &self.chart.nodes[k as usize];
                self.values.push(child.value);
                score += child.score;
                apps += child.apps;
            }
            let ctx = Ctx { text, lexed: &self.lexed, start: s, end: end as usize, times: &self.times };
            let Some(value) = (rule.build)(&self.values, &ctx) else { continue };
            let k0 = self.chart.kids.len() as u32;
            let node = Node {
                start: s as u32,
                end,
                dim: rule.dim,
                rule: r,
                score,
                apps,
                value,
                kids: (k0, k0 + kids_len),
                next: NONE,
            };
            if insert(&mut self.chart, node) {
                self.chart.kids.extend_from_slice(kids);
            }
        }
    }

    /// Chooses the nodes for `output`: the candidates the grammar keeps
    /// that do not cut a written token, longest first (then by score,
    /// fewer applications, earlier start and lower rule index), each taken
    /// when it overlaps none taken before. Appends them to `out` in text
    /// order.
    ///
    /// A number that lies inside a chosen ordinal is part of that ordinal,
    /// not a value the text states (`twenty` in `the twenty first`), so it
    /// is left out unless the grammar's [`Grammar::shadows`] says the
    /// ordinal leaves its numbers standing.
    pub fn select(&mut self, text: &str, output: Output, out: &mut Vec<Mention>) {
        self.shadowed.clear();
        if output == Output::Number {
            let mut shadows = std::mem::take(&mut self.shadows);
            shadows.clear();
            self.pick(text, Output::Ordinal, &mut shadows);
            self.shadowed.resize(self.lexed.len(), NONE);
            for (i, m) in shadows.iter().enumerate() {
                let node = &self.chart.nodes[m.node as usize];
                if self.grammar.shadows(node) {
                    self.shadowed[node.start as usize..node.end as usize].fill(i as u32);
                }
            }
            self.shadows = shadows;
        }
        self.pick(text, output, out);
    }

    fn pick(&mut self, text: &str, output: Output, out: &mut Vec<Mention>) {
        let n = self.lexed.len();
        let lex = &self.lexed.lexemes;
        self.cands.clear();
        for (id, node) in self.chart.nodes.iter().enumerate() {
            if node.dim == Dim::DEAD || self.grammar.output(node.dim) != Some(output) {
                continue;
            }
            let (s, e) = (node.start as usize, node.end as usize);
            // A node may not start or end inside a word glued to a numeral
            // (`29km`, `x10`).
            if lex[s].glued && s > 0 && lex[s - 1].is_alnum() && lex[s].is_alnum() {
                continue;
            }
            if e < n && lex[e].glued && lex[e].is_alnum() && lex[e - 1].is_alnum() {
                continue;
            }
            // Chosen ordinals do not overlap, so one index at both ends
            // means one ordinal covers the whole span.
            if output == Output::Number && !self.shadowed.is_empty() {
                let owner = self.shadowed[s];
                if owner != NONE && owner == self.shadowed[e - 1] {
                    continue;
                }
            }
            let ctx = Ctx { text, lexed: &self.lexed, start: s, end: e, times: &self.times };
            if !self.grammar.keep(output, node, &ctx) {
                continue;
            }
            self.cands.push((id as u32, lex[s].start, lex[e - 1].end));
        }
        let nodes = &self.chart.nodes;
        self.cands.sort_by(|a, b| {
            let (x, y) = (&nodes[a.0 as usize], &nodes[b.0 as usize]);
            (b.2 - b.1)
                .cmp(&(a.2 - a.1))
                .then(y.score.cmp(&x.score))
                .then(x.apps.cmp(&y.apps))
                .then(a.1.cmp(&b.1))
                .then(x.rule.cmp(&y.rule))
                .then(a.0.cmp(&b.0))
        });
        self.taken.clear();
        self.taken.resize(n, false);
        let from = out.len();
        for &(id, start, end) in &self.cands {
            let node = &nodes[id as usize];
            let span = node.start as usize..node.end as usize;
            if self.taken[span.clone()].iter().any(|&t| t) {
                continue;
            }
            self.taken[span].fill(true);
            out.push(Mention { start, end, value: node.value, node: id });
        }
        out[from..].sort_by_key(|p| p.start);
    }
}

/// Adds `node` to its cell, unless an equal or better node with the same
/// value is there, or the cell is full of better ones. A node it beats is
/// marked dead. Returns whether it was added.
fn insert(chart: &mut Chart, mut node: Node) -> bool {
    let start = node.start as usize;
    let mut count = 0;
    let mut worst: Option<u32> = None;
    let mut id = chart.head[start];
    while id != NONE {
        let n = &chart.nodes[id as usize];
        if n.dim == node.dim && n.end == node.end {
            if n.value == node.value {
                if !node.better_than(n) {
                    return false;
                }
                chart.nodes[id as usize].dim = Dim::DEAD;
                break;
            }
            count += 1;
            if worst.is_none_or(|w| chart.nodes[w as usize].better_than(n)) {
                worst = Some(id);
            }
        }
        id = n.next;
    }
    if count >= CELL {
        let w = worst.expect("a full cell has a worst node");
        if !node.better_than(&chart.nodes[w as usize]) {
            return false;
        }
        chart.nodes[w as usize].dim = Dim::DEAD;
    }
    node.next = chart.head[start];
    chart.head[start] = chart.nodes.len() as u32;
    chart.nodes.push(node);
    true
}

/// Matches `items` from lexeme `pos` on, recording each complete match.
#[allow(clippy::too_many_arguments)]
fn walk(
    chart: &Chart,
    lexed: &Lexed,
    text: &str,
    items: &[Item],
    pos: u32,
    stack: &mut Vec<u32>,
    words: u32,
    done: &mut Vec<Done>,
    done_kids: &mut Vec<u32>,
) {
    let Some(item) = items.first() else {
        done.push(Done { end: pos, kids_at: done_kids.len() as u32, kids_len: stack.len() as u32, words });
        done_kids.extend_from_slice(stack);
        return;
    };
    let p = pos as usize;
    match *item {
        Item::Word { text: w, glued, optional, exact_case } => {
            if optional {
                walk(chart, lexed, text, &items[1..], pos, stack, words, done, done_kids);
            }
            let Some(l) = lexed.lexemes.get(p) else { return };
            if glued && !l.glued {
                return;
            }
            let matches = if exact_case { lexed.raw(text, p) == w } else { lexed.word(p) == w };
            if matches {
                walk(chart, lexed, text, &items[1..], pos + 1, stack, words + 1, done, done_kids);
            }
        }
        Item::Node { dim, test, glued } => {
            let Some(l) = lexed.lexemes.get(p) else { return };
            if glued && !l.glued {
                return;
            }
            for (id, n) in chart.starting_at(p) {
                if n.dim != dim || test.is_some_and(|t| !t(&n.value)) {
                    continue;
                }
                stack.push(id);
                walk(chart, lexed, text, &items[1..], n.end, stack, words, done, done_kids);
                stack.pop();
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use std::sync::OnceLock;

    use super::*;
    use crate::engine::numeral::NumeralStyle;
    use crate::engine::rule::{Entry, Lexicon, node, word};

    const A: Dim = Dim(0);
    const B: Dim = Dim(1);
    const S: Dim = Dim(2);
    const N: Dim = Dim(3);

    /// A toy grammar: `S` counts the lexemes of an `a` followed by `b`s.
    struct Toy(OnceLock<Lexicon>);

    static TOY: Toy = Toy(OnceLock::new());

    fn count(_: &[Value], ctx: &Ctx<'_>) -> Option<Value> {
        Some(Value::Term { value: (ctx.end - ctx.start) as i64, flags: 0 })
    }

    fn ranked(n: i64) -> Option<Value> {
        Some(Value::Term { value: n, flags: 0 })
    }

    static RULES: &[Rule] = &[
        Rule { name: "s_a", dim: S, items: &[node(A)], weight: 0, build: count },
        Rule { name: "s_more", dim: S, items: &[node(S), node(B)], weight: 0, build: count },
        Rule { name: "s_ab", dim: S, items: &[node(A), node(B)], weight: 1, build: count },
        // Six readings of `c` with different values, for the cell cap.
        Rule { name: "c1", dim: S, items: &[word("c")], weight: 1, build: |_, _| ranked(1) },
        Rule { name: "c2", dim: S, items: &[word("c")], weight: 2, build: |_, _| ranked(2) },
        Rule { name: "c3", dim: S, items: &[word("c")], weight: 3, build: |_, _| ranked(3) },
        Rule { name: "c4", dim: S, items: &[word("c")], weight: 4, build: |_, _| ranked(4) },
        Rule { name: "c5", dim: S, items: &[word("c")], weight: 5, build: |_, _| ranked(5) },
        Rule { name: "c0", dim: S, items: &[word("c")], weight: 0, build: |_, _| ranked(0) },
        Rule { name: "n", dim: S, items: &[node(N)], weight: 0, build: count },
    ];

    impl Grammar for Toy {
        fn name(&self) -> &'static str {
            "toy"
        }
        fn lexicon(&self) -> &Lexicon {
            self.0.get_or_init(|| {
                Lexicon::new(&[
                    Entry { word: "a", dim: A, value: 0, flags: 0 },
                    Entry { word: "b", dim: B, value: 0, flags: 0 },
                ])
            })
        }
        fn rules(&self) -> &'static [Rule] {
            RULES
        }
        fn dims(&self) -> u16 {
            4
        }
        fn numeral(&self) -> Dim {
            N
        }
        fn style(&self) -> NumeralStyle {
            NumeralStyle { mixed_separators: false, space_groups: false, indian_groups: false, decimal_comma: false }
        }
        fn output(&self, dim: Dim) -> Option<Output> {
            (dim == S).then_some(Output::Number)
        }
    }

    fn live(p: &Parser, start: u32, end: u32) -> Vec<&Node> {
        p.chart().nodes.iter().filter(|n| n.dim == S && n.start == start && n.end == end).collect()
    }

    #[test]
    fn equal_readings_keep_the_better_one() {
        let mut p = Parser::new(&TOY);
        p.parse("a b b");
        let two = live(&p, 0, 2);
        assert_eq!(two.len(), 1, "{two:?}");
        assert_eq!(RULES[usize::from(two[0].rule)].name, "s_ab");
        assert_eq!(two[0].score, 3);
        // The longer node is built on the better one.
        let three = live(&p, 0, 3);
        assert_eq!(three.len(), 1);
        assert_eq!(p.chart().children(p.chart().nodes.iter().position(|n| n == three[0]).unwrap() as u32).len(), 2);
    }

    #[test]
    fn a_cell_keeps_the_best_four() {
        let mut p = Parser::new(&TOY);
        p.parse("c");
        let mut kept: Vec<i64> = live(&p, 0, 1).iter().map(|n| n.value.term().unwrap().0).collect();
        kept.sort();
        assert_eq!(kept, [2, 3, 4, 5]);
        let mut out = Vec::new();
        p.select("c", Output::Number, &mut out);
        assert_eq!(out.len(), 1);
        assert_eq!(out[0].value.term(), Some((5, 0)));
    }

    #[test]
    fn selection_takes_the_longest_and_does_not_overlap() {
        let mut p = Parser::new(&TOY);
        let text = "a b a";
        p.parse(text);
        let mut out = Vec::new();
        p.select(text, Output::Number, &mut out);
        let spans: Vec<(u32, u32)> = out.iter().map(|m| (m.start, m.end)).collect();
        assert_eq!(spans, [(0, 3), (4, 5)]);
    }

    #[test]
    fn selection_does_not_cut_a_written_token() {
        let mut p = Parser::new(&TOY);
        let text = "a1 b";
        p.parse(text);
        let mut out = Vec::new();
        p.select(text, Output::Number, &mut out);
        assert!(out.is_empty(), "{out:?}");
        let text = "1 a";
        p.parse(text);
        out.clear();
        p.select(text, Output::Number, &mut out);
        assert_eq!(out.len(), 2);
    }

    #[test]
    fn starting_at_lists_every_node_of_a_position() {
        let mut p = Parser::new(&TOY);
        p.parse("a b");
        let at0 = p.chart().starting_at(0).count();
        let at1 = p.chart().starting_at(1).count();
        assert!(at0 >= 3, "{at0}");
        assert_eq!(at1, 1);
        assert_eq!(p.chart().starting_at(9).count(), 0);
        assert_eq!(p.lexed().len(), 2);
        assert_eq!(p.grammar().name(), "toy");
    }

    #[test]
    #[should_panic(expected = "optional word")]
    fn a_rule_may_not_open_with_an_optional_word() {
        struct Bad;
        static BAD_RULES: &[Rule] =
            &[Rule { name: "bad", dim: S, items: &[crate::engine::rule::opt_word("x")], weight: 0, build: count }];
        impl Grammar for Bad {
            fn name(&self) -> &'static str {
                "bad"
            }
            fn lexicon(&self) -> &Lexicon {
                TOY.lexicon()
            }
            fn rules(&self) -> &'static [Rule] {
                BAD_RULES
            }
            fn dims(&self) -> u16 {
                4
            }
            fn numeral(&self) -> Dim {
                N
            }
            fn style(&self) -> NumeralStyle {
                TOY.style()
            }
            fn output(&self, _: Dim) -> Option<Output> {
                None
            }
        }
        static BAD: Bad = Bad;
        let _ = Parser::new(&BAD);
    }
}
