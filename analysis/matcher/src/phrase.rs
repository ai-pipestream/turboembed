//! The phrase matcher: spaCy's `PhraseMatcher`. A list of phrases, each a
//! run of token values, is compiled into one Aho-Corasick automaton over
//! token symbols, so the document is read once, left to right, whatever
//! the number of phrases, and every occurrence of every phrase is found,
//! overlapping and nested ones included.
//!
//! The automaton is a trie of the phrases with a failure link per node
//! (the longest proper suffix of the node's path that is also in the
//! trie) and an output link (the nearest such suffix that ends a phrase).
//! A node's children are stored by symbol in one sorted array per node and
//! found by binary search; the root's, where the walk returns after every
//! mismatch, are a table indexed by symbol, one load per token.

use std::collections::HashMap;
use std::sync::Arc;

use turbo_analysis::{Tag, TagSet};

use crate::attr::{Attr, Flag};
use crate::matcher::{Match, MatchError};
use crate::pattern::PatternError;
use crate::seq::{Buf, Sequence};
use crate::vocab::{Fnv, NONE, Sym, Vocab};

/// What a phrase matcher compares: a string attribute, or a flag, whose
/// phrase values are true and false.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub enum PhraseOn {
    Attr(Attr),
    Flag(Flag),
}

/// Collects phrases under labels; `build` compiles them.
pub struct PhraseMatcherBuilder {
    on: PhraseOn,
    vocab: Vocab,
    labels: Vec<String>,
    index: HashMap<String, u32>,
    removed: Vec<bool>,
    /// Every phrase's symbols, one after another.
    syms: Vec<Sym>,
    /// Per phrase: its label, `u32::MAX` once removed, and its end in
    /// `syms`.
    phrases: Vec<(u32, u32)>,
}

impl PhraseMatcherBuilder {
    pub fn new(on: PhraseOn) -> PhraseMatcherBuilder {
        PhraseMatcherBuilder {
            on,
            vocab: Vocab::default(),
            labels: Vec::new(),
            index: HashMap::new(),
            removed: Vec::new(),
            syms: Vec::new(),
            phrases: Vec::new(),
        }
    }

    fn label(&mut self, label: &str) -> Result<u32, PatternError> {
        if label.is_empty() {
            return Err(PatternError::EmptyLabel);
        }
        if let Some(&i) = self.index.get(label) {
            self.removed[i as usize] = false;
            return Ok(i);
        }
        let i = self.labels.len() as u32;
        self.labels.push(label.into());
        self.index.insert(label.into(), i);
        self.removed.push(false);
        Ok(i)
    }

    fn push(&mut self, label: &str, syms: impl IntoIterator<Item = Sym>) -> Result<(), PatternError> {
        if label.is_empty() {
            return Err(PatternError::EmptyLabel);
        }
        let before = self.syms.len();
        self.syms.extend(syms);
        if self.syms.len() == before {
            return Err(PatternError::EmptyPhrase);
        }
        let l = self.label(label)?;
        self.phrases.push((l, self.syms.len() as u32));
        Ok(())
    }

    /// Adds a phrase given as the values of the compared attribute, one
    /// per token: `["new", "york"]` for a matcher on `Attr::Lower`.
    pub fn add(&mut self, label: &str, values: &[&str]) -> Result<(), PatternError> {
        if !matches!(self.on, PhraseOn::Attr(_)) {
            return Err(PatternError::PhraseValue);
        }
        let syms: Vec<Sym> = values.iter().map(|v| self.vocab.intern(v)).collect();
        self.push(label, syms)
    }

    /// Adds a phrase of flag values, for a matcher on a flag.
    pub fn add_flags(&mut self, label: &str, values: &[bool]) -> Result<(), PatternError> {
        if !matches!(self.on, PhraseOn::Flag(_)) {
            return Err(PatternError::PhraseValue);
        }
        self.push(label, values.iter().map(|&v| Sym::from(v)))
    }

    /// Adds every token of an analyzed sequence as one phrase, by the
    /// compared attribute: a phrase list run through the same steps as
    /// the documents (spaCy's `PhraseMatcher.add` of `Doc` patterns).
    pub fn add_sequence(&mut self, label: &str, seq: &Sequence<'_>) -> Result<(), PhraseError> {
        let mut buf = Buf::default();
        let syms: Vec<Sym> = match self.on {
            PhraseOn::Attr(a) => {
                if !seq.has(a) {
                    return Err(PhraseError::Match(MatchError::Missing(a)));
                }
                (0..seq.len()).map(|i| seq.with_value(a, i, &mut buf, |v| self.vocab.intern(v))).collect()
            }
            PhraseOn::Flag(f) => {
                if !seq.has_flag(f) {
                    return Err(PhraseError::Match(MatchError::MissingFlag(f)));
                }
                seq.flags().iter().map(|&bits| Sym::from(bits & f.bit() != 0)).collect()
            }
        };
        self.push(label, syms).map_err(PhraseError::Pattern)
    }

    /// Removes a label and its phrases; false when there was none.
    pub fn remove(&mut self, label: &str) -> bool {
        match self.index.get(label) {
            Some(&i) if !self.removed[i as usize] => {
                self.removed[i as usize] = true;
                for p in &mut self.phrases {
                    if p.0 == i {
                        p.0 = u32::MAX;
                    }
                }
                true
            }
            _ => false,
        }
    }

    pub fn contains(&self, label: &str) -> bool {
        self.index.get(label).is_some_and(|&i| !self.removed[i as usize])
    }

    /// How many labels have phrases.
    pub fn len(&self) -> usize {
        self.removed.iter().filter(|&&r| !r).count()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn build(self) -> Result<PhraseMatcher, PatternError> {
        // The labels left, renumbered in the order they were added.
        let mut renumber = vec![u32::MAX; self.labels.len()];
        let mut labels = Vec::new();
        for (i, l) in self.labels.iter().enumerate() {
            if !self.removed[i] {
                renumber[i] = labels.len() as u32;
                labels.push(l.as_str());
            }
        }
        if labels.len() > usize::from(u16::MAX) + 1 {
            return Err(PatternError::Labels);
        }
        let mut phrases: Vec<(&[Sym], u16)> = Vec::new();
        let mut from = 0usize;
        for &(l, end) in &self.phrases {
            if l != u32::MAX && renumber[l as usize] != u32::MAX {
                phrases.push((&self.syms[from..end as usize], renumber[l as usize] as u16));
            }
            from = end as usize;
        }
        phrases.sort_unstable();
        phrases.dedup();
        let mut config = Fnv::default();
        config.str(&format!("{:?}", self.on));
        for l in &labels {
            config.str(l);
        }
        // The symbols are numbered in the order strings were first added,
        // so the hash is of the strings themselves.
        let mut names = vec![""; self.vocab.len()];
        for (s, &i) in self.vocab_entries() {
            names[i as usize] = s;
        }
        for (syms, l) in &phrases {
            config.u64(u64::from(*l));
            config.u64(syms.len() as u64);
            for &s in *syms {
                match self.on {
                    PhraseOn::Attr(_) => config.str(names[s as usize]),
                    PhraseOn::Flag(_) => config.u64(u64::from(s)),
                }
            }
        }
        let alphabet = match self.on {
            PhraseOn::Attr(_) => self.vocab.len(),
            PhraseOn::Flag(_) => 2,
        };
        let automaton = Automaton::build(&phrases, alphabet);
        Ok(PhraseMatcher {
            on: self.on,
            labels: TagSet::new("phrases", &labels),
            vocab: self.vocab,
            automaton,
            config: config.0,
        })
    }

    fn vocab_entries(&self) -> impl Iterator<Item = (&str, &Sym)> {
        self.vocab.entries()
    }
}

/// Why a phrase from a sequence could not be added.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum PhraseError {
    Pattern(PatternError),
    Match(MatchError),
}

impl std::fmt::Display for PhraseError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            PhraseError::Pattern(e) => e.fmt(f),
            PhraseError::Match(e) => e.fmt(f),
        }
    }
}

impl std::error::Error for PhraseError {}

/// Compiled phrases, ready to run over any number of sequences.
pub struct PhraseMatcher {
    on: PhraseOn,
    labels: Arc<TagSet>,
    vocab: Vocab,
    automaton: Automaton,
    config: u64,
}

impl PhraseMatcher {
    pub fn builder(on: PhraseOn) -> PhraseMatcherBuilder {
        PhraseMatcherBuilder::new(on)
    }

    pub fn on(&self) -> PhraseOn {
        self.on
    }

    /// The labels, in the order they were first added.
    pub fn labels(&self) -> &Arc<TagSet> {
        &self.labels
    }

    /// A hash of the phrases, for provenance.
    pub fn config_hash(&self) -> u64 {
        self.config
    }

    /// How many trie nodes the automaton has, the root included.
    pub fn nodes(&self) -> usize {
        self.automaton.nodes.len() - 1
    }

    /// Every occurrence of every phrase, overlapping ones included, in
    /// order of start, end and label. A phrase added twice under one label
    /// is reported once.
    pub fn find(&self, seq: &Sequence<'_>) -> Result<Vec<Match>, MatchError> {
        let syms = self.symbols(seq)?;
        let mut out = Vec::new();
        self.automaton.scan(&syms, |start, end, label| {
            out.push(Match { start: start as u32, end: end as u32, label: Tag(label) })
        });
        out.sort_unstable();
        Ok(out)
    }

    fn symbols(&self, seq: &Sequence<'_>) -> Result<Vec<Sym>, MatchError> {
        match self.on {
            PhraseOn::Attr(a) => {
                if !seq.has(a) {
                    return Err(MatchError::Missing(a));
                }
                let mut buf = Buf::default();
                Ok((0..seq.len()).map(|i| seq.with_value(a, i, &mut buf, |v| self.vocab.get(v))).collect())
            }
            PhraseOn::Flag(f) => {
                if !seq.has_flag(f) {
                    return Err(MatchError::MissingFlag(f));
                }
                Ok(seq.flags().iter().map(|&bits| Sym::from(bits & f.bit() != 0)).collect())
            }
        }
    }
}

const NO_NODE: u32 = u32::MAX;
const ROOT: u32 = 0;

/// One trie node, its links and the start of its children and labels in
/// the shared arrays: the fields a step reads together, in one place.
#[derive(Copy, Clone, Default)]
struct Node {
    /// Its children's start in `edges`; they end where the next node's
    /// start.
    edges: u32,
    /// Its labels' start in `labels`, likewise.
    labels: u32,
    fail: u32,
    /// The nearest node on the failure chain that ends a phrase.
    out: u32,
    depth: u32,
}

struct Automaton {
    /// The root's child per symbol.
    root: Vec<u32>,
    /// Every node, then one more whose offsets end the last node's ranges.
    nodes: Vec<Node>,
    /// Children as (symbol, node), each node's sorted by symbol.
    edges: Vec<(Sym, u32)>,
    labels: Vec<u16>,
}

impl Automaton {
    /// `phrases` sorted by symbols, then label, without repeats.
    fn build(phrases: &[(&[Sym], u16)], alphabet: usize) -> Automaton {
        // The trie, node by node in the order of the sorted phrases: each
        // phrase shares the nodes of its common prefix with the one before.
        let mut parent = vec![NO_NODE];
        let mut sym_in = vec![NONE];
        let mut depth = vec![0u32];
        let mut terms: Vec<(u32, u16)> = Vec::new();
        let mut path: Vec<u32> = vec![ROOT];
        let mut prev: &[Sym] = &[];
        for &(syms, label) in phrases {
            let shared = prev.iter().zip(syms).take_while(|(a, b)| a == b).count();
            path.truncate(shared + 1);
            for (d, &s) in syms.iter().enumerate().skip(shared) {
                let node = parent.len() as u32;
                parent.push(*path.last().expect("the root"));
                sym_in.push(s);
                depth.push(d as u32 + 1);
                path.push(node);
            }
            terms.push((*path.last().expect("a phrase node"), label));
            prev = syms;
        }
        let n = parent.len();
        let mut nodes = vec![Node { fail: ROOT, out: NO_NODE, ..Node::default() }; n + 1];
        for (node, &d) in nodes.iter_mut().zip(&depth) {
            node.depth = d;
        }

        // Children by parent. Siblings were made in symbol order, so a
        // stable placement by parent keeps them sorted.
        for &p in &parent[1..] {
            nodes[p as usize + 1].edges += 1;
        }
        for i in 0..n {
            nodes[i + 1].edges += nodes[i].edges;
        }
        let mut fill: Vec<u32> = nodes.iter().map(|x| x.edges).collect();
        let mut edges = vec![(0, 0); n - 1];
        for v in 1..n {
            let p = parent[v] as usize;
            edges[fill[p] as usize] = (sym_in[v], v as u32);
            fill[p] += 1;
        }
        let mut root = vec![NO_NODE; alphabet];
        for &(s, to) in &edges[nodes[0].edges as usize..nodes[1].edges as usize] {
            root[s as usize] = to;
        }

        terms.sort_unstable();
        for &(node, _) in &terms {
            nodes[node as usize + 1].labels += 1;
        }
        for i in 0..n {
            nodes[i + 1].labels += nodes[i].labels;
        }
        let labels = terms.iter().map(|&(_, l)| l).collect();

        let mut a = Automaton { root, nodes, edges, labels };

        // Failure and output links, shallowest nodes first, so a node's
        // parent and every shorter suffix have theirs already.
        let mut by_depth: Vec<u32> = (1..n as u32).collect();
        by_depth.sort_by_key(|&v| depth[v as usize]);
        for &v in &by_depth {
            let v = v as usize;
            let (p, s) = (parent[v], sym_in[v]);
            let f = if p == ROOT {
                ROOT
            } else {
                let mut f = a.nodes[p as usize].fail;
                loop {
                    if let Some(w) = a.goto(f, s) {
                        break w;
                    }
                    if f == ROOT {
                        break ROOT;
                    }
                    f = a.nodes[f as usize].fail;
                }
            };
            a.nodes[v].fail = f;
            a.nodes[v].out = if a.ends(f) { f } else { a.nodes[f as usize].out };
        }
        a
    }

    fn ends(&self, node: u32) -> bool {
        self.nodes[node as usize].labels != self.nodes[node as usize + 1].labels
    }

    #[inline]
    fn goto(&self, node: u32, s: Sym) -> Option<u32> {
        if node == ROOT {
            let to = *self.root.get(s as usize)?;
            return (to != NO_NODE).then_some(to);
        }
        let (a, b) = (self.nodes[node as usize].edges as usize, self.nodes[node as usize + 1].edges as usize);
        let edges = &self.edges[a..b];
        let i = if edges.len() <= 8 {
            edges.iter().position(|e| e.0 == s)?
        } else {
            edges.binary_search_by_key(&s, |e| e.0).ok()?
        };
        Some(edges[i].1)
    }

    /// Reads the symbols once, calling `emit(start, end, label)` for every
    /// phrase occurrence: by end, and at one end longest first.
    fn scan(&self, syms: &[Sym], mut emit: impl FnMut(usize, usize, u16)) {
        let mut state = ROOT;
        for (j, &s) in syms.iter().enumerate() {
            if s == NONE {
                state = ROOT;
                continue;
            }
            loop {
                if let Some(next) = self.goto(state, s) {
                    state = next;
                    break;
                }
                if state == ROOT {
                    break;
                }
                state = self.nodes[state as usize].fail;
            }
            let mut t = if self.ends(state) { state } else { self.nodes[state as usize].out };
            while t != NO_NODE {
                let node = self.nodes[t as usize];
                let start = j + 1 - node.depth as usize;
                for &l in &self.labels[node.labels as usize..self.nodes[t as usize + 1].labels as usize] {
                    emit(start, j + 1, l);
                }
                t = node.out;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn scan(phrases: &[(&[Sym], u16)], text: &[Sym]) -> Vec<(usize, usize, u16)> {
        let mut p = phrases.to_vec();
        p.sort_unstable();
        p.dedup();
        let a = Automaton::build(&p, 16);
        let mut out = Vec::new();
        a.scan(text, |s, e, l| out.push((s, e, l)));
        out.sort_unstable();
        out
    }

    /// Every occurrence by brute force, to compare the automaton with.
    fn brute(phrases: &[(&[Sym], u16)], text: &[Sym]) -> Vec<(usize, usize, u16)> {
        let mut out = Vec::new();
        for &(p, l) in phrases {
            for s in 0..text.len() {
                if text[s..].starts_with(p) {
                    out.push((s, s + p.len(), l));
                }
            }
        }
        out.sort_unstable();
        out.dedup();
        out
    }

    #[test]
    fn overlapping_and_nested_phrases() {
        // he, she, his, hers over "ushers", the textbook example.
        let (u, s, h, e, r, i) = (0, 1, 2, 3, 4, 5);
        let phrases: &[(&[Sym], u16)] = &[(&[h, e], 0), (&[s, h, e], 1), (&[h, i, s], 2), (&[h, e, r, s], 3)];
        let text = [u, s, h, e, r, s];
        assert_eq!(scan(phrases, &text), vec![(1, 4, 1), (2, 4, 0), (2, 6, 3)]);
    }

    #[test]
    fn matches_brute_force_on_every_small_case() {
        // All phrase sets over a two-symbol alphabet of length up to 3,
        // taken four at a time from a fixed list, against every text of
        // length up to 7.
        let all: Vec<Vec<Sym>> = (1..=3)
            .flat_map(|len| (0..1u32 << len).map(move |bits| (0..len).map(|k| (bits >> k) & 1).collect()))
            .collect();
        for start in 0..all.len() {
            let set: Vec<(&[Sym], u16)> =
                (0..4).map(|k| (all[(start + k * 3) % all.len()].as_slice(), k as u16 % 3)).collect();
            for len in 0..=7 {
                for bits in 0..1u32 << len {
                    let text: Vec<Sym> = (0..len).map(|k| (bits >> k) & 1).collect();
                    let mut sorted = set.clone();
                    sorted.sort_unstable();
                    sorted.dedup();
                    assert_eq!(scan(&sorted, &text), brute(&sorted, &text), "{set:?} {text:?}");
                }
            }
        }
    }

    #[test]
    fn unknown_symbols_reset_the_walk() {
        let phrases: &[(&[Sym], u16)] = &[(&[1, 2], 0)];
        assert_eq!(scan(phrases, &[1, NONE, 2, 1, 2]), vec![(3, 5, 0)]);
    }
}
