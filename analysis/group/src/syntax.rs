//! Syntactic sketches: the treelets of an item's dependency tree, with
//! entities, numbers and function words replaced by their kind, as the
//! shingles of a `Sketches<L>`. Two clauses that share their
//! predicate-argument structure sketch alike whatever names, amounts and
//! dates they carry, so `NearDuplicates<L>` and `cross_duplicates` link
//! structural templates as they link copied text, and the search engine
//! indexes the same band keys.

#[path = "syntax/nodes.rs"]
mod nodes;

use std::marker::PhantomData;

use turbo_analysis::{Annotator, Error, Layer, Refs, Span, Tag, View};
use turbo_analysis_layers::{Dependencies, Entities, MorphPosTags, Tokens, Word, WordOrTokenAnchor};

use crate::hash::{self, LANES, SimHash};
use crate::sketch::{Sketches, item_spans};
use nodes::{Graph, mix};

const ARC: u64 = 0x6172_6300_0000_0001;
const CHAIN: u64 = 0x6368_6169_6e00_0001;
const FRAME: u64 = 0x6672_616d_6500_0001;
const OUTSIDE: u64 = 0x6f75_7473_6964_6501;
const SCHEMA: u64 = 0x7363_6865_6d61_0001;

/// Reads a text view, the items of `L`, the tokens, the words (`A`:
/// `Tokens` or `Words`), their POS tags and dependencies, and, when the
/// document has them, their lemmas and its entities; writes the items'
/// `Sketches<L>` from dependency treelets.
///
/// Every word is a node anchored by its kind alone: an entity's words
/// are one node named by the entity's kind, a number is a number, a
/// content word (verb, noun, proper noun, adjective) is its lemma and
/// POS tag, and a function word is its POS tag. The item's shingles are
/// its one-hop arcs (head, relation, dependent, direction), at depth 2
/// its two-hop chains, and the core-argument frame of every head with a
/// core argument (`nsubj`, `obj`, `iobj`, `ccomp`, `xcomp`, `obl`). An
/// arc that leaves the item keeps the outside head's POS tag only.
/// Every hash is seeded with the POS, relation and entity tag sets, so
/// sketches made under different schemas never collide.
///
/// The columns mean what `Sketcher` writes: `words` the word count,
/// `exact` the sequence of lowercased forms with their tags, `shape` the
/// sequence of delexicalized nodes, `simhash` the SimHash of the
/// treelets with every occurrence counted, `minhash` the MinHash of
/// their set. `shingle` holds the treelet depth.
pub struct SyntaxSketcher<L, A: WordOrTokenAnchor = Tokens> {
    /// Treelet depth: 1 for arcs and frames, 2 to add two-hop chains.
    pub depth: u8,
    _of: PhantomData<fn() -> (L, A)>,
}

impl<L, A: WordOrTokenAnchor> SyntaxSketcher<L, A> {
    pub fn new(depth: u8) -> SyntaxSketcher<L, A> {
        SyntaxSketcher { depth, _of: PhantomData }
    }
}

impl<L, A: WordOrTokenAnchor> Default for SyntaxSketcher<L, A> {
    fn default() -> SyntaxSketcher<L, A> {
        SyntaxSketcher::new(2)
    }
}

impl<L, A: WordOrTokenAnchor> Clone for SyntaxSketcher<L, A> {
    fn clone(&self) -> Self {
        SyntaxSketcher::new(self.depth)
    }
}

/// Reused buffers for sketching many items.
#[derive(Default)]
struct Scratch {
    shingles: Vec<u64>,
    exact: Vec<u64>,
    shape: Vec<u64>,
    frame: Vec<(u16, u64)>,
    /// Per word, the item it was last stamped into.
    member: Vec<u32>,
}

impl<L: Layer, A: WordOrTokenAnchor> Annotator for SyntaxSketcher<L, A> {
    const ID: &'static str = "turbo:syntax-sketcher";
    const VERSION: &'static str = "1";
    type Reads = (View, L, Tokens, A, A::Pos, Dependencies<A>, Option<A::Lemmas>, Option<Entities>);
    type Writes = (Sketches<L>,);

    fn annotate(&self, input: Refs<'_, Self>) -> Result<(Sketches<L>,), Error> {
        let (view, units, tokens, anchor, pos, deps, lemmas, entities) = input;
        let fail = |detail: String| Error::Annotator { annotator: Self::ID, detail };
        if !(1..=2).contains(&self.depth) {
            return Err(fail(format!("a treelet depth of {}, not 1 or 2", self.depth)));
        }
        let ts = units.text_spans().ok_or_else(|| fail(format!("{} has no spans to sketch", L::NAME)))?;
        let in_view = ts.view == view.id();
        if !in_view && view.parent() != Some(ts.view) {
            return Err(fail(format!(
                "the items are in view {}, neither the read view {} nor its parent",
                ts.view.0,
                view.id().0
            )));
        }
        let words = anchor.resolved(view, tokens).map_err(fail)?;
        let seed = mix(
            mix(mix(SCHEMA, pos.tagset().content_hash()), deps.relations.content_hash()),
            entities.map_or(0, |e| e.kinds.content_hash()),
        );
        let graph = Graph::build::<A>(seed, &words, pos, deps, lemmas, entities).map_err(fail)?;
        let mut out = Sketches::empty(self.depth);
        out.minhash.reserve(ts.start.len() * LANES);
        let mut scratch = Scratch { member: vec![u32::MAX; words.len()], ..Scratch::default() };
        for i in 0..ts.start.len() {
            let (mut first, mut last) = (usize::MAX, 0usize);
            for s in item_spans(&ts, i) {
                let s = if in_view { s } else { view.alignment().span_to_child(s) };
                let (a, b) = words_in(&words, s);
                for (w, word) in words.iter().enumerate().take(b).skip(a) {
                    if word.token_span.end <= s.end {
                        scratch.member[w] = i as u32;
                        first = first.min(w);
                        last = last.max(w);
                    }
                }
            }
            let range = if first == usize::MAX { 0..0 } else { first..last + 1 };
            self.sketch_item(&graph, &words, i as u32, range, &mut scratch, &mut out);
        }
        Ok((out,))
    }

    fn config_hash(&self) -> u64 {
        u64::from(self.depth)
    }
}

/// The words whose spans start inside `s`, as a range: the words are
/// in text order.
fn words_in(words: &[Word<'_>], s: Span) -> (usize, usize) {
    let a = words.partition_point(|w| w.token_span.start < s.start);
    let b = a + words[a..].partition_point(|w| w.token_span.start < s.end);
    (a, b)
}

impl<L: Layer, A: WordOrTokenAnchor> SyntaxSketcher<L, A> {
    /// Appends item `item`'s sketch: its words are those of `range`
    /// stamped with `item` in `scratch.member`.
    fn sketch_item(
        &self,
        g: &Graph,
        words: &[Word<'_>],
        item: u32,
        range: std::ops::Range<usize>,
        scratch: &mut Scratch,
        out: &mut Sketches<L>,
    ) {
        let inside = |w: usize| scratch.member[w] == item;
        scratch.shingles.clear();
        scratch.exact.clear();
        scratch.shape.clear();
        let mut sim = SimHash::default();
        let mut count = 0u32;
        for w in range.clone() {
            if !inside(w) {
                continue;
            }
            count += 1;
            scratch.exact.push(g.exact[w]);
            if !g.is_node(w) {
                continue;
            }
            scratch.shape.push(g.shape[w]);
            let mut emit = |t: u64| {
                let t = hash::splitmix(t);
                scratch.shingles.push(t);
                sim.add(t);
            };
            if let Some(h) = g.head[w] {
                let h = h as usize;
                let head_anchor = if inside(h) { g.shape[h] } else { mix(OUTSIDE, u64::from(g.pos[h].0)) };
                let dir = u64::from(h > w);
                emit(mix(mix(mix(mix(ARC, head_anchor), rel(g, w)), g.shape[w]), dir));
                if self.depth >= 2
                    && inside(h)
                    && let Some(gp) = g.head[h]
                    && inside(gp as usize)
                {
                    let gp = gp as usize;
                    let t = mix(mix(mix(mix(mix(CHAIN, g.shape[gp]), rel(g, h)), g.shape[h]), rel(g, w)), g.shape[w]);
                    emit(t);
                }
            }
            scratch.frame.clear();
            for &c in g.children(w) {
                let c = c as usize;
                if inside(c) && g.core[g.rel[c].0 as usize] {
                    scratch.frame.push((g.rel[c].0, g.shape[c]));
                }
            }
            if !scratch.frame.is_empty() {
                scratch.frame.sort_unstable();
                let mut t = mix(FRAME, g.shape[w]);
                for &(r, s) in &scratch.frame {
                    t = mix(mix(t, u64::from(r)), s);
                }
                emit(t);
            }
        }
        debug_assert!(count as usize <= words.len());
        out.words.push(count);
        out.exact.push(hash::sequence(&scratch.exact));
        out.shape.push(hash::sequence64(&scratch.shape));
        out.simhash.push(sim.finish());
        out.minhash.extend_from_slice(&hash::minhash(&scratch.shingles));
    }
}

/// The relation of the arc from node `w` to its head, as a hash input.
#[inline]
fn rel(g: &Graph, w: usize) -> u64 {
    let Tag(r) = g.rel[w];
    u64::from(r)
}
