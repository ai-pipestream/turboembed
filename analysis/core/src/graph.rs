//! Graph layers over a document: directed, labeled, scored edges between any
//! two layers (`Edges<From, To>`) and sets of items of one layer (`Groups<L>`).

use std::fmt;
use std::sync::Arc;

use crate::layer::{Anchor, Key, KeyId, Layer, LayerName, Tag, TagSet};
use crate::span::{Ix, IxRange};

/// One directed, labeled, scored edge from an item of `From` to an item of
/// `To`.
pub struct Edge<From, To> {
    pub from: Ix<From>,
    pub to: Ix<To>,
    pub kind: Tag,
    pub score: f64,
}

impl<From, To> Edge<From, To> {
    pub const fn new(from: Ix<From>, to: Ix<To>, kind: Tag, score: f64) -> Edge<From, To> {
        Edge { from, to, kind, score }
    }
}

impl<From, To> Clone for Edge<From, To> {
    fn clone(&self) -> Self {
        *self
    }
}
impl<From, To> Copy for Edge<From, To> {}
impl<From, To> PartialEq for Edge<From, To> {
    fn eq(&self, other: &Self) -> bool {
        self.from == other.from && self.to == other.to && self.kind == other.kind && self.score == other.score
    }
}
impl<From, To> fmt::Debug for Edge<From, To> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Edge")
            .field("from", &self.from)
            .field("to", &self.to)
            .field("kind", &self.kind)
            .field("score", &self.score)
            .finish()
    }
}

/// Directed, labeled, scored edges from items of `From` to items of `To`,
/// stored by column in `(from, to, kind)` order with a secondary index
/// (`by_to`) ordered by `(to, row)` so lookups in both directions are binary
/// searches. A `(from, to, kind)` triple appears at most once: a layer that
/// repeats one is refused when it is added to a document, since the two
/// rows would be one relation with two scores.
///
/// `From` and `To` may be the same layer type (for example coreference links
/// between mentions or discourse relations between clauses) or two distinct
/// layer types (for example entity-to-token heads, quote-to-speaker links, or
/// abbreviation-to-definition pairs). Each endpoint is typed by its layer's
/// `Ix`, so an index from a third layer cannot be passed to an edge row:
///
/// ```compile_fail
/// use turbo_analysis::{Anchor, Edge, Edges, Ix, Layer, Tag, TagSet};
/// struct Sentences;
/// struct Tokens;
/// struct Entities;
/// impl Layer for Sentences {
///     const ID: &'static str = "opennlp:sentences";
///     fn anchor() -> Anchor { Anchor::Document }
///     fn len(&self) -> usize { 0 }
/// }
/// impl Layer for Tokens {
///     const ID: &'static str = "opennlp:tokens";
///     fn anchor() -> Anchor { Anchor::Document }
///     fn len(&self) -> usize { 0 }
/// }
/// impl Layer for Entities {
///     const ID: &'static str = "opennlp:entities";
///     fn anchor() -> Anchor { Anchor::Document }
///     fn len(&self) -> usize { 0 }
/// }
/// fn wrong_endpoint(s: Ix<Sentences>, t: Ix<Tokens>) -> Edges<Entities, Tokens> {
///     Edges::new(TagSet::new("rel", &["head"]), [Edge::new(s, t, Tag(0), 1.0)])
/// }
/// ```
pub struct Edges<From, To> {
    pub from_key: Key<From>,
    pub to_key: Key<To>,
    pub from: Vec<Ix<From>>,
    pub to: Vec<Ix<To>>,
    pub kind: Vec<Tag>,
    pub kinds: Arc<TagSet>,
    pub score: Vec<f64>,
    /// Row indices `0..len` ordered by `(to[r], r)`, so the incoming edges of
    /// any target item are a contiguous slice found by binary search.
    pub by_to: Vec<Ix<Edges<From, To>>>,
}

impl<From, To> Clone for Edges<From, To> {
    fn clone(&self) -> Self {
        Edges {
            from_key: self.from_key,
            to_key: self.to_key,
            from: self.from.clone(),
            to: self.to.clone(),
            kind: self.kind.clone(),
            kinds: self.kinds.clone(),
            score: self.score.clone(),
            by_to: self.by_to.clone(),
        }
    }
}

impl<From, To> fmt::Debug for Edges<From, To> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Edges")
            .field("from_key", &self.from_key)
            .field("to_key", &self.to_key)
            .field("from", &self.from)
            .field("to", &self.to)
            .field("kind", &self.kind)
            .field("kinds", &self.kinds)
            .field("score", &self.score)
            .field("by_to", &self.by_to)
            .finish()
    }
}

impl<From: Layer, To: Layer> Edges<From, To> {
    /// Builds an edge layer between the default instances of `From` and `To`
    /// (or the instances the producing step wrote or read), sorting the rows
    /// by `(from, to, kind)` and building the target index. When `From` and
    /// `To` are one layer type, both ends resolve to the same instance; use
    /// `between` to link two instances of it.
    pub fn new(kinds: Arc<TagSet>, rows: impl IntoIterator<Item = Edge<From, To>>) -> Edges<From, To> {
        Self::between(Key::DEFAULT, Key::DEFAULT, kinds, rows)
    }

    /// Builds an edge layer between the named instances `from_key` and
    /// `to_key`, sorting the rows by `(from, to, kind)` and building the
    /// target index. A named instance must be one the producing step read or
    /// wrote (its provenance's `reads` or `on`); `Document::add` refuses an
    /// edge layer into any other.
    pub fn between(
        from_key: Key<From>,
        to_key: Key<To>,
        kinds: Arc<TagSet>,
        rows: impl IntoIterator<Item = Edge<From, To>>,
    ) -> Edges<From, To> {
        let mut sorted: Vec<Edge<From, To>> = rows.into_iter().collect();
        sorted.sort_by_key(|a| (a.from, a.to, a.kind));
        let n = sorted.len();
        let mut from = Vec::with_capacity(n);
        let mut to = Vec::with_capacity(n);
        let mut kind = Vec::with_capacity(n);
        let mut score = Vec::with_capacity(n);
        for e in sorted {
            from.push(e.from);
            to.push(e.to);
            kind.push(e.kind);
            score.push(e.score);
        }
        let mut by_to: Vec<Ix<Edges<From, To>>> = (0..n as u32).map(Ix::new).collect();
        by_to.sort_unstable_by_key(|&r| (to[r.get()], r));
        Edges { from_key, to_key, from, to, kind, kinds, score, by_to }
    }

    /// Row `i` as an `Edge<From, To>`.
    pub fn get(&self, i: Ix<Self>) -> Edge<From, To> {
        let r = i.get();
        Edge { from: self.from[r], to: self.to[r], kind: self.kind[r], score: self.score[r] }
    }

    /// All edges in `(from, to, kind)` order, with their row index.
    pub fn iter(&self) -> impl ExactSizeIterator<Item = (Ix<Self>, Edge<From, To>)> + '_ {
        (0..self.from.len() as u32).map(|i| {
            let ix = Ix::new(i);
            (ix, self.get(ix))
        })
    }

    /// The contiguous range of edge rows whose source is `s`: two binary
    /// searches over `from`, no scan.
    pub fn from_source(&self, s: Ix<From>) -> IxRange<Self> {
        let a = self.from.partition_point(|&x| x < s);
        let b = a + self.from[a..].partition_point(|&x| x == s);
        IxRange::new(a as u32, b as u32)
    }

    /// The outgoing edges from source `s`, in `(to, kind)` order.
    pub fn outgoing(&self, s: Ix<From>) -> impl ExactSizeIterator<Item = (Ix<Self>, Edge<From, To>)> + '_ {
        self.from_source(s).iter().map(|i| (i, self.get(i)))
    }

    /// The row indices of edges whose target is `t`, in `(from, kind)` order:
    /// two binary searches over `by_to`, no scan.
    pub fn to_target(&self, t: Ix<To>) -> &[Ix<Self>] {
        let a = self.by_to.partition_point(|&r| self.to[r.get()] < t);
        let b = a + self.by_to[a..].partition_point(|&r| self.to[r.get()] == t);
        &self.by_to[a..b]
    }

    /// The incoming edges to target `t`, in `(from, kind)` order.
    pub fn incoming(&self, t: Ix<To>) -> impl ExactSizeIterator<Item = (Ix<Self>, Edge<From, To>)> + '_ {
        self.to_target(t).iter().copied().map(|i| (i, self.get(i)))
    }
}

/// Reusable buffers for [`Edges::reachable_from`]: a breadth-first
/// queue and a generation-stamped visited column, so repeated
/// traversals allocate nothing once the buffers have grown to the
/// largest frontier and node index they have seen.
#[derive(Clone, Debug, Default)]
pub struct ReachScratch {
    queue: Vec<u32>,
    seen: Vec<u32>,
    generation: u32,
}

impl ReachScratch {
    pub fn new() -> ReachScratch {
        ReachScratch::default()
    }

    /// Marks `node` visited in this traversal; returns whether it
    /// was already marked.
    fn mark(&mut self, node: u32) -> bool {
        let i = node as usize;
        if i >= self.seen.len() {
            self.seen.resize(i + 1, 0);
        }
        let was = self.seen[i] == self.generation;
        self.seen[i] = self.generation;
        was
    }
}

impl<L: Layer> Edges<L, L> {
    /// The items reachable from `root` following outgoing edges,
    /// breadth-first, `root` first, each item once, at most
    /// `max_hops` edges out (`max_hops == 0` yields only `root`).
    /// Cycles terminate by the visited marks in `scratch`, which is
    /// reused across calls; `out` is cleared and refilled, so after
    /// the first call neither buffer allocates for a graph no larger
    /// than one already traversed. The edges are assumed checked
    /// (as a document holds them): targets are followed as stored.
    pub fn reachable_from(&self, root: Ix<L>, max_hops: u32, scratch: &mut ReachScratch, out: &mut Vec<Ix<L>>) {
        out.clear();
        scratch.queue.clear();
        scratch.generation = scratch.generation.wrapping_add(1);
        if scratch.generation == 0 {
            scratch.seen.iter_mut().for_each(|s| *s = 0);
            scratch.generation = 1;
        }
        scratch.mark(root.get() as u32);
        scratch.queue.push(root.get() as u32);
        out.push(root);
        let mut head = 0usize;
        let mut hops = 0u32;
        while head < scratch.queue.len() && hops < max_hops {
            let level_end = scratch.queue.len();
            while head < level_end {
                let node = Ix::new(scratch.queue[head]);
                head += 1;
                for row in self.from_source(node).iter() {
                    let to = self.to[row.get()];
                    if !scratch.mark(to.get() as u32) {
                        scratch.queue.push(to.get() as u32);
                        out.push(to);
                    }
                }
            }
            hops += 1;
        }
    }
}

impl<From: Layer, To: Layer> Layer for Edges<From, To> {
    const ID: &'static str = "turbo:edges";
    const OF: Option<&'static LayerName> = Some(From::NAME);
    const TO: Option<&'static LayerName> = Some(To::NAME);

    fn anchor() -> Anchor {
        Anchor::Document
    }

    fn len(&self) -> usize {
        self.from.len()
    }

    fn indexes(&self) -> Vec<(KeyId, usize)> {
        let from_bound = self.from.last().map_or(0, |i| i.get() + 1);
        let to_bound = self.by_to.last().and_then(|r| self.to.get(r.get())).map_or(0, |i| i.get() + 1);
        vec![(self.from_key.id(), from_bound), (self.to_key.id(), to_bound)]
    }

    fn check(&self) -> Result<(), String> {
        let n = self.from.len();
        if self.to.len() != n || self.kind.len() != n || self.score.len() != n || self.by_to.len() != n {
            return Err("a target, a kind, a score and a target index for every edge".into());
        }
        if self.kind.iter().any(|k| k.0 as usize >= self.kinds.len()) {
            return Err(format!("a kind outside {}", self.kinds.name));
        }
        if !(1..n)
            .all(|i| (self.from[i - 1], self.to[i - 1], self.kind[i - 1]) < (self.from[i], self.to[i], self.kind[i]))
        {
            return Err("edges in (from, to, kind) order, each once".into());
        }
        if self.by_to.iter().any(|r| r.get() >= n) {
            return Err("a target index outside the edges".into());
        }
        if !self.by_to.windows(2).all(|w| (self.to[w[0].get()], w[0]) < (self.to[w[1].get()], w[1])) {
            return Err("target index in (to, row) order".into());
        }
        Ok(())
    }
}

/// Clusters or sets of items from layer `L` (coreference chains, duplicate
/// groups, multi-argument sets), stored flat in one `members` column and one
/// `end` offset column. Each group's members are kept in strictly increasing
/// `Ix<L>` order, and every group has at least one member: a layer with an
/// empty group is refused when it is added to a document.
pub struct Groups<L> {
    pub of_key: Key<L>,
    pub members: Vec<Ix<L>>,
    pub end: Vec<u32>,
}

impl<L> Clone for Groups<L> {
    fn clone(&self) -> Self {
        Groups { of_key: self.of_key, members: self.members.clone(), end: self.end.clone() }
    }
}

impl<L> fmt::Debug for Groups<L> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Groups")
            .field("of_key", &self.of_key)
            .field("members", &self.members)
            .field("end", &self.end)
            .finish()
    }
}

impl<L: Layer> Default for Groups<L> {
    fn default() -> Self {
        Groups { of_key: Key::DEFAULT, members: Vec::new(), end: Vec::new() }
    }
}

impl<L: Layer> Groups<L> {
    /// Builds groups over the default instance of `L` (or the instance the
    /// producing step wrote or read), sorting and deduplicating each group's
    /// members.
    pub fn new(groups: impl IntoIterator<Item = impl IntoIterator<Item = Ix<L>>>) -> Groups<L> {
        Self::of(Key::DEFAULT, groups)
    }

    /// Builds groups over the named instance `of_key`, sorting and
    /// deduplicating each group's members. As with `Edges::between`, the
    /// producing step must have read or written that instance.
    pub fn of(of_key: Key<L>, groups: impl IntoIterator<Item = impl IntoIterator<Item = Ix<L>>>) -> Groups<L> {
        let mut out = Groups { of_key, members: Vec::new(), end: Vec::new() };
        for g in groups {
            out.push(g);
        }
        out
    }

    /// Appends one group, sorting and deduplicating its members into strictly
    /// increasing `Ix<L>` order. An empty group is kept here and refused by
    /// `Layer::check`.
    pub fn push(&mut self, group: impl IntoIterator<Item = Ix<L>>) -> Ix<Self> {
        let start = self.members.len();
        self.members.extend(group);
        self.members[start..].sort_unstable();
        let mut write = start;
        for read in start..self.members.len() {
            if write == start || self.members[write - 1] != self.members[read] {
                self.members[write] = self.members[read];
                write += 1;
            }
        }
        self.members.truncate(write);
        let i = self.end.len() as u32;
        self.end.push(self.members.len() as u32);
        Ix::new(i)
    }

    /// The members of group `i`, in strictly increasing `Ix<L>` order.
    pub fn get(&self, i: Ix<Self>) -> &[Ix<L>] {
        let g = i.get();
        let start = if g == 0 { 0 } else { self.end[g - 1] as usize };
        &self.members[start..self.end[g] as usize]
    }

    /// Whether group `i` contains `item`: a binary search over the group's
    /// sorted slice.
    pub fn contains(&self, i: Ix<Self>, item: Ix<L>) -> bool {
        self.get(i).binary_search(&item).is_ok()
    }

    /// Every group containing `item`, with `item`'s rank in the
    /// group: one binary search per group over its sorted slice, no
    /// allocation and no auxiliary index, so the cost is $O(G \log
    /// n)$ in the group count for the occasional reverse lookup
    /// (the forward direction stays [`Groups::get`]).
    pub fn groups_containing(&self, item: Ix<L>) -> impl Iterator<Item = (Ix<Self>, usize)> + '_ {
        self.iter().filter_map(move |(ix, members)| members.binary_search(&item).ok().map(|rank| (ix, rank)))
    }

    /// Every group with its index and its sorted slice of members.
    pub fn iter(&self) -> impl ExactSizeIterator<Item = (Ix<Self>, &[Ix<L>])> + '_ {
        (0..self.end.len() as u32).map(|i| {
            let ix = Ix::new(i);
            (ix, self.get(ix))
        })
    }
}

impl<L: Layer> Layer for Groups<L> {
    const ID: &'static str = "turbo:groups";
    const OF: Option<&'static LayerName> = Some(L::NAME);

    fn anchor() -> Anchor {
        Anchor::Document
    }

    fn len(&self) -> usize {
        self.end.len()
    }

    fn indexes(&self) -> Vec<(KeyId, usize)> {
        // Each group is sorted, so its last member is its largest.
        let bound = self.iter().filter_map(|(_, g)| g.last()).map(|i| i.get() + 1).max().unwrap_or(0);
        vec![(self.of_key.id(), bound)]
    }

    fn check(&self) -> Result<(), String> {
        if self.end.is_empty() {
            if !self.members.is_empty() {
                return Err("members with no groups".into());
            }
            return Ok(());
        }
        if *self.end.last().unwrap() as usize != self.members.len() {
            return Err("group ends do not match members".into());
        }
        if self.end[0] == 0 || !self.end.windows(2).all(|w| w[0] < w[1]) {
            return Err("group ends in order, every group with a member".into());
        }
        for g in 0..self.end.len() {
            if !self.get(Ix::new(g as u32)).windows(2).all(|w| w[0] < w[1]) {
                return Err(format!("group {g} members in order, each once"));
            }
        }
        Ok(())
    }
}
