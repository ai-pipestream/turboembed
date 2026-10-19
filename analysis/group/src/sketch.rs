//! Sketches: a fixed-size summary of each item's words, aligned to the
//! items, that near-duplicate grouping and boilerplate detection compare
//! instead of the text.

use std::marker::PhantomData;

use turbo_analysis::{Anchor, Annotator, Error, Ix, Layer, LayerName, Refs, Span, TextSpans, View};

use crate::hash::{self, LANES, SimHash};

/// The sketch of each item of `L`. An item's words are the maximal runs
/// of alphanumeric characters in its text, fragment by fragment for a
/// discontinuous item, so no word joins two fragments; whitespace and
/// punctuation are no part of any word. The columns, item by item:
///
/// - `words`: how many words it holds.
/// - `exact`: a 128-bit hash of its word sequence. Two items with the
///   same words in the same order have the same hash, whatever lies
///   between the words.
/// - `shape`: a 64-bit hash of the same sequence with every number word
///   (`2024`, `٣`) replaced by one placeholder, so "Page 3 of 10" and
///   "Page 4 of 10" have one shape.
/// - `simhash`: the 64-bit SimHash of its shingles, each occurrence
///   counted.
/// - `minhash`: `LANES` values per item, the MinHash of its set of
///   shingles, row i at `minhash[i * LANES..(i + 1) * LANES]`.
///
/// A shingle is a run of `shingle` consecutive words; an item with fewer
/// words is one shingle of all of them. An item with no words has no
/// shingle: its SimHash is 0 and its MinHash lanes are all `u32::MAX`.
pub struct Sketches<L> {
    pub shingle: u8,
    pub words: Vec<u32>,
    pub exact: Vec<u128>,
    pub shape: Vec<u64>,
    pub simhash: Vec<u64>,
    pub minhash: Vec<u32>,
    pub _of: PhantomData<fn() -> L>,
}

impl<L> Sketches<L> {
    pub fn empty(shingle: u8) -> Sketches<L> {
        Sketches {
            shingle,
            words: Vec::new(),
            exact: Vec::new(),
            shape: Vec::new(),
            simhash: Vec::new(),
            minhash: Vec::new(),
            _of: PhantomData,
        }
    }

    /// Item `i`'s MinHash row.
    pub fn row(&self, i: Ix<L>) -> &[u32] {
        &self.minhash[i.get() * LANES..(i.get() + 1) * LANES]
    }

    /// The estimated Jaccard similarity of two items' shingle sets.
    pub fn jaccard(&self, a: Ix<L>, b: Ix<L>) -> f64 {
        hash::jaccard(self.row(a), self.row(b))
    }

    /// The share of SimHash bits two items agree on.
    pub fn simhash_similarity(&self, a: Ix<L>, b: Ix<L>) -> f64 {
        1.0 - f64::from((self.simhash[a.get()] ^ self.simhash[b.get()]).count_ones()) / 64.0
    }

    /// Appends the sketch of one item, given its text pieces in order.
    pub fn push<'t>(&mut self, pieces: impl IntoIterator<Item = &'t str>, scratch: &mut Scratch) {
        scratch.exact.clear();
        scratch.shape.clear();
        for piece in pieces {
            hash::words(piece, |w| {
                let h = hash::word(w.as_bytes());
                scratch.exact.push(h);
                scratch.shape.push(if hash::is_number(w) { hash::NUMBER } else { h });
            });
        }
        let ws = &scratch.exact;
        let mut sim = SimHash::default();
        scratch.shingles.clear();
        hash::shingles(ws, usize::from(self.shingle), |s| {
            scratch.shingles.push(s);
            sim.add(s);
        });
        self.words.push(u32::try_from(ws.len()).unwrap_or(u32::MAX));
        self.exact.push(hash::sequence(ws));
        self.shape.push(hash::sequence64(&scratch.shape));
        self.simhash.push(sim.finish());
        self.minhash.extend_from_slice(&hash::minhash(&scratch.shingles));
    }
}

impl<L> Clone for Sketches<L> {
    fn clone(&self) -> Self {
        Sketches {
            shingle: self.shingle,
            words: self.words.clone(),
            exact: self.exact.clone(),
            shape: self.shape.clone(),
            simhash: self.simhash.clone(),
            minhash: self.minhash.clone(),
            _of: PhantomData,
        }
    }
}

impl<L> std::fmt::Debug for Sketches<L> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Sketches").field("shingle", &self.shingle).field("words", &self.words).finish_non_exhaustive()
    }
}

/// Reused buffers for sketching many items.
#[derive(Default)]
pub struct Scratch {
    exact: Vec<u64>,
    shape: Vec<u64>,
    shingles: Vec<u64>,
}

impl<L: Layer> Layer for Sketches<L> {
    const ID: &'static str = "turbo:sketches";
    const OF: Option<&'static LayerName> = Some(L::NAME);

    fn anchor() -> Anchor {
        Anchor::aligned::<L>()
    }

    fn len(&self) -> usize {
        self.words.len()
    }

    fn check(&self) -> Result<(), String> {
        let n = self.words.len();
        if self.shingle == 0 {
            return Err("a shingle of no words".into());
        }
        if self.exact.len() != n || self.shape.len() != n || self.simhash.len() != n {
            return Err("an exact hash, a shape and a SimHash for every item".into());
        }
        if n.checked_mul(LANES) != Some(self.minhash.len()) {
            return Err(format!("{} MinHash values for {n} items of {LANES} lanes", self.minhash.len()));
        }
        for i in 0..n {
            let row = &self.minhash[i * LANES..(i + 1) * LANES];
            if self.words[i] == 0 && (self.simhash[i] != 0 || row.iter().any(|&v| v != u32::MAX)) {
                return Err(format!("item {i} has no words but a shingle hash"));
            }
        }
        Ok(())
    }
}

/// Reads a text view and the items of `L`; writes their `Sketches<L>`.
///
/// The items are in the view read, or in its parent, the view it was
/// made from: then each span is mapped through the view's alignment, so
/// sketches can be read from a case folded view of blocks found on the
/// source text. Every item of `L` is sketched, empty ones included. `L`
/// must have spans. Time is linear in the bytes of the items, with a few
/// operations per shingle.
pub struct Sketcher<L> {
    /// Words per shingle: 3 by default.
    pub shingle: u8,
    _of: PhantomData<fn() -> L>,
}

impl<L> Sketcher<L> {
    pub fn new(shingle: u8) -> Sketcher<L> {
        Sketcher { shingle, _of: PhantomData }
    }
}

impl<L> Default for Sketcher<L> {
    fn default() -> Sketcher<L> {
        Sketcher::new(3)
    }
}

impl<L: Layer> Annotator for Sketcher<L> {
    const ID: &'static str = "turbo:sketcher";
    type Reads = (View, L);
    type Writes = (Sketches<L>,);

    fn annotate(&self, (view, units): Refs<'_, Self>) -> Result<(Sketches<L>,), Error> {
        let fail = |detail: String| Error::Annotator { annotator: Self::ID, detail };
        if self.shingle == 0 {
            return Err(fail("a shingle of no words".into()));
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
        let text = view.text();
        let piece = |s: Span| {
            let s = if in_view { s } else { view.alignment().span_to_child(s) };
            text.get(s.range()).ok_or_else(|| fail(format!("a span {}..{} outside the view", s.start, s.end)))
        };
        let mut out = Sketches::empty(self.shingle);
        out.minhash.reserve(ts.start.len() * LANES);
        let mut scratch = Scratch::default();
        let mut pieces: Vec<&str> = Vec::new();
        for i in 0..ts.start.len() {
            pieces.clear();
            for s in item_spans(&ts, i) {
                pieces.push(piece(s)?);
            }
            out.push(pieces.iter().copied(), &mut scratch);
        }
        Ok((out,))
    }

    fn config_hash(&self) -> u64 {
        u64::from(self.shingle)
    }
}

/// Item `i`'s spans: its fragments when it has them, else its envelope.
pub fn item_spans<'a>(ts: &'a TextSpans<'a>, i: usize) -> impl Iterator<Item = Span> + 'a {
    let frags = if ts.frag_item_end.is_empty() { 0..0 } else { ts.item_frag_range(i) };
    let whole = frags.is_empty().then(|| Span::new(ts.start[i], ts.end[i]));
    whole.into_iter().chain(frags.map(|f| Span::new(ts.frag_start[f], ts.frag_end[f])))
}
