//! Late chunking: pool already-contextualized subword vectors
//! into one vector per item of a span layer, without running the
//! encoder again per item.
//!
//! The contract that makes the result "late" rather than merely
//! pooled is enforced here, in code: the subword vectors must
//! come with their encoder windows ([`TokenVectorWindows`]), the
//! vectors and the pieces must name the same model, and what
//! happens to an item whose pieces were encoded in more than one
//! window is an explicit [`WindowPolicy`] choice, never a silent
//! degradation. An independent per-token embedding table has no
//! windows to present, so it cannot be wired in by accident.

use std::collections::BTreeSet;
use std::marker::PhantomData;

use turbo_analysis::{Annotator, Error, Ix, Layer, Overlap, Refs, Span, View};
use turbo_analysis_layers::{Subwords, TokenVectorWindows, Vectors};

/// How an item's selected subword vectors become one vector.
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq)]
pub enum Pooling {
    /// Component-wise arithmetic mean.
    #[default]
    Mean,
    /// Component-wise maximum.
    Max,
    /// Mean over the pieces the mask keeps: pieces whose surface
    /// text holds no alphanumeric character (punctuation and
    /// symbols) and pieces in the annotator's stopword set are
    /// excluded before averaging.
    MaskedMean,
}

/// What to do with an item whose pieces were encoded in more
/// than one encoder window.
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq)]
pub enum WindowPolicy {
    /// Refuse, naming the item and the windows it straddles.
    /// This is the default: a straddling item was not encoded in
    /// one pass, and pooling it anyway would silently lose the
    /// cross-boundary context late chunking exists to keep.
    #[default]
    Reject,
    /// Pool each window's pieces separately, then combine the
    /// per-window vectors weighted by their piece counts.
    SplitPool,
    /// Pool across the seam directly, accepting a vector
    /// assembled from more than one forward pass.
    Allow,
}

/// Pools `Vectors<Subwords>` into `Vectors<L>` for a
/// span-anchored layer `L` (`Chunks`, `Levels`, `Sentences`,
/// `Structure`, `Entities`, ...): one pooled row per item of
/// `L`, in item order.
///
/// A subword piece contributes to an item when its span overlaps
/// any fragment of the item's span (a piece cut by an item
/// boundary contributes to both neighbors; a zero-length piece
/// contributes to the item whose fragment contains its
/// position). Pieces whose vector row is empty never
/// contribute. An item selecting no contributing piece gets an
/// empty row (`Vectors::is_empty`), never a fabricated vector.
///
/// The output is the ordinary `Vectors<L>` layer, so to keep a
/// naive per-item embedding and the late one on one document,
/// bind this step at a named instance (for example `@late`).
#[derive(Clone, Debug)]
pub struct LateChunkAnnotator<L> {
    pub pooling: Pooling,
    pub window_policy: WindowPolicy,
    /// L2-normalize each pooled row. Off by default: the token
    /// vectors may already be normalized, and normalizing again
    /// after mean pooling is a policy choice, not a repair.
    pub normalize: bool,
    /// Surface forms excluded under [`Pooling::MaskedMean`],
    /// matched case-insensitively against a piece's whole text.
    pub stopwords: BTreeSet<String>,
    _of: PhantomData<fn() -> L>,
}

impl<L> Default for LateChunkAnnotator<L> {
    fn default() -> Self {
        LateChunkAnnotator {
            pooling: Pooling::Mean,
            window_policy: WindowPolicy::Reject,
            normalize: false,
            stopwords: BTreeSet::new(),
            _of: PhantomData,
        }
    }
}

impl<L> LateChunkAnnotator<L> {
    pub fn new(pooling: Pooling, window_policy: WindowPolicy) -> Self {
        LateChunkAnnotator { pooling, window_policy, ..Self::default() }
    }

    pub fn normalize(mut self, normalize: bool) -> Self {
        self.normalize = normalize;
        self
    }

    pub fn stopwords(mut self, words: impl IntoIterator<Item = impl Into<String>>) -> Self {
        self.stopwords = words.into_iter().map(|w| fold(&w.into())).collect();
        self
    }

    fn fail(&self, detail: String) -> Error {
        Error::Annotator { annotator: "turbo:late-chunk", detail }
    }

    /// Whether the mask drops a piece with this surface text,
    /// against a stopword set already folded by the caller
    /// (the public field is folded at use, so entries inserted
    /// directly, in any case, mask as the builder's do).
    fn masked(&self, text: &str, stopwords: &BTreeSet<String>) -> bool {
        !text.chars().any(char::is_alphanumeric) || stopwords.contains(&fold(text))
    }
}

fn fold(s: &str) -> String {
    s.trim().chars().flat_map(char::to_lowercase).collect()
}

/// Whether piece span `p` overlaps fragment `f`, half-open,
/// with a zero-length piece counting at its position. An
/// empty fragment overlaps nothing: a zero-length target
/// inside a piece must not pool that piece while the same
/// target at a piece boundary pools nothing.
fn overlaps(p: Span, f: Span) -> bool {
    if f.start == f.end {
        false
    } else if p.start == p.end {
        f.start <= p.start && p.start < f.end
    } else {
        p.start < f.end && f.start < p.end
    }
}

struct Pool {
    sum: Vec<f64>,
    max: Vec<f64>,
    count: usize,
}

impl Pool {
    fn new(dim: usize) -> Pool {
        Pool { sum: vec![0.0; dim], max: vec![f64::NEG_INFINITY; dim], count: 0 }
    }

    fn add(&mut self, row: &[f32]) {
        for (i, &v) in row.iter().enumerate() {
            let v = f64::from(v);
            self.sum[i] += v;
            if v > self.max[i] {
                self.max[i] = v;
            }
        }
        self.count += 1;
    }

    fn finish(&self, pooling: Pooling) -> Vec<f64> {
        match pooling {
            Pooling::Max => self.max.clone(),
            Pooling::Mean | Pooling::MaskedMean => self.sum.iter().map(|s| s / self.count as f64).collect(),
        }
    }
}

impl<L: Layer> Annotator for LateChunkAnnotator<L> {
    const ID: &'static str = "turbo:late-chunk";
    type Reads = (View, Vectors<Subwords>, Subwords, TokenVectorWindows, L);
    type Writes = (Vectors<L>,);

    fn annotate(&self, (view, vectors, subwords, windows, target): Refs<'_, Self>) -> Result<Self::Writes, Error> {
        let fail = |detail: String| self.fail(detail);
        vectors.check().map_err(fail)?;
        subwords.check().map_err(fail)?;
        windows.check().map_err(fail)?;
        target.check().map_err(fail)?;
        if vectors.len() != subwords.len() {
            return Err(fail(format!("{} subword vectors for {} subwords", vectors.len(), subwords.len())));
        }
        if vectors.model != subwords.model {
            return Err(fail(format!(
                "the vectors are {}'s (revision {}), the subwords {}'s (revision {}); \
                 late pooling needs the token vectors of the encoder that produced the pieces",
                vectors.model.id, vectors.model.revision, subwords.model.id, subwords.model.revision
            )));
        }
        if vectors.data.iter().any(|v| !v.is_finite()) {
            return Err(fail("a subword vector holds a non-finite value".into()));
        }
        let target_spans = target.text_spans().ok_or_else(|| fail(format!("{} has no spans to pool over", L::NAME)))?;
        let view_id = view.id();
        for (what, v) in
            [("subwords", subwords.spans.view()), ("windows", windows.spans.view()), ("target", target_spans.view)]
        {
            if v != view_id {
                return Err(fail(format!("the {what} are in view {}, the text in view {}", v.0, view_id.0)));
            }
        }
        if target_spans.start.len() != target.len() {
            return Err(fail(format!("{} spans for {} items of {}", target_spans.start.len(), target.len(), L::NAME)));
        }
        // The fragment table is indexed below; hold it to its own
        // invariants even for a layer whose check is lenient.
        target_spans.check(Overlap::Any).map_err(fail)?;
        // Every span this step reads must lie in the view text
        // on UTF-8 boundaries, in every pooling mode: a direct
        // call gets no `Document::add` to catch coordinates
        // outside the view or inside a character, and only
        // masked mean would otherwise notice while slicing.
        let text = view.text();
        let in_text = |what: &str, i: usize, start: u32, end: u32| -> Result<(), Error> {
            let (start, end) = (start as usize, end as usize);
            if end > text.len() || !text.is_char_boundary(start) || !text.is_char_boundary(end) {
                return Err(fail(format!(
                    "{what} {i} ({start}..{end}) is outside the view text or inside a character"
                )));
            }
            Ok(())
        };
        for (i, span) in windows.spans.iter() {
            in_text("window", i.get(), span.start, span.end)?;
        }
        for (i, span) in subwords.spans.iter() {
            in_text("subword", i.get(), span.start, span.end)?;
        }
        for (i, (&start, &end)) in target_spans.start.iter().zip(target_spans.end).enumerate() {
            in_text("target", i, start, end)?;
        }
        for (i, (&start, &end)) in target_spans.frag_start.iter().zip(target_spans.frag_end).enumerate() {
            in_text("target fragment", i, start, end)?;
        }
        // Fold the configured stopwords at use, so entries
        // inserted through the public field in any case mask
        // exactly as the builder's folded entries do.
        let stopwords: BTreeSet<String> = self.stopwords.iter().map(|w| fold(w)).collect();

        // The window holding each piece, by binary search over
        // the ordered windows. A piece no window contains means
        // the windows do not describe these vectors.
        let (win_starts, win_ends) = windows.spans.columns();
        let (piece_starts, piece_ends) = subwords.spans.columns();
        let mut piece_window = Vec::with_capacity(subwords.len());
        for (i, (&start, &end)) in piece_starts.iter().zip(piece_ends).enumerate() {
            let at = win_starts.partition_point(|&s| s <= start);
            let held = at.checked_sub(1).filter(|&w| end <= win_ends[w]);
            let Some(w) = held else {
                return Err(fail(format!(
                    "subword {i} ({start}..{end}) lies in no encoder window; \
                     the windows do not describe these vectors"
                )));
            };
            piece_window.push(w);
        }

        let dim = vectors.dim as usize;
        let cells = target.len().checked_mul(dim).ok_or_else(|| fail("the pooled layer is too large".into()))?;
        let mut data = vec![0.0f32; cells];
        let mut empty = Vec::new();
        for item in 0..target.len() {
            // This item's fragments (its envelope when it is
            // continuous), and the piece range overlapping the
            // envelope, by binary search over piece ends/starts.
            let fragments: Vec<Span> = if target_spans.frag_item_end.is_empty() {
                vec![Span::new(target_spans.start[item], target_spans.end[item])]
            } else {
                let r = target_spans.item_frag_range(item);
                r.map(|f| Span::new(target_spans.frag_start[f], target_spans.frag_end[f])).collect()
            };
            let (env_start, env_end) = (target_spans.start[item], target_spans.end[item]);
            let mut lo = piece_ends.partition_point(|&e| e < env_start);
            while lo < piece_ends.len() && piece_ends[lo] == env_start && piece_starts[lo] < env_start {
                lo += 1;
            }
            let hi = piece_starts.partition_point(|&s| s < env_end).max(lo);

            // Pool per encoder window; the policy decides what
            // more than one non-empty window group means.
            let mut groups: Vec<(usize, Pool)> = Vec::new();
            for piece in lo..hi {
                let span = Span::new(piece_starts[piece], piece_ends[piece]);
                if !fragments.iter().any(|&f| overlaps(span, f)) {
                    continue;
                }
                let ix = Ix::<Subwords>::new(piece as u32);
                if vectors.is_empty(ix) {
                    continue;
                }
                if self.pooling == Pooling::MaskedMean {
                    let surface = text.get(span.range()).ok_or_else(|| {
                        fail(format!("subword {piece} ({}..{}) is outside the view text", span.start, span.end))
                    })?;
                    if self.masked(surface, &stopwords) {
                        continue;
                    }
                }
                let w = piece_window[piece];
                match groups.last_mut() {
                    Some((last, pool)) if *last == w => pool.add(vectors.row(ix)),
                    _ => {
                        // Pieces are in text order and windows do
                        // not overlap, so groups stay sorted; a
                        // window seen again would mean the piece
                        // order broke, which `check` excludes.
                        let mut pool = Pool::new(dim);
                        pool.add(vectors.row(ix));
                        groups.push((w, pool));
                    }
                }
            }
            let pooled = match groups.len() {
                0 => None,
                1 => Some(groups.pop().unwrap().1.finish(self.pooling)),
                _ => match self.window_policy {
                    WindowPolicy::Reject => {
                        let named: Vec<String> =
                            groups.iter().map(|(w, _)| format!("{}..{}", win_starts[*w], win_ends[*w])).collect();
                        return Err(fail(format!(
                            "item {item} of {} ({env_start}..{env_end}) straddles encoder windows [{}]; \
                             choose WindowPolicy::SplitPool or WindowPolicy::Allow to pool it anyway",
                            L::NAME,
                            named.join(", ")
                        )));
                    }
                    WindowPolicy::Allow => {
                        let mut all = Pool::new(dim);
                        for (_, pool) in &groups {
                            for i in 0..dim {
                                all.sum[i] += pool.sum[i];
                                if pool.max[i] > all.max[i] {
                                    all.max[i] = pool.max[i];
                                }
                            }
                            all.count += pool.count;
                        }
                        Some(all.finish(self.pooling))
                    }
                    WindowPolicy::SplitPool => {
                        let total: usize = groups.iter().map(|(_, p)| p.count).sum();
                        let mut combined = vec![0.0f64; dim];
                        for (_, pool) in &groups {
                            let weight = pool.count as f64 / total as f64;
                            for (i, v) in pool.finish(self.pooling).iter().enumerate() {
                                combined[i] += weight * v;
                            }
                        }
                        Some(combined)
                    }
                },
            };
            let Some(mut row) = pooled else {
                empty.push(Ix::<L>::new(item as u32));
                continue;
            };
            if self.normalize {
                let norm = row.iter().map(|v| v * v).sum::<f64>().sqrt();
                if norm > 0.0 {
                    row.iter_mut().for_each(|v| *v /= norm);
                }
            }
            if row.iter().any(|v| !v.is_finite()) {
                return Err(fail(format!("pooling item {item} produced a non-finite value")));
            }
            for (i, v) in row.iter().enumerate() {
                data[item * dim + i] = *v as f32;
            }
        }
        let out = Vectors::new(vectors.dim, data, empty, vectors.model.clone());
        out.check().map_err(fail)?;
        Ok((out,))
    }

    fn config_hash(&self) -> u64 {
        fn feed(hash: &mut u64, bytes: &[u8]) {
            for &b in bytes {
                *hash = (*hash ^ u64::from(b)).wrapping_mul(0x100000001b3);
            }
        }
        let mut hash = 0xcbf29ce484222325u64;
        feed(&mut hash, Self::ID.as_bytes());
        feed(&mut hash, &[self.pooling as u8, self.window_policy as u8, u8::from(self.normalize)]);
        // Folded, matching how the set is read at use: a word
        // inserted in any case hashes as its folded form.
        let folded: BTreeSet<String> = self.stopwords.iter().map(|w| fold(w)).collect();
        for word in &folded {
            feed(&mut hash, word.as_bytes());
            feed(&mut hash, &[0]);
        }
        hash
    }
}
