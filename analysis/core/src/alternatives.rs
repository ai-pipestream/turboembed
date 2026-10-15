//! Multiple readings of the same text: token lattices with position lengths
//! (`Alternatives<L>`) and n-best sequences for aligned layers (`NBest<L>`).
//!
//! A single-path layer such as `Tokens` or `PosTags` holds one reading of the
//! document so that downstream annotators can read a flat slice without
//! graph traversal. When a tokenizer produces a word lattice (Sudachi split
//! modes A, B and C, compound words beside their parts, or a synonym graph
//! where one alternative spans multiple token positions) or a beam search
//! keeps its top-k outcome sequences, those readings live in a companion
//! layer (`Alternatives<Tokens>` or `NBest<PosTags>`) alongside the 1-best
//! default layer.

use std::fmt;
use std::marker::PhantomData;
use std::ops::{BitOr, BitOrAssign};
use std::sync::Arc;

use crate::layer::{Anchor, Layer, LayerName, Tag, TagSet};
use crate::span::{Ix, IxRange, Span, Spans};
use crate::text::ViewId;

/// A named reading mode in a token lattice.
///
/// Morphological analyzers with multi-granular segmentation (such as Sudachi)
/// assign each lattice edge to one or more split modes (`A`, `B`, `C`), while
/// search analyzers and Viterbi segmenters mark the primary 1-best path with
/// [`SplitMode::Best`].
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
#[repr(u8)]
pub enum SplitMode {
    /// The primary 1-best path through the lattice.
    Best = 0,
    /// Short unit (Sudachi Mode A): the finest morphological split, suited for
    /// keyword search recall.
    A = 1,
    /// Middle unit (Sudachi Mode B): intermediate compound granularity.
    B = 2,
    /// Long unit (Sudachi Mode C): named entities and full compounds kept
    /// whole.
    C = 3,
}

impl SplitMode {
    pub const ALL: [SplitMode; 4] = [SplitMode::Best, SplitMode::A, SplitMode::B, SplitMode::C];

    #[inline]
    pub const fn bit(self) -> u8 {
        1u8 << (self as u8)
    }
}

/// A compact bitset of [`SplitMode`]s (plus up to four custom mode bits `4..8`)
/// indicating which readings include a lattice alternative.
#[derive(Copy, Clone, Default, PartialEq, Eq, Hash)]
pub struct ModeSet(pub u8);

impl ModeSet {
    pub const EMPTY: ModeSet = ModeSet(0);
    pub const BEST: ModeSet = ModeSet(SplitMode::Best.bit());
    pub const A: ModeSet = ModeSet(SplitMode::A.bit());
    pub const B: ModeSet = ModeSet(SplitMode::B.bit());
    pub const C: ModeSet = ModeSet(SplitMode::C.bit());
    pub const ALL: ModeSet = ModeSet(0b0000_1111);

    #[inline]
    pub const fn from_mode(mode: SplitMode) -> ModeSet {
        ModeSet(mode.bit())
    }

    #[inline]
    pub const fn with(self, mode: SplitMode) -> ModeSet {
        ModeSet(self.0 | mode.bit())
    }

    #[inline]
    pub const fn contains(self, mode: SplitMode) -> bool {
        (self.0 & mode.bit()) != 0
    }

    #[inline]
    pub const fn is_empty(self) -> bool {
        self.0 == 0
    }
}

impl BitOr for ModeSet {
    type Output = ModeSet;

    #[inline]
    fn bitor(self, rhs: ModeSet) -> ModeSet {
        ModeSet(self.0 | rhs.0)
    }
}

impl BitOr<SplitMode> for ModeSet {
    type Output = ModeSet;

    #[inline]
    fn bitor(self, rhs: SplitMode) -> ModeSet {
        self.with(rhs)
    }
}

impl BitOr for SplitMode {
    type Output = ModeSet;

    #[inline]
    fn bitor(self, rhs: SplitMode) -> ModeSet {
        ModeSet(self.bit() | rhs.bit())
    }
}

impl BitOrAssign for ModeSet {
    #[inline]
    fn bitor_assign(&mut self, rhs: ModeSet) {
        self.0 |= rhs.0;
    }
}

impl BitOrAssign<SplitMode> for ModeSet {
    #[inline]
    fn bitor_assign(&mut self, rhs: SplitMode) {
        self.0 |= rhs.bit();
    }
}

impl fmt::Debug for ModeSet {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "ModeSet({:#04x})", self.0)
    }
}

/// One row of [`Alternatives<L>`]: a candidate token or span covering
/// `position..position + position_length` in the lattice.
#[derive(Clone, Debug, PartialEq)]
pub struct Alternative<'a> {
    pub span: Span,
    pub position: u32,
    pub position_length: u32,
    pub mode: ModeSet,
    pub score: Option<f64>,
    pub form: Option<&'a str>,
}

/// A position-length lattice of alternative readings for a positional layer `L`
/// (typically `Tokens`), stored by column: `turbo:alternatives<L>`.
///
/// The lattice is a directed acyclic graph over positions `0..=num_positions`.
/// Each row `i` is an edge from `position[i]` to
/// `position[i] + position_length[i]` with UTF-8 byte span `spans.get(i)` in
/// `spans.view()`, a [`ModeSet`] naming the readings that include it, an
/// optional score (or negative cost), and an optional surface/normalized form
/// (for synonym graphs where an alternative's text is not the raw substring of
/// the view).
///
/// ### Byte-tiling invariant (`Layer::check`)
///
/// Every valid lattice satisfies:
/// 1. Alternatives are ordered by non-decreasing `position`, with
///    `position_length >= 1` and `position + position_length <= num_positions`.
/// 2. Every alternative lies on at least one complete path from position `0` to
///    `num_positions` (no unreachable or dead-end edges), and every internal
///    position boundary `1..num_positions` is visited by at least one path (no
///    phantom positions).
/// 3. Every position `p` has a consistent start byte `start_at[p]` across all
///    edges leaving `p`, and every position `q` has a consistent end byte
///    `end_at[q]` across all edges entering `q`, with
///    `start_at[p] <= end_at[p + 1]` and `end_at[p] <= start_at[p]` at every
///    internal boundary `1..num_positions`. Consequently, every path from `0`
///    to `num_positions` covers the exact same unit position bytes (and, when
///    positions abut with `end_at[p] == start_at[p]`, partitions the byte range
///    `start_at[0]..end_at[num_positions]` with zero gaps and zero overlaps).
pub struct Alternatives<L> {
    pub spans: Spans<Alternatives<L>>,
    pub position: Vec<u32>,
    pub position_length: Vec<u32>,
    pub num_positions: u32,
    pub mode: Vec<ModeSet>,
    pub score: Option<Vec<f64>>,
    pub form: Option<Vec<Option<Box<str>>>>,
    _of: PhantomData<fn() -> L>,
}

impl<L> Clone for Alternatives<L> {
    fn clone(&self) -> Self {
        Alternatives {
            spans: self.spans.clone(),
            position: self.position.clone(),
            position_length: self.position_length.clone(),
            num_positions: self.num_positions,
            mode: self.mode.clone(),
            score: self.score.clone(),
            form: self.form.clone(),
            _of: PhantomData,
        }
    }
}

impl<L> fmt::Debug for Alternatives<L> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Alternatives")
            .field("spans", &self.spans)
            .field("position", &self.position)
            .field("position_length", &self.position_length)
            .field("num_positions", &self.num_positions)
            .field("mode", &self.mode)
            .field("score", &self.score)
            .field("form", &self.form)
            .finish()
    }
}

impl<L> Alternatives<L> {
    /// Creates an empty lattice over `view`.
    pub fn new(view: ViewId) -> Alternatives<L> {
        Alternatives {
            spans: Spans::new(view),
            position: Vec::new(),
            position_length: Vec::new(),
            num_positions: 0,
            mode: Vec::new(),
            score: None,
            form: None,
            _of: PhantomData,
        }
    }

    /// Appends an alternative with its span, start position, position length
    /// and mode set, updating `num_positions` to at least
    /// `position + position_length`.
    pub fn push(&mut self, span: Span, position: u32, position_length: u32, mode: ModeSet) -> Ix<Self> {
        let ix = self.spans.push(span);
        self.position.push(position);
        self.position_length.push(position_length);
        self.mode.push(mode);
        let end_pos = position.saturating_add(position_length);
        if end_pos > self.num_positions {
            self.num_positions = end_pos;
        }
        if let Some(scores) = &mut self.score {
            scores.push(0.0);
        }
        if let Some(forms) = &mut self.form {
            forms.push(None);
        }
        ix
    }

    /// Appends an alternative with an explicit score and optional surface/synonym form.
    pub fn push_full(
        &mut self,
        span: Span,
        position: u32,
        position_length: u32,
        mode: ModeSet,
        score: Option<f64>,
        form: Option<&str>,
    ) -> Ix<Self> {
        let n_before = self.spans.len();
        let ix = self.spans.push(span);
        self.position.push(position);
        self.position_length.push(position_length);
        self.mode.push(mode);
        let end_pos = position.saturating_add(position_length);
        if end_pos > self.num_positions {
            self.num_positions = end_pos;
        }
        match (&mut self.score, score) {
            (Some(scores), Some(s)) => scores.push(s),
            (Some(scores), None) => scores.push(0.0),
            (None, Some(s)) => {
                let mut scores = vec![0.0; n_before];
                scores.push(s);
                self.score = Some(scores);
            }
            (None, None) => {}
        }
        match (&mut self.form, form) {
            (Some(forms), f) => forms.push(f.map(Box::from)),
            (None, Some(f)) => {
                let mut forms = vec![None; n_before];
                forms.push(Some(Box::from(f)));
                self.form = Some(forms);
            }
            (None, None) => {}
        }
        ix
    }

    /// Reads row `i`.
    pub fn get(&self, i: Ix<Self>) -> Alternative<'_> {
        let r = i.get();
        Alternative {
            span: self.spans.get(i),
            position: self.position[r],
            position_length: self.position_length[r],
            mode: self.mode[r],
            score: self.score.as_ref().map(|s| s[r]),
            form: self.form.as_ref().and_then(|f| f[r].as_deref()),
        }
    }

    /// The contiguous range of alternatives starting at `pos`: two binary
    /// searches over `position`.
    pub fn at_position(&self, pos: u32) -> IxRange<Self> {
        let a = self.position.partition_point(|&p| p < pos);
        let b = a + self.position[a..].partition_point(|&p| p == pos);
        IxRange::new(a as u32, b as u32)
    }

    /// Recovers the single path from position `0` to `num_positions` for
    /// `mode`. At each position `p`, among the alternatives at `p` whose
    /// [`ModeSet`] contains `mode` and that can reach `num_positions` within
    /// `mode`, picks the highest-scored one (when `score` is present),
    /// breaking ties by mode granularity (`SplitMode::A` prefers the shortest
    /// `position_length`, `SplitMode::C` the longest, and `Best`/`B` the
    /// earliest row). Returns `None` if `mode` does not form a complete path
    /// from `0` to `num_positions`.
    pub fn path_for_mode(&self, mode: SplitMode) -> Option<Vec<Ix<Self>>> {
        let p_len = self.num_positions as usize;
        if p_len == 0 {
            return Some(Vec::new());
        }
        // Backward reachability within `mode` so we never step into a dead end.
        let mut can_reach_end = vec![false; p_len + 1];
        can_reach_end[p_len] = true;
        for i in (0..self.spans.len()).rev() {
            if !self.mode[i].contains(mode) {
                continue;
            }
            let u = self.position[i] as usize;
            let v = u.saturating_add(self.position_length[i] as usize);
            if v <= p_len && can_reach_end[v] {
                can_reach_end[u] = true;
            }
        }
        if !can_reach_end[0] {
            return None;
        }
        let mut out = Vec::new();
        let mut pos = 0u32;
        while pos < self.num_positions {
            let candidates = self.at_position(pos);
            let mut chosen: Option<Ix<Self>> = None;
            for ix in candidates.iter() {
                let r = ix.get();
                if !self.mode[r].contains(mode) {
                    continue;
                }
                let next_pos = pos + self.position_length[r];
                if (next_pos as usize) > p_len || !can_reach_end[next_pos as usize] {
                    continue;
                }
                chosen = Some(match chosen {
                    None => ix,
                    Some(prev) => {
                        if self.prefer_in_mode(r, prev.get(), mode) {
                            ix
                        } else {
                            prev
                        }
                    }
                });
            }
            let step = chosen?;
            pos += self.position_length[step.get()];
            out.push(step);
        }
        Some(out)
    }

    fn prefer_in_mode(&self, cand: usize, current: usize, mode: SplitMode) -> bool {
        if let Some(scores) = &self.score {
            let cmp = scores[cand].total_cmp(&scores[current]);
            if cmp != std::cmp::Ordering::Equal {
                return cmp.is_gt();
            }
        }
        let (len_c, len_cur) = (self.position_length[cand], self.position_length[current]);
        match mode {
            SplitMode::A if len_c != len_cur => len_c < len_cur,
            SplitMode::C if len_c != len_cur => len_c > len_cur,
            _ => cand < current,
        }
    }

    /// Returns the single best path from `0` to `num_positions`:
    /// - If `SplitMode::Best` forms a complete path, returns that path.
    /// - Otherwise finds the path from `0` to `num_positions` maximizing the
    ///   sum of `score` (or minimizing the number of steps, breaking ties by
    ///   earliest row index, when `score` is `None`).
    pub fn best_path(&self) -> Vec<Ix<Self>> {
        if let Some(path) = self.path_for_mode(SplitMode::Best) {
            return path;
        }
        let p_len = self.num_positions as usize;
        if p_len == 0 {
            return Vec::new();
        }
        let mut best_score = vec![f64::NEG_INFINITY; p_len + 1];
        let mut best_edge: Vec<Option<Ix<Self>>> = vec![None; p_len + 1];
        best_score[p_len] = 0.0;
        for i in (0..self.spans.len()).rev() {
            let u = self.position[i] as usize;
            let v = u.saturating_add(self.position_length[i] as usize);
            if v > p_len || best_score[v] == f64::NEG_INFINITY {
                continue;
            }
            let edge_score = self.score.as_ref().map_or(0.0, |s| s[i]);
            let cand_score = edge_score + best_score[v];
            if cand_score >= best_score[u] {
                best_score[u] = cand_score;
                best_edge[u] = Some(Ix::new(i as u32));
            }
        }
        let mut out = Vec::new();
        let mut pos = 0usize;
        while pos < p_len {
            let Some(edge) = best_edge[pos] else { break };
            out.push(edge);
            pos += self.position_length[edge.get()] as usize;
        }
        out
    }

    /// Enumerates every valid path from position `0` to `num_positions` in
    /// row order.
    pub fn all_paths(&self) -> Vec<Vec<Ix<Self>>> {
        if self.num_positions == 0 {
            return vec![Vec::new()];
        }
        let mut out = Vec::new();
        let mut current = Vec::new();
        self.collect_paths(0, &mut current, &mut out);
        out
    }

    fn collect_paths(&self, pos: u32, current: &mut Vec<Ix<Self>>, out: &mut Vec<Vec<Ix<Self>>>) {
        if pos == self.num_positions {
            out.push(current.clone());
            return;
        }
        for ix in self.at_position(pos).iter() {
            let next = pos + self.position_length[ix.get()];
            if next <= self.num_positions {
                current.push(ix);
                self.collect_paths(next, current, out);
                current.pop();
            }
        }
    }

    /// The spans of `path`, in path order.
    pub fn path_spans(&self, path: &[Ix<Self>]) -> Vec<Span> {
        path.iter().map(|&ix| self.spans.get(ix)).collect()
    }

    /// Returns the maximal contiguous covered byte intervals of `path`.
    ///
    /// Abutting spans along the path are merged, while any multi-position
    /// alternative bridging across an inter-token gap (`end_at[p] <
    /// start_at[p]`) is decomposed into its constituent position spans before
    /// coalescing. For any valid lattice (`check().is_ok()`), every complete
    /// path from `0` to `num_positions` returns the exact same covered byte
    /// intervals as [`best_path`](Self::best_path).
    pub fn covered_bytes(&self, path: &[Ix<Self>]) -> Vec<Span> {
        let p_len = self.num_positions as usize;
        let (starts, ends) = self.spans.columns();
        let mut start_at = vec![0u32; p_len + 1];
        let mut end_at = vec![0u32; p_len + 1];
        for i in 0..self.spans.len() {
            let u = self.position[i] as usize;
            let v = (u + self.position_length[i] as usize).min(p_len);
            start_at[u] = starts[i];
            end_at[v] = ends[i];
        }
        let mut out: Vec<Span> = Vec::new();
        let mut push_coalesced = |s: Span| match out.last_mut() {
            Some(prev) if prev.end == s.start => prev.end = s.end,
            _ => out.push(s),
        };
        for &ix in path {
            let r = ix.get();
            let u = self.position[r] as usize;
            let v = (u + self.position_length[r] as usize).min(p_len);
            for p in u..v {
                push_coalesced(Span::new(start_at[p], end_at[p + 1]));
            }
        }
        out
    }
}

impl<L: Layer> Layer for Alternatives<L> {
    const ID: &'static str = "turbo:alternatives";
    const OF: Option<&'static LayerName> = Some(L::NAME);

    fn anchor() -> Anchor {
        Anchor::Text
    }

    fn len(&self) -> usize {
        self.spans.len()
    }

    fn text_spans(&self) -> Option<(ViewId, &[u32], &[u32])> {
        let (start, end) = self.spans.columns();
        Some((self.spans.view(), start, end))
    }

    fn check(&self) -> Result<(), String> {
        let n = self.spans.len();
        if self.position.len() != n || self.position_length.len() != n || self.mode.len() != n {
            return Err("a position, a position length and a mode set for every alternative".into());
        }
        if let Some(scores) = &self.score {
            if scores.len() != n {
                return Err("a score for every alternative".into());
            }
            if scores.iter().any(|s| s.is_nan()) {
                return Err("no NaN alternative score".into());
            }
        }
        if let Some(forms) = &self.form
            && forms.len() != n
        {
            return Err("a form slot for every alternative".into());
        }
        if n == 0 {
            if self.num_positions != 0 {
                return Err("zero positions when there are no alternatives".into());
            }
            return Ok(());
        }
        if self.num_positions == 0 {
            return Err("a positive number of positions when alternatives are present".into());
        }
        let (starts, ends) = self.spans.columns();
        for i in 0..n {
            if starts[i] > ends[i] {
                return Err(format!("alternative {i} ends before it starts: {}..{}", starts[i], ends[i]));
            }
            if i > 0 && self.position[i] < self.position[i - 1] {
                return Err(format!("alternative {i} at position {} is before the one before it", self.position[i]));
            }
            if self.position_length[i] == 0 {
                return Err(format!("alternative {i} has a position length of zero"));
            }
            let end_pos = self.position[i]
                .checked_add(self.position_length[i])
                .ok_or_else(|| format!("alternative {i} position + position_length overflows u32"))?;
            if end_pos > self.num_positions {
                return Err(format!(
                    "alternative {i} ends at position {end_pos} past num_positions {}",
                    self.num_positions
                ));
            }
        }

        let p_len = self.num_positions as usize;
        // Forward reachability from 0 and backward reachability from `num_positions`.
        let mut from_start = vec![false; p_len + 1];
        from_start[0] = true;
        for i in 0..n {
            let u = self.position[i] as usize;
            let v = u + self.position_length[i] as usize;
            if from_start[u] {
                from_start[v] = true;
            }
        }
        if !from_start[p_len] {
            return Err(format!("no path from position 0 to position {}", self.num_positions));
        }
        let mut to_end = vec![false; p_len + 1];
        to_end[p_len] = true;
        for i in (0..n).rev() {
            let u = self.position[i] as usize;
            let v = u + self.position_length[i] as usize;
            if to_end[v] {
                to_end[u] = true;
            }
        }
        for i in 0..n {
            let u = self.position[i] as usize;
            let v = u + self.position_length[i] as usize;
            if !from_start[u] || !to_end[v] {
                return Err(format!("alternative {i} ({u}..{v}) is not on any complete path through the lattice"));
            }
        }

        // Exact byte-tiling check:
        // 1. Every position `u` (`0..p_len`) must have a unique start byte `start_at[u]`.
        // 2. Every position `v` (`1..=p_len`) must have a unique end byte `end_at[v]`.
        // 3. Every internal boundary `p` (`1..p_len`) must be anchored (`end_at[p] <= start_at[p]`),
        //    and every unit position `p` (`0..p_len`) must satisfy `start_at[p] <= end_at[p + 1]`.
        let mut start_at: Vec<Option<u32>> = vec![None; p_len + 1];
        let mut end_at: Vec<Option<u32>> = vec![None; p_len + 1];
        for i in 0..n {
            let u = self.position[i] as usize;
            let v = u + self.position_length[i] as usize;
            let (s, e) = (starts[i], ends[i]);
            match start_at[u] {
                None => start_at[u] = Some(s),
                Some(prev) if prev != s => {
                    return Err(format!(
                        "alternative {i} at position {u} starts at byte {s}, but another alternative at position {u} \
                         starts at byte {prev}"
                    ));
                }
                _ => {}
            }
            match end_at[v] {
                None => end_at[v] = Some(e),
                Some(prev) if prev != e => {
                    return Err(format!(
                        "alternative {i} ending at position {v} ends at byte {e}, but another alternative ending at \
                         position {v} ends at byte {prev}"
                    ));
                }
                _ => {}
            }
        }
        for p in 1..p_len {
            let (Some(e), Some(s)) = (end_at[p], start_at[p]) else {
                return Err(format!("position boundary {p} has no path starting and ending at it"));
            };
            if e > s {
                return Err(format!(
                    "position boundary {p} overlaps in bytes: incoming ends at {e} past outgoing start at {s}"
                ));
            }
        }
        for p in 0..p_len {
            let (Some(s), Some(e)) = (start_at[p], end_at[p + 1]) else {
                return Err(format!("position {p} has no byte bounds"));
            };
            if s > e {
                return Err(format!("position {p} starts at byte {s} after its end at byte {e}"));
            }
        }
        Ok(())
    }
}

/// Top-k outcome sequences for an aligned layer `L` (for example `PosTags`,
/// `ChunkTags`, or `Tokens`), stored flat in row-major `[candidates, items]`
/// order: `turbo:nbest<L>`.
///
/// The layer is aligned to `L` (`Anchor::aligned::<L>()`), and its `len()` is
/// `items` (the length of `L`), so `Document::add` verifies that every
/// candidate sequence has one tag per item of `L`. Candidate sequences are
/// stored best-first in non-increasing order of `score`.
#[derive(Clone, Debug)]
pub struct NBest<L> {
    pub tagset: Arc<TagSet>,
    /// The number of items in `L` (the length of each candidate sequence).
    pub items: usize,
    /// The sequence score (for example sum of log-probabilities) of each
    /// candidate sequence, best first.
    pub score: Vec<f64>,
    /// Flat `[candidates * items]` tag table: candidate `rank` has tags
    /// `&tags[rank * items..(rank + 1) * items]`.
    pub tags: Vec<Tag>,
    /// Optional flat `[candidates * items]` per-item probability table in
    /// `0.0..=1.0`.
    pub prob: Option<Vec<f64>>,
    _of: PhantomData<fn() -> L>,
}

impl<L> NBest<L> {
    /// Creates an empty `NBest<L>` for a sequence of `items` items.
    pub fn new(tagset: Arc<TagSet>, items: usize) -> NBest<L> {
        NBest { tagset, items, score: Vec::new(), tags: Vec::new(), prob: None, _of: PhantomData }
    }

    /// Appends one candidate sequence of length `self.items` with its sequence
    /// `score` and optional per-item probabilities.
    pub fn push(&mut self, score: f64, tags: &[Tag], probs: Option<&[f64]>) -> Result<usize, String> {
        if tags.len() != self.items {
            return Err(format!("expected {} tags for candidate sequence, got {}", self.items, tags.len()));
        }
        if let Some(p) = probs
            && p.len() != self.items
        {
            return Err(format!("expected {} probabilities for candidate sequence, got {}", self.items, p.len()));
        }
        let rank = self.score.len();
        self.score.push(score);
        self.tags.extend_from_slice(tags);
        match (&mut self.prob, probs) {
            (Some(table), Some(p)) => table.extend_from_slice(p),
            (Some(_), None) => {
                return Err("cannot mix candidates with and without per-item probabilities".into());
            }
            (None, Some(p)) => {
                if rank > 0 {
                    return Err("cannot mix candidates without and with per-item probabilities".into());
                }
                self.prob = Some(p.to_vec());
            }
            (None, None) => {}
        }
        Ok(rank)
    }

    /// The number of candidate sequences ($k$).
    pub fn candidates(&self) -> usize {
        self.score.len()
    }

    /// The score of candidate sequence `rank`.
    pub fn score_of(&self, rank: usize) -> f64 {
        self.score[rank]
    }

    /// The tag sequence of candidate `rank`, one tag per item of `L`.
    pub fn tags_of(&self, rank: usize) -> &[Tag] {
        &self.tags[rank * self.items..(rank + 1) * self.items]
    }

    /// The per-item probabilities of candidate `rank`, when present.
    pub fn probs_of(&self, rank: usize) -> Option<&[f64]> {
        self.prob.as_ref().map(|p| &p[rank * self.items..(rank + 1) * self.items])
    }

    /// The tag and optional probability assigned to `item` in candidate `rank`.
    pub fn at(&self, rank: usize, item: Ix<L>) -> (Tag, Option<f64>) {
        let idx = rank * self.items + item.get();
        (self.tags[idx], self.prob.as_ref().map(|p| p[idx]))
    }
}

impl<L: Layer> Layer for NBest<L> {
    const ID: &'static str = "turbo:nbest";
    const OF: Option<&'static LayerName> = Some(L::NAME);

    fn anchor() -> Anchor {
        Anchor::aligned::<L>()
    }

    fn len(&self) -> usize {
        self.items
    }

    fn check(&self) -> Result<(), String> {
        let k = self.score.len();
        let expected = k.checked_mul(self.items).ok_or_else(|| "candidates * items overflows usize".to_string())?;
        if self.tags.len() != expected {
            return Err(format!("{} tags for {} candidates of {} items", self.tags.len(), k, self.items));
        }
        if self.score.iter().any(|s| s.is_nan()) {
            return Err("no NaN sequence score".into());
        }
        if !self.score.windows(2).all(|w| w[0] >= w[1]) {
            return Err("candidate sequences in non-increasing score order (best first)".into());
        }
        if self.tags.iter().any(|t| t.0 as usize >= self.tagset.len()) {
            return Err(format!("a tag outside {}", self.tagset.name));
        }
        if let Some(prob) = &self.prob {
            if prob.len() != expected {
                return Err("a probability for every candidate tag".into());
            }
            if prob.iter().any(|v| !(0.0..=1.0).contains(v)) {
                return Err("a probability between 0 and 1".into());
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::doc::Document;
    use crate::layer::{Key, Provenance};

    struct Tok;
    impl Layer for Tok {
        const ID: &'static str = "opennlp:tokens";
        fn len(&self) -> usize {
            5
        }
    }

    #[test]
    fn sudachi_modes_a_b_c_round_trip_from_one_lattice() {
        // Sentence: "東京都選挙管理委員会に行った" (42 UTF-8 bytes, 14 kanji/kana chars, 3 bytes each)
        // Finest positions (Mode A, 8 positions):
        //   0: "東京" (0..6)      [A, B]
        //   1: "都"   (6..9)      [A, B]
        //   2: "選挙" (9..15)     [A, B]
        //   3: "管理" (15..21)    [A, B]
        //   4: "委員" (21..27)    [A]
        //   5: "会"   (27..30)    [A]
        //   6: "に"   (30..33)    [A, B, C, Best]
        //   7: "行った" (33..42)  [A, B, C, Best]
        // Mode B compounds:
        //   4..6 (pos=4, len=2): "委員会" (21..30) [B, Best]
        //   0..2 (pos=0, len=2): "東京都" (0..9)   [C, Best]
        //   2..4 (pos=2, len=2): "選挙管理" (9..21) [Best]
        // Mode C compounds:
        //   2..6 (pos=2, len=4): "選挙管理委員会" (9..30) [C]
        let text = "東京都選挙管理委員会に行った";
        assert_eq!(text.len(), 42);
        let mut doc = Document::new(text);
        let mut lat = Alternatives::<Tok>::new(ViewId::SOURCE);

        let a = SplitMode::A;
        let b = SplitMode::B;
        let c = SplitMode::C;
        let best = SplitMode::Best;

        // Position 0
        lat.push(Span::new(0, 6), 0, 1, a | b);
        lat.push(Span::new(0, 9), 0, 2, c | best);
        // Position 1
        lat.push(Span::new(6, 9), 1, 1, a | b);
        // Position 2
        lat.push(Span::new(9, 15), 2, 1, a | b);
        lat.push(Span::new(9, 21), 2, 2, ModeSet::from_mode(best));
        lat.push(Span::new(9, 30), 2, 4, ModeSet::from_mode(c));
        // Position 3
        lat.push(Span::new(15, 21), 3, 1, a | b);
        // Position 4
        lat.push(Span::new(21, 27), 4, 1, ModeSet::from_mode(a));
        lat.push(Span::new(21, 30), 4, 2, b | best);
        // Position 5
        lat.push(Span::new(27, 30), 5, 1, ModeSet::from_mode(a));
        // Position 6
        lat.push(Span::new(30, 33), 6, 1, ModeSet::ALL);
        // Position 7
        lat.push(Span::new(33, 42), 7, 1, ModeSet::ALL);

        assert_eq!(lat.check(), Ok(()));

        let surfaces = |path: &[Ix<Alternatives<Tok>>]| -> Vec<&str> {
            path.iter()
                .map(|&ix| {
                    let s = lat.spans.get(ix);
                    &text[s.start as usize..s.end as usize]
                })
                .collect()
        };

        let path_a = lat.path_for_mode(SplitMode::A).unwrap();
        assert_eq!(surfaces(&path_a), ["東京", "都", "選挙", "管理", "委員", "会", "に", "行った"]);

        let path_b = lat.path_for_mode(SplitMode::B).unwrap();
        assert_eq!(surfaces(&path_b), ["東京", "都", "選挙", "管理", "委員会", "に", "行った"]);

        let path_c = lat.path_for_mode(SplitMode::C).unwrap();
        assert_eq!(surfaces(&path_c), ["東京都", "選挙管理委員会", "に", "行った"]);

        let path_best = lat.best_path();
        assert_eq!(surfaces(&path_best), ["東京都", "選挙管理", "委員会", "に", "行った"]);

        // Every path in the lattice tiles the exact bytes 0..42 (2 ways across 0..2 * 5 ways across 2..6 = 10 paths).
        let all = lat.all_paths();
        assert_eq!(all.len(), 10);
        let best_bytes = lat.covered_bytes(&path_best);
        assert_eq!(best_bytes, [Span::new(0, 42)]);
        for p in &all {
            assert_eq!(lat.covered_bytes(p), best_bytes);
            let spans = lat.path_spans(p);
            assert_eq!(spans.first().unwrap().start, 0);
            assert_eq!(spans.last().unwrap().end, 42);
            assert!(spans.windows(2).all(|w| w[0].end == w[1].start));
        }

        doc.add(&Key::DEFAULT, lat, Provenance::by("test")).unwrap();
        assert_eq!(doc.get::<Alternatives<Tok>>().unwrap().num_positions, 8);
    }

    #[test]
    fn synonym_graph_ny_and_new_york_spans_two_positions() {
        // "in new york today"
        // pos 0: "in"    (0..2, len 1)
        // pos 1: "new"   (3..6, len 1)
        // pos 1: "ny"    (3..11, len 2, synonym form "ny")
        // pos 2: "york"  (7..11, len 1)
        // pos 3: "today" (12..17, len 1)
        let text = "in new york today";
        let mut doc = Document::new(text);
        let mut lat = Alternatives::<Tok>::new(ViewId::SOURCE);
        lat.push_full(Span::new(0, 2), 0, 1, ModeSet::BEST | SplitMode::A | SplitMode::C, Some(1.0), None);
        lat.push_full(Span::new(3, 6), 1, 1, ModeSet::BEST | SplitMode::A, Some(1.0), None);
        lat.push_full(Span::new(3, 11), 1, 2, ModeSet::from_mode(SplitMode::C), Some(0.9), Some("ny"));
        lat.push_full(Span::new(7, 11), 2, 1, ModeSet::BEST | SplitMode::A, Some(1.0), None);
        lat.push_full(Span::new(12, 17), 3, 1, ModeSet::BEST | SplitMode::A | SplitMode::C, Some(1.0), None);

        assert_eq!(lat.check(), Ok(()));
        let token_forms = |path: &[Ix<Alternatives<Tok>>]| -> Vec<&str> {
            path.iter()
                .map(|&ix| {
                    let alt = lat.get(ix);
                    alt.form.unwrap_or(&text[alt.span.start as usize..alt.span.end as usize])
                })
                .collect()
        };
        let expanded = lat.path_for_mode(SplitMode::A).unwrap();
        assert_eq!(token_forms(&expanded), ["in", "new", "york", "today"]);

        let contracted = lat.path_for_mode(SplitMode::C).unwrap();
        assert_eq!(token_forms(&contracted), ["in", "ny", "today"]);
        assert_eq!(lat.get(contracted[1]).position_length, 2);
        assert_eq!(lat.get(contracted[1]).span, Span::new(3, 11));

        // Both paths cover the exact same token bytes.
        let best = lat.best_path();
        assert_eq!(lat.covered_bytes(&expanded), lat.covered_bytes(&best));
        assert_eq!(lat.covered_bytes(&contracted), lat.covered_bytes(&best));

        doc.add(&Key::DEFAULT, lat, Provenance::by("test")).unwrap();
    }

    #[test]
    fn property_every_path_in_random_lattices_covers_the_same_bytes_as_best_path() {
        // Deterministic LCG generating 60 random multi-branch lattices with
        // both contiguous and whitespace-gapped boundaries and multi-position
        // arcs of length 1..=4.
        let mut seed = 0x9E37_79B9_7F4A_7C15u64;
        let mut next_u32 = || -> u32 {
            seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            (seed >> 32) as u32
        };

        for case in 0..60 {
            let num_pos = 3 + (next_u32() % 8) as usize;
            let contiguous = case % 2 == 0;
            let mut pos_start = Vec::with_capacity(num_pos);
            let mut pos_end = Vec::with_capacity(num_pos);
            let mut cursor = 0u32;
            for _ in 0..num_pos {
                let len = 1 + (next_u32() % 5);
                pos_start.push(cursor);
                pos_end.push(cursor + len);
                cursor += len + if contiguous { 0 } else { next_u32() % 2 };
            }

            let mut lat = Alternatives::<Tok>::new(ViewId::SOURCE);
            for p in 0..num_pos {
                // Always include the unit edge (p, p + 1) so every boundary is anchored.
                let score = f64::from(next_u32() % 100) * 0.01;
                lat.push_full(Span::new(pos_start[p], pos_end[p]), p as u32, 1, ModeSet::A, Some(score), None);
                // Randomly add multi-position edges of length 2..=4.
                for span_len in 2..=4 {
                    if p + span_len <= num_pos && (next_u32() % 3 == 0) {
                        let score = f64::from(next_u32() % 250) * 0.01;
                        let mode = if span_len == 2 { ModeSet::B } else { ModeSet::C };
                        lat.push_full(
                            Span::new(pos_start[p], pos_end[p + span_len - 1]),
                            p as u32,
                            span_len as u32,
                            mode,
                            Some(score),
                            None,
                        );
                    }
                }
            }

            assert_eq!(lat.check(), Ok(()), "case {case} failed check");
            let best = lat.best_path();
            assert!(!best.is_empty());
            let want_bytes = lat.covered_bytes(&best);
            let paths = lat.all_paths();
            assert!(!paths.is_empty());
            for path in &paths {
                assert_eq!(lat.covered_bytes(path), want_bytes, "case {case} path {path:?}");
                if contiguous {
                    assert_eq!(lat.covered_bytes(path), [Span::new(0, cursor)]);
                }
            }
        }
    }

    #[test]
    fn check_refuses_broken_tiling_dead_ends_and_unanchored_positions() {
        // 1. Mismatched start byte at position 0.
        let mut bad = Alternatives::<Tok>::new(ViewId::SOURCE);
        bad.push(Span::new(0, 3), 0, 1, ModeSet::A);
        bad.push(Span::new(1, 3), 0, 1, ModeSet::B);
        assert!(bad.check().unwrap_err().contains("starts at byte 1"));

        // 2. Mismatched end byte at position 1.
        let mut bad = Alternatives::<Tok>::new(ViewId::SOURCE);
        bad.push(Span::new(0, 3), 0, 1, ModeSet::A);
        bad.push(Span::new(0, 4), 0, 1, ModeSet::B);
        assert!(bad.check().unwrap_err().contains("ends at byte 4"));

        // 3. Overlapping byte boundary at position 1 (end_at[1] = 4 > start_at[1] = 3).
        let mut bad = Alternatives::<Tok>::new(ViewId::SOURCE);
        bad.push(Span::new(0, 4), 0, 1, ModeSet::A);
        bad.push(Span::new(3, 6), 1, 1, ModeSet::A);
        assert!(bad.check().unwrap_err().contains("overlaps in bytes"));

        // 4. Unanchored internal boundary (num_positions = 2, only a single len=2 edge).
        let mut bad = Alternatives::<Tok>::new(ViewId::SOURCE);
        bad.push(Span::new(0, 6), 0, 2, ModeSet::C);
        assert!(bad.check().unwrap_err().contains("position boundary 1 has no path"));

        // 5. Dead-end edge that cannot reach num_positions.
        let mut bad = Alternatives::<Tok>::new(ViewId::SOURCE);
        bad.push(Span::new(0, 6), 0, 2, ModeSet::C);
        bad.push(Span::new(0, 3), 0, 1, ModeSet::A);
        assert!(bad.check().unwrap_err().contains("not on any complete path"));

        // 6. Zero position_length.
        let mut bad = Alternatives::<Tok>::new(ViewId::SOURCE);
        bad.push(Span::new(0, 3), 0, 0, ModeSet::A);
        bad.num_positions = 1;
        assert!(bad.check().unwrap_err().contains("position length of zero"));
    }

    #[test]
    fn nbest_sequences_align_to_anchor_and_validate_scores_and_probs() {
        let mut doc = Document::new("dogs bark loud today .");
        doc.add(&Key::DEFAULT, Tok, Provenance::by("test")).unwrap();

        let tagset = TagSet::new("ud", &["NOUN", "VERB", "ADV", "ADJ", "PUNCT"]);
        let mut nbest = NBest::<Tok>::new(tagset.clone(), 5);
        nbest.push(-1.2, &[Tag(0), Tag(1), Tag(2), Tag(0), Tag(4)], Some(&[0.9, 0.8, 0.7, 0.9, 0.99])).unwrap();
        nbest.push(-1.8, &[Tag(0), Tag(1), Tag(3), Tag(0), Tag(4)], Some(&[0.9, 0.8, 0.3, 0.9, 0.99])).unwrap();

        assert_eq!(nbest.check(), Ok(()));
        assert_eq!(nbest.candidates(), 2);
        assert_eq!(nbest.tags_of(0), &[Tag(0), Tag(1), Tag(2), Tag(0), Tag(4)]);
        assert_eq!(nbest.at(1, Ix::new(2)), (Tag(3), Some(0.3)));

        doc.add(&Key::DEFAULT, nbest, Provenance::by("test")).unwrap();
        assert_eq!(doc.get::<NBest<Tok>>().unwrap().candidates(), 2);

        // Misaligned NBest (4 items instead of 5) is refused by Document::add.
        let mut wrong_len = NBest::<Tok>::new(tagset.clone(), 4);
        wrong_len.push(-1.0, &[Tag(0), Tag(1), Tag(2), Tag(4)], None).unwrap();
        let err = doc.add(&Key::named("wrong"), wrong_len, Provenance::by("test")).unwrap_err();
        assert!(matches!(err, crate::error::Error::Misaligned { .. }), "{err:?}");

        // Out-of-order candidate scores are refused by Layer::check.
        let mut bad_order = NBest::<Tok>::new(tagset, 5);
        bad_order.push(-2.0, &[Tag(0); 5], None).unwrap();
        bad_order.push(-1.0, &[Tag(0); 5], None).unwrap();
        assert!(bad_order.check().unwrap_err().contains("non-increasing score order"));
    }
}
