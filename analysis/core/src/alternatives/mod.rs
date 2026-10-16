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

mod nbest;

pub use nbest::NBest;

use std::fmt;
use std::marker::PhantomData;
use std::ops::{BitOr, BitOrAssign};

use crate::layer::{Anchor, Layer, LayerName};
use crate::span::{Ix, IxRange, Overlap, Span, Spans, TextSpans};
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

/// A compact bitset of [`SplitMode`]s naming the readings that include a
/// lattice alternative. Bits `4..8` are free for a producer's own readings;
/// they are carried and compared, and `path_for_mode` queries only the four
/// named modes.
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
///
/// Between abutting positions there may be a gap (`end_at[p] < start_at[p]`,
/// the whitespace between two words), and an alternative spanning several
/// positions covers the gaps between them. Only the unit position bytes are
/// the same along every path.
///
/// The queries (`at_position`, `path_for_mode`, `best_path`, `all_paths`,
/// `covered_bytes`) assume a layer that passed `check`, which `Document::add`
/// runs; a hand-built layer that did not pass it gives no meaningful answer.
/// `push` and `push_full` append in the order given, so a producer pushes in
/// non-decreasing position order.
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
    /// `position + position_length`. When the layer already carries scores,
    /// the alternative's score is `0.0`.
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

    /// The path from position `0` to `num_positions` through the alternatives
    /// whose [`ModeSet`] contains `mode`: when the mode holds several such
    /// paths, the one with the highest sum of `score` (every path, when
    /// there are no scores). Two paths with the same sum are told apart at
    /// the first position where they differ, by mode granularity
    /// (`SplitMode::A` prefers the shorter `position_length`, `SplitMode::C`
    /// the longer, `Best` and `B` the earlier row), then by the earlier row.
    /// `None` when `mode` has no complete path.
    pub fn path_for_mode(&self, mode: SplitMode) -> Option<Vec<Ix<Self>>> {
        self.best_path_through(
            |r| self.mode[r].contains(mode),
            |r| self.score.as_ref().map_or(0.0, |s| s[r]),
            Some(mode),
        )
    }

    /// The single best path from `0` to `num_positions`: the path of
    /// `SplitMode::Best` ([`path_for_mode`](Self::path_for_mode)) when it
    /// has one, else the path over all alternatives with the highest sum of
    /// `score`, or with the fewest steps when there are no scores, ties
    /// going to the earlier row at the first position that differs. Empty
    /// when the lattice is empty.
    pub fn best_path(&self) -> Vec<Ix<Self>> {
        if let Some(path) = self.path_for_mode(SplitMode::Best) {
            return path;
        }
        let weight = |r: usize| self.score.as_ref().map_or(-1.0, |s| s[r]);
        self.best_path_through(|_| true, weight, None).unwrap_or_default()
    }

    /// The path from `0` to `num_positions` over the alternatives `allowed`
    /// maximizing the sum of `weight`: one pass over the rows from the last
    /// position back, so each position's best continuation is known before
    /// the alternatives leaving it are weighed. `None` when no path exists.
    fn best_path_through(
        &self,
        allowed: impl Fn(usize) -> bool,
        weight: impl Fn(usize) -> f64,
        mode: Option<SplitMode>,
    ) -> Option<Vec<Ix<Self>>> {
        let p_len = self.num_positions as usize;
        if p_len == 0 {
            return Some(Vec::new());
        }
        // The best sum from each position to the end, and the row that
        // starts it; `None` while no allowed path reaches the end.
        let mut best: Vec<Option<(f64, usize)>> = vec![None; p_len + 1];
        best[p_len] = Some((0.0, usize::MAX));
        for r in (0..self.spans.len()).rev() {
            if !allowed(r) {
                continue;
            }
            let u = self.position[r] as usize;
            let v = u.saturating_add(self.position_length[r] as usize);
            let Some(Some((rest, _))) = best.get(v) else { continue };
            let total = weight(r) + rest;
            let better = match best[u] {
                None => true,
                Some((cur, cur_row)) => match total.total_cmp(&cur) {
                    std::cmp::Ordering::Greater => true,
                    std::cmp::Ordering::Less => false,
                    std::cmp::Ordering::Equal => self.prefer(r, cur_row, mode),
                },
            };
            if better {
                best[u] = Some((total, r));
            }
        }
        let mut out = Vec::new();
        let mut pos = 0usize;
        while pos < p_len {
            let (_, r) = best[pos]?;
            out.push(Ix::new(r as u32));
            pos += self.position_length[r] as usize;
        }
        Some(out)
    }

    /// Whether row `cand` is taken over row `current` when both start the
    /// same position with the same sum: by granularity under `mode`, then
    /// the earlier row.
    fn prefer(&self, cand: usize, current: usize, mode: Option<SplitMode>) -> bool {
        let (len_c, len_cur) = (self.position_length[cand], self.position_length[current]);
        match mode {
            Some(SplitMode::A) if len_c != len_cur => len_c < len_cur,
            Some(SplitMode::C) if len_c != len_cur => len_c > len_cur,
            _ => cand < current,
        }
    }

    /// Every path from position `0` to `num_positions`, each in row order
    /// and the paths in the order of their first differing row. The count
    /// grows exponentially with the lattice's length (a lattice with a
    /// one-position and a two-position alternative at every position has
    /// a Fibonacci number of paths), so this suits one sentence's lattice;
    /// an indexer walks [`at_position`](Self::at_position) instead.
    pub fn all_paths(&self) -> Vec<Vec<Ix<Self>>> {
        let mut out = Vec::new();
        if self.num_positions == 0 {
            out.push(Vec::new());
            return out;
        }
        // A depth-first walk with its own stack: the path so far, and for
        // each step the rows still to try at the position it left from.
        let mut current: Vec<Ix<Self>> = Vec::new();
        let mut pending: Vec<std::ops::Range<u32>> = vec![rows(self.at_position(0))];
        while let Some(range) = pending.last_mut() {
            let Some(r) = range.next() else {
                pending.pop();
                current.pop();
                continue;
            };
            let next = self.position[r as usize] + self.position_length[r as usize];
            if next > self.num_positions {
                continue;
            }
            current.push(Ix::new(r));
            if next == self.num_positions {
                out.push(current.clone());
                current.pop();
            } else {
                pending.push(rows(self.at_position(next)));
            }
        }
        out
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

/// The row numbers of a range of alternatives.
fn rows<L>(r: IxRange<L>) -> std::ops::Range<u32> {
    r.start.get() as u32..r.end.get() as u32
}

impl<L: Layer> Layer for Alternatives<L> {
    const ID: &'static str = "turbo:alternatives";
    const OF: Option<&'static LayerName> = Some(L::NAME);
    /// Lattice alternatives leaving or crossing the same position overlap in bytes.
    const OVERLAP: Overlap = Overlap::Any;

    fn anchor() -> Anchor {
        Anchor::Text
    }

    fn len(&self) -> usize {
        self.spans.len()
    }

    fn text_spans(&self) -> Option<TextSpans<'_>> {
        Some(self.spans.text_spans())
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
            if scores.iter().any(|s| !s.is_finite()) {
                return Err("a finite score for every alternative".into());
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
        // Every position needs an alternative leaving it, so there are at
        // most as many positions as alternatives; checked before anything
        // is allocated per position.
        if self.num_positions as usize > n {
            return Err(format!("{} positions for {n} alternatives", self.num_positions));
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
        self.spans.check(Self::OVERLAP)
    }
}
