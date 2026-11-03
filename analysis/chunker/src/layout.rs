//! 2D page and bounding-box layout boundary signals (`turbo:layout-boundaries`)
//! and per-chunk layout citations (`turbo:chunk-citations`).

use turbo_analysis::{Anchor, Annotator, Bound, Error, Fnv1a, Ix, Layer, Refs};
use turbo_analysis_layers::{Boundaries, Chunks, Cut, LayoutBoxes, PageBoundingBox, Sentences};

use crate::{STRUCTURE, Weight};

/// The named [`Boundaries`] instance written by [`LayoutBoundaries`].
pub const LAYOUT: &str = "layout";

/// DP boundary cost adjustments (in [`Boundaries::UNIT`]s) at visual layout
/// transitions between adjacent sentences.
///
/// Negative values reward cutting at a transition (reducing the boundary cost);
/// positive values penalize cutting.
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct LayoutCosts {
    /// Boundary cost added at a page transition (`left.page != right.page`).
    pub page_break_bonus: i32,
    /// Boundary cost added at a same-page column transition (`right.x0 > left.x1`
    /// and `right.y0 < left.y0`).
    pub column_jump_bonus: i32,
    /// Boundary cost added at a same-page vertical gap meeting or exceeding
    /// `visual_gap_threshold` (`right.y0 - left.y1 >= visual_gap_threshold`).
    pub visual_gap_bonus: i32,
    /// Minimum vertical gap (`right.y0 - left.y1`) on the same page required to
    /// trigger `visual_gap_bonus`.
    pub visual_gap_threshold: f32,
}

impl Default for LayoutCosts {
    fn default() -> Self {
        Self { page_break_bonus: -400, column_jump_bonus: -250, visual_gap_bonus: -200, visual_gap_threshold: 18.0 }
    }
}

impl LayoutCosts {
    /// Folds the layout cost parameters into an FNV-1a configuration hash.
    pub fn config_hash(&self) -> u64 {
        let mut h = Fnv1a::new();
        for x in [
            u64::from(self.page_break_bonus as u32),
            u64::from(self.column_jump_bonus as u32),
            u64::from(self.visual_gap_bonus as u32),
            u64::from(self.visual_gap_threshold.to_bits()),
        ] {
            h.u64(x);
        }
        h.finish()
    }
}

/// Computes layout boundary costs after each sentence from `layout`.
///
/// When `layout` is `None`, every sentence boundary receives cost `0` and
/// [`Cut::Sentence`].
pub fn layout_boundaries(
    sentences: &Sentences,
    layout: Option<&LayoutBoxes>,
    costs: LayoutCosts,
) -> Result<Boundaries, String> {
    sentences.check()?;
    let n = sentences.spans.len();
    let Some(layout) = layout else {
        return Ok(Boundaries { cost: vec![0; n], reason: vec![Cut::Sentence; n] });
    };
    layout.check()?;
    if sentences.spans.view() != layout.spans.view() {
        return Err("sentences and layout boxes are in different views".into());
    }
    if !costs.visual_gap_threshold.is_finite() || costs.visual_gap_threshold < 0.0 {
        return Err("visual_gap_threshold must be finite and non-negative".into());
    }
    let mut cost = vec![0i32; n];
    let mut reason = vec![Cut::Sentence; n];
    for g in 0..n.saturating_sub(1) {
        let left_span = sentences.spans.get(Ix::new(g as u32));
        let right_span = sentences.spans.get(Ix::new((g + 1) as u32));
        let left_range = layout.overlapping_range(left_span);
        let right_range = layout.overlapping_range(right_span);
        if left_range.is_empty() || right_range.is_empty() {
            continue;
        }
        let left = layout.get(Ix::new((left_range.end.get() - 1) as u32));
        let right = layout.get(right_range.start);
        if left.page != right.page {
            cost[g] = costs.page_break_bonus;
            if costs.page_break_bonus != 0 {
                reason[g] = Cut::Section;
            }
        } else if right.x0 > left.x1 && right.y0 < left.y0 {
            cost[g] = costs.column_jump_bonus;
            if costs.column_jump_bonus != 0 {
                reason[g] = Cut::Paragraph;
            }
        } else if (right.y0 - left.y1) >= costs.visual_gap_threshold {
            cost[g] = costs.visual_gap_bonus;
            if costs.visual_gap_bonus != 0 {
                reason[g] = Cut::Paragraph;
            }
        }
    }
    Ok(Boundaries { cost, reason })
}

/// Annotator that writes layout boundary costs (`turbo:layout-boundaries`) to
/// [`Boundaries`] (typically bound to [`LAYOUT`] via [`LayoutBoundaries::bound`]).
#[derive(Copy, Clone, Debug, Default, PartialEq)]
pub struct LayoutBoundaries {
    pub costs: LayoutCosts,
}

impl LayoutBoundaries {
    /// Binds this step to write the [`LAYOUT`] instance of [`Boundaries`].
    pub fn bound(self) -> Bound<Self> {
        Bound::new(self).write(0, LAYOUT)
    }
}

impl Annotator for LayoutBoundaries {
    const ID: &'static str = "turbo:layout-boundaries";
    type Reads = (Sentences, Option<LayoutBoxes>);
    type Writes = (Boundaries,);

    fn annotate(&self, (sentences, layout): Refs<'_, Self>) -> Result<(Boundaries,), Error> {
        let out = layout_boundaries(sentences, layout, self.costs)
            .map_err(|detail| Error::Annotator { annotator: Self::ID, detail })?;
        Ok((out,))
    }

    fn config_hash(&self) -> u64 {
        self.costs.config_hash()
    }
}

/// Weighs a base [`Boundaries`] layer (e.g. structure) and a layout
/// [`Boundaries`] layer at each sentence gap:
/// `(base * wb + layout * wl) / 1000`, rounded down with `div_euclid(1000)`
/// and saturated to `i32`.
pub fn weigh_with_layout(base: &Boundaries, layout: &Boundaries, (wb, wl): (Weight, Weight)) -> Option<Boundaries> {
    let n = base.cost.len();
    if layout.cost.len() != n || base.reason.len() != n || layout.reason.len() != n {
        return None;
    }
    let cost = (0..n)
        .map(|i| {
            let sum = i128::from(base.cost[i]) * i128::from(wb.0) + i128::from(layout.cost[i]) * i128::from(wl.0);
            sum.div_euclid(1000).clamp(i128::from(i32::MIN), i128::from(i32::MAX)) as i32
        })
        .collect();
    let reason = (0..n)
        .map(|i| {
            let (b, l) = (base.reason[i], layout.reason[i]);
            if b == Cut::Sentence && wl.0 != 0 && l != Cut::Sentence { l } else { b }
        })
        .collect();
    Some(Boundaries { cost, reason })
}

/// Combines a base [`Boundaries`] signal (`STRUCTURE` by default) with the
/// optional [`LAYOUT`] boundary signal into the default [`Boundaries`] instance.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct WeighLayoutBoundaries {
    pub base: Weight,
    pub layout: Weight,
}

impl Default for WeighLayoutBoundaries {
    fn default() -> Self {
        Self { base: Weight::ONE, layout: Weight::ONE }
    }
}

impl WeighLayoutBoundaries {
    /// Binds this step to read `STRUCTURE` and `LAYOUT`.
    pub fn bound(self) -> Bound<Self> {
        Bound::new(self).read(1, STRUCTURE).read(2, LAYOUT)
    }
}

impl Annotator for WeighLayoutBoundaries {
    const ID: &'static str = "turbo:weigh-layout-boundaries";
    type Reads = (Sentences, Boundaries, Option<Boundaries>);
    type Writes = (Boundaries,);

    fn annotate(&self, (sentences, base, layout): Refs<'_, Self>) -> Result<(Boundaries,), Error> {
        let fail = |detail: String| Error::Annotator { annotator: Self::ID, detail };
        sentences.check().map_err(fail)?;
        let n = sentences.spans.len();
        if base.cost.len() != n {
            return Err(fail(format!("{} base boundaries for {n} sentences", base.cost.len())));
        }
        let zero;
        let layout_ref = match layout {
            Some(l) => {
                if l.cost.len() != n {
                    return Err(fail(format!("{} layout boundaries for {n} sentences", l.cost.len())));
                }
                l
            }
            None => {
                zero = Boundaries { cost: vec![0; n], reason: vec![Cut::Sentence; n] };
                &zero
            }
        };
        let wl = if layout.is_some() { self.layout } else { Weight(0) };
        let out = weigh_with_layout(base, layout_ref, (self.base, wl))
            .ok_or_else(|| fail("a reason for every boundary".into()))?;
        Ok((out,))
    }

    fn config_hash(&self) -> u64 {
        let mut h = Fnv1a::new();
        for w in [self.base.0, self.layout.0] {
            h.u64(u64::from(w as u32));
        }
        h.finish()
    }
}

/// Layout citation for a single chunk: its 1-indexed `(start_page, end_page)`
/// page interval and per-page bounding-box unions in ascending page order.
#[derive(Clone, Debug, PartialEq)]
pub struct ChunkCitation {
    pub chunk: Ix<Chunks>,
    pub page_span: Option<(u32, u32)>,
    pub pages: Vec<PageBoundingBox>,
}

/// Per-chunk 2D page and bounding-box citations (`turbo:chunk-citations`),
/// aligned 1-to-1 with [`Chunks`].
#[derive(Clone, Debug, PartialEq)]
pub struct ChunkCitations {
    /// 1-indexed first page of each chunk (`0` when `has_page_span[i]` is `false`).
    pub page_start: Vec<u32>,
    /// 1-indexed last page of each chunk (`0` when `has_page_span[i]` is `false`).
    pub page_end: Vec<u32>,
    /// Whether chunk `i` overlapped at least one [`LayoutBox`](turbo_analysis_layers::LayoutBox).
    pub has_page_span: Vec<bool>,
    /// CSR row offsets into `boxes` (`len == chunks.len() + 1`).
    pub box_row: Vec<u32>,
    /// Per-page union bounding boxes for each chunk, sorted by page within each row.
    pub boxes: Vec<PageBoundingBox>,
}

impl Default for ChunkCitations {
    fn default() -> Self {
        Self {
            page_start: Vec::new(),
            page_end: Vec::new(),
            has_page_span: Vec::new(),
            box_row: vec![0],
            boxes: Vec::new(),
        }
    }
}

impl ChunkCitations {
    /// Returns the `(start_page, end_page)` 1-indexed page interval for `chunk`.
    pub fn page_span(&self, chunk: Ix<Chunks>) -> Option<(u32, u32)> {
        let i = chunk.get();
        if *self.has_page_span.get(i)? { Some((self.page_start[i], self.page_end[i])) } else { None }
    }

    /// Returns the per-page bounding-box unions for `chunk`.
    pub fn page_boxes(&self, chunk: Ix<Chunks>) -> &[PageBoundingBox] {
        let i = chunk.get();
        if i + 1 >= self.box_row.len() {
            return &[];
        }
        let (a, b) = (self.box_row[i] as usize, self.box_row[i + 1] as usize);
        &self.boxes[a..b]
    }

    /// Returns the [`ChunkCitation`] for `chunk`.
    pub fn get(&self, chunk: Ix<Chunks>) -> ChunkCitation {
        ChunkCitation { chunk, page_span: self.page_span(chunk), pages: self.page_boxes(chunk).to_vec() }
    }

    /// Iterates over all chunk citations in chunk order.
    pub fn iter(&self) -> impl ExactSizeIterator<Item = ChunkCitation> + '_ {
        (0..self.has_page_span.len() as u32).map(|i| self.get(Ix::new(i)))
    }
}

impl Layer for ChunkCitations {
    const ID: &'static str = "turbo:chunk-citations";

    fn anchor() -> Anchor {
        Anchor::aligned::<Chunks>()
    }

    fn len(&self) -> usize {
        self.has_page_span.len()
    }

    fn check(&self) -> Result<(), String> {
        let n = self.has_page_span.len();
        if self.page_start.len() != n || self.page_end.len() != n {
            return Err("a page_start and page_end for every chunk".into());
        }
        if self.box_row.len() != n + 1 {
            return Err(format!("{} box row offsets for {n} chunks", self.box_row.len()));
        }
        if self.box_row.first() != Some(&0) {
            return Err("chunk citation box row offsets do not start at 0".into());
        }
        if *self.box_row.last().unwrap_or(&0) as usize != self.boxes.len() {
            return Err("chunk citation box row offsets do not end at boxes.len()".into());
        }
        if !self.box_row.windows(2).all(|w| w[0] <= w[1]) {
            return Err("chunk citation box row offsets are not monotonic".into());
        }
        for i in 0..n {
            let (a, b) = (self.box_row[i] as usize, self.box_row[i + 1] as usize);
            let row_boxes = &self.boxes[a..b];
            for pbox in row_boxes {
                if pbox.page == 0 {
                    return Err(format!("chunk {i} citation box has page 0; pages are 1-indexed"));
                }
                if ![pbox.x0, pbox.y0, pbox.x1, pbox.y1].into_iter().all(f32::is_finite) {
                    return Err(format!("chunk {i} citation box has non-finite coordinates"));
                }
                if pbox.x0 > pbox.x1 || pbox.y0 > pbox.y1 {
                    return Err(format!("chunk {i} citation box has inverted bounds"));
                }
            }
            if !row_boxes.windows(2).all(|w| w[0].page < w[1].page) {
                return Err(format!("chunk {i} citation boxes are not in strictly ascending page order"));
            }
            if self.has_page_span[i] {
                if self.page_start[i] == 0 || self.page_start[i] > self.page_end[i] {
                    return Err(format!(
                        "chunk {i} has invalid page_span {}..={}",
                        self.page_start[i], self.page_end[i]
                    ));
                }
                let Some(first) = row_boxes.first() else {
                    return Err(format!("chunk {i} has page_span but no page boxes"));
                };
                let last = row_boxes.last().unwrap();
                if first.page != self.page_start[i] || last.page != self.page_end[i] {
                    return Err(format!("chunk {i} page_span does not match its page boxes"));
                }
            } else if self.page_start[i] != 0 || self.page_end[i] != 0 || !row_boxes.is_empty() {
                return Err(format!("chunk {i} without page_span must have zero pages and no boxes"));
            }
        }
        Ok(())
    }
}

/// Computes [`ChunkCitations`] for every chunk in `chunks` from `layout`.
pub fn chunk_citations(chunks: &Chunks, layout: Option<&LayoutBoxes>) -> Result<ChunkCitations, String> {
    chunks.check()?;
    let n = chunks.spans.len();
    let Some(layout) = layout else {
        return Ok(ChunkCitations {
            page_start: vec![0; n],
            page_end: vec![0; n],
            has_page_span: vec![false; n],
            box_row: vec![0; n + 1],
            boxes: Vec::new(),
        });
    };
    layout.check()?;
    if chunks.spans.view() != layout.spans.view() {
        return Err("chunks and layout boxes are in different views".into());
    }
    let mut page_start = Vec::with_capacity(n);
    let mut page_end = Vec::with_capacity(n);
    let mut has_page_span = Vec::with_capacity(n);
    let mut box_row = Vec::with_capacity(n + 1);
    let mut boxes = Vec::new();
    box_row.push(0);
    for (_, span) in chunks.spans.iter() {
        let union = layout.union_boxes_by_page(span);
        if let (Some(first), Some(last)) = (union.first(), union.last()) {
            page_start.push(first.page);
            page_end.push(last.page);
            has_page_span.push(true);
            boxes.extend_from_slice(&union);
        } else {
            page_start.push(0);
            page_end.push(0);
            has_page_span.push(false);
        }
        box_row.push(boxes.len() as u32);
    }
    let out = ChunkCitations { page_start, page_end, has_page_span, box_row, boxes };
    out.check()?;
    Ok(out)
}

/// Annotator that builds [`ChunkCitations`] (`turbo:chunk-citations`) from
/// [`Chunks`] and optional [`LayoutBoxes`].
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq)]
pub struct ChunkCitationAnnotator;

impl Annotator for ChunkCitationAnnotator {
    const ID: &'static str = "turbo:chunk-citations";
    type Reads = (Chunks, Option<LayoutBoxes>);
    type Writes = (ChunkCitations,);

    fn annotate(&self, (chunks, layout): Refs<'_, Self>) -> Result<(ChunkCitations,), Error> {
        let out = chunk_citations(chunks, layout).map_err(|detail| Error::Annotator { annotator: Self::ID, detail })?;
        Ok((out,))
    }
}

#[cfg(test)]
mod tests;
