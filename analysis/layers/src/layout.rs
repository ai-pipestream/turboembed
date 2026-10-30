//! 2D page and bounding-box layout spans (`turbo:layout-boxes`) for PDF,
//! OCR, and Docling ingestion through normalized text views and RAG chunks.

use turbo_analysis::{Ix, IxRange, Layer, Overlap, Span, Spans, ViewId};

/// A 2D bounding box on a 1-indexed page (`page >= 1`), with `x0 <= x1` and `y0 <= y1`.
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct PageBoundingBox {
    pub page: u32,
    pub x0: f32,
    pub y0: f32,
    pub x1: f32,
    pub y1: f32,
}

impl PageBoundingBox {
    /// Creates a page bounding box.
    pub const fn new(page: u32, x0: f32, y0: f32, x1: f32, y1: f32) -> Self {
        Self { page, x0, y0, x1, y1 }
    }

    /// Horizontal width (`x1 - x0`).
    pub fn width(&self) -> f32 {
        self.x1 - self.x0
    }

    /// Vertical height (`y1 - y0`).
    pub fn height(&self) -> f32 {
        self.y1 - self.y0
    }

    /// Returns the minimal bounding box enclosing `self` and `other` on the same page.
    ///
    /// # Panics
    ///
    /// Panics if `self.page != other.page`.
    pub fn union(self, other: Self) -> Self {
        assert_eq!(self.page, other.page, "cannot union bounding boxes across different pages");
        Self {
            page: self.page,
            x0: self.x0.min(other.x0),
            y0: self.y0.min(other.y0),
            x1: self.x1.max(other.x1),
            y1: self.y1.max(other.y1),
        }
    }
}

/// One 2D layout bounding box anchored to a text [`Span`].
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct LayoutBox {
    pub span: Span,
    pub page: u32,
    pub x0: f32,
    pub y0: f32,
    pub x1: f32,
    pub y1: f32,
}

impl LayoutBox {
    /// Creates a layout box.
    pub const fn new(span: Span, page: u32, x0: f32, y0: f32, x1: f32, y1: f32) -> Self {
        Self { span, page, x0, y0, x1, y1 }
    }

    /// Extracts the page and 2D coordinates without the text span.
    pub const fn page_box(&self) -> PageBoundingBox {
        PageBoundingBox { page: self.page, x0: self.x0, y0: self.y0, x1: self.x1, y1: self.y1 }
    }

    /// Horizontal width (`x1 - x0`).
    pub fn width(&self) -> f32 {
        self.x1 - self.x0
    }

    /// Vertical height (`y1 - y0`).
    pub fn height(&self) -> f32 {
        self.y1 - self.y0
    }
}

/// Sorted 2D page and bounding-box layout spans (`turbo:layout-boxes`).
///
/// Spans are non-empty, ordered in text order, and non-overlapping
/// ([`Overlap::None`]). Page numbers are 1-indexed (`page >= 1`), and
/// coordinates are finite `f32` values satisfying `x0 <= x1` and `y0 <= y1`.
#[derive(Clone, Debug)]
pub struct LayoutBoxes {
    pub spans: Spans<LayoutBoxes>,
    pub page: Vec<u32>,
    pub x0: Vec<f32>,
    pub y0: Vec<f32>,
    pub x1: Vec<f32>,
    pub y1: Vec<f32>,
}

impl LayoutBoxes {
    /// Creates an empty `LayoutBoxes` layer on `view`.
    pub fn new(view: ViewId) -> Self {
        Self {
            spans: Spans::new(view),
            page: Vec::new(),
            x0: Vec::new(),
            y0: Vec::new(),
            x1: Vec::new(),
            y1: Vec::new(),
        }
    }

    /// Creates an empty `LayoutBoxes` layer on `view` with pre-allocated capacity.
    pub fn with_capacity(view: ViewId, capacity: usize) -> Self {
        Self {
            spans: Spans::with_capacity(view, capacity),
            page: Vec::with_capacity(capacity),
            x0: Vec::with_capacity(capacity),
            y0: Vec::with_capacity(capacity),
            x1: Vec::with_capacity(capacity),
            y1: Vec::with_capacity(capacity),
        }
    }

    /// Appends one layout box to the layer and returns its index.
    pub fn push(&mut self, span: Span, page: u32, x0: f32, y0: f32, x1: f32, y1: f32) -> Ix<LayoutBoxes> {
        self.page.push(page);
        self.x0.push(x0);
        self.y0.push(y0);
        self.x1.push(x1);
        self.y1.push(y1);
        self.spans.push(span)
    }

    /// Appends a [`LayoutBox`] value to the layer and returns its index.
    pub fn push_box(&mut self, item: LayoutBox) -> Ix<LayoutBoxes> {
        self.push(item.span, item.page, item.x0, item.y0, item.x1, item.y1)
    }

    /// Returns the [`LayoutBox`] at `i`.
    pub fn get(&self, i: Ix<LayoutBoxes>) -> LayoutBox {
        let idx = i.get();
        LayoutBox {
            span: self.spans.get(i),
            page: self.page[idx],
            x0: self.x0[idx],
            y0: self.y0[idx],
            x1: self.x1[idx],
            y1: self.y1[idx],
        }
    }

    /// Iterates over all [`LayoutBox`] items in text order.
    pub fn iter(&self) -> impl ExactSizeIterator<Item = LayoutBox> + '_ {
        (0..self.spans.len() as u32).map(|i| self.get(Ix::new(i)))
    }

    /// Binary-searches for the contiguous index range of layout boxes whose
    /// spans overlap `span`. Returns an empty range when `span` is empty or no
    /// layout box overlaps `span`.
    pub fn overlapping_range(&self, span: Span) -> IxRange<LayoutBoxes> {
        let (start, end) = self.spans.columns();
        let lo = end.partition_point(|&e| e <= span.start);
        if span.is_empty() {
            return IxRange::new(lo as u32, lo as u32);
        }
        let hi = lo + start[lo..].partition_point(|&s| s < span.end);
        IxRange::new(lo as u32, hi as u32)
    }

    /// Iterates over all [`LayoutBox`] items whose spans overlap `span`.
    pub fn overlapping(&self, span: Span) -> impl ExactSizeIterator<Item = LayoutBox> + '_ {
        self.overlapping_range(span).iter().map(|ix| self.get(ix))
    }

    /// Binary-searches for the index of the layout box covering byte `offset`
    /// (`box.span.start <= offset < box.span.end`).
    pub fn index_at(&self, offset: u32) -> Option<Ix<LayoutBoxes>> {
        let (start, end) = self.spans.columns();
        let idx = end.partition_point(|&e| e <= offset);
        if idx < start.len() && start[idx] <= offset { Some(Ix::new(idx as u32)) } else { None }
    }

    /// Binary-searches for the [`LayoutBox`] covering byte `offset`.
    pub fn covering_at(&self, offset: u32) -> Option<LayoutBox> {
        self.index_at(offset).map(|ix| self.get(ix))
    }

    /// Computes the `(min_page, max_page)` 1-indexed page interval across all
    /// layout boxes overlapping `span`, or `None` if no layout box overlaps `span`.
    pub fn page_span(&self, span: Span) -> Option<(u32, u32)> {
        let range = self.overlapping_range(span);
        if range.is_empty() {
            return None;
        }
        let pages = &self.page[range.start.get()..range.end.get()];
        let mut min_page = pages[0];
        let mut max_page = pages[0];
        for &p in &pages[1..] {
            min_page = min_page.min(p);
            max_page = max_page.max(p);
        }
        Some((min_page, max_page))
    }

    /// Computes the per-page bounding-box union across all layout boxes
    /// overlapping `span`, sorted in ascending page order.
    pub fn union_boxes_by_page(&self, span: Span) -> Vec<PageBoundingBox> {
        let range = self.overlapping_range(span);
        let mut out: Vec<PageBoundingBox> = Vec::new();
        for ix in range.iter() {
            let i = ix.get();
            let candidate =
                PageBoundingBox { page: self.page[i], x0: self.x0[i], y0: self.y0[i], x1: self.x1[i], y1: self.y1[i] };
            match out.binary_search_by_key(&candidate.page, |b| b.page) {
                Ok(pos) => out[pos] = out[pos].union(candidate),
                Err(pos) => out.insert(pos, candidate),
            }
        }
        out
    }
}

impl Layer for LayoutBoxes {
    const ID: &'static str = "turbo:layout-boxes";
    const OVERLAP: Overlap = Overlap::None;
    positional!(LayoutBoxes);

    fn check(&self) -> Result<(), String> {
        let n = self.spans.len();
        if self.page.len() != n || self.x0.len() != n || self.y0.len() != n || self.x1.len() != n || self.y1.len() != n
        {
            return Err("a page and four coordinates for every layout box".into());
        }
        self.spans.check(Self::OVERLAP)?;
        if self.spans.iter().any(|(_, s)| s.is_empty()) {
            return Err("no empty layout box span".into());
        }
        for i in 0..n {
            if self.page[i] == 0 {
                return Err(format!("layout box {i} has page 0; pages are 1-indexed"));
            }
            let coords = [self.x0[i], self.y0[i], self.x1[i], self.y1[i]];
            if coords.iter().any(|c| !c.is_finite()) {
                return Err(format!("layout box {i} has non-finite coordinates"));
            }
            if self.x0[i] > self.x1[i] || self.y0[i] > self.y1[i] {
                return Err(format!("layout box {i} has inverted bounds"));
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn binary_search_page_span_and_per_page_union_work() {
        let mut boxes = LayoutBoxes::new(ViewId::SOURCE);
        boxes.push(Span::new(0, 10), 1, 10.0, 20.0, 100.0, 40.0);
        boxes.push(Span::new(12, 25), 1, 15.0, 45.0, 110.0, 65.0);
        boxes.push(Span::new(26, 40), 2, 10.0, 15.0, 95.0, 35.0);
        assert_eq!(boxes.check(), Ok(()));

        assert_eq!(boxes.covering_at(5).unwrap().page, 1);
        assert_eq!(boxes.covering_at(11), None);
        assert_eq!(boxes.covering_at(30).unwrap().page, 2);

        assert_eq!(boxes.page_span(Span::new(0, 25)), Some((1, 1)));
        assert_eq!(boxes.page_span(Span::new(5, 30)), Some((1, 2)));
        assert_eq!(boxes.page_span(Span::new(10, 12)), None);
        assert_eq!(boxes.page_span(Span::new(5, 5)), None);

        let union = boxes.union_boxes_by_page(Span::new(5, 30));
        assert_eq!(
            union,
            vec![PageBoundingBox::new(1, 10.0, 20.0, 110.0, 65.0), PageBoundingBox::new(2, 10.0, 15.0, 95.0, 35.0),]
        );
    }

    #[test]
    fn check_rejects_invalid_pages_coords_and_overlaps() {
        let mut zero_page = LayoutBoxes::new(ViewId::SOURCE);
        zero_page.push(Span::new(0, 5), 0, 0.0, 0.0, 10.0, 10.0);
        assert!(zero_page.check().unwrap_err().contains("1-indexed"));

        let mut inverted = LayoutBoxes::new(ViewId::SOURCE);
        inverted.push(Span::new(0, 5), 1, 20.0, 0.0, 10.0, 10.0);
        assert!(inverted.check().unwrap_err().contains("inverted"));

        let mut non_finite = LayoutBoxes::new(ViewId::SOURCE);
        non_finite.push(Span::new(0, 5), 1, f32::NAN, 0.0, 10.0, 10.0);
        assert!(non_finite.check().unwrap_err().contains("non-finite"));

        let mut overlapping = LayoutBoxes::new(ViewId::SOURCE);
        overlapping.push(Span::new(0, 6), 1, 0.0, 0.0, 10.0, 10.0);
        overlapping.push(Span::new(5, 10), 1, 0.0, 10.0, 10.0, 20.0);
        assert!(overlapping.check().is_err());
    }
}
