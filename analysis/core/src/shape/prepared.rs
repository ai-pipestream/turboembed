use super::*;

/// A document view's borrowed span columns, checked once for repeated
/// segmentation. The document and columns remain immutably borrowed.
/// Ordered, non-overlapping layers use binary-search window bounds;
/// arbitrary overlapping/unordered layers retain the checked scan.
pub struct PreparedSegments<'a> {
    doc: &'a Document,
    view: ViewId,
    layers: Vec<PreparedLayer<'a>>,
}

struct PreparedLayer<'a> {
    spans: TextSpans<'a>,
    ordered: bool,
}

impl PreparedLayer<'_> {
    fn items(&self, window: Span) -> Range<usize> {
        if !self.ordered {
            return 0..self.spans.start.len();
        }
        ordered_range(self.spans.start, self.spans.end, window)
    }
}

/// Both ordered item envelopes and one item's checked fragment table
/// have monotone starts and ends. Keep empty points at either closed
/// edge, excluding non-empty spans that merely touch a window edge.
pub(super) fn ordered_range(start: &[u32], end: &[u32], window: Span) -> Range<usize> {
    let lo = start.partition_point(|&s| s < window.start).min(end.partition_point(|&e| e <= window.start));
    let hi = start.partition_point(|&s| s < window.end).max(end.partition_point(|&e| e <= window.end));
    lo..hi.max(lo)
}

impl Document {
    /// Validates every supplied column once, including endpoints outside
    /// any particular window. Keep the returned handle while paging:
    /// its ordered layers then visit only the window's candidate items.
    /// No caller-supplied "already validated" flag is trusted.
    pub fn prepare_segments<'a>(
        &'a self,
        view: ViewId,
        layers: &[(KeyId, TextSpans<'a>)],
    ) -> Result<PreparedSegments<'a>, Error> {
        let text = view_text(self, view)?;
        let mut checked = Vec::with_capacity(layers.len());
        for &(key, spans) in layers {
            validate_layer(text, view, key, spans)?;
            checked.push(PreparedLayer { spans, ordered: spans.is_ordered() });
        }
        Ok(PreparedSegments { doc: self, view, layers: checked })
    }
}

impl PreparedSegments<'_> {
    /// The same fragment/zero-width membership as
    /// [`Document::combined_segments`], without revalidating columns.
    /// Ordered layer bounds cost O(log N) per layer, followed by the
    /// candidate items/fragments and sweep/output work. Unordered or
    /// overlapping layers scan their items, preserving the same result.
    pub fn combined_segments(&self, window: &DocumentWindow) -> Result<WindowShape, Error> {
        if window.view != self.view {
            return Err(Error::Bounds {
                layer: view_key(),
                view: window.view,
                detail: format!("prepared spans are in view {}, the window is in view {}", self.view.0, window.view.0),
            });
        }
        validate_window(self.doc, window)?;
        Ok(segments::segment(
            window,
            self.layers.iter().enumerate().map(|(i, layer)| (i, layer.spans, layer.items(window.span))),
        ))
    }
}
