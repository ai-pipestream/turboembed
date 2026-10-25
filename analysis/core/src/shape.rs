//! Document shape: window pagination over one view, and the combined
//! segmentation of several text-anchored layers inside a window.
//!
//! A window is how a consumer pages through a long view: it asks for
//! `len` bytes from `from`, and gets a character-aligned span, the page
//! it falls on when the view is tiled by `len`-byte pages, and whether
//! the request ran past the end. The combined segmentation of a window
//! is the finest partition of its span such that every segment is
//! covered by a constant set of layer items: the sweep-line product of
//! the given layers' fragments. Heatmap lanes, inspectors, and
//! side-by-side comparisons render from that one partition instead of
//! re-deriving overlaps per layer pair.
//!
//! Membership is fragment-exact: a discontinuous item is a member of a
//! segment only where one of its fragments covers the segment, never
//! across its gaps. A zero-width item appears as its own zero-length
//! segment at its position, between its neighbours, and is a member of
//! no non-empty segment.

use std::collections::{BTreeMap, BTreeSet};
use std::ops::Range;

use crate::Document;
use crate::doc::View;
use crate::error::Error;
use crate::layer::{Key, KeyId};
use crate::offsets::{CodePointSpan, Utf16Span};
use crate::span::{Overlap, Span, TextSpans};
use crate::text::ViewId;

/// One page-sized window over a view, aligned to character boundaries.
///
/// `page` and `pages` count the tiling of the view by windows of the
/// requested length starting at byte 0: `page` is the tile the snapped
/// start falls in, `pages` is the tile count (at least 1, so an empty
/// view is page 0 of 1). `clamped` records that `from + len` ran past
/// the end of the view, so the span is shorter than requested.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct DocumentWindow {
    pub view: ViewId,
    pub span: Span,
    pub page: usize,
    pub pages: usize,
    pub clamped: bool,
}

impl DocumentWindow {
    /// The window's text. The window must come from `doc` (or name a
    /// span of the same view text): the span is sliced unchecked past
    /// the usual bounds checks of `str` indexing.
    pub fn text<'a>(&self, doc: &'a Document) -> &'a str {
        &doc.text(self.view)[self.span.range()]
    }

    /// The window's span in UTF-16 units, via the view's offset table.
    pub fn utf16_span(&self, doc: &Document) -> Utf16Span {
        doc.offsets(self.view).span_to_utf16_typed(doc.text(self.view), self.span)
    }

    /// The window's span in code points, via the view's offset table.
    pub fn code_point_span(&self, doc: &Document) -> CodePointSpan {
        doc.offsets(self.view).span_to_code_point_typed(doc.text(self.view), self.span)
    }
}

/// One segment of a window's combined segmentation: a maximal run of
/// bytes covered by one constant set of layer items. `members` indexes
/// [`WindowShape::members`]. A zero-length segment carries the
/// zero-width items at its position.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ShapeSegment {
    pub span: Span,
    pub members: Range<usize>,
}

/// One layer item covering a segment. `layer` is the index of the layer
/// in the slice passed to [`Document::combined_segments`]; `item` is
/// the item index within that layer.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub struct ShapeMember {
    pub layer: usize,
    pub item: u32,
}

/// The combined segmentation of one window: segments in text order,
/// covering the window span exactly (zero-length segments sit between
/// their neighbours without adding coverage), with each segment's
/// member set stored once in the flat `members` table, sorted by
/// `(layer, item)`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct WindowShape {
    pub window: DocumentWindow,
    pub segments: Vec<ShapeSegment>,
    pub members: Vec<ShapeMember>,
}

impl WindowShape {
    /// The members of one segment.
    pub fn members_of(&self, segment: &ShapeSegment) -> &[ShapeMember] {
        &self.members[segment.members.clone()]
    }
}

fn view_key() -> KeyId {
    Key::<View>::DEFAULT.id()
}

fn snap_down(text: &str, mut at: u32) -> u32 {
    while at > 0 && !text.is_char_boundary(at as usize) {
        at -= 1;
    }
    at
}

fn snap_up(text: &str, mut at: u32) -> u32 {
    let len = text.len() as u32;
    while at < len && !text.is_char_boundary(at as usize) {
        at += 1;
    }
    at
}

impl Document {
    /// A window of `len` bytes from byte `from` of `view`, snapped to
    /// character boundaries: the start snaps down to the boundary at or
    /// before `from`, the end snaps down to the boundary at or before
    /// `from + len` (so the span is never longer than `len` unless one
    /// character alone is wider, in which case that character is
    /// included whole). The end is clamped to the view's length.
    ///
    /// Refuses a view the document does not have, a zero `len`, and a
    /// `from` past the end of the view, each as [`Error::Bounds`]
    /// naming the view and the offending numbers.
    pub fn window(&self, view: ViewId, from: u32, len: u32) -> Result<DocumentWindow, Error> {
        let layer = view_key();
        if view.0 as usize >= self.view_count() {
            return Err(Error::Bounds {
                layer,
                view,
                detail: format!("the document has {} views, view {} does not exist", self.view_count(), view.0),
            });
        }
        if len == 0 {
            return Err(Error::Bounds { layer, view, detail: "window length is zero".into() });
        }
        let text = self.text(view);
        let text_len = text.len() as u32;
        if from > text_len {
            return Err(Error::Bounds {
                layer,
                view,
                detail: format!("window starts at byte {from}, past the end of the view at {text_len} bytes"),
            });
        }
        let start = snap_down(text, from);
        let wanted_end = (from as u64 + len as u64).min(text_len as u64) as u32;
        let mut end = snap_down(text, wanted_end);
        if end == start && start < text_len {
            end = snap_up(text, start + 1);
        }
        let clamped = from as u64 + len as u64 > text_len as u64;
        let pages = (text_len as u64).div_ceil(len as u64).max(1) as usize;
        let page = if text_len == 0 { 0 } else { (start / len) as usize };
        Ok(DocumentWindow { view, span: Span { start, end }, page, pages, clamped })
    }

    /// The combined segmentation of `window` under `layers`, each given
    /// as its key (for error messages) and its borrowed span columns
    /// (from [`crate::Layer::text_spans`] or [`crate::Spans::text_spans`]).
    ///
    /// Every fragment edge inside the window is a cut point; between
    /// consecutive cut points the set of covering items is constant, and
    /// adjacent runs with the same set merge into one segment. An item
    /// covers a segment only through its fragments, so the gap inside a
    /// discontinuous item is segmented by whatever else covers it.
    /// Fragments straddling the window edges are clipped to the window.
    /// Zero-width items become zero-length segments at their position
    /// (one per position, holding every zero-width item there), ordered
    /// between the segments around them; segments either side of one
    /// never merge across it.
    ///
    /// Refuses, as [`Error::Bounds`] naming the layer and item: span
    /// columns of another view, structurally inconsistent columns
    /// (checked with [`Overlap::Any`], so any overlap rule's layer is
    /// accepted), an endpoint past the end of the view text, and an
    /// endpoint off a character boundary. A window naming a view the
    /// document does not have, or a span outside that view, is refused
    /// the same way. Runs in O((F + B) log F) for F fragments meeting
    /// the window and B cut points.
    pub fn combined_segments(
        &self,
        window: &DocumentWindow,
        layers: &[(KeyId, TextSpans<'_>)],
    ) -> Result<WindowShape, Error> {
        let (view, ws, we) = (window.view, window.span.start, window.span.end);
        if view.0 as usize >= self.view_count() {
            return Err(Error::Bounds {
                layer: view_key(),
                view,
                detail: format!("the document has {} views, view {} does not exist", self.view_count(), view.0),
            });
        }
        let text = self.text(view);
        let text_len = text.len() as u32;
        if ws > we || we > text_len || !text.is_char_boundary(ws as usize) || !text.is_char_boundary(we as usize) {
            return Err(Error::Bounds {
                layer: view_key(),
                view,
                detail: format!("window span {ws}..{we} is not a character-aligned span of the {text_len}-byte view"),
            });
        }

        // Clipped non-empty fragments as sweep events, plus zero-width
        // items by position.
        let mut starts: BTreeMap<u32, Vec<ShapeMember>> = BTreeMap::new();
        let mut ends: BTreeMap<u32, Vec<ShapeMember>> = BTreeMap::new();
        let mut cuts: Vec<u32> = vec![ws, we];
        let mut zero: BTreeMap<u32, BTreeSet<ShapeMember>> = BTreeMap::new();
        for (layer_ix, (key, ts)) in layers.iter().enumerate() {
            let bounds = |detail: String| Error::Bounds { layer: *key, view, detail };
            if ts.view != view {
                return Err(bounds(format!("spans are in view {}, the window is in view {}", ts.view.0, view.0)));
            }
            ts.check(Overlap::Any).map_err(bounds)?;
            for (cols, what) in
                [(ts.start, "start"), (ts.end, "end"), (ts.frag_start, "fragment start"), (ts.frag_end, "fragment end")]
            {
                for (i, &at) in cols.iter().enumerate() {
                    if at > text_len {
                        return Err(bounds(format!(
                            "{what} {i} is byte {at}, past the end of the view at {text_len} bytes"
                        )));
                    }
                    if !text.is_char_boundary(at as usize) {
                        return Err(bounds(format!("{what} {i} is byte {at}, inside a character of the view")));
                    }
                }
            }
            for item in 0..ts.start.len() {
                let member = ShapeMember { layer: layer_ix, item: item as u32 };
                if ts.frag_item_end.is_empty() {
                    collect(ts.start[item], ts.end[item], member, ws, we, &mut starts, &mut ends, &mut cuts, &mut zero);
                } else {
                    let r = ts.item_frag_range(item);
                    for f in r {
                        collect(
                            ts.frag_start[f],
                            ts.frag_end[f],
                            member,
                            ws,
                            we,
                            &mut starts,
                            &mut ends,
                            &mut cuts,
                            &mut zero,
                        );
                    }
                }
            }
        }
        cuts.sort_unstable();
        cuts.dedup();

        let mut shape = WindowShape { window: *window, segments: Vec::new(), members: Vec::new() };
        let mut active: BTreeSet<ShapeMember> = BTreeSet::new();
        let mut zero_between = false;
        for (i, &at) in cuts.iter().enumerate() {
            if let Some(set) = zero.get(&at) {
                let from = shape.members.len();
                shape.members.extend(set.iter().copied());
                shape
                    .segments
                    .push(ShapeSegment { span: Span { start: at, end: at }, members: from..shape.members.len() });
                zero_between = true;
            }
            if let Some(gone) = ends.get(&at) {
                for m in gone {
                    active.remove(m);
                }
            }
            if let Some(fresh) = starts.get(&at) {
                for m in fresh {
                    active.insert(*m);
                }
            }
            if i + 1 < cuts.len() {
                let next = cuts[i + 1];
                let candidate: Vec<ShapeMember> = active.iter().copied().collect();
                let extends = !zero_between
                    && shape.segments.last().is_some_and(|last| {
                        last.span.end == at && shape.members[last.members.clone()] == candidate[..]
                    });
                if extends {
                    shape.segments.last_mut().expect("checked above").span.end = next;
                } else {
                    let from = shape.members.len();
                    shape.members.extend(candidate);
                    shape
                        .segments
                        .push(ShapeSegment { span: Span { start: at, end: next }, members: from..shape.members.len() });
                }
                zero_between = false;
            }
        }
        Ok(shape)
    }
}

#[allow(clippy::too_many_arguments)]
fn collect(
    start: u32,
    end: u32,
    member: ShapeMember,
    ws: u32,
    we: u32,
    starts: &mut BTreeMap<u32, Vec<ShapeMember>>,
    ends: &mut BTreeMap<u32, Vec<ShapeMember>>,
    cuts: &mut Vec<u32>,
    zero: &mut BTreeMap<u32, BTreeSet<ShapeMember>>,
) {
    if start == end {
        if (ws..=we).contains(&start) {
            zero.entry(start).or_default().insert(member);
            cuts.push(start);
        }
        return;
    }
    if end <= ws || start >= we {
        return;
    }
    let (s, e) = (start.max(ws), end.min(we));
    starts.entry(s).or_default().push(member);
    ends.entry(e).or_default().push(member);
    cuts.push(s);
    cuts.push(e);
}
