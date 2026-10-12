//! Where each sentence and chunk is embedded: the windows of a text's
//! subword pieces that the model encodes, one row each, and the pieces of
//! a window each item's vector is pooled over.
//!
//! A row holds the template's leading special tokens and the prompt's
//! prefix, then the pieces of one window, then the template's trailing
//! special tokens, so a window holds at most `width` pieces: `max_seq`
//! less that overhead. The policy, for one document:
//!
//! 1. An item covers the pieces whose span overlaps its own (`pieces_of`).
//!    An item that covers no piece (an empty span, or whitespace a
//!    WordPiece tokenizer drops) is not placed: it has no token to pool.
//! 2. An item of more than `width` pieces is cut into parts of `width`
//!    pieces from its first piece, the last part shorter. Each part is then
//!    placed like an item, and the item's vector is the mean of its parts'
//!    vectors weighted by their pieces (`Split`).
//! 3. The items and parts are sorted by their first piece, then their last.
//!    In that order each is placed in the current window when that window
//!    holds all of it; otherwise a new window starts at its first piece and
//!    runs `width` pieces, or to the end of the text, and becomes the
//!    current window.
//!
//! So every item of at most `width` pieces is pooled from one window that
//! holds it whole, and the windows of a document are in text order with
//! starts that only go up. Windows overlap where an item crosses the end
//! of the one before.

use std::ops::Range;

/// The layer an item is in.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum Which {
    Sentences,
    Chunks,
}

/// The vector one range is pooled into: item `item` of layer `which` in
/// document `doc` of a batch.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct Target {
    pub doc: u32,
    pub which: Which,
    pub item: u32,
}

/// Pieces `[start, end)` of one document's subwords: one row of the model.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct Window {
    pub doc: u32,
    pub start: u32,
    pub end: u32,
}

/// Pieces `[start, end)` of window `window`, counted from the window's
/// first piece, pooled into `target`. `whole` is false for one part of an
/// item longer than a window.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct Placed {
    pub window: u32,
    pub start: u32,
    pub end: u32,
    pub target: Target,
    pub whole: bool,
}

/// An item longer than a window, of `pieces` pieces: its vector is the sum
/// of its parts' vectors, each times its pieces, over `pieces`.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct Split {
    pub target: Target,
    pub pieces: u32,
}

/// The pieces an item spanning bytes `[start, end)` covers: those whose
/// span overlaps it, by binary search over the pieces' starts and ends,
/// which are in text order. A piece the item's edge cuts is covered, as
/// the chunker counts it on both sides. An empty span covers nothing.
pub fn pieces_of(starts: &[u32], ends: &[u32], start: u32, end: u32) -> Range<u32> {
    if start >= end {
        let at = ends.partition_point(|&e| e <= start) as u32;
        return at..at;
    }
    // The first piece that ends after the item starts, and one past the
    // last that starts before it ends.
    let lo = ends.partition_point(|&e| e <= start);
    let hi = starts.partition_point(|&s| s < end).max(lo);
    lo as u32..hi as u32
}

/// The windows and placements of a batch of documents, rebuilt for each
/// batch in buffers that are kept.
pub struct Plan {
    width: u32,
    /// One document's items and parts before they are placed: first
    /// piece, end piece, target, whole.
    pending: Vec<(u32, u32, Target, bool)>,
    pub windows: Vec<Window>,
    /// In window order: the placements of one window are a run.
    pub placed: Vec<Placed>,
    pub split: Vec<Split>,
}

impl Plan {
    /// A plan for windows of at most `width` pieces, at least 1.
    pub fn new(width: u32) -> Plan {
        assert!(width > 0, "a window holds at least one piece");
        Plan { width, pending: Vec::new(), windows: Vec::new(), placed: Vec::new(), split: Vec::new() }
    }

    pub fn width(&self) -> u32 {
        self.width
    }

    pub fn clear(&mut self) {
        self.pending.clear();
        self.windows.clear();
        self.placed.clear();
        self.split.clear();
    }

    /// Queues an item covering `pieces` for the next `place`. An item
    /// with no pieces is not queued; one longer than a window is queued
    /// as its parts.
    pub fn item(&mut self, target: Target, pieces: Range<u32>) {
        let (lo, hi) = (pieces.start, pieces.end);
        if lo >= hi {
            return;
        }
        if hi - lo <= self.width {
            self.pending.push((lo, hi, target, true));
            return;
        }
        self.split.push(Split { target, pieces: hi - lo });
        let mut a = lo;
        while a < hi {
            let b = hi.min(a + self.width);
            self.pending.push((a, b, target, false));
            a = b;
        }
    }

    /// Places every queued item of document `doc`, which has `pieces`
    /// pieces, in windows of its own.
    pub fn place(&mut self, doc: u32, pieces: u32) {
        self.pending.sort_unstable_by_key(|&(lo, hi, _, _)| (lo, hi));
        // The current window: its index, first piece and end.
        let mut cur: Option<(u32, u32, u32)> = None;
        for &(lo, hi, target, whole) in &self.pending {
            debug_assert!(hi <= pieces, "an item past the document's {pieces} pieces");
            let (w, ws) = match cur {
                Some((w, ws, we)) if lo >= ws && hi <= we => (w, ws),
                _ => {
                    let w = self.windows.len() as u32;
                    let we = pieces.min(lo + self.width);
                    self.windows.push(Window { doc, start: lo, end: we });
                    cur = Some((w, lo, we));
                    (w, lo)
                }
            };
            self.placed.push(Placed { window: w, start: lo - ws, end: hi - ws, target, whole });
        }
        self.pending.clear();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn t(item: u32) -> Target {
        Target { doc: 0, which: Which::Sentences, item }
    }

    /// Ten pieces of three bytes, a space between each: "abc abc ...".
    fn spans() -> (Vec<u32>, Vec<u32>) {
        ((0..10).map(|i| i * 4).collect(), (0..10).map(|i| i * 4 + 3).collect())
    }

    #[test]
    fn an_item_covers_every_piece_its_span_overlaps() {
        let (s, e) = spans();
        assert_eq!(pieces_of(&s, &e, 0, 3), 0..1);
        assert_eq!(pieces_of(&s, &e, 0, 7), 0..2);
        // Starting in the space before a piece, ending inside one.
        assert_eq!(pieces_of(&s, &e, 3, 9), 1..3);
        // An edge that cuts a piece covers it.
        assert_eq!(pieces_of(&s, &e, 5, 6), 1..2);
        assert_eq!(pieces_of(&s, &e, 0, 40), 0..10);
    }

    #[test]
    fn whitespace_and_empty_spans_cover_no_piece() {
        let (s, e) = spans();
        assert_eq!(pieces_of(&s, &e, 3, 4).len(), 0);
        assert_eq!(pieces_of(&s, &e, 5, 5).len(), 0, "an empty span inside a piece");
        assert_eq!(pieces_of(&s, &e, 39, 50).len(), 0, "past the last piece");
        assert_eq!(pieces_of(&[], &[], 0, 5).len(), 0);
    }

    #[test]
    fn pieces_sharing_a_span_are_covered_together() {
        // Two byte-level pieces of one four-byte character, then a word.
        let (s, e) = (vec![0, 0, 5], vec![4, 4, 8]);
        assert_eq!(pieces_of(&s, &e, 0, 4), 0..2);
        assert_eq!(pieces_of(&s, &e, 5, 8), 2..3);
    }

    #[test]
    fn items_fill_a_window_before_another_starts() {
        let mut p = Plan::new(4);
        p.item(t(0), 0..2);
        p.item(t(1), 2..4);
        p.item(t(2), 4..7);
        p.place(0, 7);
        assert_eq!(p.windows, [Window { doc: 0, start: 0, end: 4 }, Window { doc: 0, start: 4, end: 7 }]);
        let where_: Vec<_> = p.placed.iter().map(|x| (x.window, x.start, x.end, x.target.item)).collect();
        assert_eq!(where_, [(0, 0, 2, 0), (0, 2, 4, 1), (1, 0, 3, 2)]);
    }

    #[test]
    fn an_item_across_a_window_end_starts_the_next_window() {
        let mut p = Plan::new(4);
        p.item(t(0), 0..3);
        p.item(t(1), 3..6);
        p.place(0, 6);
        // The second item would cross the first window's end at 4.
        assert_eq!(p.windows[1], Window { doc: 0, start: 3, end: 6 });
        assert_eq!((p.placed[1].window, p.placed[1].start, p.placed[1].end), (1, 0, 3));
    }

    #[test]
    fn nested_items_share_the_window_of_the_outer_one() {
        // A chunk over two sentences, given after them.
        let mut p = Plan::new(8);
        p.item(t(0), 0..3);
        p.item(t(1), 3..6);
        p.item(Target { which: Which::Chunks, ..t(0) }, 0..6);
        p.place(0, 6);
        assert_eq!(p.windows.len(), 1);
        assert_eq!(p.placed.len(), 3);
    }

    #[test]
    fn an_item_longer_than_a_window_is_placed_in_parts() {
        let mut p = Plan::new(4);
        p.item(t(0), 1..11);
        p.place(0, 12);
        assert_eq!(p.split, [Split { target: t(0), pieces: 10 }]);
        let parts: Vec<_> = p.placed.iter().map(|x| (p.windows[x.window as usize].start + x.start, x.whole)).collect();
        assert_eq!(parts, [(1, false), (5, false), (9, false)]);
        let lens: Vec<_> = p.placed.iter().map(|x| x.end - x.start).collect();
        assert_eq!(lens, [4, 4, 2]);
    }

    #[test]
    fn a_part_shares_a_window_with_items_it_fits_beside() {
        let mut p = Plan::new(4);
        p.item(t(0), 0..6); // parts 0..4 and 4..6
        p.item(t(1), 4..5);
        p.place(0, 8);
        assert_eq!(p.windows.len(), 2);
        assert_eq!(p.windows[1], Window { doc: 0, start: 4, end: 8 });
        assert!(p.placed.iter().filter(|x| x.window == 1).count() == 2);
    }

    #[test]
    fn documents_get_windows_of_their_own() {
        let mut p = Plan::new(4);
        p.item(t(0), 0..2);
        p.place(0, 2);
        p.item(Target { doc: 1, ..t(0) }, 0..2);
        p.place(1, 2);
        assert_eq!(p.windows.iter().map(|w| w.doc).collect::<Vec<_>>(), [0, 1]);
        p.clear();
        assert!(p.windows.is_empty() && p.placed.is_empty() && p.split.is_empty());
    }
}
