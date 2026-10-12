//! Where each sentence and chunk is embedded: the windows of a text's
//! subword pieces that the model encodes, one row each, and the pieces of
//! a window each item's vector is pooled over.
//!
//! A row holds the template's leading special tokens and the prompt's
//! prefix, then the pieces of one window, then the template's trailing
//! special tokens, so a window holds at most `width` pieces: `max_seq`
//! less that overhead. Every window runs `width` pieces from its first,
//! or to the end of the text. The policy, for one document:
//!
//! 1. An item covers the pieces whose span overlaps its own (`pieces_of`).
//!    An item that covers no piece (an empty span, or whitespace a
//!    WordPiece tokenizer drops) is not placed: it has no token to pool.
//! 2. The windows are laid out by the finest layer the step has: the
//!    sentences when it has them, else the chunks. Of that layer, the
//!    items of at most `width` pieces are sorted by their first piece,
//!    then their last, and in that order each goes in the current window
//!    when that window holds all of it; otherwise a new window starts at
//!    its first piece and becomes the current window.
//! 3. Every other item (the chunks beside sentences, and any item of more
//!    than `width` pieces) is placed once those windows exist: first the
//!    long items of the layer that laid them out, then the others, each
//!    group in the same order. When a window holds it whole it is pooled there, in the
//!    window with the lowest first piece that does. Otherwise it is cut
//!    into parts over the windows: from its first piece `p`, the part runs
//!    to the end of the window holding `p` that reaches furthest, or to
//!    the item's end if that comes first, and the next part starts there.
//!    Only when no window holds `p` is a new one added, starting at `p`.
//!    Each of its pieces is in one part, and its vector is the mean of its
//!    parts' vectors weighted by their pieces (`Split`).
//!
//! So the windows a step lays out depend on its finest layer alone, never
//! on the longer items: a sentence gets the same window with or without
//! the chunks beside it, and a chunk adds a window only over pieces no
//! sentence's window holds.

use std::fmt;
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
/// item pooled over several windows.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct Placed {
    pub window: u32,
    pub start: u32,
    pub end: u32,
    pub target: Target,
    pub whole: bool,
}

/// An item pooled in parts, of `pieces` pieces: its vector is the sum of
/// its parts' vectors, each times its pieces, over `pieces`.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct Split {
    pub target: Target,
    pub pieces: u32,
}

/// An item queued for a document reaches past the document's pieces.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct PastTheText {
    pub target: Target,
    /// One past the item's last piece.
    pub end: u32,
    /// The document's pieces.
    pub pieces: u32,
}

impl fmt::Display for PastTheText {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{:?} item {} ends at piece {}, past the document's {} pieces",
            self.target.which, self.target.item, self.end, self.pieces
        )
    }
}

/// The pieces an item spanning bytes `[start, end)` covers: those whose
/// span overlaps it, by binary search over the pieces' starts and ends,
/// which are in text order. A piece the item's edge cuts is covered, as
/// the chunker counts it on both sides. A piece of no bytes at offset `x`
/// is covered when `start <= x < end`, so one at the item's first byte is
/// covered like one inside it, and one at its end is not. An empty span
/// covers nothing, whatever pieces sit at its offset.
pub fn pieces_of(starts: &[u32], ends: &[u32], start: u32, end: u32) -> Range<u32> {
    if start >= end {
        let at = ends.partition_point(|&e| e <= start) as u32;
        return at..at;
    }
    // The first piece that ends after the item starts, or that has no
    // bytes and sits at its start. Pieces ending at the item's start come
    // in start order, so those of no bytes come after the others.
    let mut lo = ends.partition_point(|&e| e < start);
    while lo < ends.len() && ends[lo] == start && starts[lo] < start {
        lo += 1;
    }
    // One past the last piece that starts before the item ends.
    let hi = starts.partition_point(|&s| s < end).max(lo);
    lo as u32..hi as u32
}

/// The windows and placements of a batch of documents, rebuilt for each
/// batch in buffers that are kept.
pub struct Plan {
    width: u32,
    /// One document's items that lay out its windows, before they are
    /// placed: first piece, end piece, target.
    layout: Vec<(u32, u32, Target)>,
    /// One document's other items, placed once its windows exist: the
    /// long items of the layer that laid them out (`true`) before the
    /// others, so those never depend on the others.
    rest: Vec<(bool, u32, u32, Target)>,
    /// The document's windows by first piece: (first piece, window).
    /// Every window runs `width` pieces or to the text's end, so a later
    /// first piece never has an earlier end.
    by_start: Vec<(u32, u32)>,
    /// One item's parts while it is covered: window, first piece, end.
    parts: Vec<(u32, u32, u32)>,
    pub windows: Vec<Window>,
    /// In window order: the placements of one window are a run.
    pub placed: Vec<Placed>,
    pub split: Vec<Split>,
}

impl Plan {
    /// A plan for windows of at most `width` pieces, at least 1.
    pub fn new(width: u32) -> Plan {
        assert!(width > 0, "a window holds at least one piece");
        Plan {
            width,
            layout: Vec::new(),
            rest: Vec::new(),
            by_start: Vec::new(),
            parts: Vec::new(),
            windows: Vec::new(),
            placed: Vec::new(),
            split: Vec::new(),
        }
    }

    pub fn width(&self) -> u32 {
        self.width
    }

    pub fn clear(&mut self) {
        self.layout.clear();
        self.rest.clear();
        self.by_start.clear();
        self.windows.clear();
        self.placed.clear();
        self.split.clear();
    }

    /// Queues an item of the layer that lays out the windows. One with no
    /// pieces is not queued; one longer than a window takes no part in
    /// laying them out and is placed with the other items.
    pub fn layout(&mut self, target: Target, pieces: Range<u32>) {
        let (lo, hi) = (pieces.start, pieces.end);
        if lo >= hi {
            return;
        }
        if hi - lo <= self.width {
            self.layout.push((lo, hi, target));
        } else {
            self.rest.push((true, lo, hi, target));
        }
    }

    /// Queues an item placed once the windows exist. One with no pieces
    /// is not queued.
    pub fn item(&mut self, target: Target, pieces: Range<u32>) {
        if pieces.start < pieces.end {
            self.rest.push((false, pieces.start, pieces.end, target));
        }
    }

    /// Places every queued item of document `doc`, which has `pieces`
    /// pieces, in windows of its own. An item past the last piece fails
    /// the document and nothing of it is placed.
    pub fn place(&mut self, doc: u32, pieces: u32) -> Result<(), PastTheText> {
        let layout = self.layout.iter().map(|&(_, hi, t)| (hi, t));
        let past = layout.chain(self.rest.iter().map(|&(_, _, hi, t)| (hi, t))).find(|&(hi, _)| hi > pieces);
        if let Some((end, target)) = past {
            self.layout.clear();
            self.rest.clear();
            return Err(PastTheText { target, end, pieces });
        }
        let first = self.placed.len();
        self.by_start.clear();
        self.layout.sort_by_key(|&(lo, hi, _)| (lo, hi));
        // The current window: its index, first piece and end.
        let mut cur: Option<(u32, u32, u32)> = None;
        for &(lo, hi, target) in &self.layout {
            let (w, ws) = match cur {
                Some((w, ws, we)) if lo >= ws && hi <= we => (w, ws),
                _ => {
                    let w = self.windows.len() as u32;
                    let we = pieces.min(lo + self.width);
                    self.windows.push(Window { doc, start: lo, end: we });
                    // First pieces only go up here, as the items are sorted
                    // and an item at the current first piece always fits.
                    self.by_start.push((lo, w));
                    cur = Some((w, lo, we));
                    (w, lo)
                }
            };
            self.placed.push(Placed { window: w, start: lo - ws, end: hi - ws, target, whole: true });
        }
        self.layout.clear();

        self.rest.sort_by_key(|&(lays_out, lo, hi, _)| (!lays_out, lo, hi));
        for k in 0..self.rest.len() {
            let (_, lo, hi, target) = self.rest[k];
            if let Some(w) = self.holding(lo, hi) {
                let ws = self.windows[w as usize].start;
                self.placed.push(Placed { window: w, start: lo - ws, end: hi - ws, target, whole: true });
                continue;
            }
            self.cover(doc, pieces, lo, hi);
            if let [(w, a, b)] = self.parts[..] {
                let ws = self.windows[w as usize].start;
                self.placed.push(Placed { window: w, start: a - ws, end: b - ws, target, whole: true });
                continue;
            }
            self.split.push(Split { target, pieces: hi - lo });
            for &(w, a, b) in &self.parts {
                let ws = self.windows[w as usize].start;
                self.placed.push(Placed { window: w, start: a - ws, end: b - ws, target, whole: false });
            }
        }
        self.rest.clear();
        // Windows added for the other items come after the laid out ones,
        // so their placements are put back in window order.
        self.placed[first..].sort_by_key(|x| x.window);
        Ok(())
    }

    /// The window with the lowest first piece of those holding pieces
    /// `[lo, hi)` whole. Ends never go down as first pieces go up, so the
    /// windows ending at or after `hi` are the last ones, and the first of
    /// them holds the item when any does.
    fn holding(&self, lo: u32, hi: u32) -> Option<u32> {
        let k = self.by_start.partition_point(|&(_, w)| self.windows[w as usize].end < hi);
        let &(start, w) = self.by_start.get(k)?;
        (start <= lo).then_some(w)
    }

    /// Cuts pieces `[lo, hi)` into parts over the document's windows,
    /// into `parts`, adding a window wherever none holds a piece.
    fn cover(&mut self, doc: u32, pieces: u32, lo: u32, hi: u32) {
        self.parts.clear();
        let mut p = lo;
        while p < hi {
            // The window holding p that reaches furthest is the one with the
            // last first piece at or before p, when it reaches past p.
            let k = self.by_start.partition_point(|&(s, _)| s <= p);
            let held = k.checked_sub(1).map(|k| self.by_start[k].1).filter(|&w| self.windows[w as usize].end > p);
            let w = match held {
                Some(w) => w,
                None => {
                    let w = self.windows.len() as u32;
                    self.windows.push(Window { doc, start: p, end: pieces.min(p + self.width) });
                    self.by_start.insert(k, (p, w));
                    w
                }
            };
            let end = self.windows[w as usize].end.min(hi);
            self.parts.push((w, p, end));
            p = end;
        }
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
    fn a_piece_of_no_bytes_is_covered_from_the_items_start_up_to_its_end() {
        // "abc" 0..3, a piece of no bytes at 4, "def" 4..7, "ghi" 8..11.
        let (s, e) = (vec![0, 4, 4, 8], vec![3, 4, 7, 11]);
        assert_eq!(pieces_of(&s, &e, 4, 7), 1..3, "at the item's first byte");
        assert_eq!(pieces_of(&s, &e, 3, 7), 1..3, "inside the item");
        assert_eq!(pieces_of(&s, &e, 0, 7), 0..3, "inside the item");
        assert_eq!(pieces_of(&s, &e, 0, 4), 0..1, "at the item's end");
        assert_eq!(pieces_of(&s, &e, 5, 7), 2..3);
        assert_eq!(pieces_of(&s, &e, 4, 4).len(), 0, "an empty item at its offset");
        // A piece ending where the item starts is not covered; the piece
        // of no bytes after it is.
        let (s, e) = (vec![0, 4, 4], vec![4, 4, 8]);
        assert_eq!(pieces_of(&s, &e, 4, 8), 1..3);
        assert_eq!(pieces_of(&s, &e, 4, 4).len(), 0);
    }

    fn c(item: u32) -> Target {
        Target { doc: 0, which: Which::Chunks, item }
    }

    /// Each placement as (window, first piece in the text, end, item,
    /// whole), for the items of `which`.
    fn where_(p: &Plan, which: Which) -> Vec<(u32, u32, u32, u32, bool)> {
        let mut v: Vec<_> = p
            .placed
            .iter()
            .filter(|x| x.target.which == which)
            .map(|x| {
                let ws = p.windows[x.window as usize].start;
                (x.window, ws + x.start, ws + x.end, x.target.item, x.whole)
            })
            .collect();
        v.sort_by_key(|x| (x.3, x.1));
        v
    }

    #[test]
    fn items_fill_a_window_before_another_starts() {
        let mut p = Plan::new(4);
        p.layout(t(0), 0..2);
        p.layout(t(1), 2..4);
        p.layout(t(2), 4..7);
        p.place(0, 7).unwrap();
        assert_eq!(p.windows, [Window { doc: 0, start: 0, end: 4 }, Window { doc: 0, start: 4, end: 7 }]);
        let where_: Vec<_> = p.placed.iter().map(|x| (x.window, x.start, x.end, x.target.item)).collect();
        assert_eq!(where_, [(0, 0, 2, 0), (0, 2, 4, 1), (1, 0, 3, 2)]);
    }

    #[test]
    fn an_item_across_a_window_end_starts_the_next_window() {
        let mut p = Plan::new(4);
        p.layout(t(0), 0..3);
        p.layout(t(1), 3..6);
        p.place(0, 6).unwrap();
        // The second item would cross the first window's end at 4.
        assert_eq!(p.windows[1], Window { doc: 0, start: 3, end: 6 });
        assert_eq!((p.placed[1].window, p.placed[1].start, p.placed[1].end), (1, 0, 3));
    }

    #[test]
    fn nested_items_share_the_window_of_the_outer_one() {
        // A chunk over two sentences.
        let mut p = Plan::new(8);
        p.layout(t(0), 0..3);
        p.layout(t(1), 3..6);
        p.item(c(0), 0..6);
        p.place(0, 6).unwrap();
        assert_eq!(p.windows.len(), 1);
        assert_eq!(p.placed.len(), 3);
        assert!(p.split.is_empty());
    }

    #[test]
    fn an_item_longer_than_a_window_is_placed_in_parts() {
        let mut p = Plan::new(4);
        p.layout(t(0), 1..11);
        p.place(0, 12).unwrap();
        assert_eq!(p.split, [Split { target: t(0), pieces: 10 }]);
        let parts: Vec<_> = p.placed.iter().map(|x| (p.windows[x.window as usize].start + x.start, x.whole)).collect();
        assert_eq!(parts, [(1, false), (5, false), (9, false)]);
        let lens: Vec<_> = p.placed.iter().map(|x| x.end - x.start).collect();
        assert_eq!(lens, [4, 4, 2]);
    }

    #[test]
    fn a_long_item_is_cut_over_the_laid_out_windows() {
        let mut p = Plan::new(4);
        p.layout(t(0), 0..6); // longer than a window
        p.layout(t(1), 4..5);
        p.place(0, 8).unwrap();
        // The short sentence lays out 4..8; the long one adds 0..4 and
        // takes 4..6 from the window already there.
        assert_eq!(p.windows, [Window { doc: 0, start: 4, end: 8 }, Window { doc: 0, start: 0, end: 4 }]);
        assert_eq!(where_(&p, Which::Sentences), [(1, 0, 4, 0, false), (0, 4, 6, 0, false), (0, 4, 5, 1, true)]);
        // Placements come in window order.
        assert!(p.placed.windows(2).all(|w| w[0].window <= w[1].window));
    }

    /// Sentences over every piece lay out the windows; chunks short and
    /// long, one over the whole text, add none, and the sentences are
    /// placed exactly as without them.
    #[test]
    fn chunks_add_no_window_where_the_sentences_hold_every_piece() {
        let sentences = [0..3, 3..6, 6..9, 9..12, 12..14];
        let alone = {
            let mut p = Plan::new(6);
            for (i, r) in sentences.iter().enumerate() {
                p.layout(t(i as u32), r.clone());
            }
            p.place(0, 14).unwrap();
            p
        };
        let mut p = Plan::new(6);
        for (i, r) in sentences.iter().enumerate() {
            p.layout(t(i as u32), r.clone());
        }
        for (i, r) in [0..14, 0..6, 6..12, 3..9, 2..8, 12..14].into_iter().enumerate() {
            p.item(c(i as u32), r);
        }
        p.place(0, 14).unwrap();
        assert_eq!(p.windows, alone.windows);
        assert_eq!(p.windows.len(), 3);
        assert_eq!(where_(&p, Which::Sentences), where_(&alone, Which::Sentences));
        let chunks = where_(&p, Which::Chunks);
        // The whole text: windows 0..6, 6..12, 12..14 in turn.
        assert_eq!(&chunks[..3], [(0, 0, 6, 0, false), (1, 6, 12, 0, false), (2, 12, 14, 0, false)]);
        // A chunk one window holds is pooled there whole.
        assert!(chunks.contains(&(0, 0, 6, 1, true)) && chunks.contains(&(1, 6, 12, 2, true)));
        // A short chunk across a window's end is cut at it.
        assert!(chunks.contains(&(0, 3, 6, 3, false)) && chunks.contains(&(1, 6, 9, 3, false)));
        // Each split item's parts cover each of its pieces once.
        for s in &p.split {
            let n: u32 = p.placed.iter().filter(|x| x.target == s.target).map(|x| x.end - x.start).sum();
            assert_eq!(n, s.pieces, "{:?}", s.target);
        }
    }

    /// A chunk over pieces no sentence's window holds adds windows there,
    /// starting at the first such piece.
    #[test]
    fn a_chunk_over_a_gap_between_sentences_adds_a_window() {
        let mut p = Plan::new(4);
        p.layout(t(0), 0..2);
        p.layout(t(1), 10..12);
        p.item(c(0), 0..12);
        p.item(c(1), 1..3);
        p.place(0, 12).unwrap();
        let laid = [Window { doc: 0, start: 0, end: 4 }, Window { doc: 0, start: 10, end: 12 }];
        assert_eq!(p.windows[..2], laid);
        assert_eq!(p.windows[2..], [Window { doc: 0, start: 4, end: 8 }, Window { doc: 0, start: 8, end: 12 }]);
        assert_eq!(
            where_(&p, Which::Chunks),
            [(0, 0, 4, 0, false), (2, 4, 8, 0, false), (3, 8, 12, 0, false), (0, 1, 3, 1, true)]
        );
    }

    #[test]
    fn an_item_two_windows_hold_goes_in_the_earlier() {
        let mut p = Plan::new(4);
        p.layout(t(0), 0..3);
        p.layout(t(1), 2..5);
        p.item(c(0), 2..3);
        p.item(c(1), 3..5);
        p.place(0, 6).unwrap();
        assert_eq!(p.windows.len(), 2);
        assert_eq!(where_(&p, Which::Chunks), [(0, 2, 3, 0, true), (1, 3, 5, 1, true)]);
    }

    #[test]
    fn an_item_past_the_text_fails_the_document() {
        let mut p = Plan::new(4);
        p.layout(t(0), 0..2);
        p.item(c(3), 1..9);
        let e = p.place(0, 8).unwrap_err();
        assert_eq!(e, PastTheText { target: c(3), end: 9, pieces: 8 });
        assert!(e.to_string().contains("ends at piece 9, past the document's 8 pieces"), "{e}");
        assert!(p.windows.is_empty() && p.placed.is_empty());
        // The plan goes on with the next document.
        p.layout(t(0), 0..2);
        p.place(1, 2).unwrap();
        assert_eq!(p.windows, [Window { doc: 1, start: 0, end: 2 }]);
    }

    #[test]
    fn documents_get_windows_of_their_own() {
        let mut p = Plan::new(4);
        p.layout(t(0), 0..2);
        p.place(0, 2).unwrap();
        p.layout(Target { doc: 1, ..t(0) }, 0..2);
        p.item(Target { doc: 1, ..c(0) }, 0..2);
        p.place(1, 2).unwrap();
        assert_eq!(p.windows.iter().map(|w| w.doc).collect::<Vec<_>>(), [0, 1]);
        p.clear();
        assert!(p.windows.is_empty() && p.placed.is_empty() && p.split.is_empty());
    }
}
