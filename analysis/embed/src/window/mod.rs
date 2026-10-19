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
//!
//! The units of `Levels` go through the same rules one unit at a time
//! (`crate::levels`): each unit lays out windows over its own pieces
//! alone, with `place` bounded by its last piece, and its children are
//! the other items.

use std::fmt;
use std::ops::Range;

use turbo_analysis_layers::VectorKind;

/// The layer an item is in.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum Which {
    Sentences,
    Chunks,
    /// The units of `Levels`, for the vector of this kind.
    Levels(VectorKind),
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
                    let we = pieces.min(lo.saturating_add(self.width));
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
                    self.windows.push(Window { doc, start: p, end: pieces.min(p.saturating_add(self.width)) });
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
mod tests;
