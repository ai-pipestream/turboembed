//! Window planning and most-interior window selection for per-token
//! contextual vectors (`#709`).
//!
//! When an input exceeds a single encoder window (`pieces > width`), it is
//! covered by overlapping windows of at most `width` subword pieces. Each
//! subword piece `t in 0..pieces` takes its contextual vector from the
//! window covering `t` where `t` is most interior:
//!
//! `interior_distance(t, win) = min(t - win.start, win.end - 1 - t)`
//!
//! Ties in `interior_distance` are broken deterministically in favor of the
//! earlier window.

use super::{PastTheText, Plan, Window};

/// Where one subword piece is extracted: the window of its document where
/// it is most interior (greatest distance to the window's edges, ties to
/// the earlier window).
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct TokenPlacement {
    /// The piece index in the document (`0..pieces`).
    pub piece: u32,
    /// The chosen window index in `windows`.
    pub window: u32,
    /// Offset of `piece` inside `windows[window]` (`piece - window.start`).
    pub offset: u32,
    /// Distance from `piece` to the nearest boundary of `windows[window]`:
    /// `(piece - window.start).min(window.end - 1 - piece)`.
    pub interior_distance: u32,
}

/// Default sliding-window stride for token vectors when none is specified
/// (`stride == 0`): half of `width`, at least 1.
pub fn default_token_stride(width: u32) -> u32 {
    (width / 2).max(1)
}

/// Distance of `piece` to the nearest edge of `window`, or `None` when
/// `piece` lies outside `[window.start, window.end)`.
pub fn interior_distance(piece: u32, window: Window) -> Option<u32> {
    if piece < window.start || piece >= window.end {
        return None;
    }
    let left = piece - window.start;
    let right = window.end - 1 - piece;
    Some(left.min(right))
}

/// Selects the window index in `windows` (belonging to `doc` and covering
/// `piece`) where `piece` has the greatest [`interior_distance`], breaking
/// ties in favor of the earlier window. Returns `(window_index, distance)`.
pub fn most_interior_window(doc: u32, piece: u32, windows: &[Window]) -> Option<(u32, u32)> {
    let mut best: Option<(u32, u32)> = None;
    for (idx, &win) in windows.iter().enumerate() {
        if win.doc != doc {
            continue;
        }
        let Some(dist) = interior_distance(piece, win) else { continue };
        if best.is_none_or(|(_, best_dist)| dist > best_dist) {
            best = Some((idx as u32, dist));
        }
    }
    best
}

/// Lays out deterministic sliding windows of at most `width` pieces over
/// `0..pieces` for document `doc`, stepping by `stride` pieces (`stride == 0`
/// uses [`default_token_stride`]).
pub fn plan_sliding_windows(doc: u32, pieces: u32, width: u32, stride: u32) -> Vec<Window> {
    assert!(width > 0, "a window holds at least one piece");
    if pieces == 0 {
        return Vec::new();
    }
    if pieces <= width {
        return vec![Window { doc, start: 0, end: pieces }];
    }
    let step = if stride == 0 { default_token_stride(width) } else { stride.clamp(1, width) };
    let mut windows = Vec::new();
    let mut start = 0u32;
    loop {
        let end = pieces.min(start.saturating_add(width));
        windows.push(Window { doc, start, end });
        if end == pieces {
            break;
        }
        start = start.saturating_add(step);
    }
    windows
}

/// Assigns each piece `0..pieces` of `doc` to the window in `windows` where
/// it is most interior (ties to the earlier window).
pub fn place_tokens_in_windows(doc: u32, pieces: u32, windows: &[Window]) -> Vec<TokenPlacement> {
    let mut out = Vec::with_capacity(pieces as usize);
    for piece in 0..pieces {
        let (window, dist) =
            most_interior_window(doc, piece, windows).expect("every piece in 0..pieces must be covered by a window");
        let win = windows[window as usize];
        out.push(TokenPlacement { piece, window, offset: piece - win.start, interior_distance: dist });
    }
    out
}

impl Plan {
    /// Plans windows for document `doc` (`pieces` subword pieces) and assigns
    /// each piece `0..pieces` to the window where it is most interior (greatest
    /// distance to the window's edges, ties to the earlier window).
    ///
    /// When layout items (such as sentences) have been queued on `self`, they
    /// lay out windows first and any uncovered spans are covered with sliding
    /// windows of `(self.width(), stride)`. When no layout items were queued,
    /// sliding windows of `(self.width(), stride)` cover `0..pieces` directly.
    pub fn place_tokens(&mut self, doc: u32, pieces: u32, stride: u32) -> Result<Vec<TokenPlacement>, PastTheText> {
        let layout = self.layout.iter().map(|&(_, hi, t)| (hi, t));
        let past = layout.chain(self.rest.iter().map(|&(_, _, hi, t)| (hi, t))).find(|&(hi, _)| hi > pieces);
        if let Some((end, target)) = past {
            self.layout.clear();
            self.rest.clear();
            return Err(PastTheText { target, end, pieces });
        }
        if pieces == 0 {
            self.layout.clear();
            self.rest.clear();
            return Ok(Vec::new());
        }

        let had_layout = !self.layout.is_empty() || !self.rest.is_empty();
        if !had_layout {
            self.windows.extend(plan_sliding_windows(doc, pieces, self.width, stride));
        } else {
            let doc_win_start = self.windows.len();
            self.layout.sort_by_key(|&(lo, hi, _)| (lo, hi));
            let mut cur: Option<(u32, u32)> = None;
            for &(lo, hi, _) in &self.layout {
                let fits = cur.is_some_and(|(ws, we)| lo >= ws && hi <= we);
                if !fits {
                    let we = pieces.min(lo.saturating_add(self.width));
                    self.windows.push(Window { doc, start: lo, end: we });
                    cur = Some((lo, we));
                }
            }
            self.layout.clear();
            self.rest.clear();

            // Cover any pieces not yet covered by a sentence window using
            // sliding windows of `(self.width, stride)`.
            let step = if stride == 0 { default_token_stride(self.width) } else { stride.clamp(1, self.width) };
            let mut p = 0u32;
            while p < pieces {
                if let Some(end) =
                    self.windows[doc_win_start..].iter().filter(|w| w.start <= p && p < w.end).map(|w| w.end).max()
                {
                    p = end;
                    continue;
                }
                let next_cov = self.windows[doc_win_start..]
                    .iter()
                    .filter(|w| w.start > p)
                    .map(|w| w.start)
                    .min()
                    .unwrap_or(pieces);
                let mut start = p;
                loop {
                    let end = pieces.min(start.saturating_add(self.width));
                    self.windows.push(Window { doc, start, end });
                    if end >= next_cov {
                        break;
                    }
                    start = start.saturating_add(step);
                }
                p = next_cov;
            }
            self.windows[doc_win_start..].sort_by_key(|w| (w.start, w.end));
            self.windows.dedup_by(|a, b| a.doc == doc && b.doc == doc && a.start == b.start && a.end == b.end);
        }

        Ok(place_tokens_in_windows(doc, pieces, &self.windows))
    }
}
