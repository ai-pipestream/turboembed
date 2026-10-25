//! Slicing span columns to a byte range of their view: items are
//! clipped to the range and rebased to its start, items with
//! nothing in the range drop out, and the caller gets the
//! old-to-new index remap relation layers need to re-point their
//! endpoints. Used by graph slicing
//! ([`crate::export::DocumentGraph::slice_source_span`]).

use super::Span;
use super::spans::Spans;

/// The intersection of `s` with `range`, rebased to `range.start`,
/// or `None` when they do not touch. An empty span whose point lies
/// in the closed range survives as an empty span; a non-empty span
/// that only touches a range boundary does not.
fn clip(s: Span, range: Span) -> Option<Span> {
    if s.is_empty() {
        if range.start <= s.start && s.start <= range.end {
            let at = s.start - range.start;
            return Some(Span::new(at, at));
        }
        return None;
    }
    let start = s.start.max(range.start);
    let end = s.end.min(range.end);
    if start >= end {
        return None;
    }
    Some(Span::new(start - range.start, end - range.start))
}

/// Clips every item's fragments to `range`, rebases them, coalesces
/// fragments whose gap fell outside the range, and drops items
/// with nothing left. Returns the surviving items' fragments and,
/// per old item index, its new index or `None` when dropped.
pub(crate) fn clip_items(items: &[Vec<Span>], range: Span) -> (Vec<Vec<Span>>, Vec<Option<u32>>) {
    let mut out: Vec<Vec<Span>> = Vec::new();
    let mut remap: Vec<Option<u32>> = Vec::with_capacity(items.len());
    for fragments in items {
        let mut kept: Vec<Span> = Vec::new();
        for &fragment in fragments {
            if let Some(clipped) = clip(fragment, range) {
                if let Some(last) = kept.last_mut()
                    && last.end >= clipped.start
                    && !last.is_empty()
                {
                    last.end = last.end.max(clipped.end);
                } else if kept.last().is_none_or(|last| *last != clipped) {
                    kept.push(clipped);
                }
            }
        }
        if kept.is_empty() {
            remap.push(None);
        } else {
            remap.push(Some(out.len() as u32));
            out.push(kept);
        }
    }
    (out, remap)
}

impl<L> Spans<L> {
    /// This column sliced to `range` of its view's text: items
    /// clipped and rebased to start at `range.start`, discontinuous
    /// fragments clipped and coalesced where their gap fell outside
    /// the range, items with nothing in the range dropped. Returns
    /// the sliced column (same view) and, per old item index, its
    /// new index or `None` when dropped, so index columns and
    /// relation endpoints can be re-pointed (or dropped) to match.
    /// An empty span survives exactly when its point lies in the
    /// closed range.
    pub fn sliced(&self, range: Span) -> (Spans<L>, Vec<Option<u32>>) {
        let items: Vec<Vec<Span>> = self.iter().map(|(ix, _)| self.fragments(ix).collect()).collect();
        let (clipped, remap) = clip_items(&items, range);
        let mut out = Spans::with_capacity(self.view(), clipped.len());
        for fragments in &clipped {
            match &fragments[..] {
                [one] => {
                    out.push(*one);
                }
                many => {
                    out.try_push_fragments(many).expect("clipped fragments stay separated after coalescing");
                }
            }
        }
        (out, remap)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::text::ViewId;

    struct L;

    fn spans(items: &[(u32, u32)]) -> Spans<L> {
        let mut s = Spans::new(ViewId::SOURCE);
        for &(a, b) in items {
            s.push(Span::new(a, b));
        }
        s
    }

    #[test]
    fn sliced_clips_rebases_and_remaps() {
        let s = spans(&[(0, 3), (3, 3), (4, 9), (9, 12), (12, 12), (13, 20)]);
        let (out, remap) = s.sliced(Span::new(3, 12));
        let got: Vec<Span> = out.iter().map(|(_, x)| x).collect();
        assert_eq!(got, vec![Span::new(0, 0), Span::new(1, 6), Span::new(6, 9), Span::new(9, 9)]);
        assert_eq!(remap, vec![None, Some(0), Some(1), Some(2), Some(3), None]);
    }

    #[test]
    fn sliced_coalesces_fragments_across_a_clipped_gap() {
        let mut s = Spans::<L>::new(ViewId::SOURCE);
        s.try_push_fragments(&[Span::new(0, 4), Span::new(6, 10)]).unwrap();
        s.push(Span::new(20, 22));
        let (out, remap) = s.sliced(Span::new(2, 8));
        assert_eq!(remap, vec![Some(0), None]);
        let frags: Vec<Span> = out.fragments(crate::span::Ix::new(0)).collect();
        assert_eq!(frags, vec![Span::new(0, 2), Span::new(4, 6)]);
        // A range covering only the gap drops the item.
        let (out, remap) = s.sliced(Span::new(4, 6));
        assert_eq!(out.len(), 0);
        assert_eq!(remap, vec![None, None]);
    }
}
