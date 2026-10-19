//! Boilerplate: which items of a level (structure blocks, sentences,
//! chunks) are page furniture rather than the document's own content.

use std::marker::PhantomData;

use turbo_analysis::{Anchor, Ix, Layer, LayerName, Span, TextSpans};

/// What kind of boilerplate an item is.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum BoilerplateKind {
    /// A block repeated at the top of pages or documents: a masthead, a
    /// page header.
    Header,
    /// A block repeated at the bottom: a page footer, a copyright line.
    Footer,
    /// Menus and link lists repeated across documents.
    Navigation,
    /// A long notice repeated across documents: a legal disclaimer, a
    /// confidentiality notice.
    Disclaimer,
    /// A span repeated inside one document: a "back to top" link, a
    /// repeated table caption.
    Template,
}

impl BoilerplateKind {
    /// Every kind, in wire order.
    pub const ALL: [BoilerplateKind; 5] = [
        BoilerplateKind::Header,
        BoilerplateKind::Footer,
        BoilerplateKind::Navigation,
        BoilerplateKind::Disclaimer,
        BoilerplateKind::Template,
    ];

    /// The kind's wire code.
    pub fn code(self) -> u8 {
        self as u8
    }

    /// The kind of a wire code, if it names one.
    pub fn from_code(code: u8) -> Option<BoilerplateKind> {
        BoilerplateKind::ALL.get(code as usize).copied()
    }
}

/// The boilerplate flags of the items of `L`, aligned to them: item i's
/// kind (`None` for content) and a strength in `[0, 1]`. What the
/// strength measures is the writer's: a detector writes how widely the
/// item repeats, a projector the share of an item's bytes that lie in
/// boilerplate it projects from. Readers that exclude boilerplate use
/// `kind`; readers that down-weight it use `score`.
pub struct Boilerplate<L> {
    pub kind: Vec<Option<BoilerplateKind>>,
    pub score: Vec<f32>,
    pub _of: PhantomData<fn() -> L>,
}

impl<L> Boilerplate<L> {
    pub fn new(kind: Vec<Option<BoilerplateKind>>, score: Vec<f32>) -> Boilerplate<L> {
        Boilerplate { kind, score, _of: PhantomData }
    }

    /// Whether item `i` is boilerplate.
    pub fn is_boilerplate(&self, i: Ix<L>) -> bool {
        self.kind[i.get()].is_some()
    }

    /// The boilerplate items, in item order.
    pub fn flagged(&self) -> impl Iterator<Item = Ix<L>> + '_ {
        self.kind.iter().enumerate().filter(|(_, k)| k.is_some()).map(|(i, _)| Ix::new(i as u32))
    }
}

impl<L> Clone for Boilerplate<L> {
    fn clone(&self) -> Self {
        Boilerplate::new(self.kind.clone(), self.score.clone())
    }
}

impl<L> std::fmt::Debug for Boilerplate<L> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Boilerplate").field("kind", &self.kind).field("score", &self.score).finish()
    }
}

impl<L: Layer> Layer for Boilerplate<L> {
    const ID: &'static str = "turbo:boilerplate";
    const OF: Option<&'static LayerName> = Some(L::NAME);

    fn anchor() -> Anchor {
        Anchor::aligned::<L>()
    }

    fn len(&self) -> usize {
        self.kind.len()
    }

    fn check(&self) -> Result<(), String> {
        if self.score.len() != self.kind.len() {
            return Err(format!("{} scores for {} items", self.score.len(), self.kind.len()));
        }
        if let Some(i) = self.score.iter().position(|s| !(0.0..=1.0).contains(s)) {
            return Err(format!("item {i}: a score of {} outside [0, 1]", self.score[i]));
        }
        Ok(())
    }
}

/// The bytes of a view inside boilerplate items: disjoint intervals in
/// text order, each with the kind and score of the item that covers it.
/// Where flagged items overlap, the innermost decides (the one starting
/// last, then ending first, then the later item), since a block nested
/// in another says more about its own bytes; elsewhere every byte keeps
/// its own item's kind and score, touching items included.
pub struct Covered {
    pub spans: Vec<Span>,
    pub kind: Vec<BoilerplateKind>,
    pub score: Vec<f32>,
}

impl Covered {
    /// The flagged items' spans (fragments, for a discontinuous item),
    /// cut where the deciding item changes. Refused when the flags are
    /// not aligned to the items or do not pass their check (a score that
    /// is not a number or is outside `[0, 1]`). A sweep over
    /// the span edges: time is the number of spans times its logarithm.
    pub fn new<S: Layer>(ts: &TextSpans<'_>, flags: &Boilerplate<S>) -> Result<Covered, String> {
        if flags.kind.len() != ts.start.len() {
            return Err(format!("{} boilerplate flags for {} items", flags.kind.len(), ts.start.len()));
        }
        // The scores' count and range, as the layer's own check proves them.
        Layer::check(flags)?;
        // (span, item) of every flagged fragment, by start.
        let mut items: Vec<(Span, u32)> = Vec::new();
        for (i, k) in flags.kind.iter().enumerate() {
            if k.is_some() {
                items.extend(item_spans(ts, i).filter(|s| s.start < s.end).map(|s| (s, i as u32)));
            }
        }
        items.sort_unstable_by_key(|&(s, i)| (s.start, s.end, i));
        let mut edges: Vec<u32> = items.iter().flat_map(|(s, _)| [s.start, s.end]).collect();
        edges.sort_unstable();
        edges.dedup();
        // The open items, innermost on top; an item past its end is
        // dropped when it reaches the top.
        let mut open = std::collections::BinaryHeap::new();
        let (mut next, mut c) = (0, Covered { spans: Vec::new(), kind: Vec::new(), score: Vec::new() });
        for w in edges.windows(2) {
            let (a, b) = (w[0], w[1]);
            while let Some(&(s, i)) = items.get(next).filter(|(s, _)| s.start <= a) {
                open.push((s.start, std::cmp::Reverse(s.end), i));
                next += 1;
            }
            while open.peek().is_some_and(|&(_, std::cmp::Reverse(end), _)| end <= a) {
                open.pop();
            }
            let Some(&(_, _, i)) = open.peek() else { continue };
            let (k, score) = (flags.kind[i as usize].expect("a flagged item"), flags.score[i as usize]);
            match (c.spans.last_mut(), c.kind.last(), c.score.last()) {
                (Some(last), Some(&lk), Some(&ls)) if last.end == a && lk == k && ls == score => last.end = b,
                _ => {
                    c.spans.push(Span::new(a, b));
                    c.kind.push(k);
                    c.score.push(score);
                }
            }
        }
        Ok(c)
    }

    /// The interval holding offset `x`, if any: a binary search.
    pub fn at(&self, x: u32) -> Option<usize> {
        let k = self.spans.partition_point(|s| s.end <= x);
        self.spans.get(k).is_some_and(|s| s.start <= x).then_some(k)
    }

    /// Whether offset `x` is inside a covered interval.
    pub fn contains(&self, x: u32) -> bool {
        self.at(x).is_some()
    }

    /// Adds a span's covered bytes to `bytes`, by kind in
    /// `BoilerplateKind::ALL` order, and returns the sum of its covered
    /// bytes times their interval's score.
    pub fn cover(&self, s: Span, bytes: &mut [u64; 5]) -> f64 {
        let mut k = self.spans.partition_point(|c| c.end <= s.start);
        let mut weighted = 0f64;
        while let Some(c) = self.spans.get(k).filter(|c| c.start < s.end) {
            let n = c.end.min(s.end) - c.start.max(s.start);
            bytes[self.kind[k].code() as usize] += u64::from(n);
            weighted += f64::from(n) * f64::from(self.score[k]);
            k += 1;
        }
        weighted
    }
}

/// Item `i`'s spans: its fragments when it has them, else its envelope.
fn item_spans<'a>(ts: &'a TextSpans<'a>, i: usize) -> impl Iterator<Item = Span> + 'a {
    let frags = if ts.frag_item_end.is_empty() { 0..0 } else { ts.item_frag_range(i) };
    let whole = frags.is_empty().then(|| Span::new(ts.start[i], ts.end[i]));
    whole.into_iter().chain(frags.map(|f| Span::new(ts.frag_start[f], ts.frag_end[f])))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Structure;

    #[test]
    fn codes_round_trip_and_stop_at_the_last_kind() {
        for k in BoilerplateKind::ALL {
            assert_eq!(BoilerplateKind::from_code(k.code()), Some(k));
        }
        assert_eq!(BoilerplateKind::from_code(5), None);
    }

    #[test]
    fn the_check_refuses_scores_outside_the_unit_interval_and_misaligned_columns() {
        let ok = Boilerplate::<Structure>::new(vec![None, Some(BoilerplateKind::Footer)], vec![0.0, 1.0]);
        assert_eq!(ok.check(), Ok(()));
        assert_eq!(ok.flagged().map(|i| i.get()).collect::<Vec<_>>(), [1]);
        let nan = Boilerplate::<Structure>::new(vec![None], vec![f32::NAN]);
        assert!(nan.check().unwrap_err().contains("outside"));
        let short = Boilerplate::<Structure>::new(vec![None, None], vec![0.0]);
        assert!(short.check().unwrap_err().contains("1 scores for 2"));
    }
}
