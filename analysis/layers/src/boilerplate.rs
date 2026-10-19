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
/// text order, each with the kind and score of the outermost item that
/// opened it.
pub struct Covered {
    pub spans: Vec<Span>,
    pub kind: Vec<BoilerplateKind>,
    pub score: Vec<f32>,
}

impl Covered {
    /// The union of the flagged items' spans (fragments, for a
    /// discontinuous item). Overlapping or touching items merge into the
    /// first one's interval.
    pub fn new<S>(ts: &TextSpans<'_>, flags: &Boilerplate<S>) -> Covered {
        let mut items: Vec<(Span, BoilerplateKind, f32)> = Vec::new();
        for (i, k) in flags.kind.iter().enumerate() {
            if let Some(k) = *k {
                items.extend(item_spans(ts, i).map(|s| (s, k, flags.score[i])));
            }
        }
        // Outer spans first at one start, so a merged interval keeps the
        // outermost item's kind.
        items.sort_by_key(|(s, _, _)| (s.start, std::cmp::Reverse(s.end)));
        let mut c = Covered { spans: Vec::new(), kind: Vec::new(), score: Vec::new() };
        for (s, k, score) in items {
            match c.spans.last_mut() {
                Some(last) if s.start <= last.end => last.end = last.end.max(s.end),
                _ => {
                    c.spans.push(s);
                    c.kind.push(k);
                    c.score.push(score);
                }
            }
        }
        c
    }

    /// Whether offset `x` is inside a covered interval: a binary search.
    pub fn contains(&self, x: u32) -> bool {
        let k = self.spans.partition_point(|s| s.end <= x);
        self.spans.get(k).is_some_and(|s| s.start <= x)
    }

    /// For a span: how many of its bytes are covered, the sum of covered
    /// bytes times their interval's score, and the kind of the interval
    /// covering most of it (the first, on a tie).
    pub fn cover(&self, s: Span) -> (u64, f64, Option<BoilerplateKind>) {
        let mut k = self.spans.partition_point(|c| c.end <= s.start);
        let (mut bytes, mut weighted, mut best) = (0u64, 0f64, (0u32, None));
        while let Some(c) = self.spans.get(k).filter(|c| c.start < s.end) {
            let n = c.end.min(s.end) - c.start.max(s.start);
            bytes += u64::from(n);
            weighted += f64::from(n) * f64::from(self.score[k]);
            if n > best.0 {
                best = (n, Some(self.kind[k]));
            }
            k += 1;
        }
        (bytes, weighted, best.1)
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
