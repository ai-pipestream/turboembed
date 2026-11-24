//! Reads that several layers make of their columns the same way.

use turbo_analysis::{Ix, Spans};

use crate::{DecimalRef, Strs};

/// The decimal of item `i` of a sign, digits and scale column.
pub(crate) fn decimal_at<'a>(negative: &[bool], digits: &'a Strs, scale: &[i32], i: usize) -> DecimalRef<'a> {
    DecimalRef { negative: negative[i], digits: digits.get(i), scale: scale[i].into() }
}

/// Whether two span columns are over the same view and have the same
/// fragments, item by item.
pub(crate) fn same_spans<L>(a: &Spans<L>, b: &Spans<L>) -> bool {
    a.view() == b.view() && a.len() == b.len() && a.iter().all(|(ix, _)| a.fragments(ix).eq(b.fragments(ix)))
}

/// Items `0..n` with their index, each read by `get`.
pub(crate) fn indexed<L, T>(n: usize, get: impl Fn(Ix<L>) -> T) -> impl ExactSizeIterator<Item = (Ix<L>, T)> {
    (0..n as u32).map(move |i| {
        let ix = Ix::new(i);
        (ix, get(ix))
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use turbo_analysis::{Span, ViewId};

    #[test]
    fn decimal_at_reads_one_row_of_the_columns() {
        let mut digits = Strs::default();
        digits.push("12");
        digits.push("5");
        let d = decimal_at(&[false, true], &digits, &[0, 1], 1);
        assert_eq!(d, DecimalRef { negative: true, digits: "5", scale: 1 });
    }

    #[test]
    fn same_spans_compares_view_length_and_fragments() {
        let spans = |view, items: &[&[(u32, u32)]]| {
            let mut s = Spans::<()>::new(view);
            for frags in items {
                s.push_fragments(&frags.iter().map(|&(a, b)| Span::new(a, b)).collect::<Vec<_>>());
            }
            s
        };
        let a = spans(ViewId::SOURCE, &[&[(0, 2), (4, 6)]]);
        assert!(same_spans(&a, &spans(ViewId::SOURCE, &[&[(0, 2), (4, 6)]])));
        assert!(!same_spans(&a, &spans(ViewId::SOURCE, &[&[(0, 6)]])), "same envelope, other fragments");
        assert!(!same_spans(&a, &spans(ViewId(1), &[&[(0, 2), (4, 6)]])));
        assert!(!same_spans(&a, &spans(ViewId::SOURCE, &[])), "fewer items, no panic");
    }

    #[test]
    fn indexed_pairs_each_index_with_its_item() {
        let items: Vec<_> = indexed::<(), _>(3, |ix| ix.get() * 10).collect();
        assert_eq!(items, [(Ix::new(0), 0), (Ix::new(1), 10), (Ix::new(2), 20)]);
        assert_eq!(indexed::<(), _>(4, |ix| ix).len(), 4);
    }
}
