use super::*;
use crate::ViewId;

fn measured(n: u32, contained: bool, empty_inner: bool, points: bool) -> (usize, usize) {
    let mut inner = Spans::<()>::new(ViewId::SOURCE);
    let mut outer = Spans::<()>::new(ViewId::SOURCE);
    for i in 0..n {
        inner.push(if empty_inner { Span::new(0, 0) } else { Span::new(i, i + 1) });
        outer.push(if points { Span::new(i, i) } else { Span::new(0, if i % 2 == 0 { n } else { 1 }) });
    }
    assert!(inner.is_ordered() && outer.is_start_ordered());
    ADVANCES.with(|c| c.set(0));
    SEARCH_PROBES.with(|c| c.set(0));
    let check = |s, range: IxRange<()>| {
        let want = if contained { inner.within(s) } else { inner.overlapping_range(s) };
        assert_eq!((range.start.get(), range.end.get()), (want.start.get(), want.end.get()));
    };
    if contained {
        for (_, s, range) in outer.join_contained(&inner) {
            check(s, range);
        }
    } else {
        for (_, s, range) in outer.join_overlapping(&inner) {
            check(s, range);
        }
    }
    (ADVANCES.with(|c| c.get()), SEARCH_PROBES.with(|c| c.get()))
}

#[test]
fn nested_contained_queries_never_rewind_the_forward_sweep() {
    let (small, _) = measured(512, true, false, false);
    let (large, _) = measured(1024, true, false, false);
    eprintln!("contained forward advances: n512={small}, n1024={large}");
    assert!(large <= 3 * small, "forward advances: n512={small}, n1024={large}");
    assert!(large <= 1024, "one forward pass: {large}");
}

#[test]
fn nested_overlapping_queries_never_rewind_the_forward_sweep() {
    let (small, _) = measured(512, false, false, false);
    let (large, _) = measured(1024, false, false, false);
    eprintln!("overlapping forward advances: n512={small}, n1024={large}");
    assert!(large <= 3 * small, "forward advances: n512={small}, n1024={large}");
    assert!(large <= 3 * 1024, "bounded independent sweeps: {large}");
}

#[test]
fn repeated_empty_inner_items_do_not_restart_the_lower_bound() {
    let (advances, _) = measured(1024, false, true, false);
    eprintln!("duplicate empty items: {advances} forward advances");
    assert!(advances <= 3 * 1024, "duplicate empty items: {advances} advances");
}

#[test]
fn monotonic_point_queries_sweep_without_binary_searching() {
    let (advances, probes) = measured(1024, false, false, true);
    eprintln!("point queries: {advances} forward advances, {probes} search probes");
    assert!(advances <= 3 * 1024, "point query advances: {advances}");
    assert_eq!(probes, 0, "monotonic point queries must not binary-search each item");
}

#[test]
fn point_queries_at_the_largest_offset_preserve_empty_boundary_items() {
    let mut inner = Spans::<()>::new(ViewId::SOURCE);
    inner.push(Span::new(u32::MAX - 1, u32::MAX));
    inner.push(Span::new(u32::MAX, u32::MAX));
    inner.push(Span::new(u32::MAX, u32::MAX));
    let mut outer = Spans::<()>::new(ViewId::SOURCE);
    outer.push(Span::new(u32::MAX, u32::MAX));
    let (_, _, range) = outer.join_overlapping(&inner).next().unwrap();
    assert_eq!(range.iter().map(|i| i.get()).collect::<Vec<_>>(), vec![1, 2]);
    let (_, _, range) = outer.join_contained(&inner).next().unwrap();
    assert_eq!(range.iter().map(|i| i.get()).collect::<Vec<_>>(), vec![1, 2]);
}

#[test]
fn exhaustive_small_ordered_inners_match_brute_force_with_nested_and_point_outers() {
    let mut outer = Spans::<()>::new(ViewId::SOURCE);
    for start in 0..=3 {
        // Drop from the wide bound to a point, then grow again at
        // the same start, including repeated equal-bound queries.
        for end in [3, start, 3, start] {
            outer.push(Span::new(start, end));
        }
    }
    fn check(inner: &Spans<()>, outer: &Spans<()>) {
        assert!(inner.is_ordered());
        for (_, s, range) in outer.join_contained(inner) {
            let got: Vec<_> = range.iter().map(|i| i.get()).collect();
            let want: Vec<_> =
                inner.iter().filter(|(_, t)| s.start <= t.start && t.end <= s.end).map(|(i, _)| i.get()).collect();
            assert_eq!(got, want, "contained {s:?} in {:?}", inner.columns());
        }
        for (_, s, range) in outer.join_overlapping(inner) {
            let got: Vec<_> = range.iter().map(|i| i.get()).collect();
            let want: Vec<_> = inner.iter().filter(|(_, t)| t.overlaps(s)).map(|(i, _)| i.get()).collect();
            assert_eq!(got, want, "overlapping {s:?} in {:?}", inner.columns());
        }
    }
    fn visit(items: &mut Vec<Span>, min: u32, remaining: usize, outer: &Spans<()>) {
        let mut inner = Spans::<()>::new(ViewId::SOURCE);
        for &s in items.iter() {
            inner.push(s);
        }
        check(&inner, outer);
        if remaining == 0 {
            return;
        }
        for start in min..=3 {
            for end in start..=3 {
                items.push(Span::new(start, end));
                visit(items, end, remaining - 1, outer);
                items.pop();
            }
        }
    }
    visit(&mut Vec::new(), 0, 3, &outer);
}
