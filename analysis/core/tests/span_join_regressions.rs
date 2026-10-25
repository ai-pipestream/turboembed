use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::Cell;
use turbo_analysis::*;
use turbo_analysis_layers::Entities;

struct CountAlloc;
thread_local! { static ALLOCS: Cell<Option<usize>> = const { Cell::new(None) }; }
fn count() {
    ALLOCS.with(|c| {
        if let Some(n) = c.get() {
            c.set(Some(n + 1));
        }
    });
}
unsafe impl GlobalAlloc for CountAlloc {
    unsafe fn alloc(&self, l: Layout) -> *mut u8 {
        count();
        unsafe { System.alloc(l) }
    }
    unsafe fn alloc_zeroed(&self, l: Layout) -> *mut u8 {
        count();
        unsafe { System.alloc_zeroed(l) }
    }
    unsafe fn realloc(&self, p: *mut u8, l: Layout, n: usize) -> *mut u8 {
        count();
        unsafe { System.realloc(p, l, n) }
    }
    unsafe fn dealloc(&self, p: *mut u8, l: Layout) {
        unsafe { System.dealloc(p, l) }
    }
}
#[global_allocator]
static ALLOC: CountAlloc = CountAlloc;

#[test]
fn warm_graph_join_and_alignment_loops_allocate_nothing() {
    let mut inner = Spans::<Entities>::new(ViewId::SOURCE);
    let mut outer = Spans::<Entities>::new(ViewId::SOURCE);
    for i in 0..128 {
        inner.push(Span::new(i, i + 1));
    }
    for i in 0..128 {
        outer.push(Span::new(0, if i % 2 == 0 { 128 } else { 1 }));
    }
    let kinds = TagSet::new("test", &["link"]);
    let graph =
        Edges::<Entities, Entities>::new(kinds, (0..127).map(|i| Edge::new(Ix::new(i), Ix::new(i + 1), Tag(0), 1.0)));
    let mut scratch = ReachScratch::new();
    let mut out = Vec::new();
    graph.reachable_from(Ix::new(0), 128, &mut scratch, &mut out);
    let alignment = Alignment::from_edits(vec![Edit { parent: Span::new(4, 8), child: Span::new(4, 10) }]).unwrap();
    ALLOCS.with(|c| c.set(Some(0)));
    for _ in 0..4 {
        for (_, _, range) in outer.join_contained(&inner) {
            std::hint::black_box(range);
        }
        for (_, _, range) in outer.join_overlapping(&inner) {
            std::hint::black_box(range);
        }
        graph.reachable_from(Ix::new(0), 128, &mut scratch, &mut out);
        let mut cursor = alignment.cursor();
        for i in 0..128 {
            std::hint::black_box(cursor.span_to_child(Span::new(i, i)));
        }
    }
    let allocations = ALLOCS.with(|c| c.replace(None).unwrap());
    assert_eq!(allocations, 0);
}

fn add_view(
    doc: &mut Document,
    parent: Key<View>,
    name: &'static str,
    edit: impl FnOnce(&mut ViewBuilder<'_>),
) -> Key<View> {
    let p = doc.get_key(&parent).unwrap().clone();
    let mut builder = ViewBuilder::new(p.text());
    edit(&mut builder);
    let (text, alignment) = builder.finish();
    let key = Key::<View>::named(name);
    doc.add(&key, View::derived(&p, text, alignment), Provenance::by("review")).unwrap();
    key
}

#[test]
fn utf8_point_and_character_spans_match_across_branched_views() {
    let mut doc = Document::new("aé😀z-文q");
    let v1 = add_view(&mut doc, Key::DEFAULT, "v1", |b| {
        b.keep(1);
        b.replace(3, "EE");
        b.keep(7);
        b.replace(7, "!");
        b.keep(9);
        b.replace(12, "");
    });
    let v2 = add_view(&mut doc, v1, "v2", |b| {
        b.keep(1);
        b.replace(3, "Z");
    });
    let sibling = add_view(&mut doc, Key::DEFAULT, "sibling", |b| {
        b.keep(1);
        b.replace(3, "E");
        b.keep(7);
        b.replace(8, "");
    });
    let keys = [Key::<View>::DEFAULT, v1, v2, sibling];
    for from in keys {
        let view = doc.get_key(&from).unwrap();
        let mut bounds: Vec<_> = view.text().char_indices().map(|(i, _)| i as u32).collect();
        bounds.push(view.text().len() as u32);
        let mut spans = Spans::<Entities>::new(view.id());
        for w in bounds.windows(2) {
            spans.push(Span::new(w[0], w[0]));
            spans.push(Span::new(w[0], w[1]));
        }
        let end = *bounds.last().unwrap();
        spans.push(Span::new(end, end));
        assert!(spans.is_ordered());
        for to in keys {
            let target = doc.get_key(&to).unwrap().id();
            let got = doc.project_spans_monotonic(&spans, target).unwrap();
            let want = doc.project_spans(&spans, target).unwrap();
            assert_eq!(got.columns(), want.columns());
            for ((_, s), (_, projected)) in spans.iter().zip(got.iter()) {
                assert_eq!(projected, doc.map_span(view.id(), target, s).unwrap());
            }
            let mut discontinuous = Spans::<Entities>::new(view.id());
            discontinuous
                .try_push_fragments(&[Span::new(bounds[0], bounds[1]), Span::new(bounds[bounds.len() - 2], end)])
                .unwrap();
            let a = doc.project_spans_monotonic(&discontinuous, target).unwrap();
            let b = doc.project_spans(&discontinuous, target).unwrap();
            assert_eq!(a.columns(), b.columns());
            assert_eq!(a.fragments(Ix::new(0)).collect::<Vec<_>>(), b.fragments(Ix::new(0)).collect::<Vec<_>>());
        }
    }
}
