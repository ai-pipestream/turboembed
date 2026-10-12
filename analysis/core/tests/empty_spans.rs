//! Empty spans at inserted or deleted bytes, and single offsets mapped
//! through a chain of views.
//!
//! Inserted or deleted bytes belong to neither neighbour: a start offset at
//! them goes after them and an end offset before them. An empty span there
//! stays empty and lands before them, at the smaller of the two offsets,
//! instead of becoming a span that ends before it starts.

use turbo_analysis::text::Side;
use turbo_analysis::*;

/// Adds a view made from `parent` to the document and returns it with its id set.
fn add_view(doc: &mut Document, parent: &View, name: &'static str, text: String, alignment: Alignment) -> View {
    let key = Key::<View>::named(name);
    doc.add(&key, View::derived(parent, text, alignment), Provenance::by("test:view")).unwrap();
    doc.get_key(&key).unwrap().clone()
}

fn source_view(doc: &Document) -> View {
    doc.get::<View>().unwrap().clone()
}

/// "aXb" -> "a+Xb": "+" inserted at parent offset 1, child bytes 1..2.
fn insertion() -> (String, Alignment) {
    let mut b = ViewBuilder::new("aXb");
    b.keep(1);
    b.replace(1, "+");
    b.finish()
}

/// "aXb" -> "ab": X deleted, parent bytes 1..2, at child offset 1.
fn deletion() -> (String, Alignment) {
    let mut b = ViewBuilder::new("aXb");
    b.keep(1);
    b.replace(2, "");
    b.finish()
}

#[test]
fn an_empty_span_at_an_insertion_stays_empty_before_it() {
    let (out, a) = insertion();
    assert_eq!(out, "a+Xb");
    // Parent offset 1 is the insertion point: a start goes after "+" and an
    // end before it, so the empty span lands before "+".
    assert_eq!(a.to_child(1, Side::Start), 2);
    assert_eq!(a.to_child(1, Side::End), 1);
    assert_eq!(a.span_to_child(Span::new(1, 1)), Span::new(1, 1));
    // Child offsets 1 and 2 (before and after "+") are both parent 1.
    assert_eq!(a.span_to_parent(Span::new(1, 1)), Span::new(1, 1));
    assert_eq!(a.span_to_parent(Span::new(2, 2)), Span::new(1, 1));
    // Empty spans away from the insertion shift as single offsets do.
    assert_eq!(a.span_to_child(Span::new(0, 0)), Span::new(0, 0));
    assert_eq!(a.span_to_child(Span::new(2, 2)), Span::new(3, 3));
    assert_eq!(a.span_to_parent(Span::new(3, 3)), Span::new(2, 2));
}

#[test]
fn an_empty_span_at_a_deletion_stays_empty_before_it() {
    let (out, a) = deletion();
    assert_eq!(out, "ab");
    // Child offset 1 is where X was: a start goes after X, an end before it,
    // so the empty span lands before X.
    assert_eq!(a.to_parent(1, Side::Start), 2);
    assert_eq!(a.to_parent(1, Side::End), 1);
    assert_eq!(a.span_to_parent(Span::new(1, 1)), Span::new(1, 1));
    // Parent offsets 1 and 2 (before and after X) are both child 1.
    assert_eq!(a.span_to_child(Span::new(1, 1)), Span::new(1, 1));
    assert_eq!(a.span_to_child(Span::new(2, 2)), Span::new(1, 1));
    assert_eq!(a.span_to_parent(Span::new(2, 2)), Span::new(3, 3));
}

#[test]
fn empty_spans_at_insertions_at_both_ends_of_the_text() {
    // "mid" -> "[mid]".
    let mut b = ViewBuilder::new("mid");
    b.replace(0, "[");
    b.keep(3);
    b.replace(3, "]");
    let (out, a) = b.finish();
    assert_eq!(out, "[mid]");
    assert_eq!(a.span_to_child(Span::new(0, 0)), Span::new(0, 0));
    assert_eq!(a.span_to_child(Span::new(3, 3)), Span::new(4, 4));
    for x in [0, 1] {
        assert_eq!(a.span_to_parent(Span::new(x, x)), Span::new(0, 0), "child offset {x}");
    }
    for x in [4, 5] {
        assert_eq!(a.span_to_parent(Span::new(x, x)), Span::new(3, 3), "child offset {x}");
    }
}

#[test]
fn empty_spans_at_deletions_at_both_ends_of_the_text() {
    // "[mid]" -> "mid".
    let mut b = ViewBuilder::new("[mid]");
    b.replace(1, "");
    b.keep(4);
    b.replace(5, "");
    let (out, a) = b.finish();
    assert_eq!(out, "mid");
    assert_eq!(a.span_to_parent(Span::new(0, 0)), Span::new(0, 0));
    assert_eq!(a.span_to_parent(Span::new(3, 3)), Span::new(4, 4));
    for x in [0, 1] {
        assert_eq!(a.span_to_child(Span::new(x, x)), Span::new(0, 0), "parent offset {x}");
    }
    for x in [4, 5] {
        assert_eq!(a.span_to_child(Span::new(x, x)), Span::new(3, 3), "parent offset {x}");
    }
}

// Without the reordering in `span_to_parent` and `span_to_child`, this test
// panics in `Span::new`: the start of the empty child span at 0 maps to 3,
// after the deleted text, and its end to 0, before it. (Every other test in
// this file panics the same way, and `offset_from_source` panics too when it
// maps an empty span instead of a single offset.)
#[test]
fn an_empty_span_maps_after_the_whole_text_is_deleted() {
    let mut b = ViewBuilder::new("abc");
    b.replace(3, "");
    let (out, a) = b.finish();
    assert_eq!(out, "");
    assert_eq!(a.span_to_parent(Span::new(0, 0)), Span::new(0, 0));
    for x in 0..=3 {
        assert_eq!(a.span_to_child(Span::new(x, x)), Span::new(0, 0), "parent offset {x}");
    }
    // The whole text still maps to the empty child text and back to all of it.
    assert_eq!(a.span_to_child(Span::new(0, 3)), Span::new(0, 0));

    // The same through a document, from the source to the empty view and back.
    let mut doc = Document::new("abc");
    let source = source_view(&doc);
    let v = add_view(&mut doc, &source, "empty", out, a).id();
    assert_eq!(doc.to_source(v, Span::new(0, 0)), Span::new(0, 0));
    assert_eq!(doc.from_source(v, Span::new(3, 3)), Span::new(0, 0));
    assert_eq!(doc.offset_from_source(v, 3, Side::Start), 0);
    assert_eq!(doc.offset_from_source(v, 3, Side::End), 0);
    assert_eq!(doc.offset_to_source(v, 0, Side::Start), 3);
    assert_eq!(doc.offset_to_source(v, 0, Side::End), 0);
}

#[test]
fn single_offsets_map_through_two_views_at_insertion_points() {
    // Source "ab". View A inserts "+" between a and b: "a+b". View B, made
    // from A, inserts "#" after "+": "a+#b".
    let mut doc = Document::new("ab");
    let mut b = ViewBuilder::new("ab");
    b.keep(1);
    b.replace(1, "+");
    let (text_a, align_a) = b.finish();
    assert_eq!(text_a, "a+b");
    let source = source_view(&doc);
    let view_a = add_view(&mut doc, &source, "a", text_a.clone(), align_a);
    let va = view_a.id();
    let mut b = ViewBuilder::new(&text_a);
    b.keep(2);
    b.replace(2, "#");
    let (text_b, align_b) = b.finish();
    assert_eq!(text_b, "a+#b");
    let vb = add_view(&mut doc, &view_a, "b", text_b, align_b).id();

    // Source offset 1 is the insertion point of "+". A start goes after
    // everything inserted there, through both views; an end goes before it.
    assert_eq!(doc.offset_from_source(vb, 1, Side::Start), 3);
    assert_eq!(doc.offset_from_source(vb, 1, Side::End), 1);
    assert_eq!(doc.offset_from_source(va, 1, Side::Start), 2);
    assert_eq!(doc.offset_from_source(va, 1, Side::End), 1);
    // The other source offsets shift by what was inserted before them.
    assert_eq!(doc.offset_from_source(vb, 0, Side::Start), 0);
    assert_eq!(doc.offset_from_source(vb, 2, Side::End), 4);
    // And the empty source span there stays empty, before both insertions.
    assert_eq!(doc.from_source(vb, Span::new(1, 1)), Span::new(1, 1));

    // Back up: every offset of B from 1 to 3 is source offset 1, whichever
    // side, since the bytes between them were inserted.
    for x in 1..=3 {
        assert_eq!(doc.offset_to_source(vb, x, Side::Start), 1, "B offset {x} as a start");
        assert_eq!(doc.offset_to_source(vb, x, Side::End), 1, "B offset {x} as an end");
        assert_eq!(doc.to_source(vb, Span::new(x, x)), Span::new(1, 1), "empty B span at {x}");
    }
    assert_eq!(doc.offset_to_source(vb, 4, Side::End), 2);
}

/// Every alignment `ViewBuilder` makes from up to four calls over "abc",
/// each call consuming 0 to 2 bytes and writing "", "+" or "pq".
fn every_small_alignment(mut f: impl FnMut(&str, &str, &Alignment)) {
    const WRITES: [&str; 3] = ["", "+", "pq"];
    fn go(calls: &mut Vec<(usize, &'static str)>, depth: usize, f: &mut dyn FnMut(&str, &str, &Alignment)) {
        let src = "abc";
        let mut b = ViewBuilder::new(src);
        let mut at = 0;
        for &(take, with) in calls.iter() {
            at += take;
            b.replace(at, with);
        }
        let (out, a) = b.finish();
        f(src, &out, &a);
        if depth == 0 {
            return;
        }
        let used: usize = calls.iter().map(|c| c.0).sum();
        for take in 0..=2 {
            if used + take > src.len() {
                continue;
            }
            for with in WRITES {
                calls.push((take, with));
                go(calls, depth - 1, f);
                calls.pop();
            }
        }
    }
    go(&mut Vec::new(), 4, &mut f);
}

#[test]
fn no_span_panics_and_only_empty_spans_map_out_of_order() {
    let mut count = 0;
    every_small_alignment(|src, out, a| {
        count += 1;
        for (from_len, up) in [(out.len() as u32, true), (src.len() as u32, false)] {
            let one = |x, side| if up { a.to_parent(x, side) } else { a.to_child(x, side) };
            for start in 0..=from_len {
                for end in start..=from_len {
                    let s = Span::new(start, end);
                    let got = if up { a.span_to_parent(s) } else { a.span_to_child(s) };
                    let (ms, me) = (one(start, Side::Start), one(end, Side::End));
                    if ms <= me {
                        // Includes an empty span strictly inside an edit,
                        // which widens to the whole edit as offsets do.
                        assert_eq!(got, Span::new(ms, me), "{src:?} -> {out:?}, {s:?}");
                    } else {
                        assert!(s.is_empty(), "{src:?} -> {out:?}: a non-empty {s:?} maps out of order");
                        assert_eq!(got, Span::new(me, me), "{src:?} -> {out:?}, {s:?}");
                    }
                }
            }
        }
    });
    // 1 + 9 + 9 * 9 + ... calls, minus those that run past the text.
    assert!(count > 1000, "only {count} alignments");
}
