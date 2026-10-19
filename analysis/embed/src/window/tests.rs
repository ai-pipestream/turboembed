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

#[test]
fn window_end_addition_does_not_wrap() {
    let mut p = Plan::new(10);
    p.layout(t(0), u32::MAX - 2..u32::MAX);
    p.place(0, u32::MAX).unwrap();
    assert_eq!(p.windows[0].end, u32::MAX);
    let mut p = Plan::new(10);
    p.item(t(0), u32::MAX - 2..u32::MAX);
    p.place(0, u32::MAX).unwrap();
    assert_eq!(p.windows[0].end, u32::MAX);
}
