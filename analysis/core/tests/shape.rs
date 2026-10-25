use turbo_analysis::{Document, Error, Key, KeyId, ShapeMember, Span, TextSpans, View, ViewId};

fn key() -> KeyId {
    Key::<View>::DEFAULT.id()
}

struct Cols {
    start: Vec<u32>,
    end: Vec<u32>,
    frag_start: Vec<u32>,
    frag_end: Vec<u32>,
    frag_item_end: Vec<u32>,
}

impl Cols {
    fn spans(items: &[(u32, u32)]) -> Cols {
        Cols {
            start: items.iter().map(|i| i.0).collect(),
            end: items.iter().map(|i| i.1).collect(),
            frag_start: Vec::new(),
            frag_end: Vec::new(),
            frag_item_end: Vec::new(),
        }
    }

    fn fragged(items: &[&[(u32, u32)]]) -> Cols {
        let mut c = Cols { start: vec![], end: vec![], frag_start: vec![], frag_end: vec![], frag_item_end: vec![] };
        for item in items {
            c.start.push(item[0].0);
            c.end.push(item[item.len() - 1].1);
            for f in *item {
                c.frag_start.push(f.0);
                c.frag_end.push(f.1);
            }
            c.frag_item_end.push(c.frag_start.len() as u32);
        }
        c
    }

    fn ts(&self) -> TextSpans<'_> {
        TextSpans {
            view: ViewId(0),
            start: &self.start,
            end: &self.end,
            frag_start: &self.frag_start,
            frag_end: &self.frag_end,
            frag_item_end: &self.frag_item_end,
        }
    }
}

type SegRow = (u32, u32, Vec<(usize, u32)>);

fn shape_of(doc: &Document, from: u32, len: u32, layers: &[&Cols]) -> Vec<SegRow> {
    let w = doc.window(ViewId(0), from, len).unwrap();
    let cols: Vec<(KeyId, TextSpans<'_>)> = layers.iter().map(|c| (key(), c.ts())).collect();
    let shape = doc.combined_segments(&w, &cols).unwrap();
    shape
        .segments
        .iter()
        .map(|s| (s.span.start, s.span.end, shape.members_of(s).iter().map(|m| (m.layer, m.item)).collect()))
        .collect()
}

// ---- windows ----

#[test]
fn window_pages_clamps_and_snaps() {
    // bytes: a 0..1, emoji 1..5, b 5..6, c 6..7, d 7..8
    let doc = Document::new("a\u{1F600}bcd");
    let w = doc.window(ViewId(0), 0, 3).unwrap();
    assert_eq!(w.span, Span { start: 0, end: 1 }, "end snaps down off the emoji, never past len");
    assert_eq!((w.page, w.pages, w.clamped), (0, 3, false));
    assert_eq!(w.text(&doc), "a");
    // from inside the emoji snaps down to its start; one wide char is whole.
    let w = doc.window(ViewId(0), 3, 2).unwrap();
    assert_eq!(w.span, Span { start: 1, end: 5 }, "a char wider than len is included whole");
    assert_eq!(w.text(&doc), "\u{1F600}");
    // tail window clamps.
    let w = doc.window(ViewId(0), 6, 100).unwrap();
    assert_eq!(w.span, Span { start: 6, end: 8 });
    assert!(w.clamped);
    assert_eq!(w.page, 0, "page is the len-tile of the snapped start");
    // exactly at the end: an empty span, not an error.
    let w = doc.window(ViewId(0), 8, 4).unwrap();
    assert_eq!(w.span, Span { start: 8, end: 8 });
    assert_eq!(w.text(&doc), "");
}

#[test]
fn window_offset_views_agree() {
    let doc = Document::new("a\u{1F600}b");
    let w = doc.window(ViewId(0), 1, 4).unwrap();
    let u = w.utf16_span(&doc);
    assert_eq!((u.start.0, u.end.0), (1, 3), "emoji is two UTF-16 units");
    let c = w.code_point_span(&doc);
    assert_eq!((c.start.0, c.end.0), (1, 2));
}

#[test]
fn window_refusals_name_the_numbers() {
    let doc = Document::new("abc");
    for (view, from, len, needle) in
        [(ViewId(9), 0, 1, "view 9"), (ViewId(0), 0, 0, "zero"), (ViewId(0), 4, 1, "byte 4")]
    {
        match doc.window(view, from, len) {
            Err(Error::Bounds { detail, .. }) => assert!(detail.contains(needle), "{detail} should name {needle}"),
            other => panic!("expected Bounds refusal, got {other:?}"),
        }
    }
    let empty = Document::new("");
    let w = empty.window(ViewId(0), 0, 10).unwrap();
    assert_eq!((w.span, w.page, w.pages), (Span { start: 0, end: 0 }, 0, 1));
    assert_eq!(doc.view_count(), 1);
}

// ---- combined segmentation ----

#[test]
fn overlapping_layers_partition_by_membership() {
    let doc = Document::new("0123456789");
    let a = Cols::spans(&[(1, 6), (7, 9)]);
    let b = Cols::spans(&[(3, 8)]);
    assert_eq!(
        shape_of(&doc, 0, 10, &[&a, &b]),
        vec![
            (0, 1, vec![]),
            (1, 3, vec![(0, 0)]),
            (3, 6, vec![(0, 0), (1, 0)]),
            (6, 7, vec![(1, 0)]),
            (7, 8, vec![(0, 1), (1, 0)]),
            (8, 9, vec![(0, 1)]),
            (9, 10, vec![]),
        ]
    );
}

#[test]
fn discontinuous_gap_is_not_covered_by_its_item() {
    let doc = Document::new("0123456789");
    let a = Cols::fragged(&[&[(1, 3), (6, 8)]]);
    let b = Cols::spans(&[(4, 5)]);
    assert_eq!(
        shape_of(&doc, 0, 10, &[&a, &b]),
        vec![
            (0, 1, vec![]),
            (1, 3, vec![(0, 0)]),
            (3, 4, vec![]),
            (4, 5, vec![(1, 0)]),
            (5, 6, vec![]),
            (6, 8, vec![(0, 0)]),
            (8, 10, vec![])
        ]
    );
}

#[test]
fn zero_width_items_insert_between_segments() {
    let doc = Document::new("0123456789");
    let a = Cols::spans(&[(2, 6)]);
    let z = Cols::spans(&[(0, 0), (4, 4), (10, 10)]);
    assert_eq!(
        shape_of(&doc, 0, 10, &[&a, &z]),
        vec![
            (0, 0, vec![(1, 0)]),
            (0, 2, vec![]),
            (2, 4, vec![(0, 0)]),
            (4, 4, vec![(1, 1)]),
            (4, 6, vec![(0, 0)]),
            (6, 10, vec![]),
            (10, 10, vec![(1, 2)]),
        ],
        "zero-width items hold only themselves and split their neighbours"
    );
}

#[test]
fn fragments_clip_to_the_window() {
    let doc = Document::new("0123456789");
    let a = Cols::spans(&[(0, 10), (2, 3)]);
    assert_eq!(shape_of(&doc, 4, 3, &[&a]), vec![(4, 7, vec![(0, 0)])]);
    // no layers at all: one uncovered segment; empty window: none.
    let none: &[&Cols] = &[];
    assert_eq!(shape_of(&doc, 2, 5, none), vec![(2, 7, vec![])]);
    let w = doc.window(ViewId(0), 10, 3).unwrap();
    let shape = doc.combined_segments(&w, &[(key(), a.ts())]).unwrap();
    assert!(shape.segments.is_empty());
}

#[test]
fn segmentation_matches_a_per_byte_scan() {
    let doc = Document::new("0123456789abcdef");
    let a = Cols::fragged(&[&[(1, 4), (9, 12)], &[(3, 7)], &[(14, 14)]]);
    let b = Cols::spans(&[(0, 5), (5, 5), (6, 16)]);
    let got = shape_of(&doc, 2, 12, &[&a, &b]);
    // Oracle: per byte, the items whose fragments cover it.
    let covers = |c: &Cols, item: usize, at: u32| -> bool {
        if c.frag_item_end.is_empty() {
            c.start[item] <= at && at < c.end[item]
        } else {
            c.ts().item_frag_range(item).any(|f| c.frag_start[f] <= at && at < c.frag_end[f])
        }
    };
    let mut covered = 0u32;
    for (s, e, members) in &got {
        if s == e {
            assert!(members.iter().all(|&(l, i)| {
                let c = [&a, &b][l];
                c.start[i as usize] == *s && c.end[i as usize] == *s
            }));
            continue;
        }
        assert_eq!(*s, 2 + covered, "segments tile the window in order");
        covered += e - s;
        for at in *s..*e {
            let want: Vec<(usize, u32)> = [(&a, 0usize), (&b, 1usize)]
                .iter()
                .flat_map(|(c, l)| (0..c.start.len()).filter(move |&i| covers(c, i, at)).map(move |i| (*l, i as u32)))
                .collect();
            assert_eq!(&want, members, "byte {at}");
        }
    }
    assert_eq!(covered, 12);
}

#[test]
fn segment_refusals() {
    let doc = Document::new("a\u{1F600}b"); // 6 bytes, emoji at 1..5
    let w = doc.window(ViewId(0), 0, 6).unwrap();
    // wrong view
    let a = Cols::spans(&[(0, 1)]);
    let mut ts = a.ts();
    ts.view = ViewId(3);
    assert!(matches!(doc.combined_segments(&w, &[(key(), ts)]), Err(Error::Bounds { .. })));
    // past the end
    let a = Cols::spans(&[(0, 7)]);
    match doc.combined_segments(&w, &[(key(), a.ts())]) {
        Err(Error::Bounds { detail, .. }) => assert!(detail.contains("byte 7"), "{detail}"),
        other => panic!("{other:?}"),
    }
    // inside a character
    let a = Cols::spans(&[(0, 3)]);
    match doc.combined_segments(&w, &[(key(), a.ts())]) {
        Err(Error::Bounds { detail, .. }) => assert!(detail.contains("inside a character"), "{detail}"),
        other => panic!("{other:?}"),
    }
    // structurally inconsistent columns (reversed span)
    let a = Cols::spans(&[(5, 1)]);
    assert!(matches!(doc.combined_segments(&w, &[(key(), a.ts())]), Err(Error::Bounds { .. })));
    // hand-made window outside the view
    let bad = turbo_analysis::DocumentWindow {
        view: ViewId(0),
        span: Span { start: 0, end: 99 },
        page: 0,
        pages: 1,
        clamped: false,
    };
    assert!(matches!(doc.combined_segments(&bad, &[]), Err(Error::Bounds { .. })));
    let _ = ShapeMember { layer: 0, item: 0 };
}
