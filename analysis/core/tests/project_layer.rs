//! Multi-hop cross-view bi-directional projection tests for [`ProjectableLayer`],
//! [`Document::project_layer`], [`ProjectView`], and [`OffsetTable`].

use turbo_analysis::{
    Alternatives, Analyzer, Anchor, Annotator, CodePointOffset, CodePointSpan, Document, Edge, Edges, Error, Groups,
    Ix, Key, Layer, ModeSet, Overlap, ProjectView, ProjectableLayer, Provenance, Refs, Span, Spans, Tag, TagSet,
    TextSpans, Utf16Offset, Utf16Span, View, ViewBuilder, ViewId,
};

#[derive(Clone, Debug)]
struct TestUnits {
    spans: Spans<TestUnits>,
}

impl Layer for TestUnits {
    const ID: &'static str = "test:units";
    const OVERLAP: Overlap = Overlap::Any;

    fn anchor() -> Anchor {
        Anchor::Text
    }

    fn len(&self) -> usize {
        self.spans.len()
    }

    fn text_spans(&self) -> Option<TextSpans<'_>> {
        Some(self.spans.text_spans())
    }
}

impl ProjectableLayer for TestUnits {
    fn project_view(&self, doc: &Document, to: ViewId) -> Result<Self, Error> {
        Ok(TestUnits { spans: doc.project_spans_monotonic(&self.spans, to)? })
    }
}

/// Builds a 3-view chain:
/// - `ViewId::SOURCE` (0): `"<p>Café \u{FB01}re \u{1F600} end</p>"`
///   - `"<p>"` (0..3)
///   - `"Café "` (3..9, `'é'` is 2 UTF-8 bytes at 6..8)
///   - `"\u{FB01}re "` (9..15, `'ﬁ'` is 3 UTF-8 bytes at 9..12, `"re "` is 12..15)
///   - `"\u{1F600} "` (15..20, `'😀'` is 4 UTF-8 bytes / 2 UTF-16 units at 15..19)
///   - `"end"` (20..23)
///   - `"</p>"` (23..27)
/// - `child` (`ViewId(1)`, stripping `"<p>"` and `"</p>"`):
///   `"Café \u{FB01}re \u{1F600} end"` (20 bytes)
/// - `grandchild` (`ViewId(2)`, expanding ligature `"\u{FB01}"` (3 bytes) -> `"fi"` (2 bytes)):
///   `"Café fire \u{1F600} end"` (19 bytes)
fn multi_hop_doc() -> (Document, ViewId, ViewId) {
    let src = "<p>Café \u{FB01}re \u{1F600} end</p>";
    assert_eq!(src.len(), 27);
    let mut doc = Document::new(src);
    let v0 = doc.get::<View>().unwrap().clone();

    let mut b1 = ViewBuilder::new(v0.text());
    b1.replace(3, "");
    b1.keep(23);
    b1.replace(27, "");
    let (t1, a1) = b1.finish();
    assert_eq!(t1, "Café \u{FB01}re \u{1F600} end");
    doc.add(&Key::<View>::named("child"), View::derived(&v0, t1, a1), Provenance::by("test")).unwrap();
    let v1 = doc.get_key(&Key::<View>::named("child")).unwrap().clone();

    let mut b2 = ViewBuilder::new(v1.text());
    b2.keep(6); // "Café "
    b2.replace(9, "fi"); // "\u{FB01}" -> "fi"
    let (t2, a2) = b2.finish();
    assert_eq!(t2, "Café fire \u{1F600} end");
    doc.add(&Key::<View>::named("grandchild"), View::derived(&v1, t2, a2), Provenance::by("test")).unwrap();
    let v2 = doc.get_key(&Key::<View>::named("grandchild")).unwrap().clone();

    (doc, v1.id(), v2.id())
}

#[test]
fn multi_hop_bidirectional_projection_and_utf16_codepoint_offsets() {
    let (doc, child_id, grand_id) = multi_hop_doc();

    // Define spans in grandchild ("Café fire 😀 end"):
    // 0: "Café" -> 0..5
    // 1: "fire" -> 6..10 (where "fi" came from 3-byte '\u{FB01}')
    // 2: "😀" -> 11..15
    // 3: discontinuous ["Café" (0..5), "end" (16..19)]
    let mut grand_spans = Spans::<TestUnits>::new(grand_id);
    grand_spans.push(Span::new(0, 5));
    grand_spans.push(Span::new(6, 10));
    grand_spans.push(Span::new(11, 15));
    grand_spans.try_push_fragments(&[Span::new(0, 5), Span::new(16, 19)]).unwrap();
    let grand_layer = TestUnits { spans: grand_spans };

    // Project grandchild -> child -> SOURCE -> child -> grandchild
    let child_layer = doc.project_layer(&grand_layer, child_id).unwrap();
    assert_eq!(child_layer.len(), 4);
    assert_eq!(child_layer.spans.get(Ix::new(0)), Span::new(0, 5));
    assert_eq!(child_layer.spans.get(Ix::new(1)), Span::new(6, 11)); // "\u{FB01}re" is 5 bytes in child
    assert_eq!(child_layer.spans.get(Ix::new(2)), Span::new(12, 16));
    let child_frags: Vec<Span> = child_layer.spans.fragments(Ix::new(3)).collect();
    assert_eq!(child_frags, vec![Span::new(0, 5), Span::new(17, 20)]);

    let src_layer = doc.project_layer(&child_layer, ViewId::SOURCE).unwrap();
    assert_eq!(src_layer.len(), 4);
    assert_eq!(src_layer.spans.get(Ix::new(0)), Span::new(3, 8)); // "Café" in "<p>Café \u{FB01}re \u{1F600} end</p>"
    assert_eq!(src_layer.spans.get(Ix::new(1)), Span::new(9, 14)); // "\u{FB01}re"
    assert_eq!(src_layer.spans.get(Ix::new(2)), Span::new(15, 19)); // "\u{1F600}"
    let src_frags: Vec<Span> = src_layer.spans.fragments(Ix::new(3)).collect();
    assert_eq!(src_frags, vec![Span::new(3, 8), Span::new(20, 23)]);

    // Direct grandchild -> SOURCE must match 2-hop grandchild -> child -> SOURCE
    let direct_src = doc.project_layer(&grand_layer, ViewId::SOURCE).unwrap();
    for i in 0..4 {
        let ix = Ix::new(i as u32);
        assert_eq!(direct_src.spans.get(ix), src_layer.spans.get(ix));
        let f1: Vec<Span> = direct_src.spans.fragments(ix).collect();
        let f2: Vec<Span> = src_layer.spans.fragments(ix).collect();
        assert_eq!(f1, f2);
    }

    // Round-trip back SOURCE -> child -> grandchild
    let round_child = doc.project_layer(&src_layer, child_id).unwrap();
    let round_grand = doc.project_layer(&round_child, grand_id).unwrap();
    for i in 0..4 {
        let ix = Ix::new(i as u32);
        assert_eq!(round_grand.spans.get(ix), grand_layer.spans.get(ix));
        let f1: Vec<Span> = round_grand.spans.fragments(ix).collect();
        let f2: Vec<Span> = grand_layer.spans.fragments(ix).collect();
        assert_eq!(f1, f2);
    }

    // Verify UTF-16 and Unicode code-point OffsetTable round-trip on SOURCE and grandchild
    // In SOURCE "<p>Café \u{FB01}re \u{1F600} end</p>":
    // "<p>" = 3 chars (3 UTF-16)
    // "Café " = 5 chars (5 UTF-16) -> ends at 8
    // "\u{FB01}re " = 4 chars (4 UTF-16) -> ends at 12
    // "\u{1F600}" = 1 code point (12..13), 2 UTF-16 code units (12..14)
    // " end" = 4 chars (code points 13..17, UTF-16 14..18)
    let emoji_src = src_layer.spans.get(Ix::new(2));
    let u16_emoji = doc.view_span_to_utf16(ViewId::SOURCE, emoji_src).unwrap();
    let cp_emoji = doc.view_span_to_code_point(ViewId::SOURCE, emoji_src).unwrap();
    assert_eq!(u16_emoji, Utf16Span::new(Utf16Offset(12), Utf16Offset(14)));
    assert_eq!(cp_emoji, CodePointSpan::new(CodePointOffset(12), CodePointOffset(13)));
    assert_eq!(doc.view_span_from_utf16(ViewId::SOURCE, u16_emoji).unwrap(), emoji_src);
    assert_eq!(doc.view_span_from_code_point(ViewId::SOURCE, cp_emoji).unwrap(), emoji_src);

    let end_frag = src_frags[1]; // "end" at bytes 20..23
    let u16_end = doc.view_span_to_utf16(ViewId::SOURCE, end_frag).unwrap();
    let cp_end = doc.view_span_to_code_point(ViewId::SOURCE, end_frag).unwrap();
    assert_eq!(u16_end, Utf16Span::new(Utf16Offset(15), Utf16Offset(18)));
    assert_eq!(cp_end, CodePointSpan::new(CodePointOffset(14), CodePointOffset(17)));
    assert_eq!(doc.source_utf16_to_view(grand_id, u16_end).unwrap(), Span::new(16, 19));
    assert_eq!(doc.source_code_point_to_view(grand_id, cp_end).unwrap(), Span::new(16, 19));
}

#[test]
fn alternatives_and_project_view_step_preserve_edges_and_groups() {
    let (mut doc, child_id, grand_id) = multi_hop_doc();

    // Add Alternatives<TestUnits> on grandchild and project to SOURCE
    let mut alt = Alternatives::<TestUnits>::new(grand_id);
    alt.push(Span::new(0, 5), 0, 1, ModeSet::ALL);
    alt.push(Span::new(6, 10), 1, 1, ModeSet::ALL);
    let alt_src = doc.project_layer(&alt, ViewId::SOURCE).unwrap();
    assert_eq!(alt_src.check(), Ok(()));
    assert_eq!(alt_src.spans.get(Ix::new(0)), Span::new(3, 8));
    assert_eq!(alt_src.spans.get(Ix::new(1)), Span::new(9, 14));

    // Annotator producing TestUnits on grandchild view, then ProjectView step
    // projecting to child and SOURCE while Edges and Groups reference Ix<TestUnits>.
    struct GrandUnitsAnnotator;
    impl Annotator for GrandUnitsAnnotator {
        const ID: &'static str = "test:grand_units";
        type Reads = (View,);
        type Writes = (TestUnits,);

        fn annotate(&self, (view,): Refs<'_, Self>) -> Result<Self::Writes, Error> {
            let mut spans = Spans::new(view.id());
            spans.push(Span::new(0, 5));
            spans.push(Span::new(6, 10));
            spans.push(Span::new(16, 19));
            Ok((TestUnits { spans },))
        }
    }

    let grand_key = Key::<TestUnits>::named("grand");
    let child_key = Key::<TestUnits>::named("child");
    let analyzer = Analyzer::builder()
        .given(Key::<View>::named("child"))
        .given(Key::<View>::named("grandchild"))
        .step(turbo_analysis::Bound::new(GrandUnitsAnnotator).read(0, "grandchild").write(0, "grand"))
        .step(ProjectView::to_view(grand_key, child_key, Key::<View>::named("child")).unwrap())
        .step(ProjectView::new(child_key, Key::<TestUnits>::DEFAULT).unwrap())
        .build()
        .unwrap();

    let results = analyzer.run(std::slice::from_mut(&mut doc));
    assert!(results[0].is_ok());

    let projected = doc.get::<TestUnits>().unwrap();
    assert_eq!(projected.spans.view(), ViewId::SOURCE);
    assert_eq!(projected.len(), 3);
    assert_eq!(projected.spans.get(Ix::new(0)), Span::new(3, 8));
    assert_eq!(projected.spans.get(Ix::new(1)), Span::new(9, 14));
    assert_eq!(projected.spans.get(Ix::new(2)), Span::new(20, 23));
    assert_eq!(doc.require(&child_key).unwrap().spans.view(), child_id);

    // Attach Edges<TestUnits, TestUnits> and Groups<TestUnits> to the projected SOURCE layer
    let kinds = TagSet::new("rel", &["link"]);
    let edges = Edges::<TestUnits, TestUnits>::new(
        kinds,
        [Edge { from: Ix::new(0), to: Ix::new(2), kind: Tag(0), score: 1.0 }],
    );
    doc.add(&Key::DEFAULT, edges, Provenance::by("test")).unwrap();
    let groups = Groups::<TestUnits>::new([[Ix::new(0), Ix::new(1), Ix::new(2)]]);
    doc.add(&Key::DEFAULT, groups, Provenance::by("test")).unwrap();
}
