//! Lowest Common Ancestor (LCA) cross-view bi-directional offset, span, and
//! layer projection, plus typed unit-span conversions on [`Document`].

use std::sync::Arc;

use crate::error::Error;
use crate::layer::{Key, KeyId, Layer};
use crate::offsets::{CodePointSpan, Utf16Span};
use crate::span::{Span, Spans};
use crate::text::{Alignment, Side, ViewId};

use super::{Document, View, check_spans};

mod monotonic;

impl Document {
    fn view_checked(&self, v: ViewId, layer: KeyId) -> Result<&Arc<View>, Error> {
        self.views.get(v.0 as usize).ok_or_else(|| Error::Bounds { layer, view: v, detail: "no such view".into() })
    }

    /// Returns `(lca, up_from, down_to)` where `up_from` lists the views to
    /// step up from `from` to `lca` (excluding `lca`), and `down_to` lists
    /// the views to step down from `lca` to `to` (excluding `lca`, in
    /// top-down parent-to-child order).
    fn lca_path(&self, from: ViewId, to: ViewId, layer: KeyId) -> Result<(ViewId, Vec<&View>, Vec<&View>), Error> {
        let mut from_chain: Vec<&View> = Vec::new();
        let mut cur = self.view_checked(from, layer)?.as_ref();
        from_chain.push(cur);
        while let Some(p) = cur.parent {
            cur = self.view_checked(p, layer)?.as_ref();
            from_chain.push(cur);
        }

        let mut down_to: Vec<&View> = Vec::new();
        let mut cur_to = self.view_checked(to, layer)?.as_ref();
        loop {
            if let Some(pos) = from_chain.iter().position(|v| v.id == cur_to.id) {
                let lca = cur_to.id;
                from_chain.truncate(pos);
                down_to.reverse();
                return Ok((lca, from_chain, down_to));
            }
            let Some(p) = cur_to.parent else {
                return Err(Error::Bounds { layer, view: to, detail: "views do not share a common ancestor".into() });
            };
            down_to.push(cur_to);
            cur_to = self.view_checked(p, layer)?.as_ref();
        }
    }

    /// Finds the lowest common ancestor (LCA) of views `a` and `b` in the
    /// document's view tree.
    pub fn lca_view(&self, a: ViewId, b: ViewId) -> Result<ViewId, Error> {
        let (lca, _, _) = self.lca_path(a, b, Key::<View>::DEFAULT.id())?;
        Ok(lca)
    }

    /// Composes the minimal single [`Alignment`] from view `from` (parent
    /// side) to view `to` (child side) along the path through
    /// [`Document::lca_view`].
    pub fn view_alignment(&self, from: ViewId, to: ViewId) -> Result<Alignment, Error> {
        let (_, up_from, down_to) = self.lca_path(from, to, Key::<View>::DEFAULT.id())?;
        let mut composed = Alignment::identity();
        for v in up_from {
            composed = composed.compose(&v.alignment.invert());
        }
        for v in down_to {
            composed = composed.compose(&v.alignment);
        }
        Ok(composed)
    }

    /// Maps byte offset `x` (on `side` of a span) from view `from` to view
    /// `to`, walking only up to `lca_view(from, to)` and down to `to`.
    pub fn map_offset(&self, from: ViewId, to: ViewId, mut x: u32, side: Side) -> Result<u32, Error> {
        let layer_id = Key::<View>::DEFAULT.id();
        let from_view = self.view_checked(from, layer_id)?;
        check_spans(&from_view.text, &[x], &[x]).map_err(|detail| Error::Bounds {
            layer: layer_id,
            view: from,
            detail,
        })?;
        let (_, up_from, down_to) = self.lca_path(from, to, layer_id)?;
        for v in up_from {
            x = v.alignment.to_parent(x, side);
        }
        for v in down_to {
            x = v.alignment.to_child(x, side);
        }
        Ok(x)
    }

    /// Maps byte [`Span`] `s` from view `from` to view `to`, walking only up
    /// to `lca_view(from, to)` and down to `to`.
    pub fn map_span(&self, from: ViewId, to: ViewId, mut s: Span) -> Result<Span, Error> {
        let layer_id = Key::<View>::DEFAULT.id();
        let from_view = self.view_checked(from, layer_id)?;
        check_spans(&from_view.text, &[s.start], &[s.end]).map_err(|detail| Error::Bounds {
            layer: layer_id,
            view: from,
            detail,
        })?;
        let (_, up_from, down_to) = self.lca_path(from, to, layer_id)?;
        for v in up_from {
            s = v.alignment.span_to_parent(s);
        }
        for v in down_to {
            s = v.alignment.span_to_child(s);
        }
        Ok(s)
    }

    /// Projects an entire [`Spans<L>`] column from `spans.view()` into view
    /// `to` along the LCA path, preserving item indices `Ix<L>` and
    /// discontinuous [`crate::Fragments`] invariants (coalescing fragments
    /// whose intermediate gap collapsed to zero bytes under an edit).
    pub fn project_spans<L: Layer>(&self, spans: &Spans<L>, to: ViewId) -> Result<Spans<L>, Error> {
        let (layer_id, up_from, down_to) = self.projection_path(spans, to)?;
        self.project_with(spans, to, layer_id, &mut |mut s: Span| -> Span {
            for v in &up_from {
                s = v.alignment.span_to_parent(s);
            }
            for v in &down_to {
                s = v.alignment.span_to_child(s);
            }
            s
        })
    }

    /// Converts byte [`Span`] `s` in view `v` into a [`Utf16Span`] in `v`.
    pub fn view_span_to_utf16(&self, v: ViewId, s: Span) -> Result<Utf16Span, Error> {
        let view = self.view_checked(v, Key::<View>::DEFAULT.id())?;
        check_spans(&view.text, &[s.start], &[s.end]).map_err(|detail| Error::Bounds {
            layer: Key::<View>::DEFAULT.id(),
            view: v,
            detail,
        })?;
        Ok(view.span_to_utf16(s))
    }

    /// Converts [`Utf16Span`] `s` in view `v` into a byte [`Span`] in `v`.
    pub fn view_span_from_utf16(&self, v: ViewId, s: Utf16Span) -> Result<Span, Error> {
        let view = self.view_checked(v, Key::<View>::DEFAULT.id())?;
        view.span_from_utf16(s)
    }

    /// Converts a source (view 0) [`Utf16Span`] into a byte [`Span`] in view `v`.
    pub fn source_utf16_to_view(&self, v: ViewId, s: Utf16Span) -> Result<Span, Error> {
        let _ = self.view_checked(v, Key::<View>::DEFAULT.id())?;
        let src_span = self.views[0].span_from_utf16(s)?;
        Ok(self.from_source(v, src_span))
    }

    /// Maps byte [`Span`] `s` in view `v` to source (view 0) and converts it
    /// into a source [`Utf16Span`].
    pub fn view_to_source_utf16(&self, v: ViewId, s: Span) -> Result<Utf16Span, Error> {
        let view = self.view_checked(v, Key::<View>::DEFAULT.id())?;
        check_spans(&view.text, &[s.start], &[s.end]).map_err(|detail| Error::Bounds {
            layer: Key::<View>::DEFAULT.id(),
            view: v,
            detail,
        })?;
        let src_span = self.to_source(v, s);
        Ok(self.views[0].span_to_utf16(src_span))
    }

    /// Converts byte [`Span`] `s` in view `v` into a [`CodePointSpan`] in `v`.
    pub fn view_span_to_code_point(&self, v: ViewId, s: Span) -> Result<CodePointSpan, Error> {
        let view = self.view_checked(v, Key::<View>::DEFAULT.id())?;
        check_spans(&view.text, &[s.start], &[s.end]).map_err(|detail| Error::Bounds {
            layer: Key::<View>::DEFAULT.id(),
            view: v,
            detail,
        })?;
        Ok(view.span_to_code_point(s))
    }

    /// Converts [`CodePointSpan`] `s` in view `v` into a byte [`Span`] in `v`.
    pub fn view_span_from_code_point(&self, v: ViewId, s: CodePointSpan) -> Result<Span, Error> {
        let view = self.view_checked(v, Key::<View>::DEFAULT.id())?;
        view.span_from_code_point(s)
    }

    /// Converts a source (view 0) [`CodePointSpan`] into a byte [`Span`] in view `v`.
    pub fn source_code_point_to_view(&self, v: ViewId, s: CodePointSpan) -> Result<Span, Error> {
        let _ = self.view_checked(v, Key::<View>::DEFAULT.id())?;
        let src_span = self.views[0].span_from_code_point(s)?;
        Ok(self.from_source(v, src_span))
    }

    /// Maps byte [`Span`] `s` in view `v` to source (view 0) and converts it
    /// into a source [`CodePointSpan`].
    pub fn view_to_source_code_point(&self, v: ViewId, s: Span) -> Result<CodePointSpan, Error> {
        let view = self.view_checked(v, Key::<View>::DEFAULT.id())?;
        check_spans(&view.text, &[s.start], &[s.end]).map_err(|detail| Error::Bounds {
            layer: Key::<View>::DEFAULT.id(),
            view: v,
            detail,
        })?;
        let src_span = self.to_source(v, s);
        Ok(self.views[0].span_to_code_point(src_span))
    }
}

#[cfg(test)]
mod tests {
    use crate::doc::{Document, View};
    use crate::layer::{Anchor, Key, Layer, Provenance};
    use crate::offsets::{CodePointOffset, CodePointSpan, Utf16Offset, Utf16Span};
    use crate::span::{Overlap, Span, Spans, TextSpans};
    use crate::text::{Side, ViewBuilder, ViewId};

    struct Phrases {
        spans: Spans<Phrases>,
    }

    impl Layer for Phrases {
        const ID: &'static str = "test:phrases";
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

    #[test]
    fn lca_view_and_cross_view_mapping_avoid_source_widening() {
        // View 0: "n=\u{661}\u{662}\u{663};"
        // View 1 (derived from 0): replaces all 3 Arabic-Indic digits in one
        // call with "123", so in 0<->1 the whole "123" (2..5) is a single edit.
        let mut doc = Document::new("n=\u{661}\u{662}\u{663};");
        let v0 = doc.get::<View>().unwrap().clone();
        let mut b1 = ViewBuilder::new(v0.text());
        b1.keep(2);
        b1.replace(8, "123");
        let (t1, a1) = b1.finish();
        doc.add(&Key::<View>::named("v1"), View::derived(&v0, t1, a1), Provenance::by("test")).unwrap();
        let v1 = doc.get_key(&Key::<View>::named("v1")).unwrap().clone();
        assert_eq!(v1.id(), ViewId(1));

        // View 2 (derived from 1): inserts "[" at start -> "[n=123;"
        let mut b2 = ViewBuilder::new(v1.text());
        b2.replace(0, "[");
        let (t2, a2) = b2.finish();
        doc.add(&Key::<View>::named("v2"), View::derived(&v1, t2, a2), Provenance::by("test")).unwrap();
        let v2_id = doc.get_key(&Key::<View>::named("v2")).unwrap().id();

        // View 3 (derived from 1): replaces ";" at 5..6 with "];" -> "n=123];"
        let mut b3 = ViewBuilder::new(v1.text());
        b3.keep(5);
        b3.replace(6, "];");
        let (t3, a3) = b3.finish();
        doc.add(&Key::<View>::named("v3"), View::derived(&v1, t3, a3), Provenance::by("test")).unwrap();
        let v3_id = doc.get_key(&Key::<View>::named("v3")).unwrap().id();

        assert_eq!(doc.lca_view(v2_id, v3_id).unwrap(), ViewId(1));
        assert_eq!(doc.lca_view(v2_id, v1.id()).unwrap(), ViewId(1));
        assert_eq!(doc.lca_view(v2_id, ViewId::SOURCE).unwrap(), ViewId::SOURCE);

        // In View 2 ("[n=123;"), the middle digit '2' is at 4..5.
        // Because LCA(v2, v3) is v1 (not v0), mapping 4..5 from v2 to v3
        // maps to exact '2' (3..4 in v3), never widening to "123" (2..5) via v0!
        let mid_in_v2 = Span::new(4, 5);
        assert_eq!(&doc.text(v2_id)[mid_in_v2.range()], "2");
        let mid_in_v3 = doc.map_span(v2_id, v3_id, mid_in_v2).unwrap();
        assert_eq!(mid_in_v3, Span::new(3, 4));
        assert_eq!(&doc.text(v3_id)[mid_in_v3.range()], "2");
        assert_eq!(doc.map_offset(v2_id, v3_id, 4, Side::Start).unwrap(), 3);

        // Composed view_alignment(v2, v3) agrees with map_span / map_offset.
        let align_2_to_3 = doc.view_alignment(v2_id, v3_id).unwrap();
        assert_eq!(align_2_to_3.span_to_child(mid_in_v2), mid_in_v3);
        assert_eq!(align_2_to_3.span_to_parent(mid_in_v3), mid_in_v2);
    }

    #[test]
    fn project_spans_handles_continuous_and_discontinuous_fragments() {
        // View 0: "take -- it -- off"
        // View 1: deletes first " -- " (4..8 -> " ") and deletes " -- it -- " (4..14 -> " ") in View 2.
        let mut doc = Document::new("take -- it -- off");
        let v0 = doc.get::<View>().unwrap().clone();

        let mut b1 = ViewBuilder::new(v0.text());
        b1.keep(4);
        b1.replace(8, " ");
        b1.keep(10);
        b1.replace(14, " ");
        let (t1, a1) = b1.finish();
        assert_eq!(t1, "take it off");
        doc.add(&Key::<View>::named("clean"), View::derived(&v0, t1, a1), Provenance::by("test")).unwrap();
        let v1 = doc.get_key(&Key::<View>::named("clean")).unwrap().clone();

        // View 2 (from v1 "take it off"): deletes " it " (4..8) so "take" and "off" touch ("takeoff").
        let mut b2 = ViewBuilder::new(v1.text());
        b2.keep(4);
        b2.replace(8, "");
        let (t2, a2) = b2.finish();
        assert_eq!(t2, "takeoff");
        doc.add(&Key::<View>::named("collapsed"), View::derived(&v1, t2, a2), Provenance::by("test")).unwrap();
        let v2_id = doc.get_key(&Key::<View>::named("collapsed")).unwrap().id();

        let mut spans = Spans::<Phrases>::new(ViewId::SOURCE);
        // Item 0: discontinuous "take" (0..4) + "off" (14..17)
        let ix0 = spans.push_fragments(&[Span::new(0, 4), Span::new(14, 17)]);
        // Item 1: continuous "it" (8..10)
        let ix1 = spans.push(Span::new(8, 10));

        // Project from v0 to v1 ("take it off"): "take" is 0..4, "off" is 8..11 (still discontinuous).
        let p1 = doc.project_spans(&spans, v1.id()).unwrap();
        assert_eq!(p1.view(), v1.id());
        assert!(p1.is_discontinuous(ix0));
        let frags1: Vec<Span> = p1.fragments(ix0).collect();
        assert_eq!(frags1, vec![Span::new(0, 4), Span::new(8, 11)]);
        assert_eq!(p1.get(ix1), Span::new(5, 7));

        // Project back from v1 to v0: round-trips!
        let back0 = doc.project_spans(&p1, ViewId::SOURCE).unwrap();
        let frags0: Vec<Span> = back0.fragments(ix0).collect();
        assert_eq!(frags0, vec![Span::new(0, 4), Span::new(14, 17)]);
        assert_eq!(back0.get(ix1), Span::new(8, 10));

        // Project from v0 to v2 ("takeoff"): gap between "take" (0..4) and "off" (4..7) collapsed,
        // so the two touching fragments coalesce into a single continuous span 0..7!
        let p2 = doc.project_spans(&spans, v2_id).unwrap();
        assert_eq!(p2.view(), v2_id);
        assert!(!p2.is_discontinuous(ix0));
        assert_eq!(p2.get(ix0), Span::new(0, 7));
        assert_eq!(p2.get(ix1), Span::new(4, 4));
        p2.check(Overlap::Any).unwrap();
    }

    #[test]
    fn document_typed_utf16_and_code_point_helpers_round_trip() {
        // Source: "a\u{1F600}\u{FB01}b" ('a' 0..1, '😀' 1..5, 'ﬁ' 5..8, 'b' 8..9)
        let mut doc = Document::new("a\u{1F600}\u{FB01}b");
        let v0 = doc.get::<View>().unwrap().clone();
        let mut b = ViewBuilder::new(v0.text());
        b.keep(5);
        b.replace(8, "fi");
        let (t1, a1) = b.finish();
        doc.add(&Key::<View>::named("nfkc"), View::derived(&v0, t1, a1), Provenance::by("test")).unwrap();
        let v1_id = doc.get_key(&Key::<View>::named("nfkc")).unwrap().id();

        let fi_v1 = Span::new(5, 7);
        assert_eq!(doc.view_span_to_utf16(v1_id, fi_v1).unwrap(), Utf16Span::new(Utf16Offset(3), Utf16Offset(5)));
        assert_eq!(
            doc.view_span_to_code_point(v1_id, fi_v1).unwrap(),
            CodePointSpan::new(CodePointOffset(2), CodePointOffset(4))
        );

        let src_u16 = doc.view_to_source_utf16(v1_id, fi_v1).unwrap();
        assert_eq!(src_u16, Utf16Span::new(Utf16Offset(3), Utf16Offset(4)));
        assert_eq!(doc.source_utf16_to_view(v1_id, src_u16).unwrap(), fi_v1);

        let src_cp = doc.view_to_source_code_point(v1_id, fi_v1).unwrap();
        assert_eq!(src_cp, CodePointSpan::new(CodePointOffset(2), CodePointOffset(3)));
        assert_eq!(doc.source_code_point_to_view(v1_id, src_cp).unwrap(), fi_v1);
    }

    #[test]
    fn map_offset_map_span_and_project_spans_reject_invalid_coordinates() {
        let doc = Document::new("é");
        assert!(doc.map_offset(ViewId::SOURCE, ViewId::SOURCE, 1, Side::Start).is_err());
        assert!(doc.map_offset(ViewId::SOURCE, ViewId::SOURCE, 99, Side::Start).is_err());
        assert!(doc.map_span(ViewId::SOURCE, ViewId::SOURCE, Span::new(1, 99)).is_err());
        assert!(doc.map_span(ViewId::SOURCE, ViewId::SOURCE, Span::new(0, 1)).is_err());

        let mut bad_spans = Spans::<Phrases>::new(ViewId::SOURCE);
        bad_spans.push(Span::new(1, 99));
        assert!(doc.project_spans(&bad_spans, ViewId::SOURCE).is_err());
    }

    #[test]
    fn composed_view_alignment_round_trips_through_document_graph_and_map_span_steps_sequentially() {
        use crate::export::{Codecs, DocumentGraph, export, import};

        let mut b1 = ViewBuilder::new("aXb");
        b1.keep(1);
        b1.replace(2, "");
        let (_, ab) = b1.finish();
        let mut b2 = ViewBuilder::new("ab");
        b2.keep(1);
        b2.replace(1, "+");
        let (final_text, bc) = b2.finish();
        let composed = ab.compose(&bc);

        let mut doc = Document::new("aXb");
        let v0 = doc.get::<View>().unwrap().clone();
        doc.add(&Key::<View>::named("composed"), View::derived(&v0, final_text, composed), Provenance::by("test"))
            .unwrap();
        let codecs = Codecs::new();
        let bytes = export(&doc, &codecs).unwrap().to_bytes().unwrap();
        let restored = import(&DocumentGraph::from_bytes(&bytes).unwrap(), &codecs).unwrap();
        let before_align = doc.get_key(&Key::<View>::named("composed")).unwrap().alignment();
        let after_align = restored.get_key(&Key::<View>::named("composed")).unwrap().alignment();
        assert_eq!(before_align, after_align);
        for x in 0..=3 {
            for side in [Side::Start, Side::End] {
                assert_eq!(before_align.to_child(x, side), after_align.to_child(x, side));
                assert_eq!(before_align.to_parent(x, side), after_align.to_parent(x, side));
            }
        }

        // Step-by-step empty-span projection across intermediate views:
        // "abcd" -> "axbcd" (insert 'x' at 1) -> "ybcd" (replace 0..2 with 'y').
        let mut s_doc = Document::new("abcd");
        let sv0 = s_doc.get::<View>().unwrap().clone();
        let mut sb1 = ViewBuilder::new(sv0.text());
        sb1.keep(1);
        sb1.replace(1, "x");
        let (st1, sa1) = sb1.finish();
        s_doc.add(&Key::<View>::named("v1"), View::derived(&sv0, st1, sa1.clone()), Provenance::by("test")).unwrap();
        let sv1 = s_doc.get_key(&Key::<View>::named("v1")).unwrap().clone();
        let mut sb2 = ViewBuilder::new(sv1.text());
        sb2.replace(2, "y");
        let (st2, sa2) = sb2.finish();
        s_doc.add(&Key::<View>::named("v2"), View::derived(&sv1, st2, sa2.clone()), Provenance::by("test")).unwrap();
        let sv2_id = s_doc.get_key(&Key::<View>::named("v2")).unwrap().id();

        let empty_1 = Span::new(1, 1);
        assert_eq!(sa2.span_to_child(sa1.span_to_child(empty_1)), Span::new(0, 1));
        assert_eq!(s_doc.map_span(ViewId::SOURCE, sv2_id, empty_1).unwrap(), Span::new(0, 1));
        for z in 0..=4 {
            let empty_z = Span::new(z, z);
            assert_eq!(
                s_doc.map_span(sv2_id, ViewId::SOURCE, empty_z).unwrap(),
                sa1.span_to_parent(sa2.span_to_parent(empty_z))
            );
        }
    }

    /// The cursor projection matches the binary-search projection
    /// bit-for-bit on ordered spans across a two-view chain, and the
    /// discontinuous fallback returns the same column too.
    #[test]
    fn project_spans_monotonic_matches_project_spans() {
        // Source "aa bb cc dd ee" -> v1 drops " cc" and inserts "XX"
        // at 2 -> v2 replaces "dd" with "Y".
        let mut doc = Document::new("aa bb cc dd ee");
        let v0 = doc.get::<View>().unwrap().clone();
        let mut b1 = ViewBuilder::new(v0.text());
        b1.keep(2);
        b1.replace(2, "XX");
        b1.keep(5);
        b1.replace(8, "");
        let (t1, a1) = b1.finish();
        doc.add(&Key::<View>::named("v1"), View::derived(&v0, t1, a1), Provenance::by("test")).unwrap();
        let v1 = doc.get_key(&Key::<View>::named("v1")).unwrap().clone();
        let dd = v1.text().find("dd").unwrap() as u32;
        let mut b2 = ViewBuilder::new(v1.text());
        b2.keep(dd as usize);
        b2.replace((dd + 2) as usize, "Y");
        let (t2, a2) = b2.finish();
        doc.add(&Key::<View>::named("v2"), View::derived(&v1, t2, a2), Provenance::by("test")).unwrap();
        let v2_id = doc.get_key(&Key::<View>::named("v2")).unwrap().id();

        let mut spans = Spans::<Phrases>::new(ViewId::SOURCE);
        for s in [(0, 0), (0, 2), (2, 2), (2, 3), (3, 5), (5, 5), (6, 8), (8, 9), (9, 11), (11, 11), (12, 14), (14, 14)]
        {
            spans.push(Span::new(s.0, s.1));
        }
        assert!(spans.is_ordered());
        let want = doc.project_spans(&spans, v2_id).unwrap();
        let got = doc.project_spans_monotonic(&spans, v2_id).unwrap();
        assert_eq!(got.columns(), want.columns());
        let back = doc.project_spans(&want, ViewId::SOURCE).unwrap();
        let back_mono = doc.project_spans_monotonic(&want, ViewId::SOURCE).unwrap();
        assert_eq!(back_mono.columns(), back.columns());

        // Discontinuous items delegate, still bit-for-bit.
        let mut disc = Spans::<Phrases>::new(ViewId::SOURCE);
        disc.try_push_fragments(&[Span::new(0, 2), Span::new(9, 11)]).unwrap();
        disc.push(Span::new(12, 14));
        let want = doc.project_spans(&disc, v2_id).unwrap();
        let got = doc.project_spans_monotonic(&disc, v2_id).unwrap();
        assert_eq!(got.columns(), want.columns());
        for (ix, _) in want.iter() {
            assert_eq!(got.fragments(ix).collect::<Vec<_>>(), want.fragments(ix).collect::<Vec<_>>());
        }
    }
}
