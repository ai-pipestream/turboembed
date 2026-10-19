//! Lowest Common Ancestor (LCA) cross-view bi-directional offset, span, and
//! layer projection, plus typed unit-span conversions on [`Document`].

use std::sync::Arc;

use crate::error::Error;
use crate::layer::{Key, KeyId, Layer};
use crate::offsets::{CodePointSpan, Utf16Span};
use crate::span::{Ix, Span, Spans};
use crate::text::{Alignment, Side, ViewId};

use super::{Document, View, check_spans};

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
        let (_, up_from, down_to) = self.lca_path(from, to, Key::<View>::DEFAULT.id())?;
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
        let (_, up_from, down_to) = self.lca_path(from, to, Key::<View>::DEFAULT.id())?;
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
        let layer_id = Key::<L>::DEFAULT.id();
        let (_, up_from, down_to) = self.lca_path(spans.view(), to, layer_id)?;
        let map_one = |mut s: Span| -> Span {
            for v in &up_from {
                s = v.alignment.span_to_parent(s);
            }
            for v in &down_to {
                s = v.alignment.span_to_child(s);
            }
            s
        };

        let mut out = Spans::with_capacity(to, spans.len());
        if !spans.has_discontinuous() {
            for (_, s) in spans.iter() {
                out.push(map_one(s));
            }
            return Ok(out);
        }

        let mut coalesced: Vec<Span> = Vec::new();
        for idx in 0..spans.len() {
            let ix = Ix::new(idx as u32);
            if !spans.is_discontinuous(ix) {
                out.push(map_one(spans.get(ix)));
                continue;
            }
            coalesced.clear();
            for frag in spans.fragments(ix) {
                let mapped = map_one(frag);
                if mapped.is_empty() {
                    continue;
                }
                if let Some(last) = coalesced.last_mut()
                    && last.end >= mapped.start
                {
                    last.end = last.end.max(mapped.end);
                } else {
                    coalesced.push(mapped);
                }
            }
            match coalesced.len() {
                0 => {
                    let env = map_one(spans.get(ix));
                    out.push(Span::new(env.start, env.start));
                }
                1 => {
                    out.push(coalesced[0]);
                }
                _ => {
                    out.try_push_fragments(&coalesced)
                        .map_err(|e| Error::Invalid { layer: layer_id, detail: e.to_string() })?;
                }
            }
        }
        Ok(out)
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
        // Source: "a\u{1F600}\u{FB01}b"
        // 'a' (0..1, u16 0..1, cp 0..1)
        // '😀' (1..5, u16 1..3, cp 1..2)
        // 'ﬁ' (5..8, u16 3..4, cp 2..3)
        // 'b' (8..9, u16 4..5, cp 3..4)
        let mut doc = Document::new("a\u{1F600}\u{FB01}b");
        let v0 = doc.get::<View>().unwrap().clone();
        let mut b = ViewBuilder::new(v0.text());
        b.keep(5);
        b.replace(8, "fi");
        let (t1, a1) = b.finish();
        doc.add(&Key::<View>::named("nfkc"), View::derived(&v0, t1, a1), Provenance::by("test")).unwrap();
        let v1_id = doc.get_key(&Key::<View>::named("nfkc")).unwrap().id();

        // In v1 ("a😀fib"), "fi" is bytes 5..7, UTF-16 3..5, code points 2..4.
        let fi_v1 = Span::new(5, 7);
        assert_eq!(doc.view_span_to_utf16(v1_id, fi_v1).unwrap(), Utf16Span::new(Utf16Offset(3), Utf16Offset(5)));
        assert_eq!(
            doc.view_span_to_code_point(v1_id, fi_v1).unwrap(),
            CodePointSpan::new(CodePointOffset(2), CodePointOffset(4))
        );

        // Mapping "fi" in v1 to source UTF-16 / code-point spans gives 'ﬁ' (u16 3..4, cp 2..3).
        let src_u16 = doc.view_to_source_utf16(v1_id, fi_v1).unwrap();
        assert_eq!(src_u16, Utf16Span::new(Utf16Offset(3), Utf16Offset(4)));
        assert_eq!(doc.source_utf16_to_view(v1_id, src_u16).unwrap(), fi_v1);

        let src_cp = doc.view_to_source_code_point(v1_id, fi_v1).unwrap();
        assert_eq!(src_cp, CodePointSpan::new(CodePointOffset(2), CodePointOffset(3)));
        assert_eq!(doc.source_code_point_to_view(v1_id, src_cp).unwrap(), fi_v1);
    }
}
