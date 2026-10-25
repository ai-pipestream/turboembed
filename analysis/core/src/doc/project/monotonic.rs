//! The shared span-projection body and its forward-cursor variant:
//! [`Document::project_spans`] maps each span by binary search, and
//! [`Document::project_spans_monotonic`] maps ordered spans through
//! one [`AlignmentCursor`] per view on the path.

use crate::error::Error;
use crate::layer::{Key, KeyId, Layer};
use crate::span::{Ix, Span, Spans};
use crate::text::{AlignmentCursor, ViewId};

use super::super::{Document, View, check_spans};

impl Document {
    /// The bounds checks and LCA path shared by the span projections:
    /// the views to walk up from `spans.view()` and down to `to`.
    pub(super) fn projection_path<L: Layer>(
        &self,
        spans: &Spans<L>,
        to: ViewId,
    ) -> Result<(KeyId, Vec<&View>, Vec<&View>), Error> {
        let layer_id = Key::<L>::DEFAULT.id();
        let from = spans.view();
        let from_view = self.view_checked(from, layer_id)?;
        let ts = spans.text_spans();
        check_spans(&from_view.text, ts.start, ts.end).map_err(|detail| Error::Bounds {
            layer: layer_id,
            view: from,
            detail,
        })?;
        if !ts.frag_item_end.is_empty() {
            check_spans(&from_view.text, ts.frag_start, ts.frag_end).map_err(|detail| Error::Bounds {
                layer: layer_id,
                view: from,
                detail,
            })?;
        }
        let (_, up_from, down_to) = self.lca_path(from, to, layer_id)?;
        Ok((layer_id, up_from, down_to))
    }

    /// [`Document::project_spans`] with how one span is mapped given:
    /// the item-for-item walk and fragment coalescing are shared, the
    /// caller chooses binary searches or forward cursors.
    pub(super) fn project_with<L: Layer>(
        &self,
        spans: &Spans<L>,
        to: ViewId,
        layer_id: KeyId,
        map_one: &mut dyn FnMut(Span) -> Span,
    ) -> Result<Spans<L>, Error> {
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

    /// Projects an entire [`Spans<L>`] column as
    /// [`Document::project_spans`] does, mapping each span through
    /// one forward [`AlignmentCursor`] per view on the LCA path
    /// instead of a binary search per offset: $O(\text{edits} +
    /// \text{spans})$ per view. The spans must be ordered without
    /// overlaps ([`Spans::is_ordered`]), so envelope and fragment
    /// offsets never decrease along the walk; a layer that is not
    /// ordered, or carries discontinuous items (whose envelopes the
    /// coalescing walk maps after their fragments, against the
    /// cursor's forward rule), is projected by
    /// [`Document::project_spans`] itself. Results match
    /// `project_spans` bit-for-bit either way.
    pub fn project_spans_monotonic<L: Layer>(&self, spans: &Spans<L>, to: ViewId) -> Result<Spans<L>, Error> {
        if !spans.is_ordered() || spans.has_discontinuous() {
            return self.project_spans(spans, to);
        }
        let (layer_id, up_from, down_to) = self.projection_path(spans, to)?;
        let mut up: Vec<AlignmentCursor<'_>> = up_from.iter().map(|v| v.alignment.cursor()).collect();
        let mut down: Vec<AlignmentCursor<'_>> = down_to.iter().map(|v| v.alignment.cursor()).collect();
        self.project_with(spans, to, layer_id, &mut |mut s: Span| -> Span {
            for c in &mut up {
                s = c.span_to_parent(s);
            }
            for c in &mut down {
                s = c.span_to_child(s);
            }
            s
        })
    }
}
