//! Native spans on the result's view with what projection did to them
//! (issue #1907, item 7): every `SpanFragment` says whether it is the
//! native span exactly, widened over a lossy edit, or an empty anchor on
//! deleted text, straight from `Document::map_span_outcome` and
//! `Document::project_fragments`; a coalesced fragment lists its native
//! members by index, each with its own outcome.

use turbo_analysis::{Ix, ProjectedFragment, Span, SpanOutcome, Spans, ViewId};

use super::ProjectionCx;
use crate::proto::result::{FragmentMember, SpanFragment, SpanOutcome as WireOutcome};

impl SpanFragment {
    /// An exact fragment with no members: a span on the result's own
    /// view, or one the message does not project.
    pub fn exact(start: u32, end: u32) -> SpanFragment {
        SpanFragment { start, end, outcome: WireOutcome::Exact as i32, members: Vec::new() }
    }
}

/// The wire value of a native outcome.
pub fn outcome_to_proto(outcome: SpanOutcome) -> WireOutcome {
    match outcome {
        SpanOutcome::Exact => WireOutcome::Exact,
        SpanOutcome::Widened => WireOutcome::Widened,
        SpanOutcome::Deleted => WireOutcome::Deleted,
    }
}

impl ProjectionCx<'_> {
    /// `span` of `layer_view` on the target view as one fragment with
    /// its outcome: exact on the target's own view, else what
    /// `Document::map_span_outcome` says. A span the document refuses to
    /// map is kept for `document_to_typed` to fail with.
    pub(crate) fn project_wire_span(&self, layer_view: ViewId, span: Span) -> SpanFragment {
        if layer_view == self.target_view {
            let (start, end) = self.offsetter.to_wire(self.target_text, span);
            return SpanFragment::exact(start, end);
        }
        match self.doc.map_span_outcome(layer_view, self.target_view, span) {
            Ok(p) => {
                let (start, end) = self.offsetter.to_wire(self.target_text, p.span);
                SpanFragment { start, end, outcome: outcome_to_proto(p.outcome) as i32, members: Vec::new() }
            }
            Err(_) => self.refuse(layer_view, span),
        }
    }

    /// An item's fragments on the target view: on the target's own view
    /// each native fragment as it is, exact; onto another view what
    /// `Document::project_fragments` gives, overlapping or touching
    /// projected fragments coalesced into one that lists its members.
    pub(crate) fn project_spans<L>(&self, spans: &Spans<L>, item: Ix<L>) -> Vec<SpanFragment> {
        let view = spans.view();
        if view == self.target_view {
            return spans.fragments(item).map(|f| self.project_wire_span(view, f)).collect();
        }
        self.project_fragments(view, spans.fragments(item))
    }

    /// `fragments` of `layer_view`, in the order given, on the target
    /// view through `Document::project_fragments`, whatever the view:
    /// members index the order given. For spans gathered from more than
    /// one item, where overlaps must coalesce even on the target's own
    /// view.
    pub(crate) fn project_fragments(
        &self,
        layer_view: ViewId,
        fragments: impl Iterator<Item = Span>,
    ) -> Vec<SpanFragment> {
        let fragments: Vec<Span> = fragments.collect();
        match self.doc.project_fragments(layer_view, self.target_view, fragments.iter().copied()) {
            Ok(projected) => projected.into_iter().map(|p| self.wire_fragment(p)).collect(),
            Err(_) => vec![self.refuse(layer_view, fragments.first().copied().unwrap_or(Span::new(0, 0)))],
        }
    }

    fn wire_fragment(&self, p: ProjectedFragment) -> SpanFragment {
        let (start, end) = self.offsetter.to_wire(self.target_text, p.span);
        let members = if p.is_coalesced() {
            p.members
                .iter()
                .map(|&(index, outcome)| FragmentMember { index, outcome: outcome_to_proto(outcome) as i32 })
                .collect()
        } else {
            Vec::new()
        };
        SpanFragment { start, end, outcome: outcome_to_proto(p.outcome) as i32, members }
    }

    /// Records the first span the document refused to map and stands an
    /// empty exact fragment in for it; the projection fails before the
    /// message is served.
    fn refuse(&self, layer_view: ViewId, span: Span) -> SpanFragment {
        if self.unmapped.get().is_none() {
            self.unmapped.set(Some((layer_view, span)));
        }
        SpanFragment::exact(0, 0)
    }
}
