//! What projecting a span onto another view did to it: kept its bytes
//! exactly, widened it over a lossy edit, or landed it on deleted text
//! ([`SpanOutcome`]); and, for an item's fragments, which ones coalesced
//! into one projected fragment ([`ProjectedFragment`]).
//!
//! The outcome and the coalescing are separate facts: a coalesced
//! fragment can be exact or widened too, and keeps every member's own
//! outcome.

use crate::error::Error;
use crate::span::Span;
use crate::text::ViewId;

use super::super::Document;

/// What one projection did to a span.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub enum SpanOutcome {
    /// Mapped back, the projected span is the native one: the bytes the
    /// span stood for are exactly the bytes it covers now.
    Exact,
    /// The projected span covers more than the native one stood for: it
    /// touches an edit that does not map byte for byte.
    Widened,
    /// A non-empty native span projects onto an empty one: the target
    /// view deleted its text, and the empty span anchors where it was.
    Deleted,
}

/// A native span on the target view, with its outcome.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub struct ProjectedSpan {
    pub span: Span,
    pub outcome: SpanOutcome,
}

/// One projected fragment of an item: its span, the outcome of the whole
/// projected fragment, and the native fragments (by index, ascending)
/// that coalesced into it, each with its own outcome. More than one
/// member means the fragment is coalesced.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ProjectedFragment {
    pub span: Span,
    pub outcome: SpanOutcome,
    pub members: Vec<(u32, SpanOutcome)>,
}

impl ProjectedFragment {
    /// Whether more than one native fragment coalesced into this one.
    pub fn is_coalesced(&self) -> bool {
        self.members.len() > 1
    }
}

impl Document {
    /// `s` of view `from` on view `to`, as [`Document::map_span`] maps it,
    /// with what the mapping did: exact when mapping it back gives `s`,
    /// deleted when a non-empty `s` lands empty, widened otherwise.
    pub fn map_span_outcome(&self, from: ViewId, to: ViewId, s: Span) -> Result<ProjectedSpan, Error> {
        let span = self.map_span(from, to, s)?;
        let outcome = if !s.is_empty() && span.is_empty() {
            SpanOutcome::Deleted
        } else if self.map_span(to, from, span)? == s {
            SpanOutcome::Exact
        } else {
            SpanOutcome::Widened
        };
        Ok(ProjectedSpan { span, outcome })
    }

    /// An item's `fragments` of view `from` on view `to`. Projected
    /// fragments that overlap, or an empty one inside or touching another,
    /// coalesce into one, listing their members; the coalesced fragment
    /// is widened when any member is, else exact when any member is, else
    /// deleted. The result is in text order.
    pub fn project_fragments(
        &self,
        from: ViewId,
        to: ViewId,
        fragments: impl IntoIterator<Item = Span>,
    ) -> Result<Vec<ProjectedFragment>, Error> {
        let mut projected = Vec::new();
        for (i, f) in fragments.into_iter().enumerate() {
            let index = u32::try_from(i).map_err(|_| Error::Bounds {
                layer: crate::layer::Key::<crate::View>::DEFAULT.id(),
                view: from,
                detail: "more than 2^32 fragments".into(),
            })?;
            projected.push((index, self.map_span_outcome(from, to, f)?));
        }
        projected.sort_by_key(|(i, p)| (p.span.start, p.span.end, *i));
        let mut out: Vec<ProjectedFragment> = Vec::with_capacity(projected.len());
        for (i, p) in projected {
            if let Some(last) = out.last_mut()
                && joins(last.span, p.span)
            {
                last.span = Span::new(last.span.start, last.span.end.max(p.span.end));
                last.members.push((i, p.outcome));
                last.members.sort_unstable_by_key(|(m, _)| *m);
                last.outcome = merged(&last.members);
                continue;
            }
            out.push(ProjectedFragment { span: p.span, outcome: p.outcome, members: vec![(i, p.outcome)] });
        }
        Ok(out)
    }
}

/// Whether `next`, starting at or after `last`'s start, coalesces into
/// it: the two overlap, or one is empty and touches the other.
fn joins(last: Span, next: Span) -> bool {
    next.start < last.end || ((last.is_empty() || next.is_empty()) && next.start <= last.end)
}

fn merged(members: &[(u32, SpanOutcome)]) -> SpanOutcome {
    let any = |o| members.iter().any(|(_, m)| *m == o);
    if any(SpanOutcome::Widened) {
        SpanOutcome::Widened
    } else if any(SpanOutcome::Exact) {
        SpanOutcome::Exact
    } else {
        SpanOutcome::Deleted
    }
}

#[cfg(test)]
#[path = "outcome_tests.rs"]
mod tests;
