//! Replacing mentions with labels, OpenNLP's `PiiRewrite`, as a derived
//! view: the new text is a [`View`] of the text the mentions were found
//! in, written with a [`ViewBuilder`], so its alignment maps every offset
//! both ways. The rewrite also records each replacement in a
//! [`PiiEdits`] layer over that view and each label in a [`Pii`] layer
//! over it.

use std::fmt;

use turbo_analysis::{Alignment, Ix, PolicyDigest, Span, View, ViewBuilder, ViewId};
use turbo_analysis_layers::{
    CustomKinds, DetectorError, KindError, Pii, PiiEdit, PiiEdits, PiiMention, PiiRule, PiiStrategyKind, mention_order,
};

/// Mentions a rewrite cannot replace.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum RewriteError {
    /// A span past the end of the text or not on character boundaries.
    Span(Span),
    /// Two mentions overlap, so one replacement would cut the other.
    Overlap(Span, Span),
    /// A mention has an empty normalized form that cannot be tokenized.
    EmptyValue(Span),
    /// A mention's kind is a custom kind outside the layer's table.
    Kind(Span, KindError),
    /// A mention's detector is outside the layer's table, or the label
    /// layer's table is full.
    Detector(Span, String),
    /// The pii layer is over view `layer`, not over `view`, the view being
    /// rewritten, so its spans are offsets of another text.
    ForeignView { layer: ViewId, view: ViewId },
    /// A reversible strategy could not encrypt the mention, for the reason
    /// given.
    Encrypt(Span, String),
}

impl fmt::Display for RewriteError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            RewriteError::Span(s) => write!(f, "the span {}..{} is not a range of the text", s.start, s.end),
            RewriteError::Overlap(a, b) => {
                write!(f, "mentions must not overlap: {}..{} and {}..{}", a.start, a.end, b.start, b.end)
            }
            RewriteError::EmptyValue(s) => {
                write!(f, "mention at {}..{} has an empty normalized form", s.start, s.end)
            }
            RewriteError::Kind(s, e) => write!(f, "mention at {}..{}: {e}", s.start, s.end),
            RewriteError::Detector(s, e) => write!(f, "mention at {}..{}: {e}", s.start, s.end),
            RewriteError::Encrypt(s, e) => write!(f, "mention at {}..{} cannot be encrypted: {e}", s.start, s.end),
            RewriteError::ForeignView { layer, view } => {
                write!(f, "the pii layer is over view {}, not view {}", layer.0, view.0)
            }
        }
    }
}

impl std::error::Error for RewriteError {}

/// A coalesced union span, the individual overlapping mentions that
/// formed it, and their indexes in the pii layer, ascending.
pub(crate) struct MentionCluster<'a> {
    pub(crate) merged: PiiMention<'a>,
    pub(crate) members: Vec<PiiMention<'a>>,
    pub(crate) refs: Vec<Ix<Pii>>,
}

/// Which rule chose the strategy of one replacement, the id its policy
/// gives that rule (empty for none), and the strategy.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub(crate) struct EditOrigin<'r> {
    pub(crate) rule: PiiRule,
    pub(crate) strategy: PiiStrategyKind,
    pub(crate) rule_id: &'r str,
}

impl EditOrigin<'static> {
    /// A replacement by the only strategy of a single-strategy rewrite.
    pub(crate) const fn only(strategy: PiiStrategyKind) -> EditOrigin<'static> {
        EditOrigin { rule: PiiRule::Default, strategy, rule_id: "" }
    }
}

/// A text with its mentions replaced by labels: the derived view, its
/// edits layer and its label layer.
///
/// The view's id, and the view the two layers are over, is
/// [`ViewId::PENDING`] until a document takes the view; a pipeline step
/// ([`super::RedactedView`]) stores the view and its edits together, and
/// the document binds the edits to the id it gives the view.
pub struct PiiRewrite {
    view: View,
    labels: Pii,
    edits: PiiEdits,
    parent_len: usize,
}

/// Every mention of `pii` with its index.
fn indexed(pii: &Pii) -> impl Iterator<Item = (Ix<Pii>, PiiMention<'_>)> {
    pii.spans.iter().map(|(i, _)| (i, pii.get(i)))
}

impl PiiRewrite {
    /// Validates every mention's span in `text` and groups overlapping
    /// mentions into disjoint union clusters, keeping both the top-ranked
    /// merged mention and all constituent mentions with their indexes.
    pub(crate) fn group_clusters<'a>(
        text: &str,
        kinds: &CustomKinds,
        mentions: impl IntoIterator<Item = (Ix<Pii>, PiiMention<'a>)>,
    ) -> Result<Vec<MentionCluster<'a>>, RewriteError> {
        let mut ordered: Vec<(Ix<Pii>, PiiMention<'a>)> = mentions.into_iter().collect();
        for (_, m) in &ordered {
            let (start, end) = (m.span.start as usize, m.span.end as usize);
            if start >= end || end > text.len() || !text.is_char_boundary(start) || !text.is_char_boundary(end) {
                return Err(RewriteError::Span(m.span));
            }
            kinds.check(m.kind).map_err(|e| RewriteError::Kind(m.span, e))?;
        }
        ordered.sort_by_key(|(_, m)| mention_order(m.span.start, m.span.end, m.kind));
        let mut out: Vec<MentionCluster<'a>> = Vec::with_capacity(ordered.len());
        let mut best_key = ((u16::MAX, u32::MAX), u32::MAX, std::cmp::Reverse(0u32));
        for (ix, m) in ordered {
            // The winner of a cluster: the smallest kind priority, then the
            // earliest start, then the longest span.
            let key = (m.kind.order_key(), m.span.start, std::cmp::Reverse(m.span.end - m.span.start));
            if let Some(last) = out.last_mut()
                && m.span.start < last.merged.span.end
            {
                last.merged.span.end = last.merged.span.end.max(m.span.end);
                if key < best_key {
                    last.merged.kind = m.kind;
                    last.merged.detector = m.detector;
                    last.merged.evidence = m.evidence;
                    last.merged.score = m.score;
                    last.merged.normalized = m.normalized;
                    best_key = key;
                }
                last.members.push(m);
                last.refs.push(ix);
            } else {
                best_key = key;
                out.push(MentionCluster { merged: m, members: vec![m], refs: vec![ix] });
            }
        }
        for c in &mut out {
            c.refs.sort_unstable();
        }
        Ok(out)
    }

    /// Replaces every cluster of the mentions of `pii`, a layer over
    /// `src`, with the label `write` writes for it, in text order, through
    /// a [`ViewBuilder`] over `src`: one edit per cluster. A cluster of
    /// more than one mention that `refuses` is refused, before any label
    /// is written. `write` names the rule and strategy of each label.
    pub(crate) fn derive<'a, 'r>(
        src: &View,
        pii: &'a Pii,
        refuses: impl Fn(&MentionCluster<'a>) -> bool,
        policy_revision: u64,
        mut write: impl FnMut(&MentionCluster<'a>, &mut String) -> Result<EditOrigin<'r>, RewriteError>,
    ) -> Result<PiiRewrite, RewriteError> {
        let layer = pii.spans.view();
        if layer != src.id() {
            return Err(RewriteError::ForeignView { layer, view: src.id() });
        }
        let text = src.text();
        let clusters = Self::group_clusters(text, &pii.custom_kinds, indexed(pii))?;
        if let Some(c) = clusters.iter().find(|c| c.members.len() > 1 && refuses(c)) {
            return Err(RewriteError::Overlap(c.members[0].span, c.members[1].span));
        }
        let mut out = ViewBuilder::new(text);
        let mut origins = Vec::with_capacity(clusters.len());
        let mut buf = String::new();
        for c in &clusters {
            let span = c.merged.span;
            buf.clear();
            origins.push(write(c, &mut buf)?);
            if buf.is_empty() {
                return Err(RewriteError::EmptyValue(span));
            }
            out.keep(span.start as usize);
            out.replace(span.end as usize, &buf);
        }
        let (new_text, alignment) = out.finish();
        // Every cluster replaced non-empty bytes, so each recorded one
        // edit, in order.
        debug_assert_eq!(alignment.edits().len(), clusters.len());
        let mut labels = Pii::with_custom_kinds(ViewId::PENDING, pii.custom_kinds.clone());
        let mut edits = PiiEdits::new(ViewId::PENDING, policy_revision);
        for ((c, origin), e) in clusters.iter().zip(&origins).zip(alignment.edits()) {
            let m = c.merged;
            // The label carries its source mention's detector, facts and
            // score on; the detector is re-registered in the label layer's
            // own table.
            let (id, version) = pii.detectors.get(m.detector).ok_or_else(|| {
                RewriteError::Detector(m.span, DetectorError::Unknown(m.detector.index()).to_string())
            })?;
            let detector = labels.detector(id, version).map_err(|e| RewriteError::Detector(m.span, e.to_string()))?;
            labels.push(PiiMention {
                span: e.child,
                kind: m.kind,
                detector,
                evidence: m.evidence,
                score: m.score,
                normalized: &new_text[e.child.start as usize..e.child.end as usize],
            });
            edits.push(PiiEdit {
                span: e.child,
                source: e.parent,
                members: &c.refs,
                rule: origin.rule,
                strategy: origin.strategy,
                rule_id: origin.rule_id,
            });
        }
        Ok(PiiRewrite { view: View::derived(src, new_text, alignment), labels, edits, parent_len: text.len() })
    }

    /// The derived view.
    pub fn view(&self) -> &View {
        &self.view
    }

    /// The new text.
    pub fn text(&self) -> &str {
        self.view.text()
    }

    pub fn into_text(self) -> String {
        self.view.text().to_owned()
    }

    /// How the new text lines up with the text the mentions were in.
    pub fn alignment(&self) -> &Alignment {
        self.view.alignment()
    }

    /// The labels as mentions of the new text, in text order, each with
    /// its label as the normalized form and the kind, detector, evidence
    /// and score of the mention it replaced. The layer is over [`ViewId::PENDING`].
    pub fn labels(&self) -> &Pii {
        &self.labels
    }

    /// One edit per replacement, over [`ViewId::PENDING`].
    pub fn edits(&self) -> &PiiEdits {
        &self.edits
    }

    /// This rewrite as made by the redaction policy with digest `digest`,
    /// which every edit then names.
    pub fn with_policy_digest(mut self, digest: PolicyDigest) -> PiiRewrite {
        self.edits.policy_digest = Some(digest);
        self
    }

    /// The derived view and its edits, for a step to store together.
    pub fn into_parts(self) -> (View, PiiEdits) {
        (self.view, self.edits)
    }

    /// A span of the original text in the new text. A non-empty span that
    /// touches a replaced mention takes in its whole label, so the result
    /// covers every byte the span stood for; a span outside every mention
    /// keeps its bytes. Refused past the end of the original text.
    pub fn to_derived(&self, span: Span) -> Result<Span, RewriteError> {
        if span.start > span.end || span.end as usize > self.parent_len {
            return Err(RewriteError::Span(span));
        }
        Ok(self.alignment().span_to_child(span))
    }

    /// A span of the new text in the original text, by the same rules:
    /// a span touching a label takes in the whole mention it replaced.
    pub fn to_original(&self, span: Span) -> Result<Span, RewriteError> {
        if span.start > span.end || span.end as usize > self.text().len() {
            return Err(RewriteError::Span(span));
        }
        Ok(self.alignment().span_to_parent(span))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use turbo_analysis::{Document, Layer};
    use turbo_analysis_layers::{PiiDetectorRef, PiiEvidence, PiiKind};

    const SHAPE: &[PiiEvidence] = &[PiiEvidence::Shape];

    fn src(text: &str) -> View {
        Document::new(text).get::<View>().expect("the source view").clone()
    }

    /// A layer whose detector table holds the test detector at 0.
    fn layer(mentions: &[PiiMention<'_>]) -> Pii {
        let mut pii = Pii::new(ViewId::SOURCE);
        pii.detector("test", "1").unwrap();
        for &m in mentions {
            pii.push(m);
        }
        pii
    }

    fn email(start: u32, end: u32) -> PiiMention<'static> {
        PiiMention {
            span: Span::new(start, end),
            kind: PiiKind::Email,
            detector: PiiDetectorRef::new(0),
            evidence: SHAPE,
            score: None,
            normalized: "a@b.com",
        }
    }

    fn label(
        text: &'static str,
    ) -> impl FnMut(&MentionCluster<'_>, &mut String) -> Result<EditOrigin<'static>, RewriteError> {
        move |_, buf| {
            buf.push_str(text);
            Ok(EditOrigin::only(PiiStrategyKind::Static))
        }
    }

    #[test]
    fn derive_rejects_empty_spans_and_empty_labels_and_keeps_labels_valid() {
        let view = src("0123456789");
        assert_eq!(
            PiiRewrite::derive(&view, &layer(&[email(3, 3)]), |_| false, 0, label("EMAIL-1")).err(),
            Some(RewriteError::Span(Span::new(3, 3)))
        );
        let one = layer(&[email(3, 6)]);
        assert_eq!(
            PiiRewrite::derive(&view, &one, |_| false, 0, label("")).err(),
            Some(RewriteError::EmptyValue(Span::new(3, 6)))
        );
        let ok = PiiRewrite::derive(&view, &one, |_| false, 9, label("EMAIL-1")).unwrap();
        assert_eq!(ok.text(), "012EMAIL-16789");
        assert_eq!(ok.labels().check(), Ok(()));
        assert_eq!(ok.labels().detectors.get(ok.labels().detector[0]), Some(("test", "1")));
        assert_eq!(ok.edits().check(), Ok(()));
        assert_eq!(ok.view().parent(), Some(ViewId::SOURCE));
        assert_eq!((ok.labels().spans.view(), ok.edits().spans.view()), (ViewId::PENDING, ViewId::PENDING));
        let e = ok.edits().get(Ix::new(0));
        assert_eq!(
            (e.source, e.span, e.members, ok.edits().policy_revision),
            (Span::new(3, 6), Span::new(3, 10), &[Ix::new(0)][..], 9)
        );
    }

    #[test]
    fn a_mention_with_a_detector_outside_the_table_is_refused() {
        let foreign = layer(&[PiiMention { detector: PiiDetectorRef::new(4), ..email(3, 6) }]);
        assert!(matches!(
            PiiRewrite::derive(&src("0123456789"), &foreign, |_| false, 0, label("X")),
            Err(RewriteError::Detector(..))
        ));
    }

    #[test]
    fn a_layer_over_another_view_is_refused() {
        let mut other = layer(&[email(3, 6)]);
        other.spans = turbo_analysis::Spans::new(ViewId(2));
        other.spans.push(Span::new(3, 6));
        assert_eq!(
            PiiRewrite::derive(&src("0123456789"), &other, |_| false, 0, label("X")).err(),
            Some(RewriteError::ForeignView { layer: ViewId(2), view: ViewId::SOURCE })
        );
    }

    #[test]
    fn both_directions_go_through_the_alignment_and_are_bounded() {
        let r = PiiRewrite::derive(&src("ab 0123 cd"), &layer(&[email(3, 7)]), |_| false, 0, label("[E]")).unwrap();
        assert_eq!(r.text(), "ab [E] cd");
        assert_eq!(r.to_derived(Span::new(0, 2)), Ok(Span::new(0, 2)));
        assert_eq!(r.to_derived(Span::new(4, 5)), Ok(Span::new(3, 6)));
        assert_eq!(r.to_derived(Span::new(8, 10)), Ok(Span::new(7, 9)));
        assert_eq!(r.to_original(Span::new(4, 5)), Ok(Span::new(3, 7)));
        assert_eq!(r.to_derived(Span::new(0, 11)), Err(RewriteError::Span(Span::new(0, 11))));
        assert_eq!(r.to_original(Span::new(0, 10)), Err(RewriteError::Span(Span::new(0, 10))));
    }

    fn mention(start: u32, end: u32, kind: PiiKind) -> (Ix<Pii>, PiiMention<'static>) {
        (
            Ix::new(start),
            PiiMention {
                span: Span::new(start, end),
                kind,
                detector: PiiDetectorRef::new(0),
                evidence: SHAPE,
                score: None,
                normalized: "v",
            },
        )
    }

    #[test]
    fn a_cluster_winner_is_the_smallest_priority_not_the_smallest_id_or_earliest_span() {
        // Phone (id 30) starts first and is longer, Email (id 5) has the
        // smaller priority: the merged mention takes Email's kind.
        let kinds = CustomKinds::from_names(["ticket"]).unwrap();
        let ticket = kinds.find("ticket").unwrap();
        let text = "0123456789";
        let clusters =
            PiiRewrite::group_clusters(text, &kinds, [mention(0, 9, PiiKind::Phone), mention(2, 6, PiiKind::Email)])
                .unwrap();
        assert_eq!(clusters.len(), 1);
        assert_eq!((clusters[0].merged.kind, clusters[0].merged.span), (PiiKind::Email, Span::new(0, 9)));
        assert_eq!(clusters[0].refs, vec![Ix::new(0), Ix::new(2)]);
        // Every built-in kind beats a custom kind, whatever its code.
        let clusters =
            PiiRewrite::group_clusters(text, &kinds, [mention(0, 9, ticket), mention(3, 5, PiiKind::Phone)]).unwrap();
        assert_eq!(clusters[0].merged.kind, PiiKind::Phone);
        // A custom kind outside the table is refused before any grouping.
        let stray = PiiKind::Custom(turbo_analysis_layers::CustomKindId::new(4));
        assert_eq!(
            PiiRewrite::group_clusters(text, &kinds, [mention(0, 9, stray)]).err(),
            Some(RewriteError::Kind(Span::new(0, 9), KindError::UnknownCustom(4)))
        );
    }
}
