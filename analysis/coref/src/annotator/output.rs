//! The annotator's output layers from a finished resolver.

use std::sync::Arc;

use turbo_analysis::{Edge, Edges, Groups, IxRange, Spans, TagSet, ViewId};

use super::Resolution;
use crate::input::ix;
use crate::layer::{Agreement, Mentions};
use crate::sieve::{Resolver, Sieve};

/// The tag set of the edges' kinds: one label per [`Sieve`], in
/// [`Sieve::ALL`] order.
pub fn sieve_tags() -> Arc<TagSet> {
    let labels: Vec<&str> = Sieve::ALL.iter().map(|s| s.label()).collect();
    TagSet::new("coref-sieve", &labels)
}

fn mentions(view: ViewId, resolver: &Resolver<'_>) -> Result<Mentions, String> {
    let n = resolver.mentions.len();
    let mut layer = Mentions {
        spans: Spans::with_capacity(view, n),
        sentence: Vec::with_capacity(n),
        tokens: Vec::with_capacity(n),
        head: Vec::with_capacity(n),
        entity: Some(Vec::with_capacity(n)),
        kind: Vec::with_capacity(n),
        origin: Vec::with_capacity(n),
        agreement: Vec::with_capacity(n),
    };
    for m in &resolver.mentions {
        match &m.fragments {
            Some(fragments) => {
                layer.spans.try_push_fragments(fragments).map_err(|e| format!("mention fragments: {e:?}"))?;
            }
            None => {
                layer.spans.push(m.span);
            }
        }
        layer.sentence.push(ix(m.sentence));
        layer.tokens.push(IxRange::new(m.first as u32, m.last as u32 + 1));
        layer.head.push(ix(m.head_token));
        layer.entity.as_mut().expect("made with an entity column").push(m.entity.map(|e| ix(e as usize)));
        layer.kind.push(m.kind());
        layer.origin.push(m.origin);
        layer.agreement.push(Agreement::new(m.number, m.gender, m.animacy, m.person));
    }
    Ok(layer)
}

/// The three layers of an annotation: the mentions, every chain
/// (singletons included) in first-mention order, and the resolved links.
pub(super) fn layers(view: ViewId, resolver: &mut Resolver<'_>) -> Result<Resolution, String> {
    let layer = mentions(view, resolver)?;
    let chains = resolver.clusters.chains();
    let groups = Groups::new(chains.into_iter().map(|c| c.into_iter().map(ix)));
    let tags = sieve_tags();
    let rows: Vec<Edge<Mentions, Mentions>> = resolver
        .links
        .iter()
        .map(|l| {
            let kind = tags.tag(l.sieve.label()).expect("every sieve has a tag");
            Edge::new(ix(l.anaphor), ix(l.antecedent), kind, l.score)
        })
        .collect();
    Ok((layer, groups, Edges::new(tags, rows)))
}

/// The layers of a document without tokens: no mentions, no chains.
pub(super) fn empty(view: ViewId) -> (Mentions, Groups<Mentions>, Edges<Mentions, Mentions>) {
    let layer = Mentions {
        spans: Spans::new(view),
        sentence: Vec::new(),
        tokens: Vec::new(),
        head: Vec::new(),
        entity: Some(Vec::new()),
        kind: Vec::new(),
        origin: Vec::new(),
        agreement: Vec::new(),
    };
    let none: Vec<Vec<turbo_analysis::Ix<Mentions>>> = Vec::new();
    (layer, Groups::new(none), Edges::new(sieve_tags(), Vec::new()))
}
