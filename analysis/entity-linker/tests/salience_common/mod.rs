//! Documents built straight from hand-labelled mentions for the salience
//! tests: each sentence is a list of (entity name, head role), written
//! as `Name w` per mention and closed with `.`, and the consolidated
//! entities are made from the labels directly, one per name, in order of
//! first mention. No tagger, parser or consolidator is in the loop.

#![allow(dead_code)]

use std::collections::BTreeMap;

use turbo_analysis::{Document, Key, Provenance, Span, Spans, ViewId};
use turbo_analysis_layers::{
    Block, ConsolidatedMention, DocumentEntities, DocumentEntity, Sentences, Structure, SyntacticRole,
    SyntacticRoleCounts,
};

pub use SyntacticRole::{Modifier as M, Object as O, Other as X, Subject as S};

pub type Sentence = &'static [(&'static str, SyntacticRole)];

/// An entity's first-seen order and its labelled mentions.
type Labelled = (usize, Vec<(Span, SyntacticRole)>);

/// The document, with `Sentences` and `DocumentEntities`.
pub fn doc(sentences: &[Sentence]) -> Document {
    build(sentences, None, true)
}

/// The document, with `DocumentEntities`, `Sentences` when
/// `with_sentences`, and sentence `heading` as a `Structure` heading.
pub fn build(sentences: &[Sentence], heading: Option<usize>, with_sentences: bool) -> Document {
    let mut text = String::new();
    let mut sent_spans = Vec::new();
    let mut mentions: BTreeMap<&str, Labelled> = BTreeMap::new();
    for s in sentences {
        let start = text.len() as u32;
        for &(name, role) in s.iter() {
            let at = text.len() as u32;
            text.push_str(name);
            let n = mentions.len();
            mentions.entry(name).or_insert((n, Vec::new())).1.push((Span::new(at, text.len() as u32), role));
            text.push_str(" w ");
        }
        text.push('.');
        sent_spans.push(Span::new(start, text.len() as u32));
        text.push(' ');
    }
    let mut ordered: Vec<(&str, Labelled)> = mentions.into_iter().collect();
    ordered.sort_by_key(|(_, (first, _))| *first);
    let mut graph = DocumentEntities::new(ViewId(0));
    for (name, (_, ms)) in ordered {
        let mut role_counts = SyntacticRoleCounts::default();
        ms.iter().for_each(|(_, r)| role_counts.add(*r));
        let mentions: Vec<ConsolidatedMention> = ms
            .iter()
            .enumerate()
            .map(|(i, &(span, role))| ConsolidatedMention {
                span,
                entity: None,
                is_representative: i == 0,
                role: Some(role),
            })
            .collect();
        graph.entities.push(DocumentEntity {
            canonical_name: name.into(),
            kind: "ORG".into(),
            kb_id: None,
            geo_coord: None,
            country_code: None,
            mentions: mentions.into(),
            role_counts,
            mean_sentiment: 0.0,
            sentiment_mentions: 0,
            confidence: 1.0,
        });
    }
    let mut d = Document::new(text.as_str());
    let mut spans = Spans::new(ViewId(0));
    sent_spans.iter().for_each(|s| {
        spans.push(*s);
    });
    if with_sentences {
        d.add(&Key::<Sentences>::DEFAULT, Sentences { spans }, Provenance::by("test:sentences")).unwrap();
    }
    if let Some(h) = heading {
        let mut spans = Spans::new(ViewId(0));
        spans.push(sent_spans[h]);
        let s = Structure { spans, kind: vec![Block::Heading], parent: vec![None], depth: vec![1] };
        d.add(&Key::<Structure>::DEFAULT, s, Provenance::by("test:structure")).unwrap();
    }
    d.add(&Key::<DocumentEntities>::DEFAULT, graph, Provenance::by("test:entities")).unwrap();
    d
}

/// The canonical names, most salient first.
pub fn ranked(d: &Document, s: &turbo_analysis_layers::EntitySalience) -> Vec<String> {
    let g = d.require(&Key::<DocumentEntities>::DEFAULT).unwrap();
    s.top_k(usize::MAX).iter().map(|r| g.entities[r.entity_ix as usize].canonical_name.to_string()).collect()
}
