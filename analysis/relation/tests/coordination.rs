//! A long coordination: every entity is a `conj` sibling of one
//! head, so a sibling pattern matches every ordered pair. The
//! mention count is quadratic in the entity count; collecting the
//! edges must stay linear in the mentions.

use turbo_analysis::{Annotator, Document, Ix, IxRange, Span, Spans, Tag, TagSet, View, ViewId};
use turbo_analysis_layers::{Dependencies, Entities, Sentences, Tokens};
use turbo_relation::{RelationAnnotator, RelationPattern};

#[test]
fn sibling_pattern_over_a_long_coordination() {
    let n = 300usize;
    let text: String = std::iter::once("h").chain(std::iter::repeat_n(" e", n)).collect();
    let doc = Document::new(text.as_str());
    let mut tokens = Spans::new(ViewId::SOURCE);
    let mut entity_spans = Spans::new(ViewId::SOURCE);
    tokens.push(Span::new(0, 1));
    for i in 0..n as u32 {
        tokens.push(Span::new(2 + 2 * i, 3 + 2 * i));
        entity_spans.push(Span::new(2 + 2 * i, 3 + 2 * i));
    }
    let mut sentence = Spans::new(ViewId::SOURCE);
    sentence.push(Span::new(0, text.len() as u32));
    let sentences = Sentences { spans: sentence };
    let tokens = Tokens { spans: tokens, sentence: Some(vec![Ix::new(0); n + 1]) };
    let deps = Dependencies {
        head: std::iter::once(None).chain(std::iter::repeat_n(Some(Ix::new(0)), n)).collect(),
        relation: std::iter::once(Tag(0)).chain(std::iter::repeat_n(Tag(1), n)).collect(),
        relations: TagSet::new("dependency", &["root", "conj"]),
    };
    let entities = Entities {
        spans: entity_spans,
        tokens: (1..=n as u32).map(|i| IxRange::new(i, i + 1)).collect(),
        kind: vec![Tag(0); n],
        kinds: TagSet::new("entity", &["item"]),
        score: vec![1.0; n],
    };
    let annotator: RelationAnnotator =
        RelationAnnotator::new(vec![RelationPattern::new("sibling", "<conj >conj").unwrap()]).unwrap();
    let view = doc.get::<View>().unwrap();
    let (relations, edges) =
        annotator.annotate((view, &sentences, &tokens, &tokens, &deps, &entities, None, None)).unwrap();
    assert_eq!(relations.subject.len(), n * (n - 1));
    assert_eq!(edges.from, relations.subject);
    assert_eq!(edges.to, relations.object);
}
