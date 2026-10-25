//! `DocumentEntities::check` holds the orders its docs promise: mentions in
//! text order, entities by first mention, relations by subject, object and
//! kind.

use turbo_analysis::{Layer, Span, ViewId};
use turbo_analysis_layers::{
    ConsolidatedMention, DocumentEntities, DocumentEntity, EntityRelation, SyntacticRoleCounts,
};

fn mention(start: u32, end: u32) -> ConsolidatedMention {
    ConsolidatedMention { span: Span { start, end }, entity: None, is_representative: false, role: None }
}

fn entity(spans: &[(u32, u32)]) -> DocumentEntity {
    DocumentEntity {
        canonical_name: "x".into(),
        kind: "person".into(),
        kb_id: None,
        geo_coord: None,
        country_code: None,
        mentions: spans.iter().map(|&(s, e)| mention(s, e)).collect(),
        role_counts: SyntacticRoleCounts::default(),
        mean_sentiment: 0.0,
        sentiment_mentions: 0,
        confidence: 0.5,
    }
}

fn relation(subject: u32, object: u32, kind: &str) -> EntityRelation {
    EntityRelation { subject, object, kind: kind.into(), score: 0.5, support: 1 }
}

fn graph(entities: Vec<DocumentEntity>, relations: Vec<EntityRelation>) -> DocumentEntities {
    DocumentEntities { view: ViewId::SOURCE, entities, relations }
}

#[test]
fn ordered_entities_mentions_and_relations_pass() {
    let g = graph(
        vec![entity(&[(0, 3), (0, 3), (10, 12)]), entity(&[(5, 8)])],
        vec![relation(0, 1, "a"), relation(0, 1, "b"), relation(1, 0, "a")],
    );
    assert_eq!(g.check(), Ok(()));
}

#[test]
fn mentions_out_of_text_order_are_refused() {
    let g = graph(vec![entity(&[(10, 12), (0, 3)])], Vec::new());
    assert!(g.check().unwrap_err().contains("out of text order"));
}

#[test]
fn entities_out_of_first_mention_order_are_refused() {
    let g = graph(vec![entity(&[(5, 8)]), entity(&[(0, 3)])], Vec::new());
    assert!(g.check().unwrap_err().contains("first mention"));
}

#[test]
fn unsorted_or_repeated_relations_are_refused() {
    let two = || vec![entity(&[(0, 3)]), entity(&[(5, 8)])];
    let unsorted = graph(two(), vec![relation(1, 0, "a"), relation(0, 1, "a")]);
    assert!(unsorted.check().unwrap_err().contains("not sorted"));
    let repeated = graph(two(), vec![relation(0, 1, "a"), relation(0, 1, "a")]);
    assert!(repeated.check().unwrap_err().contains("not sorted"));
}
