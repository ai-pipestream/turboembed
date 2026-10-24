//! Coreference chain fixes (#1047): transitive edge union,
//! entity edges, and sub-sentence clause chunk antecedents.

use turbo_analysis::{
    Analyzer, Document, Edge, Edges, Ix, IxRange, Key, Provenance, Span, Spans, Tag, TagSet, View, ViewId,
};
use turbo_analysis_layers::{Chunks, Cut, Sentences, Tokens};
use turbo_chunker::{
    ChunkAntecedentBuilder, ChunkAntecedents, MentionKind, Mentions, chunk_antecedents, extract_chains,
};

fn spans<L>(pairs: &[(u32, u32)]) -> Spans<L> {
    let mut s = Spans::new(ViewId::SOURCE);
    for &(a, b) in pairs {
        s.push(Span::new(a, b));
    }
    s
}

fn chunks(list: &[(u32, u32, u32, u32)]) -> Chunks {
    Chunks {
        spans: spans(&list.iter().map(|&(a, b, _, _)| (a, b)).collect::<Vec<_>>()),
        level: vec![0; list.len()],
        parent: vec![None; list.len()],
        sentences: list.iter().map(|&(_, _, a, b)| IxRange::new(a, b)).collect(),
        cut: vec![Cut::DocumentStart; list.len()],
        cost: vec![0; list.len()],
        model_tokens: vec![1; list.len()],
        over_budget: vec![false; list.len()],
        prompt_overhead: 0,
    }
}

/// "Ada Lovelace wrote notes. She published them. Her work endured."
fn fixture() -> (Sentences, Mentions) {
    let sentences = Sentences { spans: spans(&[(0, 25), (26, 45), (46, 63)]) };
    let mut mentions = Mentions::new(ViewId::SOURCE);
    mentions.push(Span::new(0, 12), Ix::new(0), IxRange::new(0, 2), Ix::new(1), MentionKind::Proper);
    mentions.push(Span::new(26, 29), Ix::new(1), IxRange::new(5, 6), Ix::new(5), MentionKind::Pronominal);
    mentions.push(Span::new(46, 49), Ix::new(2), IxRange::new(9, 10), Ix::new(9), MentionKind::Pronominal);
    (sentences, mentions)
}

fn edges(pairs: &[(u32, u32)]) -> Edges<Mentions, Mentions> {
    Edges::<Mentions, Mentions>::between(
        Key::DEFAULT,
        Key::DEFAULT,
        TagSet::new("coref", &["antecedent"]),
        pairs.iter().map(|&(f, t)| Edge { from: Ix::new(f), to: Ix::new(t), kind: Tag(0), score: 0.9 }),
    )
}

#[test]
fn sequential_edges_union_into_one_chain() {
    let (sentences, mentions) = fixture();
    let e = edges(&[(1, 0), (2, 1)]);
    let chains = extract_chains(&sentences, None, Some(&mentions), None, Some(&e), None, None, None);
    assert_eq!(chains.clusters.len(), 1, "clusters: {:?}", chains.clusters.len());
    assert_eq!(chains.clusters[0].len(), 3);
}

#[test]
fn her_resolves_to_the_name_not_to_she() {
    let (sentences, mentions) = fixture();
    let e = edges(&[(1, 0), (2, 1)]);
    let ch = chunks(&[(0, 25, 0, 1), (26, 63, 1, 3)]);
    let out = chunk_antecedents(&sentences, &ch, None, Some(&mentions), None, Some(&e), None, None, None);
    let links = out.links_of(Ix::new(1));
    assert_eq!(links.len(), 1);
    assert_eq!(links[0].antecedent_span, Span::new(0, 12), "Ada Lovelace, not She");
    assert_eq!(links[0].mention_span, Span::new(26, 29));
}

#[test]
fn clause_chunk_resolves_to_earlier_clause_of_same_sentence() {
    // "Ada Lovelace wrote notes, and she published them." (one sentence)
    let sentences = Sentences { spans: spans(&[(0, 49)]) };
    let mut mentions = Mentions::new(ViewId::SOURCE);
    mentions.push(Span::new(0, 12), Ix::new(0), IxRange::new(0, 2), Ix::new(1), MentionKind::Proper);
    mentions.push(Span::new(30, 33), Ix::new(0), IxRange::new(6, 7), Ix::new(6), MentionKind::Pronominal);
    let e = edges(&[(1, 0)]);
    let ch = chunks(&[(0, 25, 0, 1), (26, 49, 0, 1)]);
    let out = chunk_antecedents(&sentences, &ch, None, Some(&mentions), None, Some(&e), None, None, None);
    assert!(out.links_of(Ix::new(0)).is_empty());
    let links = out.links_of(Ix::new(1));
    assert_eq!(links.len(), 1);
    assert_eq!(links[0].antecedent_span, Span::new(0, 12));
    assert_eq!(links[0].mention_span, Span::new(30, 33));
}

#[test]
fn entity_edges_alone_form_a_chain() {
    use turbo_analysis_layers::Entities;
    let (sentences, _) = fixture();
    let entities = Entities {
        spans: spans(&[(0, 12), (26, 29), (46, 49)]),
        tokens: vec![IxRange::new(0, 2), IxRange::new(5, 6), IxRange::new(9, 10)],
        kind: vec![Tag(0), Tag(1), Tag(1)],
        kinds: TagSet::new("ner", &["PERSON", "PRON"]),
        score: vec![1.0, 1.0, 1.0],
    };
    let ee = Edges::<Entities, Entities>::between(
        Key::DEFAULT,
        Key::DEFAULT,
        TagSet::new("coref", &["antecedent"]),
        [
            Edge { from: Ix::new(1), to: Ix::new(0), kind: Tag(0), score: 0.9 },
            Edge { from: Ix::new(2), to: Ix::new(1), kind: Tag(0), score: 0.9 },
        ],
    );
    let chains = extract_chains(&sentences, None, None, None, None, Some(&entities), None, Some(&ee));
    assert_eq!(chains.clusters.len(), 1);
    assert_eq!(chains.clusters[0].len(), 3);
    let ch = chunks(&[(0, 25, 0, 1), (26, 63, 1, 3)]);
    let out = chunk_antecedents(&sentences, &ch, None, None, None, None, Some(&entities), None, Some(&ee));
    let links = out.links_of(Ix::new(1));
    assert_eq!(links.len(), 1);
    assert_eq!(links[0].antecedent_span, Span::new(0, 12));
}

#[test]
fn entity_edges_reach_the_builder_through_the_analyzer() {
    // The builder, not just chunk_antecedents: with only Entities and
    // Edges<Entities, Entities> given, the pronoun chunk still gets the
    // entity antecedent. Dropping the builder's entity-edges read (the
    // original bug) makes this fail while the direct calls above pass.
    use turbo_analysis_layers::Entities;
    let text = "Ada Lovelace wrote notes. She published them. Her work endured.";
    let (sentences, _) = fixture();
    let entities = Entities {
        spans: spans(&[(0, 12), (26, 29), (46, 49)]),
        tokens: vec![IxRange::new(0, 2), IxRange::new(5, 6), IxRange::new(9, 10)],
        kind: vec![Tag(0), Tag(1), Tag(1)],
        kinds: TagSet::new("ner", &["PERSON", "PRON"]),
        score: vec![1.0, 1.0, 1.0],
    };
    let ee = Edges::<Entities, Entities>::between(
        Key::DEFAULT,
        Key::DEFAULT,
        TagSet::new("coref", &["antecedent"]),
        [
            Edge { from: Ix::new(1), to: Ix::new(0), kind: Tag(0), score: 0.9 },
            Edge { from: Ix::new(2), to: Ix::new(1), kind: Tag(0), score: 0.9 },
        ],
    );
    let mut doc = Document::new(text);
    let by = Provenance::by("t");
    doc.add(
        &Key::DEFAULT,
        Tokens {
            spans: spans(&[
                (0, 3),
                (4, 12),
                (13, 18),
                (19, 24),
                (24, 25),
                (26, 29),
                (30, 39),
                (40, 44),
                (44, 45),
                (46, 49),
                (50, 54),
            ]),
            sentence: None,
        },
        by.clone(),
    )
    .unwrap();
    doc.add(&Key::DEFAULT, sentences, by.clone()).unwrap();
    doc.add(&Key::DEFAULT, chunks(&[(0, 25, 0, 1), (26, 63, 1, 3)]), by.clone()).unwrap();
    doc.add(&Key::DEFAULT, entities, by.clone()).unwrap();
    doc.add(&Key::DEFAULT, ee, by).unwrap();
    let analyzer = Analyzer::builder()
        .given(Key::<View>::DEFAULT)
        .given(Key::<Tokens>::DEFAULT)
        .given(Key::<Sentences>::DEFAULT)
        .given(Key::<Chunks>::DEFAULT)
        .given(Key::<Entities>::DEFAULT)
        .given(Key::<Edges<Entities, Entities>>::DEFAULT)
        .add(ChunkAntecedentBuilder)
        .build()
        .unwrap();
    analyzer.run(std::slice::from_mut(&mut doc))[0].as_ref().unwrap();
    let out = doc.get::<ChunkAntecedents>().unwrap();
    let links = out.links_of(Ix::new(1));
    assert_eq!(links.len(), 1);
    assert_eq!(links[0].antecedent_span, Span::new(0, 12));
    assert_eq!(links[0].mention_span, Span::new(26, 29));
}
