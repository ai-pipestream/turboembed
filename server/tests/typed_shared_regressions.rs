mod common;
use turbo_analysis::export::export;
use turbo_analysis::{
    Document, Evidence, EvidenceStep, IxRange, Key, Layer, ModelRef, NBest, Provenance, Span, Spans, Tag, TagSet, View,
    ViewBuilder, ViewId,
};
use turbo_analysis_layers::{Entities, PosTags, SparseTerms, Tokens};
use turbo_entity_linker::LinkedEntities;
use turbo_kserve::proto::result::{
    DocumentGraphRequest, InspectDocumentGraphRequest, OffsetMode, SpanFragment, TypedDocumentResult,
    typed_annotation::Value,
};
use turbo_kserve::results::document::codecs;

fn spans<L>(view: ViewId, pairs: &[(u32, u32)]) -> Spans<L> {
    let mut out = Spans::new(view);
    for &(a, b) in pairs {
        out.push(Span::new(a, b));
    }
    out
}
async fn project(client: &mut common::Analysis, doc: &Document, mode: OffsetMode) -> TypedDocumentResult {
    let graph = export(doc, codecs()).unwrap().to_bytes().unwrap();
    let graph = client
        .round_trip_document_graph(DocumentGraphRequest { graph: graph.into() })
        .await
        .unwrap()
        .into_inner()
        .graph;
    client
        .inspect_document_graph(InspectDocumentGraphRequest {
            doc_id: "r2".into(),
            graph: graph.to_vec(),
            view: 0,
            offsets: mode as i32,
        })
        .await
        .unwrap()
        .into_inner()
}
fn entity_doc() -> Document {
    let mut d = Document::new("A B C");
    d.add(
        &Key::<Tokens>::DEFAULT,
        Tokens { spans: spans(ViewId::SOURCE, &[(0, 1), (2, 3), (4, 5)]), sentence: None },
        Provenance::by("r2:tokens"),
    )
    .unwrap();
    for (key, positions, ranges) in [
        (Key::<Entities>::DEFAULT, vec![(2, 3), (4, 5)], vec![IxRange::new(1, 2), IxRange::new(2, 3)]),
        (Key::<Entities>::named("gold"), vec![(0, 1)], vec![IxRange::new(0, 1)]),
    ] {
        d.add(
            &key,
            Entities {
                spans: spans(ViewId::SOURCE, &positions),
                tokens: ranges,
                kind: vec![Tag(0); positions.len()],
                kinds: TagSet::new("ner", &["THING"]),
                score: vec![0.9; positions.len()],
            },
            Provenance {
                reads: vec![Key::<Tokens>::DEFAULT.id()],
                ..Provenance::by(if key.name.is_empty() { "r2:default" } else { "r2:gold" })
            },
        )
        .unwrap();
    }
    d
}

fn nbest_doc() -> Document {
    let mut d = Document::new("é😀");
    d.add(
        &Key::<Tokens>::DEFAULT,
        Tokens { spans: spans(ViewId::SOURCE, &[(0, 2), (2, 6)]), sentence: None },
        Provenance::by("r2:tokens"),
    )
    .unwrap();
    let tags = TagSet::new("pos", &["NN"]);
    d.add(
        &Key::<PosTags>::DEFAULT,
        PosTags { tagset: tags.clone(), tags: vec![Tag(0); 2], prob: None },
        Provenance::by("r2:pos"),
    )
    .unwrap();
    let mut nb = NBest::<PosTags>::new(tags, 2);
    nb.push(-0.5, &[Tag(0); 2], Some(&[0.8, 0.7])).unwrap();
    d.add(
        &Key::<NBest<PosTags>>::DEFAULT,
        nb,
        Provenance { on: vec![Key::<PosTags>::DEFAULT.id()], ..Provenance::by("r2:nbest") },
    )
    .unwrap();
    d
}

#[tokio::test]
async fn nested_nbest_probabilities_are_validated() {
    let (server, mut client) = common::serve_no_admin().await;
    let good = project(&mut client, &nbest_doc(), OffsetMode::Utf8).await;
    client.round_trip_typed_document(good.clone()).await.unwrap();
    let mut accepted = Vec::new();
    for p in [1.5, f64::NAN] {
        let mut bad = good.clone();
        let l = bad.layers.iter_mut().find(|l| l.id.starts_with("turbo:nbest")).unwrap();
        let Some(Value::Nbest(n)) = &mut l.annotations[0].value else { panic!("nbest") };
        n.candidates[0].probability = Some(p);
        if client.round_trip_typed_document(bad).await.is_ok() {
            accepted.push(p);
        }
    }
    assert!(accepted.is_empty(), "invalid nested NBest probabilities accepted: {accepted:?}");
    server.stop().await;
}

#[tokio::test]
async fn nested_sparse_spans_and_occurrences_are_validated() {
    let (server, mut client) = common::serve_no_admin().await;
    let mut doc = Document::new("é😀");
    let mut terms =
        SparseTerms::<View>::new(ModelRef { id: "fixture".into(), revision: "1".into(), sha256: "fixture".into() });
    terms.push(1, 1.0, Some(Span::new(0, 2)));
    terms.push_occurrences(&[Span::new(0, 2)]);
    terms.unit();
    doc.add(&Key::<SparseTerms<View>>::DEFAULT, terms, Provenance::by("r2:sparse")).unwrap();
    let mut accepted = Vec::new();
    for mode in [OffsetMode::Utf8, OffsetMode::Utf16] {
        let good = project(&mut client, &doc, mode).await;
        client.round_trip_typed_document(good.clone()).await.unwrap();
        for slot in ["span", "occurrence"] {
            let mut bad = good.clone();
            let l = bad.layers.iter_mut().find(|l| l.id.starts_with("turbo:sparse")).unwrap();
            let Some(Value::SparseTerms(s)) = &mut l.annotations[0].value else { panic!("sparse") };
            let invalid = if mode == OffsetMode::Utf8 {
                SpanFragment { start: 1, end: 2 }
            } else {
                SpanFragment { start: 2, end: 3 }
            };
            if slot == "span" {
                s.entries[0].span = Some(invalid);
            } else {
                s.entries[0].occurrences[0] = invalid;
            }
            if client.round_trip_typed_document(bad).await.is_ok() {
                accepted.push(format!("{mode:?}:{slot}"));
            }
        }
    }
    assert!(accepted.is_empty(), "nested interior-codepoint/surrogate offsets accepted: {accepted:?}");
    server.stop().await;
}

#[tokio::test]
async fn nbest_spans_follow_named_pos_token_binding() {
    let (server, mut client) = common::serve_no_admin().await;
    let mut doc = Document::new("xxé");
    let source = doc.get::<View>().unwrap().clone();
    let mut b = ViewBuilder::new(source.text());
    b.replace(2, "");
    let (text, alignment) = b.finish();
    doc.add(&Key::<View>::named("trim"), View::derived(&source, text, alignment), Provenance::by("r2:view")).unwrap();
    for (key, view, pair) in
        [(Key::<Tokens>::DEFAULT, ViewId::SOURCE, (0, 1)), (Key::<Tokens>::named("gold"), ViewId(1), (0, 2))]
    {
        doc.add(&key, Tokens { spans: spans(view, &[pair]), sentence: None }, Provenance::by("r2:tokens")).unwrap();
    }
    let tags = TagSet::new("pos", &["NN"]);
    let pos = Key::<PosTags>::named("gold");
    doc.add(
        &pos,
        PosTags { tagset: tags.clone(), tags: vec![Tag(0)], prob: None },
        Provenance { on: vec![Key::<Tokens>::named("gold").id()], ..Provenance::by("r2:pos") },
    )
    .unwrap();
    let mut nb = NBest::<PosTags>::new(tags, 1);
    nb.push(-0.1, &[Tag(0)], Some(&[0.9])).unwrap();
    doc.add(&Key::<NBest<PosTags>>::DEFAULT, nb, Provenance { on: vec![pos.id()], ..Provenance::by("r2:nbest") })
        .unwrap();
    let out = project(&mut client, &doc, OffsetMode::Utf8).await;
    let nbest = out.layers.iter().find(|l| l.id.starts_with("turbo:nbest")).unwrap();
    assert_eq!(
        nbest.annotations[0].spans,
        [SpanFragment { start: 2, end: 4 }],
        "NBest used default tokens instead of its named POS anchor's tokens"
    );
    server.stop().await;
}

fn links(count: usize, prefix: &str) -> LinkedEntities {
    LinkedEntities {
        kb_id: (0..count).map(|i| Some(format!("{prefix}{i}"))).collect(),
        provider_id: vec!["fixture".into(); count],
        canonical_name: (0..count).map(|i| format!("{prefix}name{i}")).collect(),
        matched_alias: vec!["alias".into(); count],
        score: vec![0.9; count],
        nil_reason: vec![None; count],
        nil_cluster: vec![None; count],
        candidates: vec![vec![]; count],
    }
}

#[tokio::test]
async fn evidence_and_links_follow_explicit_entity_bindings_not_names() {
    let (server, mut client) = common::serve_no_admin().await;
    let mut doc = entity_doc();
    let gold = Key::<Entities>::named("gold");
    doc.add(
        &Key::<LinkedEntities>::DEFAULT,
        links(2, "default"),
        Provenance { on: vec![Key::<Entities>::DEFAULT.id()], ..Provenance::by("r2:links") },
    )
    .unwrap();
    doc.add(
        &Key::<LinkedEntities>::named("resolver-alias"),
        links(1, "gold"),
        Provenance { on: vec![gold.id()], ..Provenance::by("r2:gold-links") },
    )
    .unwrap();
    let evidence = EvidenceStep::new(gold).build(&doc).unwrap();
    doc.add(
        &Key::<Evidence<Entities>>::named("evidence-alias"),
        evidence,
        Provenance { on: vec![gold.id()], ..Provenance::by("r2:evidence") },
    )
    .unwrap();
    let out = project(&mut client, &doc, OffsetMode::Utf8).await;
    let layer = out.layers.iter().find(|l| l.id == Entities::NAME.to_string() && l.instance == "gold").unwrap();
    let Some(Value::Entity(e)) = &layer.annotations[0].value else { panic!("entity") };
    assert_eq!(e.kb_id.as_deref(), Some("gold0"));
    assert_eq!(e.canonical, "goldname0");
    assert_eq!(e.votes.len(), 1);
    assert_eq!(e.votes[0].annotator, "r2:gold");
    assert_eq!(client.round_trip_typed_document(out.clone()).await.unwrap().into_inner(), out);
    server.stop().await;
}

#[tokio::test]
async fn consolidated_entity_nested_fields_are_validated() {
    use turbo_analysis_layers::{ConsolidatedMention, DocumentEntities, DocumentEntity, SyntacticRoleCounts};
    use turbo_kserve::proto::result::EntityRelationProto;
    let (server, mut client) = common::serve_no_admin().await;
    let mut doc = Document::new("é😀");
    let mut entities = DocumentEntities::new(ViewId::SOURCE);
    entities.entities.push(DocumentEntity {
        canonical_name: "é".into(),
        kind: "THING".into(),
        kb_id: None,
        geo_coord: None,
        country_code: None,
        mentions: vec![ConsolidatedMention {
            span: Span::new(0, 2),
            entity: None,
            is_representative: true,
            role: None,
        }]
        .into_boxed_slice(),
        role_counts: SyntacticRoleCounts::default(),
        mean_sentiment: 0.0,
        sentiment_mentions: 0,
        confidence: 0.8,
    });
    doc.add(&Key::<DocumentEntities>::DEFAULT, entities, Provenance::by("r2:entities")).unwrap();
    let good = project(&mut client, &doc, OffsetMode::Utf8).await;
    client.round_trip_typed_document(good.clone()).await.unwrap();
    let mut accepted = Vec::new();
    for field in ["confidence", "mention", "relation"] {
        let mut bad = good.clone();
        let layer = bad.layers.iter_mut().find(|l| l.id == DocumentEntities::NAME.to_string()).unwrap();
        let Some(Value::DocumentEntity(e)) = &mut layer.annotations[0].value else { panic!("document entity") };
        match field {
            "confidence" => e.confidence = f32::NAN,
            "mention" => e.mentions[0].span = Some(SpanFragment { start: 1, end: 2 }),
            _ => e.relations.push(EntityRelationProto {
                subject: 99,
                object: 99,
                kind: "relation".into(),
                score: 0.9,
                support: 1,
            }),
        }
        if client.round_trip_typed_document(bad).await.is_ok() {
            accepted.push(field);
        }
    }
    assert!(accepted.is_empty(), "invalid consolidated entity payload fields accepted: {accepted:?}");
    server.stop().await;
}
