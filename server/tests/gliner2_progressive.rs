#[path = "gliner2_progressive/fixture.rs"]
mod fixture;

use std::sync::Arc;

use fixture::package;
use tokio_stream::StreamExt as _;
use turbo_analysis::export::{DocumentGraph, export, import};
use turbo_analysis::{Document, Evidence, IxRange, Key, Layer, Provenance, Span, Spans, TagSet, ViewId};
use turbo_analysis_layers::{Entities, Sentences, Tokens};
use turbo_kserve::config::DEFAULT_MAX_MESSAGE_BYTES as MAX;
use turbo_kserve::proto::result::analysis_service_client::AnalysisServiceClient;
use turbo_kserve::proto::result::analyze_progressive_event::Event;
use turbo_kserve::proto::result::*;
use turbo_kserve::results::document::codecs;
use turbo_kserve::{Serve, Server};
use turbo_relation::Relations;

const TEXT: &str = "Jobs founded Apple in Cupertino. Cook works for Apple.";

fn spans<L>(ranges: &[(u32, u32)]) -> Spans<L> {
    let mut out = Spans::new(ViewId::SOURCE);
    for &(start, end) in ranges {
        out.push(Span::new(start, end));
    }
    out
}

fn initial_graph() -> Vec<u8> {
    let mut doc = Document::new(TEXT);
    let sentences = Sentences { spans: spans(&[(0, 32), (33, 54)]) };
    let tokens = Tokens {
        spans: spans(&[
            (0, 4),
            (5, 12),
            (13, 18),
            (19, 21),
            (22, 31),
            (31, 32),
            (33, 37),
            (38, 43),
            (44, 47),
            (48, 53),
            (53, 54),
        ]),
        sentence: None,
    };
    let kinds = TagSet::new("entity-kinds", &["PERSON", "ORG", "LOC"]);
    let ranges = [(0, 4, "PERSON"), (13, 18, "ORG"), (22, 31, "LOC"), (33, 37, "PERSON"), (48, 53, "ORG")];
    let entities = Entities {
        spans: spans(&ranges.map(|(start, end, _)| (start, end))),
        tokens: [0, 2, 4, 6, 9].map(|i| IxRange::new(i, i + 1)).into_iter().collect(),
        kind: ranges.map(|(_, _, kind)| kinds.tag(kind).unwrap()).into_iter().collect(),
        kinds: Arc::clone(&kinds),
        score: vec![1.0; 5],
    };
    doc.add(&Key::<Sentences>::DEFAULT, sentences, Provenance::by("test:sentences")).unwrap();
    doc.add(&Key::<Tokens>::DEFAULT, tokens, Provenance::by("test:tokens")).unwrap();
    doc.add(&Key::<Entities>::DEFAULT, entities, Provenance::by("test:entities")).unwrap();
    export(&doc, codecs()).unwrap().to_bytes().unwrap()
}

async fn serve() -> (Server, AnalysisServiceClient<tonic::transport::Channel>, BundleCapabilities) {
    let server =
        Server::start("127.0.0.1:0".parse().unwrap(), Vec::new(), MAX, Serve { admin: true, ..Serve::default() })
            .await
            .unwrap();
    let mut client = AnalysisServiceClient::connect(format!("http://{}", server.local_addr()))
        .await
        .unwrap()
        .max_decoding_message_size(MAX)
        .max_encoding_message_size(MAX);
    let loaded = client
        .load_bundle(LoadBundleRequest {
            name: "relations".into(),
            bundle: package().to_string_lossy().into_owned(),
            device: 0,
            sessions: 1,
            precision: 0,
            max_batch: 0,
            max_seq: 0,
            select_device: false,
            user: String::new(),
        })
        .await
        .unwrap()
        .into_inner();
    let capabilities = loaded.capabilities.unwrap();
    assert_eq!(capabilities.task, "TASK_RELATION");
    assert_eq!(capabilities.head_steps, ["turbo:zero-shot-relations"]);
    let listed = client.capabilities(CapabilitiesRequest {}).await.unwrap().into_inner();
    assert_eq!(listed.bundles.len(), 1);
    assert_eq!(listed.bundles[0].task, "TASK_RELATION");
    (server, client, capabilities)
}

async fn analyze(
    client: &mut AnalysisServiceClient<tonic::transport::Channel>,
    labels: Vec<OpenRelationLabel>,
) -> DocumentGraphResponse {
    let mut stream = client
        .analyze_document_progressive(AnalyzePipelineRequest {
            doc_id: "seeded-relations".into(),
            initial_graph: initial_graph(),
            relation_model: "relations".into(),
            steps: vec!["turbo:zero-shot-relations".into()],
            relations: Some(OpenRelationSchema { labels, threshold: Some(0.0), within_sentence: true }),
            ..Default::default()
        })
        .await
        .unwrap()
        .into_inner();
    let mut complete = None;
    while let Some(event) = stream.next().await {
        if let Some(Event::Complete(done)) = event.unwrap().event {
            complete = Some(done);
        }
    }
    complete.expect("a complete relation graph")
}

fn label(name: &str, domain: &str, range: &str) -> OpenRelationLabel {
    OpenRelationLabel {
        name: name.into(),
        threshold: None,
        symmetric: false,
        domain: vec![domain.into()],
        range: vec![range.into()],
    }
}

#[tokio::test]
async fn seeded_gliner2_relations_round_trip_through_the_real_server() {
    let (server, mut client, capabilities) = serve().await;
    let complete = analyze(
        &mut client,
        vec![label("founded", "PERSON", "ORG"), label("located in", "ORG", "LOC"), label("works for", "PERSON", "ORG")],
    )
    .await;
    let doc = import(&DocumentGraph::from_bytes(&complete.graph).unwrap(), codecs()).unwrap();
    let relations = doc.get::<Relations>().unwrap_or_else(|| panic!("relations layer: {complete:?}"));
    let entities = doc.get::<Entities>().unwrap();
    for (i, expected) in ["Jobs", "Apple", "Cupertino", "Cook", "Apple"].into_iter().enumerate() {
        let span = entities.spans.get(turbo_analysis::Ix::new(i as u32));
        assert_eq!(&doc.source()[span.start as usize..span.end as usize], expected);
    }
    let edges = doc.get::<turbo_analysis::Edges<Entities, Entities>>().unwrap();
    assert_eq!(edges.from.len(), 5);
    let rows: Vec<_> = (0..relations.spans.len())
        .map(|i| {
            let (subject, object, kind, score) = relations.mention(turbo_analysis::Ix::new(i as u32));
            (subject.get(), object.get(), kind.to_owned(), score)
        })
        .collect();
    assert_eq!(rows.len(), 5, "{rows:?}");
    for (subject, object, expected) in
        [(0, 1, "founded"), (0, 1, "works for"), (1, 2, "located in"), (3, 4, "founded"), (3, 4, "works for")]
    {
        assert!(rows.iter().any(|(s, o, kind, _)| (*s, *o, kind.as_str()) == (subject, object, expected)));
    }
    for (i, (subject, object, kind, score)) in rows.iter().enumerate() {
        assert!(*subject < 5 && *object < 5 && !kind.is_empty());
        assert!((0.0..=1.0).contains(score));
        assert_eq!((edges.from[i].get(), edges.to[i].get(), edges.score[i]), (*subject, *object, *score));
        let span = relations.spans.get(turbo_analysis::Ix::new(i as u32));
        let head = entities.spans.get(turbo_analysis::Ix::new(*subject as u32));
        let tail = entities.spans.get(turbo_analysis::Ix::new(*object as u32));
        assert_eq!(span, Span::new(head.start.min(tail.start), head.end.max(tail.end)));
        assert!(doc.source().get(span.start as usize..span.end as usize).is_some());
    }
    let provenance = doc.provenance(Key::<Relations>::DEFAULT.id()).unwrap();
    assert_eq!(provenance.model.as_ref().unwrap().sha256, capabilities.model.unwrap().sha256);
    doc.get::<Evidence<Relations>>().unwrap().verify(&doc).unwrap();

    let echoed = client
        .round_trip_document_graph(DocumentGraphRequest { graph: complete.graph.clone(), pii_output: None })
        .await
        .unwrap()
        .into_inner();
    assert_eq!(echoed.graph, complete.graph);

    let negative = analyze(&mut client, vec![label("competes with", "ORG", "ORG")]).await;
    let negative_doc = import(&DocumentGraph::from_bytes(&negative.graph).unwrap(), codecs()).unwrap();
    assert_eq!(negative_doc.get::<Relations>().unwrap().len(), 0);
    assert!(negative_doc.get::<turbo_analysis::Edges<Entities, Entities>>().unwrap().from.is_empty());
    negative_doc.get::<Evidence<Relations>>().unwrap().verify(&negative_doc).unwrap();
    server.stop().await;
}
