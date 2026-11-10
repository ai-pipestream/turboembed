mod common;
#[path = "gliner_progressive/fixture.rs"]
mod fixture;

use std::path::PathBuf;

use fixture::package;
use tokio_stream::StreamExt as _;
use turbo_analysis::export::{DocumentGraph, import};
use turbo_analysis::{Evidence, Key, Layer, View};
use turbo_analysis_layers::Entities;
use turbo_gliner::{Decisions, GlinerAnnotator};
use turbo_kserve::config::DEFAULT_MAX_MESSAGE_BYTES as MAX;
use turbo_kserve::proto::result::analysis_service_client::AnalysisServiceClient;
use turbo_kserve::proto::result::analyze_progressive_event::Event;
use turbo_kserve::proto::result::*;
use turbo_kserve::results::document::codecs;
use turbo_kserve::{Serve, Server};

async fn serve_package(
    path: PathBuf,
) -> (Server, AnalysisServiceClient<tonic::transport::Channel>, BundleCapabilities) {
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
            name: "gliner".into(),
            bundle: path.to_string_lossy().into_owned(),
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
    assert_eq!(capabilities.task, "TASK_OPEN_SPAN");
    assert_eq!(capabilities.head_steps, ["turbo:gliner"]);
    assert_eq!(capabilities.schema.unwrap().max_entity_types, 32);
    let listed = client.capabilities(CapabilitiesRequest {}).await.unwrap().into_inner();
    assert_eq!(listed.bundles.len(), 1);
    assert_eq!(listed.bundles[0].task, "TASK_OPEN_SPAN");
    (server, client, capabilities)
}

async fn serve() -> (Server, AnalysisServiceClient<tonic::transport::Channel>, BundleCapabilities) {
    serve_package(package()).await
}

fn request(text: &str, with_gliner: bool) -> AnalyzePipelineRequest {
    let mut steps = vec!["turbo-sentdetect:rules".into(), "turbo-tokenize:uax29-untyped".into()];
    if with_gliner {
        steps.push("turbo:subwords".into());
        steps.push("turbo:gliner".into());
    }
    AnalyzePipelineRequest {
        doc_id: "gliner-doc".into(),
        text: text.into(),
        model: if with_gliner { "gliner".into() } else { String::new() },
        steps,
        open_span: with_gliner.then(|| OpenSpanSchema {
            labels: vec!["person".into(), "location".into()],
            threshold: Some(0.0),
            nested: false,
            prompts: Vec::new(),
        }),
        ..Default::default()
    }
}

async fn analyze(
    client: &mut AnalysisServiceClient<tonic::transport::Channel>,
    request: AnalyzePipelineRequest,
) -> DocumentGraphResponse {
    let mut stream = client.analyze_document_progressive(request).await.unwrap().into_inner();
    let mut complete = None;
    while let Some(event) = stream.next().await {
        if let Some(Event::Complete(done)) = event.unwrap().event {
            complete = Some(done);
        }
    }
    complete.expect("a complete graph")
}

#[tokio::test]
async fn skipped_tokens_skip_every_gliner_output() {
    let (server, mut client, _) = serve().await;
    let mut initial = turbo_analysis::Document::new("Ann");
    initial
        .skip(
            Key::<turbo_analysis_layers::Tokens>::DEFAULT.id(),
            "upstream declined".into(),
            turbo_analysis::Provenance::by("test:tokens"),
        )
        .unwrap();
    let initial_graph = turbo_analysis::export::export(&initial, codecs()).unwrap().to_bytes().unwrap();
    let complete = analyze(
        &mut client,
        AnalyzePipelineRequest {
            initial_graph,
            model: "gliner".into(),
            steps: vec!["turbo-tokenize:uax29-untyped".into(), "turbo:gliner".into()],
            open_span: Some(OpenSpanSchema {
                labels: vec!["person".into()],
                threshold: None,
                nested: false,
                prompts: Vec::new(),
            }),
            ..Default::default()
        },
    )
    .await;
    let doc = import(&DocumentGraph::from_bytes(&complete.graph).unwrap(), codecs()).unwrap();
    for key in [Key::<Entities>::DEFAULT.id(), Key::<Decisions>::DEFAULT.id(), Key::<Evidence<Entities>>::DEFAULT.id()]
    {
        assert!(
            matches!(doc.status(key), Some(turbo_analysis::Status::Skipped(why)) if why.contains("upstream declined")),
            "{key}: {:?}",
            doc.status(key)
        );
    }
    for layer in ["turbo:entities", "turbo:gliner-decisions", "turbo:evidence<turbo:entities>"] {
        assert!(
            complete.layers.iter().any(|row| row.layer == layer && row.status == LayerWireStatus::Skipped as i32),
            "{layer}: {:?}",
            complete.layers
        );
    }
    server.stop().await;
}

#[tokio::test]
async fn converted_package_hot_loads_and_produces_a_deterministic_evidence_graph() {
    let (server, mut client, capabilities) = serve().await;
    let text = "Zoë met 東京 in Paris. ";
    let complete = analyze(&mut client, request(&format!("{text}{}", "Word. ".repeat(80)), true)).await;
    assert!(
        complete.layers.iter().any(|l| l.layer == "turbo:entities" && l.status == LayerWireStatus::Ok as i32),
        "{:?}",
        complete.layers
    );
    assert!(complete.layers.iter().any(|l| l.layer == "turbo:gliner-decisions"));
    assert!(complete.layers.iter().any(|l| l.layer == "turbo:evidence<turbo:entities>"));

    let doc = import(&DocumentGraph::from_bytes(&complete.graph).unwrap(), codecs()).unwrap();
    let subwords = doc.get::<turbo_analysis_layers::Subwords>().unwrap();
    assert!(!subwords.spans.is_empty());
    for (_, span) in subwords.spans.iter() {
        assert!(doc.source().get(span.start as usize..span.end as usize).is_some());
    }
    let entities = doc.get::<Entities>().unwrap();
    let decisions = doc.get::<Decisions>().unwrap();
    assert_eq!(entities.len(), decisions.logits.len());
    assert!(!entities.spans.is_empty(), "the zero threshold fixture emits entities");
    let evidence = doc.get::<Evidence<Entities>>().unwrap();
    evidence.verify(&doc).unwrap();
    for (_, span) in entities.spans.iter() {
        let source = doc.source().get(span.start as usize..span.end as usize).unwrap();
        assert!(!source.is_empty());
    }
    let subword_provenance = doc.provenance(Key::<turbo_analysis_layers::Subwords>::DEFAULT.id()).unwrap();
    assert_eq!(subword_provenance.version, <turbo_gliner::GlinerSubwords as turbo_analysis::Annotator>::VERSION);
    assert_eq!(subword_provenance.model.as_ref().unwrap().sha256, capabilities.model.as_ref().unwrap().sha256);
    let provenance = doc.provenance(Key::<Entities>::DEFAULT.id()).unwrap();
    assert_eq!(provenance.model.as_ref().unwrap().sha256, capabilities.model.as_ref().unwrap().sha256);

    let echoed = client
        .round_trip_document_graph(DocumentGraphRequest { graph: complete.graph.clone(), pii_output: None })
        .await
        .unwrap()
        .into_inner();
    assert_eq!(echoed.graph, complete.graph);
    let typed = client
        .inspect_document_graph(InspectDocumentGraphRequest {
            doc_id: "gliner-doc".into(),
            graph: complete.graph.to_vec(),
            view: 0,
            offsets: OffsetMode::Utf8 as i32,
            pii_output: None,
            pii_policy: None,
            binding: None,
            operation_key: String::new(),
        })
        .await
        .unwrap()
        .into_inner();
    assert_eq!(typed.layers.iter().find(|l| l.id == "turbo:entities").unwrap().annotations.len(), entities.len());

    let again = analyze(&mut client, request(&format!("{text}{}", "Word. ".repeat(80)), true)).await;
    assert_eq!(again.graph, complete.graph);
    let mut guided_request = request(&format!("{text}{}", "Word. ".repeat(80)), true);
    guided_request.open_span.as_mut().unwrap().prompts = vec!["human being".into(), "geographic place".into()];
    let guided = analyze(&mut client, guided_request).await;
    assert_ne!(guided.graph, complete.graph);
    let guided_doc = import(&DocumentGraph::from_bytes(&guided.graph).unwrap(), codecs()).unwrap();
    assert_eq!(guided_doc.get::<Entities>().unwrap().kinds.labels().join(","), "person,location");
    let empty = analyze(&mut client, request("", true)).await;
    let empty_doc = import(&DocumentGraph::from_bytes(&empty.graph).unwrap(), codecs()).unwrap();
    assert_eq!(empty_doc.get::<Entities>().unwrap().len(), 0);
    GlinerAnnotator::evidence(&empty_doc, Key::DEFAULT, Key::DEFAULT).unwrap().verify(&empty_doc).unwrap();
    server.stop().await;
}

#[tokio::test]
async fn case_fold_projects_tokens_and_gliner_validates_its_schema() {
    let (server, mut client, _) = serve().await;
    let complete = analyze(
        &mut client,
        AnalyzePipelineRequest {
            steps: vec![
                "turbo:normalize:case-fold".into(),
                "turbo-sentdetect:rules".into(),
                "turbo-tokenize:uax29-untyped".into(),
            ],
            ..request("Straße Café", false)
        },
    )
    .await;
    let doc = import(&DocumentGraph::from_bytes(&complete.graph).unwrap(), codecs()).unwrap();
    let normalized = doc.require(&Key::<View>::named("normalized")).unwrap();
    assert_eq!(normalized.text(), "strasse café");

    let mut missing = request("Ann", true);
    missing.open_span = None;
    let error = client.analyze_document_progressive(missing).await.unwrap_err();
    assert_eq!(error.code(), tonic::Code::InvalidArgument);
    assert!(error.message().contains("open_span"));
    let mut unexpected = request("Ann", false);
    unexpected.open_span =
        Some(OpenSpanSchema { labels: vec!["person".into()], threshold: None, nested: false, prompts: Vec::new() });
    let error = client.analyze_document_progressive(unexpected).await.unwrap_err();
    assert_eq!(error.code(), tonic::Code::InvalidArgument);
    assert!(error.message().contains("turbo:gliner"));

    let mut missing_relations = request("Ann", true);
    missing_relations.steps.push("turbo:zero-shot-relations".into());
    missing_relations.relation_model = "relations".into();
    let error = client.analyze_document_progressive(missing_relations).await.unwrap_err();
    assert_eq!(error.code(), tonic::Code::InvalidArgument);
    assert!(error.message().contains("relation labels"));
    let mut unexpected_relations = request("Ann", false);
    unexpected_relations.relations = Some(OpenRelationSchema::default());
    let error = client.analyze_document_progressive(unexpected_relations).await.unwrap_err();
    assert_eq!(error.code(), tonic::Code::InvalidArgument);
    assert!(error.message().contains("turbo:zero-shot-relations"));
    server.stop().await;
}

#[tokio::test]
#[ignore = "requires GLINER_PACKAGE to name a converted release candidate"]
async fn real_package_builds_the_complete_analysis_graph() {
    let root = std::env::var("GLINER_PACKAGE").expect("GLINER_PACKAGE");
    let package = turbo_pins::require(&root);
    let (server, mut client, capabilities) = serve_package(package).await;
    let text = "Zoë visited Paris. Email test@example.com. Acme hired Ada Lovelace in London.";
    let complete = analyze(
        &mut client,
        AnalyzePipelineRequest {
            doc_id: "real-gliner".into(),
            text: text.into(),
            model: "gliner".into(),
            steps: [
                "turbo:normalize:case-fold",
                "turbo-structure:blocks",
                "turbo-sentdetect:rules",
                "turbo-tokenize:uax29-untyped",
                "turbo-stem:terms",
                "turbo-stem:stemmer",
                "turbo-pii:pii",
                "turbo:subwords",
                "turbo:gliner",
                "turbo:structure-boundaries",
                "turbo:chunker",
            ]
            .into_iter()
            .map(str::to_owned)
            .collect(),
            open_span: Some(OpenSpanSchema {
                labels: vec!["person".into(), "organization".into(), "location".into()],
                threshold: Some(0.8),
                nested: false,
                prompts: Vec::new(),
            }),
            pii_output: Some(PiiOutputOptions { full_text: true, ..Default::default() }),
            ..Default::default()
        },
    )
    .await;
    assert!(!complete.graph.is_empty());
    for layer in
        ["turbo:stems", "turbo:pii", "turbo:subwords", "turbo:entities", "turbo:gliner-decisions", "turbo:chunks"]
    {
        assert!(
            complete.layers.iter().any(|l| l.layer == layer && l.status == LayerWireStatus::Ok as i32),
            "{layer}: {:?}",
            complete.layers
        );
    }
    let doc = import(&DocumentGraph::from_bytes(&complete.graph).unwrap(), codecs()).unwrap();
    assert!(!doc.get::<turbo_analysis_layers::Stems>().unwrap().stem.is_empty());
    assert!(!doc.get::<turbo_analysis_layers::Pii>().unwrap().spans.is_empty());
    assert!(!doc.get::<turbo_analysis_layers::Chunks>().unwrap().spans.is_empty());
    let entities = doc.get::<Entities>().unwrap();
    let decisions = doc.get::<Decisions>().unwrap();
    assert_eq!(entities.len(), decisions.logits.len());
    doc.get::<Evidence<Entities>>().unwrap().verify(&doc).unwrap();
    assert_eq!(doc.require(&Key::<View>::named("normalized")).unwrap().text(), text.to_lowercase());
    let provenance = doc.provenance(Key::<Entities>::DEFAULT.id()).unwrap();
    assert_eq!(provenance.model.as_ref().unwrap().sha256, capabilities.model.unwrap().sha256);
    for (_, span) in entities.spans.iter() {
        assert!(doc.source().get(span.start as usize..span.end as usize).is_some());
    }
    let echoed = client
        .round_trip_document_graph(DocumentGraphRequest {
            graph: complete.graph.clone(),
            pii_output: Some(PiiOutputOptions { full_text: true, ..Default::default() }),
        })
        .await
        .unwrap()
        .into_inner();
    assert_eq!(echoed.graph, complete.graph);
    server.stop().await;
}

async fn serve_relation_models(
    entity_package: PathBuf,
    relation_package: PathBuf,
) -> (Server, AnalysisServiceClient<tonic::transport::Channel>, Vec<BundleCapabilities>) {
    let server =
        Server::start("127.0.0.1:0".parse().unwrap(), Vec::new(), MAX, Serve { admin: true, ..Serve::default() })
            .await
            .unwrap();
    let mut client = AnalysisServiceClient::connect(format!("http://{}", server.local_addr()))
        .await
        .unwrap()
        .max_decoding_message_size(MAX)
        .max_encoding_message_size(MAX);
    for (name, path) in [("gliner", entity_package), ("relations", relation_package)] {
        client
            .load_bundle(LoadBundleRequest {
                name: name.into(),
                bundle: path.to_string_lossy().into_owned(),
                device: 0,
                sessions: 1,
                precision: 0,
                max_batch: 0,
                max_seq: 0,
                select_device: false,
                user: String::new(),
            })
            .await
            .unwrap();
    }
    let listed = client.capabilities(CapabilitiesRequest {}).await.unwrap().into_inner();
    assert_eq!(listed.bundles.len(), 2);
    (server, client, listed.bundles)
}

#[tokio::test]
#[ignore = "requires GLINER_PACKAGE and GLINER2_PACKAGE release candidates"]
async fn entities_and_neural_relations_share_one_graph() {
    let entity_package = turbo_pins::require(std::env::var("GLINER_PACKAGE").expect("GLINER_PACKAGE"));
    let relation_package = turbo_pins::require(std::env::var("GLINER2_PACKAGE").expect("GLINER2_PACKAGE"));
    let (server, mut client, capabilities) = serve_relation_models(entity_package, relation_package).await;
    let text = "Steve Jobs founded Apple in Cupertino. Tim Cook works for Apple.";
    let complete = analyze(
        &mut client,
        AnalyzePipelineRequest {
            doc_id: "entity-relations".into(),
            text: text.into(),
            model: "gliner".into(),
            relation_model: "relations".into(),
            steps: vec![
                "turbo-sentdetect:rules".into(),
                "turbo-tokenize:uax29-untyped".into(),
                "turbo:gliner".into(),
                "turbo:zero-shot-relations".into(),
            ],
            open_span: Some(OpenSpanSchema {
                labels: vec!["person".into(), "organization".into(), "location".into()],
                threshold: Some(0.8),
                nested: false,
                prompts: Vec::new(),
            }),
            relations: Some(OpenRelationSchema {
                labels: vec![
                    OpenRelationLabel {
                        name: "founded".into(),
                        threshold: None,
                        symmetric: false,
                        domain: vec!["person".into()],
                        range: vec!["organization".into()],
                    },
                    OpenRelationLabel {
                        name: "works for".into(),
                        threshold: None,
                        symmetric: false,
                        domain: vec!["person".into()],
                        range: vec!["organization".into()],
                    },
                    OpenRelationLabel {
                        name: "located in".into(),
                        threshold: None,
                        symmetric: false,
                        domain: vec!["organization".into()],
                        range: vec!["location".into()],
                    },
                ],
                threshold: Some(0.5),
                within_sentence: true,
            }),
            ..Default::default()
        },
    )
    .await;
    let doc = import(&DocumentGraph::from_bytes(&complete.graph).unwrap(), codecs()).unwrap();
    let relations = doc.get::<turbo_relation::Relations>().unwrap();
    assert!(!relations.spans.is_empty(), "no relations: {:?}", complete.layers);
    let edges = doc.get::<turbo_analysis::Edges<Entities, Entities>>().unwrap();
    assert!(!edges.from.is_empty());
    assert!(edges.from.len() <= relations.spans.len());
    doc.get::<Evidence<turbo_relation::Relations>>().unwrap().verify(&doc).unwrap();
    for row in 0..relations.spans.len() {
        let (subject, object, kind, score) = relations.mention(turbo_analysis::Ix::new(row as u32));
        assert!(subject.get() < doc.get::<Entities>().unwrap().len());
        assert!(object.get() < doc.get::<Entities>().unwrap().len());
        assert!(!kind.is_empty());
        assert!((0.0..=1.0).contains(&score));
    }
    let relation_capability = capabilities.iter().find(|c| c.task == "TASK_RELATION").unwrap();
    let provenance = doc.provenance(Key::<turbo_relation::Relations>::DEFAULT.id()).unwrap();
    assert_eq!(provenance.model.as_ref().unwrap().sha256, relation_capability.model.as_ref().unwrap().sha256);
    let echoed = client
        .round_trip_document_graph(DocumentGraphRequest { graph: complete.graph.clone(), pii_output: None })
        .await
        .unwrap()
        .into_inner();
    assert_eq!(echoed.graph, complete.graph);
    server.stop().await;
}
