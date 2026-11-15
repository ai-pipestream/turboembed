//! Full text to document graph over the wire with the exact release
//! packages (#1948): a real server on a loopback port loads a converted
//! GLiNER package and a GLiNER2 package, and `AnalyzeDocumentProgressive`
//! runs every model-free stage, GLiNER entities, GLiNER2 relations and the
//! chunker over UTF-8, multi-sentence text. The served graph is compared
//! byte for byte with the graph the in-process Rust API writes from the
//! same packages ([`compile_analyzer_with_models`]), and checked as a graph
//! a caller reads: spans slice the source and keep their UTF-16 offsets,
//! relation edges link entities in the graph, the model digests the
//! server reports are the graph's provenance, a second run and the graph
//! round trip are byte-identical, empty text is a graph, and a document
//! longer than the relation encoder's window is refused by name. The
//! inspection calls read the same graph: `InspectDocumentGraph` gives each
//! entity at its UTF-16 offsets and each relation over the graph's entity
//! rows, with the served packages as their models, and
//! `InspectDocumentWindow` pages through the whole source with every
//! entity on a page.
//!
//! ```sh
//! GLINER_PACKAGE=... GLINER2_PACKAGE=... cargo test -p turbo-kserve \
//!     --test full_text_graph_served -- --ignored
//! ```

use std::path::{Path, PathBuf};
use std::sync::Arc;

use tokio_stream::StreamExt as _;
use turbo::deberta::package::PackageManifest;
use turbo_analysis::export::{DocumentGraph, export, import};
use turbo_analysis::{Document, Edges, Evidence, Ix, Key, ViewId};
use turbo_analysis_layers::{Chunks, Entities, Pii, Sentences, Tokens};
use turbo_gliner::{Decode, ExtractionSchema};
use turbo_kserve::api::TURBO_E_CAPACITY;
use turbo_kserve::config::DEFAULT_MAX_MESSAGE_BYTES as MAX;
use turbo_kserve::progressive::{Models, compile_analyzer_with_models};
use turbo_kserve::proto::result::analysis_service_client::AnalysisServiceClient;
use turbo_kserve::proto::result::analyze_progressive_event::Event;
use turbo_kserve::proto::result::*;
use turbo_kserve::results::document::codecs;
use turbo_kserve::{Serve, Server};
use turbo_relation::{Gliner2Relations, RelationLabel, Relations, ZeroShotRelationConfig};

type Client = AnalysisServiceClient<tonic::transport::Channel>;

/// Every model-free stage of the in-process acceptance but the place-name
/// gazetteer, whose `Entities` GLiNER writes here, then the model stages
/// and the chunker.
const STEPS: &[&str] = &[
    "turbo:normalize",
    "turbo-structure:blocks",
    "turbo-sentdetect:rules",
    "turbo-tokenize:uax29-untyped",
    "turbo-abbrev:schwartz-hearst",
    "turbo-stem:terms",
    "turbo-stem:stemmer",
    "turbo-stopword:bundled",
    "turbo-keywords:yake",
    "turbo-pii:pii",
    "turbo-numeric:numbers",
    "turbo-numeric:money",
    "turbo-numeric:temporal",
    "turbo:subwords",
    "turbo:gliner",
    "turbo:zero-shot-relations",
    "turbo:structure-boundaries",
    "turbo:lexical-boundaries",
    "turbo:chunker",
];

/// A heading and three sentences in two paragraphs, with diacritics, an
/// emoji and an ideograph outside the Basic Multilingual Plane, an NFKC
/// ligature, a currency sign, an email address and a phone number.
const TEXT: &str = "# Relatório\n\nZoë Müller \u{1F680} works for Acme Corporation in São Paulo. She paid €1,200 for \
                    the ﬁrst flat, mail zoe.muller@example.com or call +1 555 010 0777.\n\nTim Cook founded the \
                    World Health Organization office in Berlin with \u{20BB7}野 on 3 March 2024.";

const LABELS: [&str; 3] = ["person", "organization", "location"];
const THRESHOLD: f32 = 0.5;

/// (name, domain, range) of each relation label.
const RELATIONS: [(&str, &str, &str); 3] = [
    ("works for", "person", "organization"),
    ("founded", "person", "organization"),
    ("located in", "organization", "location"),
];

fn packages() -> (PathBuf, PathBuf) {
    let gliner = std::env::var("GLINER_PACKAGE").expect("GLINER_PACKAGE");
    let gliner2 = std::env::var("GLINER2_PACKAGE").expect("GLINER2_PACKAGE");
    (turbo_pins::require(gliner), turbo_pins::require(gliner2))
}

fn open_span() -> OpenSpanSchema {
    OpenSpanSchema {
        labels: LABELS.map(str::to_owned).to_vec(),
        threshold: Some(THRESHOLD),
        nested: false,
        prompts: Vec::new(),
    }
}

fn relation_schema() -> OpenRelationSchema {
    OpenRelationSchema {
        labels: RELATIONS
            .map(|(name, domain, range)| OpenRelationLabel {
                name: name.into(),
                threshold: None,
                symmetric: false,
                domain: vec![domain.into()],
                range: vec![range.into()],
            })
            .to_vec(),
        threshold: Some(THRESHOLD),
        within_sentence: true,
    }
}

/// The same packages and schemas, opened in process as the server opens a
/// loaded package.
fn models(gliner: &Path, gliner2: &Path) -> Models {
    let entity = turbo_gliner::package::Manifest::load(gliner).unwrap();
    let schema = ExtractionSchema::with_prompts(
        LABELS.map(str::to_owned).to_vec(),
        LABELS.map(str::to_owned).to_vec(),
        THRESHOLD,
        Decode::Flat,
    )
    .unwrap();
    let (identity, encoder) = PackageManifest::load(&gliner2.join("encoder")).unwrap();
    let relation =
        Gliner2Relations::load_with(encoder.into(), gliner2, &identity.source_repository, &identity.source_revision)
            .unwrap();
    let labels = RELATIONS
        .map(|(name, domain, range)| RelationLabel {
            name: name.into(),
            threshold: None,
            symmetric: false,
            domain: Some(vec![domain.into()]),
            range: Some(vec![range.into()]),
        })
        .to_vec();
    let config = ZeroShotRelationConfig {
        threshold: THRESHOLD.into(),
        within_sentence: true,
        ..ZeroShotRelationConfig::new(labels)
    };
    Models { gliner: Some((Arc::new(entity), schema)), relations: Some((Arc::new(relation), config)) }
}

/// The graph the in-process API writes, and the first failed stage's
/// error if one failed.
fn in_process(models: Models, text: &str) -> (Vec<u8>, Option<String>) {
    let mut doc = Document::new(text);
    let analyzer = compile_analyzer_with_models(&doc, STEPS, models).unwrap();
    let failed = analyzer.run(std::slice::from_mut(&mut doc)).into_iter().find_map(|r| r.err().map(|e| e.to_string()));
    (export(&doc, codecs()).unwrap().to_bytes().unwrap(), failed)
}

async fn serve(gliner: &Path, gliner2: &Path) -> (Server, Client, Vec<BundleCapabilities>) {
    let server =
        Server::start("127.0.0.1:0".parse().unwrap(), Vec::new(), MAX, Serve { admin: true, ..Serve::default() })
            .await
            .unwrap();
    let mut client = AnalysisServiceClient::connect(format!("http://{}", server.local_addr()))
        .await
        .unwrap()
        .max_decoding_message_size(MAX)
        .max_encoding_message_size(MAX);
    for (name, path) in [("gliner", gliner), ("relations", gliner2)] {
        let request = LoadBundleRequest {
            name: name.into(),
            bundle: path.to_string_lossy().into_owned(),
            sessions: 1,
            ..Default::default()
        };
        client.load_bundle(request).await.unwrap();
    }
    let listed = client.capabilities(CapabilitiesRequest {}).await.unwrap().into_inner();
    (server, client, listed.bundles)
}

/// The served complete graph and the steps that failed on the way.
async fn served(client: &mut Client, text: &str) -> (DocumentGraphResponse, Vec<AnalysisStepFailure>) {
    let request = AnalyzePipelineRequest {
        doc_id: "full-text-graph".into(),
        text: text.into(),
        model: "gliner".into(),
        relation_model: "relations".into(),
        steps: STEPS.iter().map(|s| (*s).to_owned()).collect(),
        open_span: Some(open_span()),
        relations: Some(relation_schema()),
        pii_output: Some(PiiOutputOptions { full_text: true, ..Default::default() }),
        ..Default::default()
    };
    let mut stream = client.analyze_document_progressive(request).await.unwrap().into_inner();
    let (mut complete, mut failures) = (None, Vec::new());
    while let Some(event) = stream.next().await {
        match event.unwrap().event {
            Some(Event::Complete(done)) => complete = Some(done),
            Some(Event::StepFailed(f)) => failures.push(f),
            _ => {}
        }
    }
    (complete.expect("a complete graph"), failures)
}

fn slice(doc: &Document, start: u32, end: u32) -> &str {
    doc.source().get(start as usize..end as usize).expect("a span on char boundaries")
}

#[tokio::test]
#[ignore = "requires GLINER_PACKAGE and GLINER2_PACKAGE release candidates"]
async fn the_served_graph_is_the_in_process_graph() {
    let (gliner, gliner2) = packages();
    let (local, failed) = in_process(models(&gliner, &gliner2), TEXT);
    assert_eq!(failed, None);
    let (server, mut client, capabilities) = serve(&gliner, &gliner2).await;
    let (complete, failures) = served(&mut client, TEXT).await;
    assert!(failures.is_empty(), "{failures:?}");
    for layer in &complete.layers {
        assert_eq!(layer.status, LayerWireStatus::Ok as i32, "{layer:?}");
    }
    assert!(complete.graph == local, "the served graph differs from the in-process graph");
    assert_eq!(served(&mut client, TEXT).await.0.graph, complete.graph, "a second run");

    let doc = import(&DocumentGraph::from_bytes(&complete.graph).unwrap(), codecs()).unwrap();
    assert_eq!(doc.source(), TEXT);
    let entities = doc.get::<Entities>().unwrap();
    let names: Vec<_> = entities.spans.iter().map(|(_, s)| slice(&doc, s.start, s.end)).collect();
    for name in ["Acme Corporation", "Berlin"] {
        assert!(names.contains(&name), "entity {name:?} in {names:?}");
    }
    let pii: Vec<_> = doc.get::<Pii>().unwrap().spans.iter().map(|(_, s)| slice(&doc, s.start, s.end)).collect();
    assert!(pii.contains(&"zoe.muller@example.com"), "{pii:?}");
    assert_eq!(doc.get::<Sentences>().unwrap().spans.len(), 3, "three sentences");
    assert!(!doc.get::<Chunks>().unwrap().spans.is_empty());
    doc.get::<Evidence<Entities>>().unwrap().verify(&doc).unwrap();

    let relations = doc.get::<Relations>().unwrap();
    assert!(!relations.spans.is_empty(), "no relations");
    let edges = doc.get::<Edges<Entities, Entities>>().unwrap();
    assert!(!edges.from.is_empty() && edges.from.len() <= relations.spans.len());
    for row in 0..relations.spans.len() {
        let (subject, object, kind, score) = relations.mention(Ix::new(row as u32));
        assert!(
            subject.get() < entities.spans.len() && object.get() < entities.spans.len(),
            "row {row} links entities"
        );
        assert!(RELATIONS.iter().any(|(name, _, _)| *name == kind), "{kind}");
        assert!((0.0..=1.0).contains(&score));
    }
    doc.get::<Evidence<Relations>>().unwrap().verify(&doc).unwrap();

    // Each span keeps the UTF-16 offset the text itself counts: the emoji
    // and the ideograph are one code point and two UTF-16 units each.
    let units = |at: u32| doc.source()[..at as usize].encode_utf16().count() as u32;
    let token_spans = doc.get::<Tokens>().unwrap().spans.iter().map(|(_, s)| s);
    for s in token_spans.chain(entities.spans.iter().map(|(_, s)| s)) {
        let u = doc.view_to_source_utf16(ViewId::SOURCE, s).unwrap();
        assert_eq!((u.start.0, u.end.0), (units(s.start), units(s.end)), "{s:?}");
        assert_eq!(doc.source_utf16_to_view(ViewId::SOURCE, u).unwrap(), s);
    }

    // The digests the server reports are the graph's model provenance.
    for (relation, key) in [(false, Key::<Entities>::DEFAULT.id()), (true, Key::<Relations>::DEFAULT.id())] {
        let capability = capabilities.iter().find(|c| (c.task == "TASK_RELATION") == relation).unwrap();
        let reported = &capability.model.as_ref().unwrap().sha256;
        assert_eq!(&doc.provenance(key).unwrap().model.as_ref().unwrap().sha256, reported, "{}", capability.name);
    }

    let echoed = client
        .round_trip_document_graph(DocumentGraphRequest {
            graph: complete.graph.clone(),
            pii_output: Some(PiiOutputOptions { full_text: true, ..Default::default() }),
        })
        .await
        .unwrap()
        .into_inner();
    assert_eq!(echoed.graph, complete.graph, "the graph round trip");
    server.stop().await;
}

#[tokio::test]
#[ignore = "requires GLINER_PACKAGE and GLINER2_PACKAGE release candidates"]
async fn empty_text_and_a_document_past_the_relation_window_match_in_process() {
    let (gliner, gliner2) = packages();
    let (server, mut client, _) = serve(&gliner, &gliner2).await;
    let (empty, failures) = served(&mut client, "").await;
    assert!(failures.is_empty(), "{failures:?}");
    assert!(empty.graph == in_process(models(&gliner, &gliner2), "").0, "empty text");

    // One sentence of 366 words, under GLiNER's 384, whose pieces are more
    // than the relation encoder's 512. GLiNER2 refuses it by its limit, in
    // process and served, and every other stage still writes the same graph.
    let long = format!("Tim Cook works for Acme Corporation{}.", " near Constantinople".repeat(180));
    let (local, failed) = in_process(models(&gliner, &gliner2), &long);
    let failed = failed.expect("the relation step refuses");
    assert!(failed.contains("max_sequence"), "{failed}");
    let (complete, failures) = served(&mut client, &long).await;
    let [failure] = &failures[..] else { panic!("one failed step: {failures:?}") };
    let wire = failure.failure.as_ref().unwrap();
    assert_eq!(wire.turbo_code, TURBO_E_CAPACITY, "{wire:?}");
    assert!(wire.message.contains("max_sequence 512"), "{wire:?}");
    assert!(complete.graph == local, "the served graph of a refused document differs from in process");
    server.stop().await;
}

#[test]
fn a_model_step_without_its_model_is_refused_by_name_in_process() {
    let doc = Document::new(TEXT);
    let refused = compile_analyzer_with_models(&doc, STEPS, Models::default()).err().expect("refused");
    assert_eq!(refused.message, "turbo:gliner requires its model");
    let steps: Vec<_> = STEPS.iter().copied().filter(|s| *s != "turbo:gliner").collect();
    let refused = compile_analyzer_with_models(&doc, &steps, Models::default()).err().expect("refused");
    assert_eq!(refused.message, "turbo:zero-shot-relations requires its model");
}

#[tokio::test]
#[ignore = "requires GLINER_PACKAGE and GLINER2_PACKAGE release candidates"]
async fn the_inspection_apis_read_the_served_graph_as_it_is() {
    let (gliner, gliner2) = packages();
    let (server, mut client, capabilities) = serve(&gliner, &gliner2).await;
    let graph = served(&mut client, TEXT).await.0.graph;
    let bytes: Vec<u8> = graph.to_vec();
    let doc = import(&DocumentGraph::from_bytes(&graph).unwrap(), codecs()).unwrap();
    let (entities, relations) = (doc.get::<Entities>().unwrap(), doc.get::<Relations>().unwrap());
    // Empty layers would let every check below pass on nothing.
    assert!(!entities.spans.is_empty() && !relations.spans.is_empty(), "the graph has entities and relations");

    // InspectDocumentGraph in UTF-16 units: every entity at the offsets the
    // graph maps it to, every relation linking the graph's entity rows.
    let typed = client
        .inspect_document_graph(InspectDocumentGraphRequest {
            doc_id: "full-text-graph".into(),
            graph: bytes.clone(),
            offsets: OffsetMode::Utf16 as i32,
            // The text holds PII, so its layers are read with the text.
            pii_output: Some(PiiOutputOptions { full_text: true, ..Default::default() }),
            ..Default::default()
        })
        .await
        .unwrap()
        .into_inner();
    let layer = |id: &str| typed.layers.iter().find(|l| l.id == id && l.instance.is_empty()).expect(id);
    let typed_entities = layer("turbo:entities");
    assert_eq!(typed_entities.annotations.len(), entities.spans.len());
    for (a, (_, s)) in typed_entities.annotations.iter().zip(entities.spans.iter()) {
        let u = doc.view_to_source_utf16(ViewId::SOURCE, s).unwrap();
        assert_eq!((a.spans[0].start, a.spans[0].end), (u.start.0, u.end.0), "{s:?}");
    }
    let typed_relations = layer("turbo:relations");
    assert_eq!(typed_relations.annotations.len(), relations.spans.len());
    for (row, a) in typed_relations.annotations.iter().enumerate() {
        let Some(typed_annotation::Value::Relation(r)) = &a.value else { panic!("row {row} is a relation") };
        let (subject, object, kind, _) = relations.mention(Ix::new(row as u32));
        assert_eq!(
            (r.subject_entity, r.object_entity, r.kind.as_str()),
            (subject.get() as u32, object.get() as u32, kind)
        );
    }
    // Each model layer names the package the server reports as its model.
    for (typed_layer, relation) in [(typed_entities, false), (typed_relations, true)] {
        let capability = capabilities.iter().find(|c| (c.task == "TASK_RELATION") == relation).unwrap();
        let digest = &capability.model.as_ref().unwrap().sha256;
        let named = typed_layer.model_refs.iter().filter_map(|r| typed.models.iter().find(|m| &m.model_ref == r));
        let bundles: Vec<_> = named.filter_map(|m| m.bundle.as_ref().map(|b| b.sha256.clone())).collect();
        assert!(bundles.contains(digest), "{}: {bundles:?}", typed_layer.id);
    }

    // InspectDocumentWindow, a page of 64 bytes at a time: the pages
    // together are the source, and each entity is a segment's annotation.
    let mut text = String::new();
    let mut labelled = std::collections::BTreeSet::new();
    let mut page = 0;
    loop {
        let w = client
            .inspect_document_window(DocumentWindowRequest {
                graph: bytes.clone(),
                offsets: OffsetMode::Utf8 as i32,
                page_size: 64,
                page,
                filter_layers: vec!["turbo:entities".into()],
                pii_output: Some(PiiOutputOptions { full_text: true, ..Default::default() }),
                ..Default::default()
            })
            .await
            .unwrap()
            .into_inner();
        let window = w.window.unwrap();
        text.push_str(&window.text);
        for s in &w.segments {
            labelled.extend(s.annotations.iter().filter(|a| a.layer == "turbo:entities").map(|a| a.item));
        }
        page += 1;
        if page >= window.page_count {
            break;
        }
    }
    assert_eq!(text, TEXT, "the pages are the source");
    assert_eq!(labelled.len(), entities.spans.len(), "every entity is on a page");
    server.stop().await;
}
