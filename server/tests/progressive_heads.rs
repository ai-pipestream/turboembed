//! The bundle head steps of `AnalyzeDocumentProgressive` (#1775) on the
//! small BERT in testdata with seeded heads: the `annotate-v1` profile
//! writes each head's layer exactly as the tagger crate's annotators
//! write it, the all-heads step runs what the bundle provides, a bundle
//! without a head is refused naming it, `Capabilities` lists each
//! bundle's head steps, and every head step of a request decodes from one
//! encoder pass over the document.

#[path = "progressive_heads/bundle.rs"]
mod bundle;
mod common;

use std::path::Path;

use bundle::*;
use common::*;
use serde_json::json;
use tokio_stream::StreamExt as _;
use tonic::Code;
use turbo_analysis::export::{DocumentGraph, import};
use turbo_analysis::{Analyzer, Document, Key, Layer, View};
use turbo_analysis_embed::{Device as EncoderDevice, EncoderOptions};
use turbo_analysis_layers::{Dependencies, Entities, Lemmas, PosTags, Sentences, Tokens};
use turbo_kserve::config::{DEFAULT_MAX_MESSAGE_BYTES as MAX, Device, ModelConfig};
use turbo_kserve::progressive::ANNOTATE_V1;
use turbo_kserve::progressive::heads::{GroupStep, HeadGroup, HeadStep};
use turbo_kserve::proto::result::analysis_service_client::AnalysisServiceClient;
use turbo_kserve::proto::result::analyze_progressive_event::Event;
use turbo_kserve::proto::result::*;
use turbo_kserve::results::document::codecs;
use turbo_kserve::results::typed::inspect_document_graph;
use turbo_kserve::{Serve, Server};
use turbo_relation::Relations;
use turbo_sentdetect::RuleSentenceDetector;
use turbo_tagger::{MultiTask, RelationTagger};
use turbo_tokenize::uax29::UntypedWordTokenizer;

type All = (PosTags, Entities, Dependencies<Tokens>, Lemmas);

const DOC: &str = "Ann met Bob in Paris, and the quiet old dog slept by the river. Bob walked the dog while Ann \
                   watched the river and the old dog.";

fn model(bundle: &Path, name: &str) -> ModelConfig {
    ModelConfig {
        bundle: bundle.to_str().unwrap().to_string(),
        name: Some(name.into()),
        device: Device::Index(cpu()),
        precision: 0,
        max_batch: 0,
        max_seq: 0,
        sessions: 1,
    }
}

async fn serve_models(models: Vec<ModelConfig>) -> (Server, AnalysisServiceClient<tonic::transport::Channel>) {
    let server = Server::start("127.0.0.1:0".parse().unwrap(), models, MAX, Serve { admin: true, ..Serve::default() })
        .await
        .unwrap();
    server.load().await.unwrap();
    let client = AnalysisServiceClient::connect(format!("http://{}", server.local_addr()))
        .await
        .unwrap()
        .max_decoding_message_size(MAX)
        .max_encoding_message_size(MAX);
    (server, client)
}

fn request(doc_id: &str, profile: &str, steps: &[&str]) -> AnalyzePipelineRequest {
    AnalyzePipelineRequest {
        doc_id: doc_id.into(),
        text: DOC.into(),
        source_bytes: Vec::new(),
        offsets: OffsetMode::Utf8 as i32,
        profile: profile.into(),
        steps: steps.iter().map(|s| s.to_string()).collect(),
        model: String::new(),
        initial_graph: Vec::new(),
    }
}

async fn run(client: &mut Analysis, req: AnalyzePipelineRequest) -> Vec<AnalyzeProgressiveEvent> {
    let mut stream = client.analyze_document_progressive(req).await.unwrap().into_inner();
    let mut events = Vec::new();
    while let Some(ev) = stream.next().await {
        events.push(ev.unwrap());
    }
    events
}

fn complete(events: &[AnalyzeProgressiveEvent]) -> &DocumentGraphResponse {
    let Some(Event::Complete(complete)) = &events.last().unwrap().event else { panic!("no Complete event") };
    complete
}

fn document(complete: &DocumentGraphResponse) -> Document {
    import(&DocumentGraph::from_bytes(&complete.graph).unwrap(), codecs()).unwrap()
}

fn encoder_options() -> EncoderOptions {
    EncoderOptions { device: EncoderDevice::Index(cpu()), ..EncoderOptions::default() }
}

fn spans(e: &Entities) -> Vec<(u32, u32)> {
    e.spans.iter().map(|(_, s)| (s.start, s.end)).collect()
}

#[tokio::test]
async fn annotate_v1_writes_each_head_layer_as_the_tagger_annotators_do() {
    let dir = bundle("annotate-v1", |_| {});
    let (server, mut client) = serve_models(vec![model(&dir, "heads")]).await;
    let events = run(&mut client, request("annotate", "annotate-v1", &[])).await;

    assert_eq!(events.len(), ANNOTATE_V1.steps.len() + 2);
    let Some(Event::Started(started)) = &events[0].event else { panic!("no Started event") };
    assert_eq!(started.step_ids.len(), ANNOTATE_V1.steps.len());
    for step in ANNOTATE_V1.steps {
        assert!(started.step_ids.iter().any(|s| s == step), "{step} planned: {:?}", started.step_ids);
    }
    let complete = complete(&events);
    let produced: Vec<&str> = complete.layers.iter().map(|l| l.layer.as_str()).collect();
    assert_eq!(produced, ANNOTATE_V1.layers);
    for l in &complete.layers {
        assert_eq!(l.status, LayerWireStatus::Ok as i32, "layer {} failed: {}", l.layer, l.status_detail);
    }

    // Field for field what the tagger crate writes over the same view,
    // tokens and sentences, from one encoder pass of its own.
    let doc = document(complete);
    let view = doc.require(&Key::<View>::DEFAULT).unwrap();
    let tokens = doc.get::<Tokens>().unwrap();
    let sentences = doc.get::<Sentences>().unwrap();
    assert!(sentences.spans.len() >= 2, "two sentences in the document");
    let multi = MultiTask::<All>::open(&dir, &[None; 4], &encoder_options()).unwrap();
    let (pos, ents, deps, lemmas) = multi.layers(view, tokens, Some(sentences), None).unwrap();

    let got = doc.get::<PosTags>().unwrap();
    assert_eq!(got.tags, pos.tags);
    assert_eq!(got.prob, pos.prob);
    assert_eq!(got.tagset.labels(), pos.tagset.labels());
    let got = doc.get::<Entities>().unwrap();
    assert_eq!((spans(got), &got.tokens, &got.kind, &got.score), (spans(&ents), &ents.tokens, &ents.kind, &ents.score));
    assert_eq!(got.kinds.labels(), ents.kinds.labels());
    let got = doc.get::<Dependencies<Tokens>>().unwrap();
    assert_eq!((&got.head, &got.relation), (&deps.head, &deps.relation));
    assert_eq!(got.relations.labels(), deps.relations.labels());
    let got = doc.get::<Lemmas>().unwrap();
    assert_eq!(got.lemma.iter().collect::<Vec<_>>(), lemmas.lemma.iter().collect::<Vec<_>>());
    assert_eq!(got.lemma.len(), tokens.spans.len());

    // The typed projection carries every head layer, one annotation a
    // token for the aligned ones and one an entity.
    let typed = inspect_document_graph(&InspectDocumentGraphRequest {
        doc_id: "annotate".into(),
        graph: complete.graph.to_vec(),
        view: 0,
        offsets: OffsetMode::Utf8 as i32,
    })
    .unwrap();
    let count = |id: &str| typed.layers.iter().find(|l| l.id == id).map(|l| l.annotations.len());
    assert_eq!(count(PosTags::ID), Some(tokens.spans.len()));
    assert_eq!(count(Lemmas::ID), Some(tokens.spans.len()));
    assert_eq!(count(Dependencies::<Tokens>::ID), Some(tokens.spans.len()));
    assert_eq!(count(Entities::ID), Some(ents.spans.len()));
    assert_ne!(ents.spans.len(), 0, "the seeded entity head marks something");

    server.stop().await;
}

#[tokio::test]
async fn the_all_heads_step_runs_every_head_the_bundle_provides() {
    let dir = bundle("all-heads", |m| m["heads"].as_array_mut().unwrap().push(relation_head(&["PER", "LOC"])));
    let (server, mut client) = serve_models(vec![model(&dir, "heads")]).await;
    let steps = ["turbo-sentdetect:rules", "turbo-tokenize:uax29-untyped", "turbo-tagger:heads"];
    let events = run(&mut client, request("all", "", &steps)).await;

    let Some(Event::Started(started)) = &events[0].event else { panic!("no Started event") };
    let mut planned: Vec<&str> = started.step_ids.iter().map(String::as_str).collect();
    planned.sort_unstable();
    let mut expected = vec![
        "turbo-sentdetect:rules",
        "turbo-tokenize:uax29-untyped",
        "turbo-tagger:pos",
        "turbo-tagger:entities",
        "turbo-tagger:dependencies",
        "turbo-tagger:lemmas",
        "turbo-tagger:relations",
    ];
    expected.sort_unstable();
    assert_eq!(planned, expected);
    assert_eq!(events.len(), expected.len() + 2);
    let complete = complete(&events);
    for l in &complete.layers {
        assert_eq!(l.status, LayerWireStatus::Ok as i32, "layer {} failed: {}", l.layer, l.status_detail);
    }

    // The relation head read the entity layer the entity head wrote, and
    // writes what the tagger crate's annotator writes over it.
    let doc = document(complete);
    let view = doc.require(&Key::<View>::DEFAULT).unwrap();
    let (tokens, sentences, ents) =
        (doc.get::<Tokens>().unwrap(), doc.get::<Sentences>().unwrap(), doc.get::<Entities>().unwrap());
    let tagger = RelationTagger::open(&dir, None, &encoder_options()).unwrap();
    let (relations, _) = tagger.relations(view, tokens, ents, Some(sentences), None).unwrap();
    let got = doc.get::<Relations>().unwrap();
    assert_eq!(got.subject, relations.subject);
    assert_eq!(got.object, relations.object);
    assert_eq!(got.kind, relations.kind);
    assert_eq!(got.score, relations.score);
    assert_eq!(got.kinds.labels(), relations.kinds.labels());

    server.stop().await;
}

#[tokio::test]
async fn capabilities_list_the_head_steps_of_each_loaded_bundle() {
    let with_relations =
        bundle("caps-relations", |m| m["heads"].as_array_mut().unwrap().push(relation_head(&["PER", "LOC"])));
    let (server, mut client) =
        serve_models(vec![model(&common::tiny(), "plain"), model(&with_relations, "heads")]).await;
    let caps = client.capabilities(CapabilitiesRequest {}).await.unwrap().into_inner();
    let steps = |name: &str| caps.bundles.iter().find(|b| b.name == name).unwrap().head_steps.clone();
    assert!(steps("plain").is_empty());
    assert_eq!(
        steps("heads"),
        [
            "turbo-tagger:pos",
            "turbo-tagger:entities",
            "turbo-tagger:dependencies",
            "turbo-tagger:lemmas",
            "turbo-tagger:relations"
        ]
    );
    let profile = caps.profiles.iter().find(|p| p.name == "annotate-v1").unwrap();
    assert_eq!(profile.required_bundles, ANNOTATE_V1.required_bundles);

    // With the plain bundle loaded too, the profile runs on the one with
    // the heads.
    let events = run(&mut client, request("pick", "annotate-v1", &[])).await;
    let complete = complete(&events);
    assert!(complete.layers.iter().any(|l| l.layer == PosTags::ID && l.status == LayerWireStatus::Ok as i32));

    server.stop().await;
}

#[tokio::test]
async fn a_bundle_without_the_head_is_refused_naming_the_head() {
    let (server, mut client) = serve().await;
    let err = client.analyze_document_progressive(request("no-heads", "annotate-v1", &[])).await.unwrap_err();
    assert_eq!(err.code(), Code::NotFound);
    assert!(err.message().contains("head.pos"), "{}", err.message());

    let steps = ["turbo-sentdetect:rules", "turbo-tokenize:uax29-untyped", "turbo-tagger:lemmas"];
    let err = client.analyze_document_progressive(request("no-lemma", "", &steps)).await.unwrap_err();
    assert_eq!(err.code(), Code::NotFound);
    assert!(err.message().contains("head.lemmas"), "{}", err.message());

    let steps = ["turbo-sentdetect:rules", "turbo-tokenize:uax29-untyped", "turbo-tagger:heads"];
    let err = client.analyze_document_progressive(request("no-any", "", &steps)).await.unwrap_err();
    assert_eq!(err.code(), Code::NotFound);
    assert!(err.message().contains("carries no head"), "{}", err.message());
    server.stop().await;

    // A bundle whose entity head is a span classifier serves the span
    // step, not the entity step the profile requires.
    let span_head = json!({
        "name": "mentions", "layer": "turboembed:entities", "kind": "HEAD_SPAN_CLASSIFIER",
        "labels": ["O", "PER", "LOC"], "scheme": "SCHEME_TAGS", "word_pooling": "WORD_POOLING_FIRST",
        "decode": "DECODE_SPANS_FLAT", "span": {"max_words": 3, "width_dim": 4},
        "artifact": "weights-f32", "tensor_prefix": "heads.mentions"
    });
    let dir = bundle("spans", |m| m["heads"][1] = span_head);
    let (server, mut client) = serve_models(vec![model(&dir, "spans")]).await;
    let caps = client.capabilities(CapabilitiesRequest {}).await.unwrap().into_inner();
    assert_eq!(
        caps.bundles[0].head_steps,
        ["turbo-tagger:pos", "turbo-tagger:dependencies", "turbo-tagger:lemmas", "turbo-tagger:spans"]
    );
    let err = client.analyze_document_progressive(request("spans", "annotate-v1", &[])).await.unwrap_err();
    assert_eq!(err.code(), Code::NotFound);
    assert!(err.message().contains("head.entities"), "{}", err.message());
    let steps = ["turbo-sentdetect:rules", "turbo-tokenize:uax29-untyped", "turbo-tagger:spans"];
    let events = run(&mut client, request("spans-ok", "", &steps)).await;
    let complete = complete(&events);
    assert!(complete.layers.iter().any(|l| l.layer == Entities::ID && l.status == LayerWireStatus::Ok as i32));
    server.stop().await;
}

/// The head steps of `steps` run over `text` in one analyzer, and the
/// rows their shared group read.
fn rows_read(dir: &Path, steps: &[HeadStep], text: &str) -> u64 {
    let o = EncoderOptions { max_seq: 8, max_batch: 3, ..encoder_options() };
    let group = HeadGroup::open(dir, steps, &o).unwrap();
    let mut builder = Analyzer::builder().add(RuleSentenceDetector::new("eng")).add(UntypedWordTokenizer::default());
    for &s in steps {
        builder = match s {
            HeadStep::Pos => builder.add(GroupStep::<PosTags>::new(group.clone())),
            HeadStep::Entities => builder.add(GroupStep::<Entities>::new(group.clone())),
            HeadStep::Dependencies => builder.add(GroupStep::<Dependencies<Tokens>>::new(group.clone())),
            HeadStep::Lemmas => builder.add(GroupStep::<Lemmas>::new(group.clone())),
            other => panic!("{other:?} is not a shared step"),
        };
    }
    let analyzer = builder.build().unwrap();
    let mut docs = vec![Document::new(text)];
    for r in analyzer.run(&mut docs) {
        r.unwrap();
    }
    assert!(docs[0].get::<Tokens>().is_some());
    group.rows()
}

#[test]
fn one_encoder_pass_serves_every_head_step_of_a_request() {
    let dir = bundle("rows", |_| {});
    // Rows of six pieces beside the special tokens: each sentence is read
    // in several overlapping windows, several rows a batch.
    let one = rows_read(&dir, &[HeadStep::Pos], DOC);
    assert!(one > 2, "the sentences are read in windows: {one} rows");
    let all = [HeadStep::Pos, HeadStep::Entities, HeadStep::Dependencies, HeadStep::Lemmas];
    assert_eq!(rows_read(&dir, &all, DOC), one);
    assert_eq!(rows_read(&dir, &[HeadStep::Entities, HeadStep::Lemmas], DOC), one);
}
