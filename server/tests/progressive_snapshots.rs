//! Every progressive snapshot remains a transportable graph when a real
//! annotator fails before another required producer finishes. No test
//! annotators, fault hooks, sleeps or elapsed-time assertions are used.

use std::sync::atomic::AtomicBool;

use tokio_stream::StreamExt;
use turbo_analysis::export::{DocumentGraph, export, import};
use turbo_analysis::{Analyzer, Bound, Document, Key, Status};
use turbo_analysis_layers::{Sentences, Terms};
use turbo_keywords::Keywords;
use turbo_kserve::progressive::compile_analyzer;
use turbo_kserve::proto::result::analysis_service_client::AnalysisServiceClient;
use turbo_kserve::proto::result::analyze_progressive_event::Event;
use turbo_kserve::proto::result::{AnalyzePipelineRequest, LayerWireStatus};
use turbo_kserve::results::document::codecs;
use turbo_kserve::{Serve, Server};
use turbo_normalize::{NormalizeStep, UnicodeForm};
use turbo_sentdetect::NewlineDetector;
use turbo_tokenize::uax29::UntypedWordTokenizer;

const STEPS: &[&str] = &["turbo-sentdetect:rules", "turbo-stem:terms", "turbo-keywords:yake", "turbo-structure:blocks"];

fn seed() -> Document {
    let text = "# Heading\n\n".to_owned() + &"Café grew. ".repeat(4096);
    let a = Analyzer::builder()
        .step(Bound::new(NormalizeStep::new(UnicodeForm::nfc())).write(0, "nfc"))
        .step(Bound::new(NewlineDetector::default()).read(0, "nfc").write(0, "nfc"))
        .step(Bound::new(UntypedWordTokenizer::default()).read(0, "nfc").read(1, "nfc"))
        .build()
        .unwrap();
    let mut d = Document::new(text);
    assert_eq!(a.run(std::slice::from_mut(&mut d)), [Ok(())]);
    assert!(d.get::<Sentences>().is_none());
    assert!(d.get_key(&Key::<Sentences>::named("nfc")).is_some());
    import(&export(&d, codecs()).unwrap(), codecs()).unwrap()
}

fn graph(d: &Document) -> Vec<u8> {
    export(d, codecs()).unwrap().to_bytes().unwrap()
}

fn check_snapshot(bytes: &[u8]) -> Result<(), String> {
    let g = DocumentGraph::from_bytes(bytes).map_err(|e| e.to_string())?;
    let restored = import(&g, codecs()).map_err(|e| e.to_string())?;
    assert_eq!(graph(&restored), bytes);
    Ok(())
}

#[test]
fn scheduler_failure_cascade_keeps_every_snapshot_importable() {
    let initial = seed();
    let a = compile_analyzer(&initial, STEPS).unwrap();
    let mut expected = initial.clone();
    assert!(a.run(std::slice::from_mut(&mut expected))[0].is_err());
    let mut actual = initial;
    let mut invalid = Vec::new();
    let mut reported_skip = false;
    let status = a.run_progressive(std::slice::from_mut(&mut actual), &AtomicBool::new(false), |e| {
        if e.timing.step_id == "turbo-stem:terms" {
            assert!(e.failure.is_some());
            reported_skip = e.skipped_dependents.contains(&"turbo-keywords:yake");
        }
        if let Err(error) = check_snapshot(&graph(&e.docs[0])) {
            invalid.push((e.timing.step_id, error));
        }
    });
    assert!(status[0].is_err());
    assert!(reported_skip, "the failure must still advertise its skipped dependent");
    assert!(matches!(actual.status(Key::<Terms>::DEFAULT.id()), Some(Status::Failed(_))));
    assert!(matches!(actual.status(Key::<Keywords>::DEFAULT.id()), Some(Status::Skipped(_))));
    assert_eq!(graph(&actual), graph(&expected));
    assert!(invalid.is_empty(), "scheduler snapshots cannot cross the graph transport: {invalid:?}");
}

#[tokio::test]
async fn rpc_failure_cascade_keeps_every_layer_batch_importable() {
    let initial = seed();
    let mut expected = initial.clone();
    let a = compile_analyzer(&initial, STEPS).unwrap();
    assert!(a.run(std::slice::from_mut(&mut expected))[0].is_err());
    let server = Server::start(
        "127.0.0.1:0".parse().unwrap(),
        vec![],
        8 * 1024 * 1024,
        Serve { admin: true, ..Default::default() },
    )
    .await
    .unwrap();
    let mut client = AnalysisServiceClient::connect(format!("http://{}", server.local_addr())).await.unwrap();
    let request = AnalyzePipelineRequest {
        initial_graph: graph(&initial),
        steps: STEPS.iter().map(|s| (*s).to_owned()).collect(),
        ..Default::default()
    };
    let mut stream = client.analyze_document_progressive(request).await.unwrap().into_inner();
    let mut sequence = 0;
    let mut invalid = Vec::new();
    let mut reported_skip = false;
    let mut complete = None;
    while let Some(event) = stream.next().await {
        let event = event.unwrap();
        sequence += 1;
        assert_eq!(event.sequence, sequence);
        match event.event.unwrap() {
            Event::Started(_) => assert_eq!(sequence, 1),
            Event::StepFailed(f) => {
                assert_eq!(f.step_id, "turbo-stem:terms");
                reported_skip = f.skipped_dependents.iter().any(|s| s == "turbo-keywords:yake");
            }
            Event::LayersReady(batch) => {
                if let Err(error) = check_snapshot(&batch.graph) {
                    invalid.push((batch.completed_step, error));
                }
            }
            Event::Complete(c) => {
                assert!(
                    c.layers.iter().any(|l| l.layer == "turbo:keywords" && l.status == LayerWireStatus::Skipped as i32)
                );
                complete = Some(c);
            }
        }
    }
    server.stop().await;
    assert!(reported_skip);
    let complete = complete.expect("a failed branch still yields a terminal graph");
    check_snapshot(&complete.graph).unwrap();
    assert_eq!(complete.graph, graph(&expected));
    assert!(invalid.is_empty(), "RPC layer batches cannot cross the graph transport: {invalid:?}");
}
