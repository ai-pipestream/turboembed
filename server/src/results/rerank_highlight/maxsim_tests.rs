//! The maxsim highlight steps under a call that ends between them.

use std::sync::atomic::Ordering;

use crate::api::Runtime;

use super::*;
use crate::config::{DEFAULT_MAX_MESSAGE_BYTES, Device, ModelConfig};
use crate::proto::result::RerankCandidateInputProto;
use crate::{Serve, Server};

async fn serve_embed() -> Server {
    let rt = Runtime::create().unwrap();
    let cpu = (0..16)
        .find(|&i| rt.device_info(i).map(|d| d.kind == turbo::TURBO_DEVICE_CPU).unwrap_or(false))
        .expect("a CPU");
    let bundle = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../testdata/tiny-bert-bundle");
    let model = ModelConfig {
        name: Some("embed".into()),
        bundle: bundle.to_str().unwrap().into(),
        device: Device::Index(cpu),
        sessions: 1,
        precision: 0,
        max_batch: 0,
        max_seq: 0,
    };
    let server =
        Server::start("127.0.0.1:0".parse().unwrap(), vec![model], DEFAULT_MAX_MESSAGE_BYTES, Serve::default())
            .await
            .unwrap();
    server.load().await.unwrap();
    server
}

fn request() -> RerankAndHighlightRequest {
    let candidate =
        |id: &str, text: &str| RerankCandidateInputProto { doc_id: id.into(), text: text.into(), ..Default::default() };
    RerankAndHighlightRequest {
        query: "password reset".into(),
        candidates: vec![
            candidate("a", "To reset a password, open the settings page. Then pick a new one."),
            candidate("b", "The cafeteria menu changes on Monday. Nothing about accounts here."),
        ],
        ..Default::default()
    }
}

fn top() -> Vec<RerankedCandidateProto> {
    (0..2).map(|i| RerankedCandidateProto { input_index: i, ..Default::default() }).collect()
}

#[tokio::test]
async fn a_call_that_ends_while_the_query_runs_encodes_no_candidate() {
    let server = serve_embed().await;
    let state = &server.state;
    let (_, loaded) = state.ready("embed", "").unwrap();
    let req = request();
    let deadline = Deadline::none();
    let held = loaded.take().unwrap();
    held.reserve_token_ranges().unwrap();
    let run = Run { state, loaded: &loaded, slot: held.slot(), deadline: &deadline };

    let query = query_vectors(&run, &req).unwrap();
    let runs_after_query = state.stream.runs.load(Ordering::Relaxed);
    assert!(runs_after_query > 0, "the query ran on the session");

    // The call ends here, as a cancelled or timed-out RPC ends it.
    deadline.trip();
    let cancelled_before = loaded.metrics.cancelled_before_exec();
    let mut ranked = top();
    let err = candidate_highlights(&run, &req, OffsetMode::Utf8, &mut ranked, 2, &query)
        .expect_err("the candidates of an ended call are not encoded");
    assert!(err.message.contains("deadline"), "{}", err.message);
    assert_eq!(state.stream.runs.load(Ordering::Relaxed), runs_after_query, "no candidate window ran");
    assert_eq!(loaded.metrics.cancelled_before_exec(), cancelled_before + 1);
    assert!(ranked.iter().all(|r| r.highlights.is_empty()));

    // The session goes back to the pool for the next caller.
    drop(held);
    assert!(loaded.take().is_ok(), "the session is free again");
}

#[tokio::test]
async fn a_live_call_highlights_every_candidate() {
    let server = serve_embed().await;
    let state = &server.state;
    let (_, loaded) = state.ready("embed", "").unwrap();
    let req = request();
    let deadline = Deadline::none();
    let held = loaded.take().unwrap();
    held.reserve_token_ranges().unwrap();
    let run = Run { state, loaded: &loaded, slot: held.slot(), deadline: &deadline };

    let query = query_vectors(&run, &req).unwrap();
    let runs_after_query = state.stream.runs.load(Ordering::Relaxed);
    let mut ranked = top();
    candidate_highlights(&run, &req, OffsetMode::Utf8, &mut ranked, 2, &query).unwrap();
    assert!(state.stream.runs.load(Ordering::Relaxed) > runs_after_query, "the candidates ran");
    assert!(ranked.iter().all(|r| !r.highlights.is_empty()), "{ranked:?}");
}

#[tokio::test]
async fn a_call_that_ended_before_the_query_runs_nothing() {
    let server = serve_embed().await;
    let state = &server.state;
    let (_, loaded) = state.ready("embed", "").unwrap();
    let deadline = Deadline::none();
    deadline.trip();
    let held = loaded.take().unwrap();
    let run = Run { state, loaded: &loaded, slot: held.slot(), deadline: &deadline };
    let cancelled_before = loaded.metrics.cancelled_before_exec();
    let err = query_vectors(&run, &request()).err().expect("the query of an ended call is not encoded");
    assert!(err.message.contains("deadline"), "{}", err.message);
    assert_eq!(state.stream.runs.load(Ordering::Relaxed), 0);
    assert_eq!(loaded.metrics.cancelled_before_exec(), cancelled_before + 1);
}
