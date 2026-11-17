use std::path::Path;
use std::sync::Arc;
use std::sync::atomic::Ordering;

use crate::config::{Device, ModelConfig};
use crate::deadline::Deadline;
use crate::offsets::OffsetMode;
use crate::priority::Priority;
use crate::proto::result::{RerankAndHighlightRequest, RerankCandidateInputProto, RerankedCandidateProto};
use crate::state::State;

use super::run_contextual_maxsim_highlights;

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn contextual_maxsim_cancel_during_query_skips_candidates_and_releases_session() {
    let bundle = Path::new(env!("CARGO_MANIFEST_DIR")).join("../testdata/tiny-bert-bundle");
    let cfg = ModelConfig {
        bundle: bundle.to_str().unwrap().to_owned(),
        name: Some("embed".into()),
        device: Device::Index(0),
        precision: 0,
        max_batch: 0,
        max_seq: 0,
        sessions: 1,
    };
    let state = Arc::new(State::new(vec!["embed".into()], vec![cfg]));
    state.load().unwrap();

    // Long multi-window query so query execution runs multiple batches, giving
    // the cancellation watcher a deterministic window during query encoding.
    let long_query = "account security password recovery ".repeat(80);
    let candidates: Vec<RerankCandidateInputProto> = (0..8)
        .map(|i| RerankCandidateInputProto {
            doc_id: format!("d{i}"),
            text: format!("Candidate {i} discusses account security and password recovery."),
            ..Default::default()
        })
        .collect();
    let top: Vec<RerankedCandidateProto> = (0..8)
        .map(|i| RerankedCandidateProto { doc_id: format!("d{i}"), input_index: i as u32, ..Default::default() })
        .collect();
    let req = Arc::new(RerankAndHighlightRequest {
        query: long_query,
        embed_model: "embed".into(),
        candidates,
        ..Default::default()
    });

    let deadline = Deadline::none();
    let state_w = Arc::clone(&state);
    let deadline_w = deadline.clone();
    let watcher = tokio::spawn(async move {
        while state_w.stream.runs.load(Ordering::Acquire) == 0 {
            tokio::task::yield_now().await;
        }
        deadline_w.trip();
    });

    let res = run_contextual_maxsim_highlights(
        &state,
        "embed",
        &req,
        OffsetMode::Utf8,
        top,
        2,
        Priority::Interactive,
        &deadline,
    )
    .await;
    watcher.await.unwrap();

    assert!(res.is_err(), "expected cancellation error, got {res:?}");
    let (_, loaded) = state.ready("embed", "").unwrap();
    // Session must be released and immediately available after cancellation.
    let held = loaded.acquire_async(Priority::Interactive, &Deadline::none()).await.unwrap();
    drop(held);
}
