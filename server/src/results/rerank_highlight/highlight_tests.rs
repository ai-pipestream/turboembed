//! The maxsim highlight call under a deadline that trips while it runs: no
//! model execution starts after the trip, in the query or in the
//! candidates, and the session goes back to the pool.

use std::path::Path;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use turbo_analysis::ModelRef;

use crate::api::Result;
use crate::config::{Device, ModelConfig};
use crate::deadline::Deadline;
use crate::offsets::OffsetMode;
use crate::priority::Priority;
use crate::proto::result::{RerankAndHighlightRequest, RerankCandidateInputProto, RerankedCandidateProto};
use crate::state::State;

use super::run_contextual_maxsim_highlights;

/// A text of `words` words. One model execution of the small bundle takes
/// about 2,000 of them, so a text of several times that takes several, and
/// one that stops early can be told from one that ran on.
fn long_text(words: usize) -> String {
    "account security password recovery ".repeat(words / 4)
}

/// The small embed bundle on one session.
fn embed_state() -> Arc<State> {
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
    state
}

/// A call's request and its ranked candidates, every candidate ranked.
fn call(query: String, candidates: Vec<String>) -> (Arc<RerankAndHighlightRequest>, Vec<RerankedCandidateProto>) {
    let top = (0..candidates.len())
        .map(|i| RerankedCandidateProto { doc_id: format!("d{i}"), input_index: i as u32, ..Default::default() })
        .collect();
    let candidates = candidates
        .into_iter()
        .enumerate()
        .map(|(i, text)| RerankCandidateInputProto { doc_id: format!("d{i}"), text, ..Default::default() })
        .collect();
    let req = RerankAndHighlightRequest { query, embed_model: "embed".into(), candidates, ..Default::default() };
    (Arc::new(req), top)
}

fn runs(state: &State) -> u64 {
    state.stream.runs.load(Ordering::Acquire)
}

/// What one call did: its answer, the model executions started when its
/// deadline had just tripped, and those started when it returned.
struct Ended {
    res: Result<(ModelRef, Vec<RerankedCandidateProto>)>,
    at_trip: u64,
    at_end: u64,
}

/// Runs the call on a fresh state and trips its deadline once `after` model
/// executions have started, or once the call has returned without that
/// many. `None` lets the call run to its end.
async fn run(query: String, candidates: Vec<String>, after: Option<u64>) -> Ended {
    let state = embed_state();
    let (req, top) = call(query, candidates);
    let deadline = Deadline::none();
    let returned = Arc::new(AtomicBool::new(false));
    let (state_w, deadline_w, returned_w) = (Arc::clone(&state), deadline.clone(), Arc::clone(&returned));
    let watcher = tokio::spawn(async move {
        let Some(after) = after else { return 0 };
        while runs(&state_w) < after && !returned_w.load(Ordering::Acquire) {
            tokio::task::yield_now().await;
        }
        deadline_w.trip();
        runs(&state_w)
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
    returned.store(true, Ordering::Release);
    let at_trip = watcher.await.unwrap();
    let at_end = runs(&state);
    // The session is back in the pool, whatever the call answered.
    let (_, loaded) = state.ready("embed", "").unwrap();
    drop(loaded.acquire_async(Priority::Interactive, &Deadline::none()).await.unwrap());
    Ended { res, at_trip, at_end }
}

/// An execution is counted after the last check of the call's flag before
/// it, so one that had passed the check may still be counted after the
/// trip. A second one means a step did not read the call's flag.
fn assert_nothing_started_after_the_trip(ended: &Ended) {
    let Ended { at_trip, at_end, .. } = ended;
    assert!(at_end <= &(at_trip + 1), "{at_end} executions started, {at_trip} of them when the call ended");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn contextual_maxsim_cancel_during_query_skips_candidates_and_releases_session() {
    let query = long_text(10_000);
    let candidates = |n: usize| (0..n).map(|i| format!("Candidate {i} discusses account security.")).collect();
    let alone = run(query.clone(), Vec::new(), None).await;
    alone.res.as_ref().expect("the query alone runs");
    assert!(alone.at_end >= 4, "the query takes {} executions; stopping it early could not be seen", alone.at_end);

    let ended = run(query, candidates(8), Some(1)).await;
    assert_nothing_started_after_the_trip(&ended);
    // The call answers only if the query and the candidates had all started
    // before it ended.
    assert!(ended.res.is_err() || ended.at_trip > alone.at_end, "answered after {} executions", ended.at_trip);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn contextual_maxsim_cancel_during_candidates_stops_the_remaining_windows() {
    let query = || "password recovery".to_owned();
    let candidates = || vec![long_text(1_500); 8];
    let query_runs = run(query(), Vec::new(), None).await.at_end;
    let whole = run(query(), candidates(), None).await;
    let highlighted = whole.res.as_ref().expect("the live call answers");
    assert!(highlighted.1.iter().all(|c| !c.highlights.is_empty()), "every candidate has a highlight");
    assert!(
        whole.at_end >= query_runs + 4,
        "the candidates take {} executions; stopping them early could not be seen",
        whole.at_end - query_runs
    );

    // The deadline trips when the first candidate execution has started.
    let ended = run(query(), candidates(), Some(query_runs + 1)).await;
    assert_nothing_started_after_the_trip(&ended);
    // The call answers only if every execution had started before it ended.
    assert!(ended.res.is_err() || ended.at_trip == whole.at_end, "answered after {} executions", ended.at_trip);
}
