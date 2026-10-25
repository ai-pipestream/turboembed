//! The reviewers' regression probes for the streaming bulk analysis
//! call (issue #722), over a real client with the server in-process on
//! the CPU: byte caps reserve real bytes and refuse oversized
//! documents at once; failures and out-of-order results hold their
//! queue room until answered, bounded by the advertised caps; waiting
//! for a session never blocks cancellation; and an unload plus reload
//! under one name never mixes loads in one batch. Nothing is mocked.

mod common;

use std::time::{Duration, Instant};

use turbo_kserve::proto::result::analyze_stream_response::Outcome;
use turbo_kserve::proto::result::{AnalyzeOptions, CapabilitiesRequest, LoadBundleRequest, UnloadBundleRequest};
use turbo_kserve::stream;

use common::*;

/// A text just under two MiB with its id and model name, so two
/// documents fit the 4 MiB stream byte cap and the third cannot.
fn two_mb_less(id: &str) -> String {
    let overhead = id.len() + NAME.len();
    "x".repeat((2 << 20) - 64 - overhead)
}

#[tokio::test]
async fn byte_pressure_blocks_admission_while_item_room_remains() {
    let (s, mut c) = serve().await;
    let list = c.capabilities(CapabilitiesRequest {}).await.unwrap().into_inner();
    let byte_cap = list.server.unwrap().stream.unwrap().max_queued_bytes_per_stream as usize;

    // Two documents reserve the whole byte cap; item permits remain
    // (256 a stream), and the third document is held by backpressure.
    let mut stream = open_stream(&mut c, 1, 0, 1000).await;
    for i in 0..3 {
        stream.send(&format!("big-{i}"), NAME, &two_mb_less(&format!("big-{i}")), None).await;
    }
    let sampled = Instant::now();
    while sampled.elapsed() < Duration::from_millis(800) {
        let stats = s.stream_stats();
        assert!(stats.queued_bytes <= byte_cap, "queued_bytes over the cap: {stats:?}");
        assert!(stats.queued_items <= 2, "a third document was admitted past the byte cap: {stats:?}");
        tokio::time::sleep(Duration::from_millis(50)).await;
    }
    // The stream is healthy: all three answer once the input closes and
    // the reads begin.
    let mut stream = stream.half_close();
    let mut answered = 0;
    while stream.responses.message().await.unwrap().is_some() {
        answered += 1;
    }
    assert_eq!(answered, 3);
}

#[tokio::test]
async fn a_document_over_the_byte_cap_refuses_at_once_naming_the_limit() {
    let (_s, mut c) = serve().await;
    let mut stream = open_stream(&mut c, 0, 0, 0).await;
    stream.send("too-big", NAME, &"x".repeat(stream::MAX_QUEUED_BYTES_PER_STREAM + 1), None).await;
    stream.send("fine", NAME, "small and ordinary", None).await;
    let answers = stream.half_close().collect().await;
    assert_eq!(answers.len(), 2);
    match &answers[0].1 {
        Outcome::Failure(f) => {
            assert_eq!(f.turbo_code, 771, "TURBO_E_CAPACITY");
            assert!(f.message.contains("max_queued_bytes_per_stream"), "{}", f.message);
        }
        o => panic!("too-big: {o:?}"),
    }
    match &answers[1].1 {
        Outcome::Vectors(v) => assert_eq!(v.dim, 32),
        o => panic!("fine: {o:?}"),
    }
}

#[tokio::test]
async fn a_window_of_one_bounds_failures_waiting_on_an_in_flight_document() {
    let (s, mut c) = serve().await;
    // Sequence 0 takes a slow path (one second's batch linger); three
    // hundred validation failures arrive behind it. Every failure holds
    // its queue room until its answer is sent, so the reader blocks at
    // the advertised cap instead of buffering without bound.
    let mut stream = open_stream(&mut c, 0, 1, 1000).await;
    stream.send("d0", NAME, "the slow first document", None).await;
    let n = 500;
    let sent = std::sync::Arc::new(std::sync::atomic::AtomicUsize::new(0));
    let sender = tokio::spawn({
        let tx = stream.tx.clone();
        let sent = sent.clone();
        async move {
            for i in 0..n {
                tx.send(stream_doc(&format!("bad-{i}"), "served-nowhere", "x", None)).await.unwrap();
                sent.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            }
        }
    });
    // The failures hold their queue room while they wait for sequence
    // 0, so the queue fills to the advertised cap and holds there; the
    // rest of the flood waits unread below. Without that, the queue
    // would stay at the one in-flight document while the ordered buffer
    // grew without bound.
    let cap = stream::MAX_QUEUED_ITEMS_PER_STREAM;
    let filled = tokio::time::timeout(Duration::from_secs(5), async {
        while s.stream_stats().queued_items < cap {
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
    })
    .await;
    assert!(filled.is_ok(), "the queue never filled: the failures hold no room");
    assert_eq!(s.stream_stats().queued_items, cap);
    sender.await.unwrap();
    assert_eq!(sent.load(std::sync::atomic::Ordering::Relaxed), n);

    // Once the first document answers, everything drains in order.
    let mut stream = stream.half_close();
    let mut answers = Vec::new();
    while let Some(r) = stream.responses.message().await.unwrap() {
        answers.push((r.id.clone(), r.outcome.unwrap()));
    }
    assert_eq!(answers.len(), 1 + n);
    assert_eq!(answers[0].0, "d0");
    for (k, (id, outcome)) in answers.iter().enumerate().skip(1) {
        assert_eq!(id, &format!("bad-{}", k - 1));
        match outcome {
            Outcome::Failure(f) => assert_eq!(f.turbo_code, 1280),
            o => panic!("{id}: {o:?}"),
        }
    }
}

#[tokio::test]
async fn cancelling_while_the_session_pool_is_held_drops_the_stream() {
    // One session, held by another caller the whole time.
    let s = turbo_kserve::Server::start(
        "127.0.0.1:0".parse().unwrap(),
        vec![turbo_kserve::config::ModelConfig {
            bundle: tiny().to_str().unwrap().to_string(),
            name: None,
            device: turbo_kserve::config::Device::Index(cpu()),
            precision: 0,
            max_batch: 0,
            max_seq: 0,
            sessions: 1,
        }],
        turbo_kserve::config::DEFAULT_MAX_MESSAGE_BYTES,
        turbo_kserve::Serve::default(),
    )
    .await
    .unwrap();
    s.load().await.unwrap();
    let _held = s.hold(NAME).expect("the only session");
    let before = s.stream_stats();
    let mut c = turbo_kserve::proto::result::analysis_service_client::AnalysisServiceClient::connect(format!(
        "http://{}",
        s.local_addr()
    ))
    .await
    .unwrap();
    let mut stream = open_stream(&mut c, 1, 1, 0).await;
    stream.send("stuck", NAME, "no session will ever come", None).await;
    tokio::time::sleep(Duration::from_millis(50)).await;
    drop(stream.tx);
    drop(stream.responses);
    // The engine observes the cancellation without waiting for the
    // session: the queue is empty while the external holder is held.
    let start = Instant::now();
    let drained = tokio::time::timeout(Duration::from_secs(2), async {
        while s.stream_stats().queued_items > 0 {
            tokio::time::sleep(Duration::from_millis(5)).await;
        }
    })
    .await;
    assert!(drained.is_ok(), "the queue never drained after cancellation");
    assert!(start.elapsed() < Duration::from_secs(2), "cancellation waited on the session pool");
    assert_eq!(s.stream_stats().queued_items, 0);
    assert_eq!(s.stream_stats().runs, before.runs);
}

#[tokio::test]
async fn unload_and_reload_under_one_name_never_share_a_batch() {
    let (s, mut c) = serve().await;
    let before = s.stream_stats().runs;
    let none = AnalyzeOptions { truncate: 1, ..Default::default() };
    // A document admitted against the first load waits a long linger;
    // swap the load under the same name before it runs.
    let mut stream = open_stream(&mut c, 0, 0, 1500).await;
    stream.send("old", NAME, "hello there", Some(none)).await;
    c.unload_bundle(UnloadBundleRequest { name: NAME.into() }).await.unwrap();
    c.load_bundle(LoadBundleRequest {
        name: NAME.into(),
        bundle: tiny().to_str().unwrap().to_string(),
        device: cpu(),
        sessions: 1,
        precision: 0,
        max_batch: 0,
        max_seq: 8,
        select_device: false,
    })
    .await
    .unwrap();
    // Over the new load's max_seq with truncation off: the
    // single-document call refuses it, and so must the stream.
    let long = "the quick brown fox jumps over the lazy dog and runs far away today";
    stream.send("new", NAME, long, Some(none)).await;
    let answers = stream.half_close().collect().await;
    assert_eq!(answers.len(), 2);
    assert_eq!(answers[0].0, "old");
    match &answers[0].1 {
        Outcome::Vectors(v) => assert_eq!(v.dim, 32, "the old document ran on the load it was admitted against"),
        o => panic!("old: {o:?}"),
    }
    assert_eq!(answers[1].0, "new");
    match &answers[1].1 {
        Outcome::Failure(f) => assert_eq!(f.turbo_code, 771, "the new load refused the over-length document"),
        o => panic!("new: {o:?}"),
    }
    // Two batches ran as two keys: the old document alone.
    assert_eq!(s.stream_stats().runs - before, 1);
}

#[tokio::test]
async fn input_order_dispatches_the_earliest_sequence_first() {
    // With a window of one, seq 0 on the lexicographically later key
    // must still run before seq 1: a buffered response holds its window
    // permit until every earlier sequence is answered, so dispatching
    // the later sequence first would deadlock waiting for it.
    let (_s, mut c) = serve().await;
    let mut stream = open_stream(&mut c, 0, 1, 50).await;
    let later_key = AnalyzeOptions { normalize: 2, ..Default::default() };
    let earlier_key = AnalyzeOptions { normalize: 1, ..Default::default() };
    stream.send("d0", NAME, "first by sequence", Some(later_key)).await;
    stream.send("d1", NAME, "second by sequence", Some(earlier_key)).await;
    let answers = tokio::time::timeout(Duration::from_secs(30), stream.half_close().collect())
        .await
        .expect("input order deadlocked on the key ordering");
    let ids: Vec<String> = answers.iter().map(|(id, _)| id.clone()).collect();
    assert_eq!(ids, vec!["d0", "d1"]);
    for (_, outcome) in answers {
        match outcome {
            Outcome::Vectors(v) => assert_eq!(v.dim, 32),
            o => panic!("{o:?}"),
        }
    }
}

#[tokio::test]
async fn oversized_documents_hold_queue_room_like_everything_else() {
    // An over-byte-cap refusal keeps one item and its id's bytes until
    // its answer is sent, so a flood of oversized documents is bounded
    // by the same caps as documents, not buffered without accounting.
    let (s, mut c) = serve().await;
    let mut stream = open_stream(&mut c, 0, 2, 1000).await;
    stream.send("d0", NAME, "the slow first document", None).await;
    let big = "x".repeat(stream::MAX_QUEUED_BYTES_PER_STREAM + 1);
    let n = 8;
    for i in 0..n {
        stream.send(&format!("big-{i}"), NAME, &big, None).await;
    }
    // Every oversized refusal holds its room while it waits behind
    // sequence 0: the queue counts them all.
    let counted = tokio::time::timeout(Duration::from_secs(5), async {
        while s.stream_stats().queued_items < 1 + n {
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
    })
    .await;
    assert!(counted.is_ok(), "the oversized refusals hold no queue room");
    assert!(s.stream_stats().queued_bytes <= stream::MAX_QUEUED_BYTES_PER_STREAM);
    let answers = stream.half_close().collect().await;
    assert_eq!(answers.len(), 1 + n);
    assert_eq!(answers[0].0, "d0");
    match &answers[0].1 {
        Outcome::Vectors(v) => assert_eq!(v.dim, 32),
        o => panic!("d0: {o:?}"),
    }
    for (id, outcome) in answers.iter().skip(1) {
        match outcome {
            Outcome::Failure(f) => {
                assert_eq!(f.turbo_code, 771, "{id}");
                assert!(f.message.contains("max_queued_bytes_per_stream"), "{}", f.message);
            }
            o => panic!("{id}: {o:?}"),
        }
    }
}

#[tokio::test]
async fn a_partially_parked_window_still_runs_the_earliest_pending_document() {
    let (_s, mut c) = serve().await;
    // d0 and d4 share one options key (normalize 1), the rest another: two
    // dispatch groups, window 4, input order. The first round runs d0, d4,
    // d1, d2; d0, d1, d2 answer and free their permits while d4's answer
    // parks its permit in the ordered buffer behind the unanswered d3. The
    // pending group of four must then dispatch three (the permits actually
    // free), not deadlock asking for all four.
    let mut stream = open_stream(&mut c, 0, 4, 500).await;
    for (i, normalize) in [1, 0, 0, 0, 1, 0, 0, 0].into_iter().enumerate() {
        let options = AnalyzeOptions { normalize, ..Default::default() };
        stream.send(&format!("d{i}"), NAME, "a short document", Some(options)).await;
    }
    let answers = tokio::time::timeout(Duration::from_secs(10), stream.half_close().collect())
        .await
        .expect("a window permit parked in the ordered buffer deadlocked the stream");
    assert_eq!(answers.len(), 8);
    for (k, (id, outcome)) in answers.iter().enumerate() {
        assert_eq!(id, &format!("d{k}"));
        match outcome {
            Outcome::Vectors(v) => assert_eq!(v.dim, 32, "{id}"),
            o => panic!("{id}: {o:?}"),
        }
    }
}

#[tokio::test]
async fn a_refusal_retains_only_the_id_bytes_it_charges() {
    let (_s, mut c) = serve().await;
    // An id over the stream's byte cap on its own: the refusal retains the
    // id only up to the cap, exactly the charge its room claims, instead of
    // holding the whole id behind a clamped charge.
    let big_id = "i".repeat(2 * stream::MAX_QUEUED_BYTES_PER_STREAM);
    let mut stream = open_stream(&mut c, 0, 1, 0).await;
    stream.send(&big_id, NAME, "x", None).await;
    let answers = stream.half_close().collect().await;
    assert_eq!(answers.len(), 1);
    let (id, outcome) = &answers[0];
    assert_eq!(id.len(), stream::MAX_QUEUED_BYTES_PER_STREAM);
    match outcome {
        Outcome::Failure(f) => {
            assert_eq!(f.turbo_code, 771, "TURBO_E_CAPACITY");
            assert!(f.message.contains("max_queued_bytes_per_stream"), "{}", f.message);
        }
        o => panic!("big id: {o:?}"),
    }
}
