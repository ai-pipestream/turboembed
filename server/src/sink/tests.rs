use super::*;
use std::sync::{Arc, Barrier};

use crate::proto::result::{AnalyzeFailure, AnalyzeVectors};

fn answer(id: &str, failed: bool) -> AnalyzeStreamResponse {
    let outcome = match failed {
        true => Outcome::Failure(AnalyzeFailure::default()),
        false => Outcome::Vectors(AnalyzeVectors { dim: 1, data: vec![0; 4].into(), ..Default::default() }),
    };
    AnalyzeStreamResponse { id: id.into(), outcome: Some(outcome), binding: None }
}

fn config(items: usize, bytes: u64) -> SinkConfig {
    SinkConfig {
        endpoint: "http://127.0.0.1:1".into(),
        max_queue_items: items,
        max_queue_bytes: bytes,
        documents: false,
        typed: None,
        pii_output: Default::default(),
    }
}

// On one thread the draining task cannot run between offers, so the
// queue holds what its bounds let in and the rest is dropped.
#[tokio::test(flavor = "current_thread")]
async fn a_full_queue_drops_and_counts_without_waiting() {
    let f = SinkForwarder::start(&config(2, 1 << 20)).unwrap();
    for i in 0..5 {
        f.offer(&answer(&i.to_string(), false));
    }
    f.offer(&answer("failed", true));
    let s = f.stats();
    assert_eq!((s.offered, s.dropped, s.queued), (5, 3, 2), "{s:?}");
    assert!(s.queued_bytes > 0);
    let f = SinkForwarder::start(&config(16, 8)).unwrap();
    f.offer(&answer("too big", false));
    let s = f.stats();
    assert_eq!((s.offered, s.dropped, s.queued, s.queued_bytes), (1, 1, 0, 0), "{s:?}");
    assert!(SinkForwarder::start(&config(0, 8)).is_err());
    assert!(SinkForwarder::start(&SinkConfig { endpoint: "not a uri".into(), ..config(1, 1) }).is_err());
    let typed = SinkTyped { view: 1 << 16, offsets: crate::proto::result::OffsetMode::Utf8 };
    assert!(SinkForwarder::start(&SinkConfig { typed: Some(typed), ..config(1, 1) }).is_err());
    assert!(SinkForwarder::start_with(&config(1, 1), Some(typed)).is_err());
}

/// close takes the sender under the same lock as enqueue's offered count
/// and send. Pausing an offer after it has counted offered, then starting
/// close, must still let that offer into the queue: close blocks on the
/// lock, the send finishes, and the item is not counted dropped.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn close_waits_out_an_in_flight_offer_instead_of_dropping_it() {
    let f = SinkForwarder::start(&config(8, 1 << 20)).unwrap();
    let entered = Arc::new(Barrier::new(2));
    let finish = Arc::new(Barrier::new(2));
    {
        let entered = entered.clone();
        let finish = finish.clone();
        *f.enqueue_gap.lock().unwrap_or_else(|p| p.into_inner()) = Some(Box::new(move || {
            entered.wait();
            finish.wait();
        }));
    }
    let offer = {
        let f = f.clone();
        std::thread::spawn(move || f.offer(&answer("in-flight", false)))
    };
    entered.wait();
    assert_eq!(f.stats().offered, 1, "the offer counted before the send");
    let close = {
        let f = f.clone();
        tokio::spawn(async move { f.close(Duration::from_millis(400)).await })
    };
    // close is blocked on the sender lock held by the offer; release the
    // offer so it can send, then close takes the sender and drains.
    finish.wait();
    offer.join().unwrap();
    close.await.unwrap();
    let s = f.stats();
    assert_eq!(s.offered, 1, "{s:?}");
    assert_eq!(s.dropped, 0, "an in-flight offer must not be dropped by close: {s:?}");
    assert_eq!(s.forwarded + s.abandoned + s.queued, 1, "{s:?}");
}

/// Offers that finished before close took the sender stay in the queue
/// for close to drain or abandon; they are not counted dropped.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn offers_finished_before_close_are_not_dropped() {
    let f = SinkForwarder::start(&config(64, 1 << 20)).unwrap();
    let start = Arc::new(Barrier::new(8));
    let mut offers = Vec::new();
    for i in 0..8 {
        let f = f.clone();
        let start = start.clone();
        offers.push(std::thread::spawn(move || {
            start.wait();
            f.offer(&answer(&i.to_string(), false));
        }));
    }
    for o in offers {
        o.join().unwrap();
    }
    assert_eq!(f.stats().queued, 8);
    f.close(Duration::from_millis(400)).await;
    let s = f.stats();
    assert_eq!((s.offered, s.dropped, s.abandoned, s.queued), (8, 0, 8, 0), "{s:?}");
}

/// The enqueue gap hook lives on the forwarder, so a barrier parked on
/// one instance cannot stall offers on another running in parallel.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn enqueue_gap_on_one_forwarder_does_not_stall_another() {
    let parked = SinkForwarder::start(&config(8, 1 << 20)).unwrap();
    let free = SinkForwarder::start(&config(8, 1 << 20)).unwrap();
    let entered = Arc::new(Barrier::new(2));
    let finish = Arc::new(Barrier::new(2));
    {
        let entered = entered.clone();
        let finish = finish.clone();
        *parked.enqueue_gap.lock().unwrap_or_else(|p| p.into_inner()) = Some(Box::new(move || {
            entered.wait();
            finish.wait();
        }));
    }
    let offer_parked = {
        let f = parked.clone();
        std::thread::spawn(move || f.offer(&answer("parked", false)))
    };
    entered.wait();
    assert_eq!(parked.stats().offered, 1);
    for i in 0..5 {
        free.offer(&answer(&i.to_string(), false));
    }
    assert_eq!(free.stats().offered, 5, "{:?}", free.stats());
    assert_eq!(free.stats().dropped, 0, "{:?}", free.stats());
    finish.wait();
    offer_parked.join().unwrap();
    parked.close(Duration::from_millis(400)).await;
    free.close(Duration::from_millis(400)).await;
    assert_eq!(parked.stats().dropped, 0, "{:?}", parked.stats());
    assert_eq!(free.stats().dropped, 0, "{:?}", free.stats());
}
