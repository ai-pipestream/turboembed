//! Stopping the demo while calls are open (#2332): an open stream ends as
//! `unavailable` when the demo stops, and a call its caller never finishes
//! holds the stop no longer than the grace given.

mod common;

use std::time::{Duration, Instant};

use common::*;
use tokio::io::AsyncWriteExt;
use tokio::net::TcpStream;
use turbo_kserve::Serve;
use turbo_kserve::proto::search::{CollectionEvent, WatchCollectionRequest};
use turbo_kserve_demo::STOP_GRACE;

const NAME: &str = "tiny-bert-bundle";

#[tokio::test(flavor = "multi_thread")]
async fn an_open_watch_ends_when_the_demo_stops() {
    let s = server(&[NAME], Serve::default()).await;
    let d = demo(&s, &[]).await;
    let path = "/turbo.search.v1.TurboSearchService/WatchCollection";
    // A watch on a collection nothing writes to: no message ever comes.
    let watch = stream(&d, path, &WatchCollectionRequest { collection_name: "quiet".into(), min_oov_ratio: 0.0 }).await;

    let began = Instant::now();
    let clean = d.stop_within(STOP_GRACE).await;
    let took = began.elapsed();
    assert!(clean, "every call ended by itself");
    assert!(took < STOP_GRACE, "the stop did not wait out the grace: {took:?}");

    let (events, end) =
        tokio::time::timeout(Duration::from_secs(10), watch.frames::<CollectionEvent>()).await.expect("the watch ends");
    assert!(events.is_empty());
    assert_eq!(end, r#"{"error":{"code":"unavailable","message":"the demo is stopping"}}"#);
    // The demo dropped its call, so the server's stop has nothing to wait for.
    assert!(s.stop_within(turbo_kserve::STOP_GRACE).await);
}

#[tokio::test(flavor = "multi_thread")]
async fn an_open_watch_ends_through_the_demo_when_the_server_stops() {
    let s = server(&[NAME], Serve::default()).await;
    let d = demo(&s, &[]).await;
    let path = "/turbo.search.v1.TurboSearchService/WatchCollection";
    let watch = stream(&d, path, &WatchCollectionRequest { collection_name: "quiet".into(), min_oov_ratio: 0.0 }).await;
    // The watch is open at the server before it stops.
    tokio::time::sleep(Duration::from_millis(200)).await;

    assert!(s.stop_within(turbo_kserve::STOP_GRACE).await, "the server's stop ended the watch");
    let (events, end) =
        tokio::time::timeout(Duration::from_secs(10), watch.frames::<CollectionEvent>()).await.expect("the watch ends");
    assert!(events.is_empty());
    assert_eq!(end, r#"{"error":{"code":"unavailable","message":"the server is stopping"}}"#);
    assert!(d.stop_within(STOP_GRACE).await);
}

#[tokio::test(flavor = "multi_thread")]
async fn a_request_its_caller_never_finishes_is_cut_after_the_grace() {
    let s = server(&[NAME], Serve::default()).await;
    let d = demo(&s, &[]).await;
    // A call that promises 64 bytes of body and sends 3: the demo waits
    // for the rest, which never comes.
    let mut conn = TcpStream::connect(d.local_addr()).await.unwrap();
    let head = "POST /turbo.result.v1.AnalysisService/Capabilities HTTP/1.1\r\nHost: demo\r\n\
                Content-Type: application/proto\r\nContent-Length: 64\r\n\r\nabc";
    conn.write_all(head.as_bytes()).await.unwrap();
    conn.flush().await.unwrap();
    // The request has reached the demo before it stops.
    tokio::time::sleep(Duration::from_millis(200)).await;

    let grace = Duration::from_millis(300);
    let began = Instant::now();
    let clean = d.stop_within(grace).await;
    let took = began.elapsed();
    assert!(!clean, "the held call was cut");
    assert!(took >= grace, "the stop waited the grace: {took:?}");
    assert!(took < STOP_GRACE, "the stop ended soon after the grace: {took:?}");
    drop(conn);
    assert!(s.stop_within(turbo_kserve::STOP_GRACE).await);
}
