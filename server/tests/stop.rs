//! Stopping the server while calls are open (#2332): a watch, which has no
//! end of its own, ends when the server stops, and a call its caller never
//! closes holds the stop no longer than the grace given.

mod common;

use std::time::{Duration, Instant};

use tonic::Code;
use turbo_kserve::STOP_GRACE;
use turbo_kserve::proto::search::WatchCollectionRequest;
use turbo_kserve::proto::search::turbo_search_service_client::TurboSearchServiceClient;

#[tokio::test(flavor = "multi_thread")]
async fn an_open_watch_ends_when_the_server_stops() {
    let (server, _analysis) = common::serve().await;
    let mut search = TurboSearchServiceClient::connect(format!("http://{}", server.local_addr())).await.unwrap();
    let watch = WatchCollectionRequest { collection_name: "watched".into(), min_oov_ratio: 0.0 };
    let mut first = search.watch_collection(watch.clone()).await.unwrap().into_inner();
    let mut second = search.watch_collection(watch).await.unwrap().into_inner();

    let began = Instant::now();
    let clean = server.stop_within(STOP_GRACE).await;
    let took = began.elapsed();
    assert!(clean, "every call ended by itself");
    assert!(took < STOP_GRACE, "the stop did not wait out the grace: {took:?}");

    for stream in [&mut first, &mut second] {
        let end = common::within("the watch ends", Duration::from_secs(10), stream.message()).await.unwrap_err();
        assert_eq!(end.code(), Code::Unavailable);
        assert_eq!(end.message(), "the server is stopping");
    }
}

#[tokio::test(flavor = "multi_thread")]
async fn a_call_its_caller_never_closes_is_cut_after_the_grace() {
    let (server, mut analysis) = common::serve().await;
    // An AnalyzeStream whose sending half stays open: the caller neither
    // sends a document nor closes, so the call never ends by itself.
    let held = common::open_stream(&mut analysis, 0, 1, 0).await;

    let grace = Duration::from_millis(300);
    let began = Instant::now();
    let clean = server.stop_within(grace).await;
    let took = began.elapsed();
    assert!(!clean, "the held call was cut");
    assert!(took >= grace, "the stop waited the grace: {took:?}");
    assert!(took < STOP_GRACE, "the stop ended soon after the grace: {took:?}");
    drop(held);
}

#[tokio::test(flavor = "multi_thread")]
async fn a_stop_with_nothing_open_is_clean() {
    let (server, analysis) = common::serve().await;
    drop(analysis);
    assert!(server.stop_within(STOP_GRACE).await);
}
