//! `turbo-kserve health` against the server in-process with
//! testdata/tiny-bert-bundle: NOT_SERVING while it loads, SERVING once
//! ready, a model by name, and an unknown name refused.

#[path = "kserve/mod.rs"]
mod kserve;

use std::time::Duration;

use kserve::*;
use turbo_kserve::probe::{self, Probe};
use turbo_kserve::proto::health::health_check_response::ServingStatus;

fn probe(s: &Server, service: &str) -> Probe {
    Probe { addr: s.local_addr(), service: service.into(), timeout: Duration::from_secs(5) }
}

#[tokio::test]
async fn the_health_probe_follows_load_and_drain() {
    let s =
        Server::start("127.0.0.1:0".parse().unwrap(), vec![model(&tiny(), 1)], MAX, Serve::default()).await.unwrap();
    assert_eq!(probe::check(&probe(&s, "")).await.unwrap(), ServingStatus::NotServing);
    s.load().await.unwrap();
    assert_eq!(probe::check(&probe(&s, "")).await.unwrap(), ServingStatus::Serving);
    assert_eq!(probe::check(&probe(&s, NAME)).await.unwrap(), ServingStatus::Serving);
    let e = probe::check(&probe(&s, "nope")).await.unwrap_err();
    assert!(e.starts_with("Check \"nope\": NotFound"), "{e}");
    s.drain();
    assert_eq!(probe::check(&probe(&s, "")).await.unwrap(), ServingStatus::NotServing);
    s.stop().await;
}
