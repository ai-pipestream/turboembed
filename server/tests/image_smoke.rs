//! The release image's smoke test (.github/workflows/release.yml, image):
//! a turbo-kserve already running elsewhere, at TURBO_KSERVE_SMOKE_ADDR,
//! with testdata/tiny-bert-bundle mounted as a bundle of that name, turns
//! ready and embeds the bundle's reference texts to its reference
//! vectors. Without the variable there is nothing to test and it passes.

#[path = "kserve/mod.rs"]
mod kserve;

use std::time::{Duration, Instant};

use kserve::*;
use turbo_kserve::proto::grpc_inference_service_client::GrpcInferenceServiceClient;

#[tokio::test]
async fn a_running_image_serves_the_tiny_bundle() {
    let Ok(addr) = std::env::var("TURBO_KSERVE_SMOKE_ADDR") else {
        eprintln!("TURBO_KSERVE_SMOKE_ADDR is not set; no image to check");
        return;
    };
    let deadline = Instant::now() + Duration::from_secs(120);
    let mut c = loop {
        match GrpcInferenceServiceClient::connect(format!("http://{addr}")).await {
            Ok(c) => break c,
            Err(e) if Instant::now() < deadline => {
                eprintln!("waiting for {addr}: {e}");
                tokio::time::sleep(Duration::from_secs(1)).await;
            }
            Err(e) => panic!("{addr}: {e}"),
        }
    };
    while !c.server_ready(ServerReadyRequest {}).await.map(|r| r.into_inner().ready).unwrap_or(false) {
        assert!(Instant::now() < deadline, "{addr} did not turn ready in 120 s");
        tokio::time::sleep(Duration::from_secs(1)).await;
    }
    let cases = reference();
    for role in ["PROMPT_NONE", "PROMPT_QUERY", "PROMPT_DOCUMENT"] {
        let group: Vec<&Case> = cases.iter().filter(|k| k.role == role).collect();
        assert!(!group.is_empty(), "{role}");
        let t: Vec<&str> = group.iter().map(|k| k.text.as_str()).collect();
        let mut r = texts(&t);
        if role != "PROMPT_NONE" {
            r = with(r, "prompt_role", st(role));
        }
        for (k, v) in group.iter().zip(embed(&mut c, r).await) {
            matches(&format!("{role} `{}`", k.text), &v, &k.vector);
        }
    }
}
