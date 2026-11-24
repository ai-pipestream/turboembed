//! A streaming call that answers every document with a failure is still
//! one OK request, so the request counter alone showed it as healthy
//! (issue #2517 F11). Each answered document is counted by outcome and
//! turbo status.

mod common;

use common::*;
use turbo_kserve::proto::result::AnalyzeOptions;

fn line(m: &str, want: &str) -> bool {
    m.lines().any(|l| l == want)
}

#[tokio::test]
async fn every_stream_answer_is_counted_by_its_outcome() {
    let (s, mut c) = serve().await;
    let mut stream = open_stream(&mut c, 0, 0, 0).await;
    // Multi-resolution options on a plain vectors request are refused
    // for that document alone.
    let refused = AnalyzeOptions { matryoshka_dims: vec![8], ..Default::default() };
    stream.send("ok-0", NAME, "a text that embeds", None).await;
    stream.send("ok-1", NAME, "another text that embeds", None).await;
    stream.send("bad", NAME, "a text with an option it cannot take", Some(refused)).await;
    let got = by_id(stream.half_close()).await;
    assert_eq!(got.len(), 3);
    let m = s.metrics();
    let ok = "turbo_kserve_documents_total{method=\"AnalyzeStream\",outcome=\"ok\",turbo_code=\"TURBO_OK\"} 2";
    let failed = "turbo_kserve_documents_total{method=\"AnalyzeStream\",outcome=\"failed\",turbo_code=\"TURBO_E_UNSUPPORTED_OPTION\"} 1";
    assert!(line(&m, ok) && line(&m, failed), "{m}");
    assert!(line(&m, "# TYPE turbo_kserve_documents_total counter"), "{m}");
    assert!(line(&m, "# TYPE turbo_kserve_steps_total counter"), "{m}");
    s.stop().await;
}
