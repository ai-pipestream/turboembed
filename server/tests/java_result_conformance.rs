//! The generated Java client against a Rust mTLS server and audit service.

#[allow(dead_code)]
#[path = "pii_release/certs.rs"]
mod certs;
mod common;
#[allow(dead_code)]
#[path = "pii_release/fixture.rs"]
mod fixture;

use std::process::Command;
use std::time::Duration;

use turbo_kserve::proto::audit::PiiAccessForm;

#[tokio::test]
#[ignore = "run by scripts/package-proto-stubs.sh with packaged Java stubs"]
async fn java_round_trips_results_with_acknowledged_token_release() {
    let classpath = std::env::var_os("TURBO_JAVA_CONFORMANCE_CP").expect("packaged Java client classpath");
    let files = fixture::Files::create("java");
    for (name, pem) in [
        ("ca.pem", certs::SERVER_CA_PEM),
        ("client.pem", certs::CLIENT_CERT_PEM),
        ("client.key", certs::CLIENT_KEY_PEM),
    ] {
        std::fs::write(files.dir.join(name), pem).unwrap();
    }

    let (audit, endpoint) = fixture::audit(fixture::Answer::Ack).await;
    let grant = format!("{}:token:email", certs::client_fingerprint());
    let release = fixture::release(&files, &[grant], endpoint, Duration::from_secs(5));
    let server = fixture::server(&files, release, fixture::open_policy()).await;
    let address = format!("localhost:{}", server.local_addr().port());
    let dir = files.dir.clone();
    let status = tokio::task::spawn_blocking(move || {
        Command::new("java")
            .current_dir(concat!(env!("CARGO_MANIFEST_DIR"), "/.."))
            .arg("-cp")
            .arg(classpath)
            .arg("ai.pipestream.turboembed.DocumentSmoke")
            .arg(address)
            .args([dir.join("ca.pem"), dir.join("client.pem"), dir.join("client.key")])
            .status()
            .expect("start Java conformance client")
    })
    .await
    .unwrap();
    assert!(status.success(), "Java conformance failed: {status}");

    let events = audit.events.lock().unwrap();
    assert_eq!(events.len(), 1, "only the explicit token request is audited");
    let event = events.values().next().unwrap();
    assert_eq!(event.service_fingerprint, certs::fingerprint_bytes());
    assert!(!event.mentions.is_empty());
    assert!(event.mentions.iter().all(|m| m.form == PiiAccessForm::Token as i32));
}
