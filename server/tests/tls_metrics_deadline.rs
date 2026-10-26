//! Integration tests for issue #1300: TLS and mTLS handshake enforcement,
//! Prometheus `/metrics` exposition, and gRPC `grpc-timeout` deadline
//! propagation across unary and streaming RPCs without leaking sessions.

mod common;

use std::path::PathBuf;
use std::time::Duration;

use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::TcpStream;
use tokio_stream::wrappers::ReceiverStream;
use tonic::metadata::MetadataValue;
use tonic::transport::{Certificate, Channel, ClientTlsConfig, Endpoint, Identity};
use tonic::{Code, Request};
use turbo_kserve::config::{DEFAULT_MAX_MESSAGE_BYTES as MAX, Device, ModelConfig};
use turbo_kserve::proto::ServerReadyRequest;
use turbo_kserve::proto::grpc_inference_service_client::GrpcInferenceServiceClient;
use turbo_kserve::proto::result::analysis_service_client::AnalysisServiceClient;
use turbo_kserve::proto::result::{
    AnalyzePipelineRequest, CapabilitiesRequest, LoadBundleRequest, UnloadBundleRequest,
};
use turbo_kserve::tls::TlsConfig;
use turbo_kserve::{Serve, Server};

use common::*;

const SERVER_CA_PEM: &str = "-----BEGIN CERTIFICATE-----\n\
MIIBhDCCASqgAwIBAgIUAVogSpXO+xbeQ4x9o1xRs9pbHo8wCgYIKoZIzj0EAwIw\n\
HzEdMBsGA1UEAwwUdHVyYm8tdGVzdC1zZXJ2ZXItY2EwIBcNMjYxMDI2MDY0NjM3\n\
WhgPMjEyNjEwMDIwNjQ2MzdaMB8xHTAbBgNVBAMMFHR1cmJvLXRlc3Qtc2VydmVy\n\
LWNhMFkwEwYHKoZIzj0CAQYIKoZIzj0DAQcDQgAE9sFFiocjmxKXQ0kCtYZ7tmSY\n\
4hYeE1i66YfvRcbSiufRzGTY7h3xjboIymPNxoeEgvgk1231MfGd0vS3Xc7Qk6NC\n\
MEAwHQYDVR0OBBYEFOFpRbGCcyAKmJ0s5z3Ahii6GYDdMA8GA1UdEwEB/wQFMAMB\n\
Af8wDgYDVR0PAQH/BAQDAgEGMAoGCCqGSM49BAMCA0gAMEUCIQCYhj6qYO018b+k\n\
THo9t4rdisRMQVK5qXp1b3TTiqwu0wIgPFW9G8RUUEgj76ET9gx6m2PWA+7QmoHs\n\
czkEzHeFhYU=\n\
-----END CERTIFICATE-----\n";

const SERVER_CERT_PEM: &str = "-----BEGIN CERTIFICATE-----\n\
MIIBhDCCASugAwIBAgIUMDz+0icDFMibjkseSNPtVqwKB3MwCgYIKoZIzj0EAwIw\n\
HzEdMBsGA1UEAwwUdHVyYm8tdGVzdC1zZXJ2ZXItY2EwIBcNMjYxMDI2MDY0NjM3\n\
WhgPMjEyNjEwMDIwNjQ2MzdaMBQxEjAQBgNVBAMMCWxvY2FsaG9zdDBZMBMGByqG\n\
SM49AgEGCCqGSM49AwEHA0IABOi+ACPVzeNQULfW6uWG0DxJwUaESso6RcyeDKLF\n\
i2luj3jexu9SFUiZAXxeNuiYmD2NblgiY75pp/D2SNGD2BajTjBMMAkGA1UdEwQC\n\
MAAwDgYDVR0PAQH/BAQDAgWgMBMGA1UdJQQMMAoGCCsGAQUFBwMBMBoGA1UdEQQT\n\
MBGCCWxvY2FsaG9zdIcEfwAAATAKBggqhkjOPQQDAgNHADBEAiAyBIr/0nJz51E0\n\
0JHXUsmLmZKA3KgLL/mZu33rcPWbYwIgAgeirtnliflpsQbdl0I3BqLVt3Jo8PuU\n\
3eF5aNpH2eI=\n\
-----END CERTIFICATE-----\n";

const SERVER_KEY_PEM: &str = "-----BEGIN PRIVATE KEY-----\n\
MIGHAgEAMBMGByqGSM49AgEGCCqGSM49AwEHBG0wawIBAQQgXTzexqw3bQIvfaiE\n\
kQgrSussEkBOWa4hXiu6rEivRoahRANCAATovgAj1c3jUFC31urlhtA8ScFGhErK\n\
OkXMngyixYtpbo943sbvUhVImQF8XjbomJg9jW5YImO+aafw9kjRg9gW\n\
-----END PRIVATE KEY-----\n";

const CLIENT_CA_PEM: &str = "-----BEGIN CERTIFICATE-----\n\
MIIBhDCCASqgAwIBAgIUCwI4eaomEa2NQucISG+RmGkANgswCgYIKoZIzj0EAwIw\n\
HzEdMBsGA1UEAwwUdHVyYm8tdGVzdC1jbGllbnQtY2EwIBcNMjYxMDI2MDY0NjM3\n\
WhgPMjEyNjEwMDIwNjQ2MzdaMB8xHTAbBgNVBAMMFHR1cmJvLXRlc3QtY2xpZW50\n\
LWNhMFkwEwYHKoZIzj0CAQYIKoZIzj0DAQcDQgAE8jVI3Hds8oK1Yg/bFVmhD4on\n\
IicKOIljXpmgDbkWCrdQ+vGIWz+QGzaogoxPSEEgSJh2wiXDGjvFOnZEHhfL/qNC\n\
MEAwHQYDVR0OBBYEFBrce0+O9No7TUNZYFQdhqUK2y16MA8GA1UdEwEB/wQFMAMB\n\
Af8wDgYDVR0PAQH/BAQDAgEGMAoGCCqGSM49BAMCA0gAMEUCIQDjkLhdXJhECicj\n\
CqsrrCAAVxBJs/fN0e1ltw/WdwCFewIgBJYooOLNq6lwZDv3cGk8x22fQIQCMmVK\n\
+48ct49bQE0=\n\
-----END CERTIFICATE-----\n";

const CLIENT_CERT_PEM: &str = "-----BEGIN CERTIFICATE-----\n\
MIIBcDCCARegAwIBAgIUMQ8BSBhmrVSddhRHRzewospjnBUwCgYIKoZIzj0EAwIw\n\
HzEdMBsGA1UEAwwUdHVyYm8tdGVzdC1jbGllbnQtY2EwIBcNMjYxMDI2MDY0NjM3\n\
WhgPMjEyNjEwMDIwNjQ2MzdaMBwxGjAYBgNVBAMMEXR1cmJvLXRlc3QtY2xpZW50\n\
MFkwEwYHKoZIzj0CAQYIKoZIzj0DAQcDQgAE4hKHzD/yw8WF02D3olK6XqebknlB\n\
IweseYnw8uJQ42Lureyzoewzgu/abKz2w48EWVasOawgDU92BIetnHODX6MyMDAw\n\
CQYDVR0TBAIwADAOBgNVHQ8BAf8EBAMCB4AwEwYDVR0lBAwwCgYIKwYBBQUHAwIw\n\
CgYIKoZIzj0EAwIDRwAwRAIgIQPpzjXnFD9yNfeudzAerg4a7ec2T1jCANbPtUDw\n\
MKkCIFas6p9vhLm1k80xLmLuJtbOEJFhHhhEaSnv4KGmlTy1\n\
-----END CERTIFICATE-----\n";

const CLIENT_KEY_PEM: &str = "-----BEGIN PRIVATE KEY-----\n\
MIGHAgEAMBMGByqGSM49AgEGCCqGSM49AwEHBG0wawIBAQQgJi4WvkLcjdwN4Jlb\n\
Os9NOaLaxBg5BVnrCfvk/9S66d+hRANCAATiEofMP/LDxYXTYPeiUrpep5uSeUEj\n\
B6x5ifDy4lDjYu6t7LOh7DOC79psrPbDjwRZVqw5rCANT3YEh62cc4Nf\n\
-----END PRIVATE KEY-----\n";

static TLS_SERIAL: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);

struct CertDir {
    dir: PathBuf,
    cert: PathBuf,
    key: PathBuf,
    client_ca: PathBuf,
}

impl CertDir {
    fn create() -> Self {
        let id =
            format!("tls-{}-{}", std::process::id(), TLS_SERIAL.fetch_add(1, std::sync::atomic::Ordering::Relaxed));
        let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).parent().unwrap();
        let dir = root.join("target/server-tls-tests").join(id);
        std::fs::create_dir_all(&dir).unwrap();
        let cert = dir.join("server.pem");
        let key = dir.join("server.key");
        let client_ca = dir.join("client_ca.pem");
        std::fs::write(&cert, SERVER_CERT_PEM).unwrap();
        std::fs::write(&key, SERVER_KEY_PEM).unwrap();
        std::fs::write(&client_ca, CLIENT_CA_PEM).unwrap();
        Self { dir, cert, key, client_ca }
    }
}

impl Drop for CertDir {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.dir);
    }
}

fn model_cfg(sessions: u32) -> ModelConfig {
    ModelConfig {
        bundle: tiny().to_str().unwrap().to_string(),
        name: None,
        device: Device::Index(cpu()),
        precision: 0,
        max_batch: 0,
        max_seq: 0,
        sessions,
    }
}

async fn connect_tls(
    addr: std::net::SocketAddr,
    with_client_identity: bool,
) -> Result<Channel, tonic::transport::Error> {
    let mut tls = ClientTlsConfig::new().domain_name("localhost").ca_certificate(Certificate::from_pem(SERVER_CA_PEM));
    if with_client_identity {
        tls = tls.identity(Identity::from_pem(CLIENT_CERT_PEM, CLIENT_KEY_PEM));
    }
    Endpoint::from_shared(format!("https://localhost:{}", addr.port()))
        .unwrap()
        .tls_config(tls)
        .unwrap()
        .connect()
        .await
}

#[tokio::test]
async fn tls_and_mtls_handshake_and_client_certificate_enforcement() {
    let certs = CertDir::create();

    // 1. Server TLS without client CA: client with Server CA succeeds.
    let s_tls = Server::start(
        "127.0.0.1:0".parse().unwrap(),
        vec![model_cfg(1)],
        MAX,
        Serve {
            tls: Some(TlsConfig { cert: certs.cert.clone(), key: certs.key.clone(), client_ca: None }),
            ..Serve::default()
        },
    )
    .await
    .unwrap();
    s_tls.load().await.unwrap();

    let ch = connect_tls(s_tls.local_addr(), false).await.expect("TLS handshake succeeds");
    let mut k = GrpcInferenceServiceClient::new(ch);
    assert!(k.server_ready(ServerReadyRequest {}).await.unwrap().into_inner().ready);
    assert_eq!(embed_shape(&mut k, NAME, &[b"hello over tls"]).await, vec![1, 32]);
    s_tls.stop().await;

    // 2. Mutual TLS with `--tls-client-ca`: client without certificate is rejected,
    //    client with valid client certificate succeeds.
    let s_mtls = Server::start(
        "127.0.0.1:0".parse().unwrap(),
        vec![model_cfg(1)],
        MAX,
        Serve {
            tls: Some(TlsConfig {
                cert: certs.cert.clone(),
                key: certs.key.clone(),
                client_ca: Some(certs.client_ca.clone()),
            }),
            ..Serve::default()
        },
    )
    .await
    .unwrap();
    s_mtls.load().await.unwrap();

    let no_cert_failed = match connect_tls(s_mtls.local_addr(), false).await {
        Err(_) => true,
        Ok(ch) => {
            let mut k = GrpcInferenceServiceClient::new(ch);
            k.server_ready(ServerReadyRequest {}).await.is_err()
        }
    };
    assert!(no_cert_failed, "mTLS server must reject a client that presents no certificate");

    let ch_mtls = connect_tls(s_mtls.local_addr(), true).await.expect("mTLS handshake with client cert succeeds");
    let mut k_mtls = GrpcInferenceServiceClient::new(ch_mtls);
    assert!(k_mtls.server_ready(ServerReadyRequest {}).await.unwrap().into_inner().ready);
    assert_eq!(embed_shape(&mut k_mtls, NAME, &[b"hello over mtls"]).await, vec![1, 32]);
    s_mtls.stop().await;
}

async fn scrape_http(addr: std::net::SocketAddr, path: &str) -> String {
    let mut stream = TcpStream::connect(addr).await.unwrap();
    let req = format!("GET {path} HTTP/1.1\r\nHost: localhost\r\nConnection: close\r\n\r\n");
    stream.write_all(req.as_bytes()).await.unwrap();
    let mut buf = Vec::new();
    stream.read_to_end(&mut buf).await.unwrap();
    String::from_utf8(buf).unwrap()
}

#[tokio::test]
async fn prometheus_metrics_endpoint_exposes_all_server_metric_families() {
    let s = Server::start(
        "127.0.0.1:0".parse().unwrap(),
        vec![model_cfg(2)],
        MAX,
        Serve { admin: true, metrics_listen: Some("127.0.0.1:0".parse().unwrap()), ..Serve::default() },
    )
    .await
    .unwrap();
    s.load().await.unwrap();

    let mut k = kserve(&s).await;
    assert_eq!(embed_shape(&mut k, NAME, &[b"one", b"two"]).await, vec![2, 32]);

    let mut c = AnalysisServiceClient::connect(format!("http://{}", s.local_addr()))
        .await
        .unwrap()
        .max_decoding_message_size(MAX);
    let _ = c.capabilities(CapabilitiesRequest {}).await.unwrap();

    let mut st = open_stream(&mut c, 0, 2, 0).await;
    st.send("doc-0", NAME, "streaming metrics check", None).await;
    let answers = st.half_close().collect().await;
    assert_eq!(answers.len(), 1);

    c.load_bundle(LoadBundleRequest {
        name: "extra".into(),
        bundle: tiny().to_str().unwrap().to_string(),
        device: cpu(),
        sessions: 1,
        precision: 0,
        max_batch: 0,
        max_seq: 0,
        select_device: false,
    })
    .await
    .unwrap();
    c.unload_bundle(UnloadBundleRequest { name: "extra".into() }).await.unwrap();

    let metrics_addr = s.metrics_addr().expect("metrics listener bound");
    let raw = scrape_http(metrics_addr, "/metrics").await;
    assert!(raw.starts_with("HTTP/1.1 200 OK\r\n"), "unexpected status line: {raw}");
    assert!(
        raw.contains("Content-Type: text/plain; version=0.0.4; charset=utf-8\r\n"),
        "missing Prometheus content-type header: {raw}"
    );

    for expected in [
        "turbo_kserve_rpc_requests_total{service=\"inference.GRPCInferenceService\",method=\"ModelInfer\",code=\"OK\"} 1",
        "turbo_kserve_rpc_requests_total{service=\"turbo.result.v1.AnalysisService\",method=\"AnalyzeStream\",code=\"OK\"} 1",
        "turbo_kserve_rpc_duration_seconds_bucket{service=\"inference.GRPCInferenceService\",method=\"ModelInfer\",le=\"+Inf\"} 1",
        "turbo_kserve_inflight_streams 0",
        "turbo_kserve_inflight_documents 0",
        "turbo_kserve_queued_documents 0",
        "turbo_kserve_queued_bytes 0",
        "turbo_kserve_batch_size_count 2",
        "turbo_kserve_device_queue_depth{device=\"0\"} 0",
        "turbo_kserve_device_active_sessions{device=\"0\"} 0",
        "turbo_kserve_device_idle_sessions{device=\"0\"} 2",
        "turbo_kserve_bundle_loads_total{status=\"ok\"} 2",
        "turbo_kserve_bundle_unloads_total 1",
        "turbo_kserve_loaded_bundles 1",
    ] {
        assert!(raw.contains(expected), "missing expected metric line {expected:?} in:\n{raw}");
    }

    let not_found = scrape_http(metrics_addr, "/unknown").await;
    assert!(not_found.starts_with("HTTP/1.1 404 Not Found\r\n"), "unexpected 404 response: {not_found}");
    s.stop().await;
}

#[tokio::test]
async fn grpc_timeout_expires_on_unary_and_mid_stream_without_leaking_sessions() {
    let s = Server::start(
        "127.0.0.1:0".parse().unwrap(),
        vec![ModelConfig {
            bundle: tiny().to_str().unwrap().to_string(),
            name: None,
            device: Device::Index(cpu()),
            precision: 0,
            max_batch: 2,
            max_seq: 0,
            sessions: 1,
        }],
        MAX,
        Serve::default(),
    )
    .await
    .unwrap();
    s.load().await.unwrap();

    // 1. Unary ModelInfer with a 12-millisecond grpc-timeout returns DEADLINE_EXCEEDED
    //    and leaves the single session idle.
    let mut k = kserve(&s).await;
    assert!(k.server_ready(ServerReadyRequest {}).await.unwrap().into_inner().ready);
    let mut req = Request::new(texts_req(NAME, &[b"deadline test"]));
    req.metadata_mut().insert("grpc-timeout", MetadataValue::from_static("12m"));
    let err = k.model_infer(req).await.expect_err("12m timeout must expire");
    assert_eq!(err.code(), Code::DeadlineExceeded, "unary ModelInfer status: {err:?}");
    drop(s.hold(NAME).expect("unary deadline expiry must not leak the model session"));

    // 2. Streaming AnalyzeDocumentProgressive with a 12-millisecond grpc-timeout returns
    //    DEADLINE_EXCEEDED and leaves the single session idle.
    let mut c = AnalysisServiceClient::connect(format!("http://{}", s.local_addr()))
        .await
        .unwrap()
        .max_decoding_message_size(MAX);
    let mut prog_req = Request::new(AnalyzePipelineRequest {
        doc_id: "prog-deadline".into(),
        text: "hello world".into(),
        ..Default::default()
    });
    prog_req.metadata_mut().insert("grpc-timeout", MetadataValue::from_static("12m"));
    let prog_err = c.analyze_document_progressive(prog_req).await.expect_err("12m timeout must expire");
    assert_eq!(prog_err.code(), Code::DeadlineExceeded, "progressive status: {prog_err:?}");
    drop(s.hold(NAME).expect("progressive deadline expiry must not leak the model session"));

    // 3. Streaming AnalyzeStream with max_batch=2, sessions=1, and a deadline that
    //    expires mid-stream after the first full batch (doc-0, doc-1) executes while
    //    doc-2 is queued in the stream: returns DEADLINE_EXCEEDED, drains queued
    //    items to 0, and returns the single session to the idle pool.
    let (tx, rx) = tokio::sync::mpsc::channel(64);
    tx.send(stream_header(0, 16, 20)).await.unwrap();
    tx.send(stream_doc("doc-0", NAME, "stream deadline batch 1 a", None)).await.unwrap();
    tx.send(stream_doc("doc-1", NAME, "stream deadline batch 1 b", None)).await.unwrap();

    let mut stream_req = Request::new(ReceiverStream::new(rx));
    stream_req.metadata_mut().insert("grpc-timeout", MetadataValue::from_static("400m"));
    let mut responses = c.analyze_stream(stream_req).await.unwrap().into_inner();
    let mut answered = 0usize;
    let mut terminal_err = None;
    let mut held_after_batch = None;
    while let Some(item) = responses.message().await.transpose() {
        match item {
            Ok(_) => {
                answered += 1;
                if answered == 2 {
                    held_after_batch = Some(s.hold(NAME).expect("session idle after first batch"));
                    tx.send(stream_doc("doc-2", NAME, "stream deadline queued doc", None)).await.unwrap();
                }
            }
            Err(status) => {
                terminal_err = Some(status);
                break;
            }
        }
    }
    drop(tx);
    drop(held_after_batch);
    assert_eq!(answered, 2, "first full batch of 2 documents completed before deadline");
    let status = terminal_err.expect("AnalyzeStream must terminate with DEADLINE_EXCEEDED");
    assert_eq!(status.code(), Code::DeadlineExceeded, "stream status: {status:?}");

    tokio::time::timeout(Duration::from_secs(2), async {
        while s.stream_stats().queued_items > 0 || !s.metrics().contains("turbo_kserve_inflight_streams 0") {
            tokio::time::sleep(Duration::from_millis(5)).await;
        }
    })
    .await
    .expect("stream queue and inflight_streams must drain to 0 after deadline");

    assert_eq!(s.stream_stats().queued_items, 0);
    assert_eq!(s.stream_stats().queued_bytes, 0);
    let rendered = s.metrics();
    assert!(rendered.contains("turbo_kserve_inflight_streams 0"), "{rendered}");
    assert!(rendered.contains("turbo_kserve_inflight_documents 0"), "{rendered}");
    for expected in [
        "turbo_kserve_rpc_requests_total{service=\"inference.GRPCInferenceService\",method=\"ServerReady\",code=\"OK\"} 1",
        "turbo_kserve_rpc_requests_total{service=\"inference.GRPCInferenceService\",method=\"ModelInfer\",code=\"DEADLINE_EXCEEDED\"} 1",
        "turbo_kserve_rpc_requests_total{service=\"turbo.result.v1.AnalysisService\",method=\"AnalyzeDocumentProgressive\",code=\"DEADLINE_EXCEEDED\"} 1",
        "turbo_kserve_rpc_requests_total{service=\"turbo.result.v1.AnalysisService\",method=\"AnalyzeStream\",code=\"DEADLINE_EXCEEDED\"} 1",
    ] {
        assert!(rendered.contains(expected), "missing {expected:?} in:\n{rendered}");
    }
    assert!(
        !rendered.contains("method=\"ModelInfer\",code=\"OK\"")
            && !rendered.contains("method=\"AnalyzeDocumentProgressive\",code=\"OK\"")
            && !rendered.contains("method=\"AnalyzeStream\",code=\"OK\""),
        "expired RPCs must never record code=OK:\n{rendered}"
    );
    drop(s.hold(NAME).expect("AnalyzeStream deadline expiry must release the session"));

    // 4. Aborted AnalyzeStream (client drops the open stream after receiving one batch
    //    without half-closing, so no grpc-status trailer is ever emitted) records
    //    code="CANCELLED" rather than code="OK".
    let (abort_tx, abort_rx) = tokio::sync::mpsc::channel(16);
    abort_tx.send(stream_header(0, 1, 0)).await.unwrap();
    abort_tx.send(stream_doc("abort-0", NAME, "aborted stream doc", None)).await.unwrap();
    let mut abort_stream = c.analyze_stream(ReceiverStream::new(abort_rx)).await.unwrap().into_inner();
    let first = abort_stream.message().await.unwrap().expect("first batch item arrives");
    assert_eq!(first.id, "abort-0");
    drop(abort_stream);
    drop(abort_tx);

    tokio::time::timeout(Duration::from_secs(2), async {
        while !s.metrics().contains(
            "turbo_kserve_rpc_requests_total{service=\"turbo.result.v1.AnalysisService\",method=\"AnalyzeStream\",code=\"CANCELLED\"} 1",
        ) || !s.metrics().contains("turbo_kserve_inflight_streams 0")
        {
            tokio::time::sleep(Duration::from_millis(5)).await;
        }
    })
    .await
    .expect("aborted AnalyzeStream must record CANCELLED and drain inflight_streams to 0");

    let after_abort = s.metrics();
    assert!(
        !after_abort.contains("method=\"AnalyzeStream\",code=\"OK\""),
        "aborted stream must not record code=OK:\n{after_abort}"
    );
    drop(s.hold(NAME).expect("aborted AnalyzeStream must release the session"));

    s.stop().await;
}
