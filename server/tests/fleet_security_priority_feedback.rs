//! Integration tests for `#2346`: mTLS + constant-time bearer role separation,
//! two-lane `PRIORITY_INTERACTIVE` vs `PRIORITY_BULK` scheduling under load,
//! pre-execution deadline cancellation (`turbo_kserve_cancelled_before_exec_total`),
//! and deterministic `TCIDF001` corpus DF + `TCENT001` Voronoi centroid feedback.

mod common;

use std::time::Duration;

use tokio_stream::StreamExt;
use tokio_stream::wrappers::ReceiverStream;
use tonic::metadata::MetadataValue;
use tonic::transport::{Certificate, Channel, ClientTlsConfig, Endpoint, Identity};
use tonic::{Code, Request};
use turbo_analysis::CorpusIdf;
use turbo_analysis_embed::VoronoiCentroids;
use turbo_kserve::config::{DEFAULT_MAX_MESSAGE_BYTES as MAX, Device, ModelConfig};
use turbo_kserve::proto::ServerReadyRequest;
use turbo_kserve::proto::grpc_inference_service_client::GrpcInferenceServiceClient;
use turbo_kserve::proto::result::analysis_service_client::AnalysisServiceClient;
use turbo_kserve::proto::result::{AnalyzeQueryRequest, QueryDenseTargetProto};
use turbo_kserve::proto::vocabulary::turbo_vocabulary_service_client::TurboVocabularyServiceClient;
use turbo_kserve::proto::vocabulary::*;
use turbo_kserve::{AuthConfig, PRIORITY_HEADER, Serve, Server, TlsConfig};

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

fn scratch(tag: &str) -> turbo_scratch::Scratch {
    let dir = std::env::temp_dir().join(format!("turbo-2346-{tag}-{}", std::process::id()));
    let s = turbo_scratch::Scratch::at(dir);
    std::fs::create_dir_all(s.path()).unwrap();
    s
}

fn loopback() -> std::net::SocketAddr {
    "127.0.0.1:0".parse().unwrap()
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

fn with_bearer<T>(mut req: Request<T>, token: &str) -> Request<T> {
    let val = MetadataValue::try_from(format!("Bearer {token}")).unwrap();
    req.metadata_mut().insert("authorization", val);
    req
}

#[tokio::test]
async fn mtls_and_bearer_auth_separate_infer_from_admin_principals() {
    let tmp = scratch("mtls-auth");
    let (cert, key, ca, vocab_dir) =
        (tmp.join("server.pem"), tmp.join("server.key"), tmp.join("client_ca.pem"), tmp.join("vocab"));
    std::fs::write(&cert, SERVER_CERT_PEM).unwrap();
    std::fs::write(&key, SERVER_KEY_PEM).unwrap();
    std::fs::write(&ca, CLIENT_CA_PEM).unwrap();

    let auth = AuthConfig::from_tokens(["infer-secret-alpha"], ["admin-secret-omega"]).unwrap();
    let serve = Serve {
        admin: true,
        vocabulary: Some(vocab_dir),
        tls: Some(TlsConfig { cert, key, client_ca: Some(ca) }),
        auth: Some(auth),
        ..Serve::default()
    };
    let server = Server::start(loopback(), vec![model_cfg(1)], MAX, serve).await.unwrap();
    server.load().await.unwrap();
    let addr = server.local_addr();

    // 1. Missing client certificate fails mTLS.
    let no_cert_tls =
        ClientTlsConfig::new().domain_name("localhost").ca_certificate(Certificate::from_pem(SERVER_CA_PEM));
    let no_cert_ep =
        Endpoint::from_shared(format!("https://localhost:{}", addr.port())).unwrap().tls_config(no_cert_tls).unwrap();
    if let Ok(ch) = no_cert_ep.connect().await {
        let mut c = GrpcInferenceServiceClient::new(ch);
        assert!(c.server_ready(ServerReadyRequest {}).await.is_err());
    }

    // 2. Valid mTLS client certificate.
    let mtls = ClientTlsConfig::new()
        .domain_name("localhost")
        .ca_certificate(Certificate::from_pem(SERVER_CA_PEM))
        .identity(Identity::from_pem(CLIENT_CERT_PEM, CLIENT_KEY_PEM));
    let channel: Channel = Endpoint::from_shared(format!("https://localhost:{}", addr.port()))
        .unwrap()
        .tls_config(mtls)
        .unwrap()
        .connect()
        .await
        .unwrap();

    let mut kserve = GrpcInferenceServiceClient::new(channel.clone());
    let mut vocab = TurboVocabularyServiceClient::new(channel);

    // Missing token -> UNAUTHENTICATED.
    let err = kserve.server_ready(ServerReadyRequest {}).await.unwrap_err();
    assert_eq!(err.code(), Code::Unauthenticated);

    // Wrong token -> UNAUTHENTICATED.
    let err = kserve.server_ready(with_bearer(Request::new(ServerReadyRequest {}), "wrong-token")).await.unwrap_err();
    assert_eq!(err.code(), Code::Unauthenticated);

    // Infer token -> read/infer succeeds, admin state-mutating RPC gets PERMISSION_DENIED.
    let ok = kserve.server_ready(with_bearer(Request::new(ServerReadyRequest {}), "infer-secret-alpha")).await.unwrap();
    assert!(ok.into_inner().ready);

    let stats_frames = vec![ImportCorpusStatisticsRequest {
        frame: Some(import_corpus_statistics_request::Frame::Header(CorpusStatisticsHeader {
            language: "en".into(),
            doc_count: 100,
            name: "shard-0".into(),
        })),
    }];
    let denied = vocab
        .import_corpus_statistics(with_bearer(
            Request::new(tokio_stream::iter(stats_frames.clone())),
            "infer-secret-alpha",
        ))
        .await
        .unwrap_err();
    assert_eq!(denied.code(), Code::PermissionDenied);

    // Admin token -> both admin write and read RPCs succeed.
    let admin_frames = vec![
        stats_frames[0].clone(),
        ImportCorpusStatisticsRequest {
            frame: Some(import_corpus_statistics_request::Frame::Batch(CorpusTermFrequencyBatch {
                entries: vec![CorpusTermFrequencyEntry { term: "quasar".into(), doc_freq: 2 }],
            })),
        },
    ];
    let created = vocab
        .import_corpus_statistics(with_bearer(Request::new(tokio_stream::iter(admin_frames)), "admin-secret-omega"))
        .await
        .unwrap()
        .into_inner();
    assert_eq!(created.term_count, 1);

    server.stop().await;
}

async fn wait_queue_depths(server: &Server, model: &str, want: (usize, usize)) {
    let deadline = tokio::time::Instant::now() + Duration::from_secs(5);
    loop {
        if server.priority_queue_depths(model).unwrap() == want {
            return;
        }
        assert!(tokio::time::Instant::now() < deadline, "timed out waiting for queue depths {want:?}");
        tokio::time::sleep(Duration::from_millis(1)).await;
    }
}

fn spawn_lane_infer(
    endpoint: &str,
    lane: &'static str,
    order: &std::sync::Arc<std::sync::Mutex<Vec<&'static str>>>,
) -> tokio::task::JoinHandle<()> {
    let ep = endpoint.to_owned();
    let ord = std::sync::Arc::clone(order);
    tokio::spawn(async move {
        let mut ks = GrpcInferenceServiceClient::connect(ep).await.unwrap();
        let mut req = Request::new(texts_req(NAME, &[b"lane-check"]));
        req.metadata_mut().insert(PRIORITY_HEADER, MetadataValue::from_static(lane));
        ks.model_infer(req).await.unwrap();
        ord.lock().unwrap().push(lane);
    })
}

#[tokio::test]
async fn priority_interactive_stays_flat_under_saturated_bulk_stream() {
    let server = Server::start(loopback(), vec![model_cfg(1)], MAX, Serve::default()).await.unwrap();
    server.load().await.unwrap();
    let endpoint = format!("http://{}", server.local_addr());

    let mut query_client = AnalysisServiceClient::connect(endpoint.clone()).await.unwrap();
    let make_query = |i: usize| AnalyzeQueryRequest {
        query: format!("distributed vector search latency probe {i} under $50"),
        dense_targets: vec![QueryDenseTargetProto { model: NAME.into(), ..Default::default() }],
        ..Default::default()
    };

    // 1. Deterministic head-of-line preemption: while the single session is held,
    //    enqueue 3 bulk ModelInfer calls, wait until all 3 are queued, then enqueue
    //    1 interactive call and wait until it is queued before releasing `held`.
    let held = server.hold(NAME).unwrap();
    let order = std::sync::Arc::new(std::sync::Mutex::new(Vec::<&'static str>::new()));
    let mut waiters: Vec<_> = (0..3).map(|_| spawn_lane_infer(&endpoint, "bulk", &order)).collect();
    wait_queue_depths(&server, NAME, (0, 3)).await;
    waiters.push(spawn_lane_infer(&endpoint, "interactive", &order));
    wait_queue_depths(&server, NAME, (1, 3)).await;
    drop(held);
    for w in waiters {
        w.await.unwrap();
    }
    assert_eq!(order.lock().unwrap()[0], "interactive", "interactive lane must preempt queued bulk work");

    // 2. Deterministic starvation bound: enqueue MAX_INTERACTIVE_BURST + 2 interactive
    //    calls and 1 bulk call while the session is held. Bulk must be granted
    //    within MAX_INTERACTIVE_BURST grants rather than starving.
    let burst = turbo_kserve::MAX_INTERACTIVE_BURST as usize;
    let held = server.hold(NAME).unwrap();
    let burst_order = std::sync::Arc::new(std::sync::Mutex::new(Vec::<&'static str>::new()));
    let mut burst_waiters: Vec<_> =
        (0..(burst + 2)).map(|_| spawn_lane_infer(&endpoint, "interactive", &burst_order)).collect();
    burst_waiters.push(spawn_lane_infer(&endpoint, "bulk", &burst_order));
    wait_queue_depths(&server, NAME, (burst + 2, 1)).await;
    drop(held);
    for w in burst_waiters {
        w.await.unwrap();
    }
    let bulk_pos = burst_order.lock().unwrap().iter().position(|&l| l == "bulk").unwrap();
    assert!(bulk_pos <= burst, "bulk starved: granted at index {bulk_pos} (limit {burst})");

    // 3. Concurrent AnalyzeStream + AnalyzeQuery: both complete cleanly under saturation.
    query_client.analyze_query(make_query(0)).await.unwrap();
    let stream_ep = endpoint.clone();
    let bulk_handle = tokio::spawn(async move {
        let mut bulk_client = AnalysisServiceClient::connect(stream_ep).await.unwrap();
        let (tx, rx) = tokio::sync::mpsc::channel(32);
        tx.send(stream_header(1, 16, 1)).await.unwrap();
        let writer = tokio::spawn(async move {
            for d in 0..80 {
                let text = format!("background ingest document {d} with lexical and neural content ").repeat(8);
                tx.send(stream_doc(&format!("doc-{d}"), NAME, &text, None)).await.unwrap();
            }
        });
        let mut out = bulk_client.analyze_stream(ReceiverStream::new(rx)).await.unwrap().into_inner();
        let mut count = 0usize;
        while let Some(item) = out.next().await {
            assert!(item.unwrap().outcome.is_some());
            count += 1;
        }
        writer.await.unwrap();
        count
    });

    for i in 100..116 {
        let resp = query_client.analyze_query(make_query(i)).await.unwrap().into_inner();
        assert_eq!(resp.dense_results.len(), 1);
    }

    let bulk_done = bulk_handle.await.unwrap();
    assert_eq!(bulk_done, 80);

    server.stop().await;
}

async fn wire_priority_deadline_status(server: &Server, timeout_hdr: &str) -> String {
    use prost::Message as _;
    let socket = tokio::net::TcpStream::connect(server.local_addr()).await.unwrap();
    let (mut sender, conn) = h2::client::handshake(socket).await.unwrap();
    let conn = tokio::spawn(conn);
    let request = tonic::codegen::http::Request::builder()
        .method("POST")
        .uri(format!("http://{}/inference.GRPCInferenceService/ModelInfer", server.local_addr()))
        .header("content-type", "application/grpc")
        .header("te", "trailers")
        .header(PRIORITY_HEADER, "interactive")
        .header("grpc-timeout", timeout_hdr)
        .body(())
        .unwrap();
    let encoded = texts_req(NAME, &[b"queued-deadline"]).encode_to_vec();
    let mut frame = vec![0];
    frame.extend_from_slice(&u32::try_from(encoded.len()).unwrap().to_be_bytes());
    frame.extend_from_slice(&encoded);
    let (response, mut body) = sender.send_request(request, false).unwrap();
    body.send_data(frame.into(), true).unwrap();
    let resp = tokio::time::timeout(Duration::from_secs(2), response).await.unwrap().unwrap();
    let mut status = resp.headers().get("grpc-status").map(|v| v.to_str().unwrap().to_owned());
    let mut resp_body = resp.into_body();
    if status.is_none()
        && let Ok(Some(trailers)) = resp_body.trailers().await
    {
        status = trailers.get("grpc-status").map(|v| v.to_str().unwrap().to_owned());
    }
    conn.abort();
    status.expect("server returned grpc-status")
}

#[tokio::test]
async fn expired_deadlines_skip_device_execution_and_increment_metric() {
    let server = Server::start(loopback(), vec![model_cfg(1)], MAX, Serve::default()).await.unwrap();
    server.load().await.unwrap();
    let endpoint = format!("http://{}", server.local_addr());

    let held = server.hold(NAME).unwrap();
    let runs_before = server.stream_stats().runs;

    // 1. Queued wire requests whose grpc-timeout expires while waiting for the session
    //    return DEADLINE_EXCEEDED (grpc-status = 4) without running on the device.
    for _ in 0..2 {
        assert_eq!(wire_priority_deadline_status(&server, "10m").await, "4");
    }

    // 2. Queued interactive calls whose client cancels the stream mid-wait also wake
    //    their SessionPool ticket without running on the device.
    let mut kserve = GrpcInferenceServiceClient::connect(endpoint).await.unwrap();
    for i in 0..2 {
        let mut req = Request::new(texts_req(NAME, &[format!("cancel-{i}").as_bytes()]));
        req.metadata_mut().insert(PRIORITY_HEADER, MetadataValue::from_static("interactive"));
        assert!(tokio::time::timeout(Duration::from_millis(10), kserve.model_infer(req)).await.is_err());
    }

    wait_for("cancelled_before_exec counter reaches 4", Duration::from_millis(500), || {
        server.metrics().contains("turbo_kserve_cancelled_before_exec_total 4")
    })
    .await;
    assert_eq!(server.stream_stats().runs, runs_before, "expired/cancelled queued requests must not run on device");

    drop(held);
    let mut ok_req = Request::new(texts_req(NAME, &[b"after-release"]));
    ok_req.metadata_mut().insert(PRIORITY_HEADER, MetadataValue::from_static("interactive"));
    let res = kserve.model_infer(ok_req).await.unwrap().into_inner();
    assert_eq!(res.outputs.len(), 1);

    server.stop().await;
}

async fn download_artifact(vocab: &mut TurboVocabularyServiceClient<Channel>, artifact_id: String) -> Vec<u8> {
    let mut dl = vocab.download_vocabulary(DownloadVocabularyRequest { artifact_id }).await.unwrap().into_inner();
    let mut out = Vec::new();
    while let Some(chunk) = dl.next().await {
        if let Some(vocabulary_artifact_chunk::Frame::Data(b)) = chunk.unwrap().frame {
            out.extend_from_slice(&b);
        }
    }
    out
}

#[tokio::test]
async fn corpus_df_and_voronoi_centroids_round_trip_deterministically() {
    let tmp = scratch("feedback-store");
    let serve = Serve { admin: true, vocabulary: Some(tmp.path().to_path_buf()), ..Serve::default() };
    let server = Server::start(loopback(), vec![model_cfg(1)], MAX, serve).await.unwrap();
    server.load().await.unwrap();
    let endpoint = format!("http://{}", server.local_addr());

    let mut vocab = TurboVocabularyServiceClient::connect(endpoint).await.unwrap();

    // 1. Corpus DF statistics (TCIDF001) upload, listing, and download round-trip.
    let df_frames = vec![
        ImportCorpusStatisticsRequest {
            frame: Some(import_corpus_statistics_request::Frame::Header(CorpusStatisticsHeader {
                language: "en".into(),
                doc_count: 50_000,
                name: "collection-v1".into(),
            })),
        },
        ImportCorpusStatisticsRequest {
            frame: Some(import_corpus_statistics_request::Frame::Batch(CorpusTermFrequencyBatch {
                entries: vec![
                    CorpusTermFrequencyEntry { term: "the".into(), doc_freq: 49_000 },
                    CorpusTermFrequencyEntry { term: "magnetar".into(), doc_freq: 5 },
                ],
            })),
        },
    ];
    let d1 = vocab.import_corpus_statistics(tokio_stream::iter(df_frames.clone())).await.unwrap().into_inner();
    let d2 = vocab.import_corpus_statistics(tokio_stream::iter(df_frames)).await.unwrap().into_inner();
    assert_eq!(d1, d2);

    let listed_df = vocab.list_corpus_statistics(ListCorpusStatisticsRequest {}).await.unwrap().into_inner();
    assert_eq!(listed_df.statistics, vec![d1.clone()]);

    let cidf_bytes = download_artifact(&mut vocab, d1.artifact_id.clone()).await;
    let decoded_idf = CorpusIdf::from_bytes(&cidf_bytes).unwrap();
    assert_eq!(decoded_idf.doc_count(), 50_000);
    assert!(decoded_idf.idf("magnetar") > decoded_idf.idf("the") * 10.0);

    // 2. Voronoi centroids (TCENT001) training, listing, and download round-trip.
    let mut sample_vals = Vec::new();
    for i in 0..10 {
        let j = (i as f32) * 0.01;
        sample_vals.extend_from_slice(&[1.0, j, 0.0, 0.0]);
        sample_vals.extend_from_slice(&[0.0, 1.0, j, 0.0]);
        sample_vals.extend_from_slice(&[0.0, 0.0, 1.0, j]);
    }
    let vor_frames = vec![
        TrainVoronoiCentroidsRequest {
            frame: Some(train_voronoi_centroids_request::Frame::Header(TrainVoronoiCentroidsHeader {
                model_id: "tiny-bert".into(),
                model_revision: "r1".into(),
                model_sha256: "deadbeef".into(),
                dim: 4,
                num_centroids: 3,
                max_iters: 25,
                seed: 11,
                tag_set_name: "shards".into(),
                temperature: 0.1,
            })),
        },
        TrainVoronoiCentroidsRequest {
            frame: Some(train_voronoi_centroids_request::Frame::Batch(VoronoiSampleBatch { values: sample_vals })),
        },
    ];
    let v1 = vocab.train_voronoi_centroids(tokio_stream::iter(vor_frames.clone())).await.unwrap().into_inner();
    let v2 = vocab.train_voronoi_centroids(tokio_stream::iter(vor_frames)).await.unwrap().into_inner();
    assert_eq!(v1, v2);
    assert!(v1.mean_cosine > 0.99);

    let listed_v = vocab.list_voronoi_centroids(ListVoronoiCentroidsRequest {}).await.unwrap().into_inner();
    assert_eq!(listed_v.centroids, vec![v1.clone()]);

    let tcent_bytes = download_artifact(&mut vocab, v1.artifact_id.clone()).await;
    let vor = VoronoiCentroids::from_bytes(&tcent_bytes).unwrap();
    let probes = vor.route_top_probes(&[0.0, 0.99, 0.01, 0.0], 2).unwrap();
    assert_eq!(probes.len(), 2);
    assert!(probes[0].1 > 0.99);

    server.stop().await;
}
