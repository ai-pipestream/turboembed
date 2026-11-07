//! An in-process audit service, an mTLS turbo-kserve with PII release, the
//! client that calls it, and a graph holding two planted values.

use std::collections::BTreeMap;
use std::net::SocketAddr;
use std::path::PathBuf;
use std::sync::{Arc, Mutex};
use std::time::Duration;

use tonic::transport::{Certificate, Channel, ClientTlsConfig, Endpoint, Identity};
use tonic::{Request, Response, Status};
use turbo_analysis::export::export;
use turbo_analysis::{Document, Key, Provenance, Span, ViewId};
use turbo_kserve::config::{DEFAULT_MAX_MESSAGE_BYTES as MAX, Device, ModelConfig};
use turbo_kserve::proto::audit::turbo_audit_service_server::{TurboAuditService, TurboAuditServiceServer};
use turbo_kserve::proto::audit::{PiiAccessEvent, RecordPiiAccessRequest, RecordPiiAccessResponse};
use turbo_kserve::proto::result::analysis_service_client::AnalysisServiceClient;
use turbo_kserve::release::ReleaseConfig;
use turbo_kserve::release::caps::{Form, Forms, Grants};
use turbo_kserve::release::keys::{KeyRef, TokenKeyFile};
use turbo_kserve::results::disclosure::PiiOutputPolicy;
use turbo_kserve::results::document::codecs;
use turbo_kserve::tls::TlsConfig;
use turbo_kserve::{Serve, Server};
use turbo_pii::{Pii, PiiEvidence, PiiKind, PiiMention};

use super::certs::*;
use crate::common;

/// The planted values: an address and a phone number, normalized as given.
pub const EMAIL: &str = "alice.planted@example.com";
pub const PHONE: &str = "15550100777";
pub const TEXT: &str = "Mail alice.planted@example.com or call 15550100777 today.";
/// What the token key file holds; it must never reach any output.
pub const KEY_BYTES: &[u8] = b"planted-token-key-bytes-0123456789abcdef";

/// How the audit service answers.
#[derive(Copy, Clone, PartialEq, Eq)]
pub enum Answer {
    Ack,
    Fail,
    WrongId,
    Slow,
}

/// The audit service: every recorded event by id, and every call.
#[derive(Clone)]
pub struct Audit {
    pub events: Arc<Mutex<BTreeMap<String, PiiAccessEvent>>>,
    pub calls: Arc<Mutex<Vec<Vec<u8>>>>,
    answer: Answer,
}

#[tonic::async_trait]
impl TurboAuditService for Audit {
    async fn record_pii_access(
        &self,
        req: Request<RecordPiiAccessRequest>,
    ) -> Result<Response<RecordPiiAccessResponse>, Status> {
        let req = req.into_inner();
        self.calls.lock().unwrap().push(prost::Message::encode_to_vec(&req));
        let event = req.event.ok_or_else(|| Status::invalid_argument("no event"))?;
        let id = event.event_id.clone();
        match self.answer {
            Answer::Fail => return Err(Status::unavailable("audit store down")),
            Answer::WrongId => return Ok(Response::new(RecordPiiAccessResponse { event_id: "other".into() })),
            Answer::Slow => tokio::time::sleep(Duration::from_secs(5)).await,
            Answer::Ack => {}
        }
        self.events.lock().unwrap().entry(id.clone()).or_insert(event);
        Ok(Response::new(RecordPiiAccessResponse { event_id: id }))
    }
}

/// Starts an audit service answering `answer`, and its address.
pub async fn audit(answer: Answer) -> (Audit, SocketAddr) {
    let audit = Audit { events: Arc::default(), calls: Arc::default(), answer };
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    let svc = TurboAuditServiceServer::new(audit.clone());
    tokio::spawn(async move {
        let incoming = tonic::transport::server::TcpIncoming::from(listener);
        tonic::transport::Server::builder().add_service(svc).serve_with_incoming(incoming).await.unwrap();
    });
    (audit, addr)
}

/// A scratch directory holding the TLS files and the token key.
pub struct Files {
    pub dir: PathBuf,
    pub tls: TlsConfig,
    pub key: PathBuf,
}

impl Files {
    pub fn create(tag: &str) -> Files {
        let dir = PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join(format!("pii-release-{}-{tag}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let w = |name: &str, body: &[u8]| {
            std::fs::write(dir.join(name), body).unwrap();
            dir.join(name)
        };
        let tls = TlsConfig {
            cert: w("server.pem", SERVER_CERT_PEM.as_bytes()),
            key: w("server.key", SERVER_KEY_PEM.as_bytes()),
            client_ca: Some(w("client_ca.pem", CLIENT_CA_PEM.as_bytes())),
        };
        let key = w("token.key", KEY_BYTES);
        Files { dir, tls, key }
    }
}

impl Drop for Files {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.dir);
    }
}

/// The release settings granting `grants` (each `FINGERPRINT:FORMS:KINDS`)
/// with the token key in `files` and the audit service at `audit`.
pub fn release(files: &Files, grants: &[String], audit: SocketAddr, timeout: Duration) -> ReleaseConfig {
    let mut g = Grants::default();
    for spec in grants {
        g.add(spec).unwrap();
    }
    ReleaseConfig {
        grants: g,
        token_keys: vec![TokenKeyFile { key: KeyRef { id: "tok".into(), version: 4 }, path: files.key.clone() }],
        active_token_key: None,
        audit_endpoint: format!("http://{audit}"),
        audit_timeout: timeout,
    }
}

/// Every output option and both value forms.
pub fn open_policy() -> PiiOutputPolicy {
    PiiOutputPolicy { values: Forms::of(Form::Token).with(Form::Clear), ..PiiOutputPolicy::ALL }
}

fn model() -> ModelConfig {
    ModelConfig {
        bundle: common::tiny().to_str().unwrap().into(),
        name: None,
        device: Device::Index(common::cpu()),
        precision: 0,
        max_batch: 0,
        max_seq: 0,
        sessions: 1,
    }
}

/// An mTLS server with `release` under `policy`.
pub async fn server(files: &Files, release: ReleaseConfig, policy: PiiOutputPolicy) -> Server {
    let serve =
        Serve { tls: Some(files.tls.clone()), pii_output: policy, pii_release: Some(release), ..Serve::default() };
    Server::start("127.0.0.1:0".parse().unwrap(), vec![model()], MAX, serve).await.unwrap()
}

/// A client over TLS, presenting the client certificate when `identity`.
pub async fn connect(s: &Server, identity: bool) -> Result<AnalysisServiceClient<Channel>, tonic::transport::Error> {
    let mut tls = ClientTlsConfig::new().domain_name("localhost").ca_certificate(Certificate::from_pem(SERVER_CA_PEM));
    if identity {
        tls = tls.identity(Identity::from_pem(CLIENT_CERT_PEM, CLIENT_KEY_PEM));
    }
    let ch = Endpoint::from_shared(format!("https://localhost:{}", s.local_addr().port()))
        .unwrap()
        .tls_config(tls)
        .unwrap()
        .connect()
        .await?;
    Ok(AnalysisServiceClient::new(ch))
}

/// The graph of TEXT with its email (mention 0) and phone (mention 1).
pub fn graph() -> Vec<u8> {
    let mut doc = Document::new(TEXT);
    let mut pii = Pii::new(ViewId::SOURCE);
    let detector = pii.detector("test", "1").unwrap();
    let evidence = [PiiEvidence::Shape];
    for (value, kind) in [(EMAIL, PiiKind::Email), (PHONE, PiiKind::Phone)] {
        let at = TEXT.find(value).unwrap() as u32;
        let span = Span::new(at, at + value.len() as u32);
        pii.push(PiiMention { span, kind, detector, evidence: &evidence, score: None, normalized: value });
    }
    doc.add(&Key::DEFAULT, pii, Provenance::by("turbo-pii:pii")).unwrap();
    export(&doc, codecs()).unwrap().to_bytes().unwrap()
}

/// Whether `bytes` hold `needle` anywhere.
pub fn holds(bytes: &[u8], needle: &[u8]) -> bool {
    bytes.windows(needle.len()).any(|w| w == needle)
}
