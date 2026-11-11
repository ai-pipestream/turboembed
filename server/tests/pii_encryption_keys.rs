//! `--pii-encryption-key` without `--pii-grant` (#2111). Encrypting during
//! redaction releases nothing, so a server with no release still loads
//! its encryption keys, and a redaction policy's AES-256-GCM and FF1 arms
//! resolve through them at start. Without the key the policy fails start;
//! keys given both on their own and in a release config are refused. Such
//! keys serve the operator's named policies only: an inline policy naming
//! one is refused, since a caller choosing its own FF1 tweak would get a
//! deterministic encryption oracle on a server that need not authenticate
//! it.

#![cfg(feature = "pii-reversible")]

#[path = "pii_policy/common.rs"]
mod common;

use std::path::{Path, PathBuf};

use common::*;
use pb::analyze_progressive_event::Event;
use pb::pii_redaction_selection::Policy;
use pb::pii_rewrite_strategy::Strategy;
use prost::Message;
use tokio_stream::StreamExt;
use turbo_analysis::export::{DocumentGraph, import};
use turbo_analysis::{Document, Key};
use turbo_kserve::config::{DEFAULT_MAX_MESSAGE_BYTES as MAX, Device, ModelConfig};
use turbo_kserve::pii_policy::PolicyFiles;
use turbo_kserve::proto::result as pb;
use turbo_kserve::proto::result::analysis_service_client::AnalysisServiceClient;
use turbo_kserve::release::ReleaseConfig;
use turbo_kserve::release::keys::{KeyRef, TokenKeyFile};
use turbo_kserve::results::document::codecs;
use turbo_kserve::{Serve, Server};
use turbo_pii::PiiEdits;
use turbo_pii::rewrite::REDACTED;

fn model() -> ModelConfig {
    let bundle = Path::new(env!("CARGO_MANIFEST_DIR")).join("../testdata/tiny-bert-bundle");
    ModelConfig {
        bundle: bundle.to_str().unwrap().into(),
        name: None,
        device: Device::Index(0),
        precision: 0,
        max_batch: 0,
        max_seq: 0,
        sessions: 1,
    }
}

async fn start(serve: Serve) -> Result<Server, String> {
    Server::start("127.0.0.1:0".parse().unwrap(), vec![model()], MAX, serve).await
}

/// A 32-byte encryption key file registered as `vault:1`.
fn key(name: &str) -> TokenKeyFile {
    let dir: PathBuf = Path::new(env!("CARGO_TARGET_TMPDIR")).join(format!("enc-{name}-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("vault-1.key");
    std::fs::write(&path, [7u8; 32]).unwrap();
    TokenKeyFile { key: KeyRef { id: "vault".into(), version: 1 }, path }
}

/// Emails encrypted with AES-256-GCM, US SSNs with FF1 over digits, both
/// under `vault:1`.
fn redaction() -> pb::RedactionPolicy {
    let aes = Strategy::Aes256Gcm(pb::PiiAesGcmStrategy { key: key_ref("vault", 1), context: "doc".into() });
    let ff1 = Strategy::Ff1(pb::PiiFf1Strategy {
        key: key_ref("vault", 1),
        alphabet: pb::PiiFf1Alphabet::Digits as i32,
        tweak: "ssn".into(),
    });
    policy(vec![
        rule("email", builtin(pb::PiiKindId::Email), vec![entry(aes)]),
        rule("ssn", builtin(pb::PiiKindId::UsSsn), vec![entry(ff1)]),
    ])
}

fn policies() -> PolicyFiles {
    PolicyFiles(vec![redaction().encode_to_vec()])
}

/// The redacted text of a short document under `sel`, and each edit's
/// rule id.
async fn redact(s: &Server, sel: Policy) -> Result<(String, Vec<String>), tonic::Status> {
    let req = pb::AnalyzePipelineRequest {
        doc_id: "enc".into(),
        text: "Mail jane@example.com, SSN 123-45-6789.".into(),
        offsets: pb::OffsetMode::Utf8 as i32,
        steps: vec!["turbo-pii:pii".into(), "turbo-pii:redact".into()],
        pii_output: Some(pb::PiiOutputOptions { full_text: true, rewritten_text: true, ..Default::default() }),
        pii_redaction: Some(pb::PiiRedactionSelection { policy: Some(sel), dry_run: false }),
        ..Default::default()
    };
    let mut c = AnalysisServiceClient::connect(format!("http://{}", s.local_addr())).await.unwrap();
    let mut stream = c.analyze_document_progressive(req).await?.into_inner();
    let mut graph = None;
    while let Some(ev) = stream.next().await {
        if let Some(Event::Complete(done)) = ev?.event {
            graph = Some(done.graph);
        }
    }
    let doc: Document = import(&DocumentGraph::from_bytes(&graph.expect("complete")).unwrap(), codecs()).unwrap();
    let view = doc.get_key(&REDACTED).expect("the redacted view").text().to_owned();
    let edits = doc.get_key(&Key::<PiiEdits>::named(REDACTED.name)).expect("its edits");
    Ok((view, edits.edits().map(|e| e.rule_id.to_owned()).collect()))
}

#[tokio::test]
async fn a_grant_less_key_serves_named_policies_but_never_an_inline_one() {
    let serve = Serve { pii_policies: policies(), pii_encryption_keys: vec![key("inline")], ..Serve::default() };
    let s = start(serve).await.unwrap();
    let (view, mut rules) = redact(&s, Policy::Name("test-policy".into())).await.expect("the named policy runs");
    assert!(!view.contains("jane@") && !view.contains("6789"), "{view}");
    rules.sort();
    assert_eq!(rules, ["email", "ssn"]);

    // A caller's own FF1 arm under the operator's key, with its own tweak:
    // refused, as is the AES-256-GCM arm, before any text is encrypted.
    for (arm, field) in [(redaction(), "aes_256_gcm.key"), (redaction_ff1_only(), "ff1.key")] {
        let e = redact(&s, Policy::Inline(arm)).await.unwrap_err();
        assert_eq!(e.code(), tonic::Code::InvalidArgument, "{e:?}");
        assert!(e.message().contains(field), "{}", e.message());
        assert!(e.message().contains("named --pii-policy policies only"), "{}", e.message());
    }
    s.stop().await;
}

/// The caller's own FF1 arm on SSNs under `vault:1`, with a tweak of its
/// choosing.
fn redaction_ff1_only() -> pb::RedactionPolicy {
    let ff1 = Strategy::Ff1(pb::PiiFf1Strategy {
        key: key_ref("vault", 1),
        alphabet: pb::PiiFf1Alphabet::Digits as i32,
        tweak: "attacker-chosen".into(),
    });
    policy(vec![rule("ssn", builtin(pb::PiiKindId::UsSsn), vec![entry(ff1)])])
}

#[tokio::test]
async fn a_policy_encrypts_under_a_key_registered_without_a_grant() {
    let serve = Serve { pii_policies: policies(), pii_encryption_keys: vec![key("alone")], ..Serve::default() };
    assert!(serve.pii_release.is_none() && serve.tls.is_none(), "no grant, no mTLS");
    let s = start(serve).await.expect("the policy's keys resolve with no release");
    s.stop().await;
}

#[tokio::test]
async fn without_the_key_the_policy_fails_start() {
    let e = start(Serve { pii_policies: policies(), ..Serve::default() }).await.err().expect("refused");
    assert!(e.contains("aes_256_gcm.key"), "{e}");
    assert!(!e.contains("vault-1.key"), "no path in the error: {e}");
}

#[tokio::test]
async fn keys_on_their_own_and_in_a_release_are_refused() {
    let release = ReleaseConfig {
        grants: Default::default(),
        token_keys: Vec::new(),
        active_token_key: None,
        encryption_keys: vec![key("both")],
        active_encryption_key: None,
        audit_endpoint: "http://audit:9".into(),
        audit_timeout: std::time::Duration::from_secs(1),
        require_recorded_policy: false,
    };
    let serve = Serve { pii_release: Some(release), pii_encryption_keys: vec![key("both")], ..Serve::default() };
    let e = start(serve).await.err().expect("refused");
    assert!(e.contains("encryption keys"), "{e}");
}

#[tokio::test]
async fn one_reference_given_twice_is_refused() {
    let serve = Serve { pii_encryption_keys: vec![key("twice-a"), key("twice-b")], ..Serve::default() };
    let e = start(serve).await.err().expect("refused");
    assert!(e.contains("given twice"), "{e}");
}
