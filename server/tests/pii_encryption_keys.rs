//! `--pii-encryption-key` without `--pii-grant` (#2111). Encrypting during
//! redaction releases nothing, so a server with no release still loads
//! its encryption keys, and a redaction policy's AES-256-GCM and FF1 arms
//! resolve through them at start. Without the key the policy fails start;
//! keys given both on their own and in a release config are refused.

#![cfg(feature = "pii-reversible")]

#[path = "pii_policy/common.rs"]
mod common;

use std::path::{Path, PathBuf};

use common::*;
use pb::pii_rewrite_strategy::Strategy;
use prost::Message;
use turbo_kserve::config::{DEFAULT_MAX_MESSAGE_BYTES as MAX, Device, ModelConfig};
use turbo_kserve::pii_policy::PolicyFiles;
use turbo_kserve::proto::result as pb;
use turbo_kserve::release::ReleaseConfig;
use turbo_kserve::release::keys::{KeyRef, TokenKeyFile};
use turbo_kserve::{Serve, Server};

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
fn policies() -> PolicyFiles {
    let aes = Strategy::Aes256Gcm(pb::PiiAesGcmStrategy { key: key_ref("vault", 1), context: "doc".into() });
    let ff1 = Strategy::Ff1(pb::PiiFf1Strategy {
        key: key_ref("vault", 1),
        alphabet: pb::PiiFf1Alphabet::Digits as i32,
        tweak: "ssn".into(),
    });
    let p = policy(vec![
        rule("email", builtin(pb::PiiKindId::Email), vec![entry(aes)]),
        rule("ssn", builtin(pb::PiiKindId::UsSsn), vec![entry(ff1)]),
    ]);
    PolicyFiles(vec![p.encode_to_vec()])
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
