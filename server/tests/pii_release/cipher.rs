//! Ciphertext release (#1885) over real mTLS: a service granted
//! `ciphertext` for a kind receives that kind's value encrypted under the
//! active encryption key, which a holder of the key decrypts; the event
//! records the form, and no value or key byte reaches the answer or the
//! audit service.

use prost::Message;
use turbo_kserve::proto::audit::PiiAccessForm;
use turbo_kserve::proto::result::{PiiValueForm, PiiWithheldReason, pii_value};
use turbo_kserve::release::caps::{Form, Forms};
use turbo_kserve::release::cipher::{CONTEXT, SCHEME};
use turbo_kserve::release::keys::{KeyRef, TokenKeyFile};
use turbo_kserve::release::{Release, ReleaseConfig};
use turbo_kserve::results::disclosure::PiiOutputPolicy;
use turbo_pii::PiiKind;
use turbo_pii::rewrite::AesGcmEncryptor;

use crate::certs::client_fingerprint;
use crate::fixture::*;
use crate::{SECOND, ask, forms, reason};

/// What the encryption key file holds; it must never reach any output.
const ENCRYPTION_KEY: [u8; 32] = *b"planted-encryption-key-32-bytes!";

fn with_encryption_key(files: &Files, config: ReleaseConfig) -> ReleaseConfig {
    let path = files.dir.join("encryption.key");
    std::fs::write(&path, ENCRYPTION_KEY).unwrap();
    let key = TokenKeyFile { key: KeyRef { id: "vault".into(), version: 2 }, path };
    ReleaseConfig { encryption_keys: vec![key], ..config }
}

#[tokio::test]
async fn a_ciphertext_grant_releases_values_the_encryption_key_decrypts() {
    let files = Files::create("cipher");
    let (audit, at) = audit(Answer::Ack).await;
    let grant = format!("{}:ciphertext:email", client_fingerprint());
    let config = with_encryption_key(&files, release(&files, &[grant], at, SECOND));
    let policy = PiiOutputPolicy { values: Forms::of(Form::Ciphertext), ..PiiOutputPolicy::ALL };
    let s = server(&files, config, policy).await;
    let mut c = connect(&s, true).await.unwrap();

    let typed = c.inspect_document_graph(ask(PiiValueForm::Ciphertext, "op-cipher")).await.unwrap().into_inner();
    let got = forms(&typed);
    let pii_value::Form::Ciphertext(sealed) = &got[0] else { panic!("a ciphertext: {got:?}") };
    assert_eq!((sealed.scheme.as_str(), sealed.key_id.as_str(), sealed.key_version), (SCHEME, "vault", 2));
    let text = std::str::from_utf8(&sealed.ciphertext).unwrap();
    let dec = AesGcmEncryptor::new(&ENCRYPTION_KEY).unwrap().with_context(CONTEXT);
    let kind = format!("BUILTIN:{}", PiiKind::Email.builtin_id().unwrap());
    assert_eq!(dec.decrypt(text).unwrap(), (kind, EMAIL.to_owned()));
    assert_eq!(reason(&got[1]), Some(PiiWithheldReason::NotGranted), "the phone kind is not granted");
    let wire = typed.encode_to_vec();
    for planted in [EMAIL.as_bytes(), PHONE.as_bytes(), &ENCRYPTION_KEY, KEY_BYTES] {
        assert!(!holds(&wire, planted), "the answer carries a planted value or key byte");
    }

    // A retry encrypts afresh under the same event.
    let again = c.inspect_document_graph(ask(PiiValueForm::Ciphertext, "op-cipher")).await.unwrap().into_inner();
    assert_ne!(forms(&again)[0], got[0], "a fresh nonce each release");
    let events = audit.events.lock().unwrap().clone();
    assert_eq!(events.len(), 1, "one release, one record");
    let e = events.values().next().unwrap();
    assert_eq!(e.mentions.len(), 1);
    assert_eq!(e.mentions[0].form, PiiAccessForm::Ciphertext as i32);
    for call in audit.calls.lock().unwrap().iter() {
        for planted in [EMAIL.as_bytes(), &ENCRYPTION_KEY, sealed.ciphertext.as_slice()] {
            assert!(!holds(call, planted), "an audit event carries a value, ciphertext or key byte");
        }
    }
    s.stop().await;
}

#[test]
fn a_ciphertext_grant_needs_an_encryption_key() {
    let files = Files::create("cipher-no-key");
    let grant = format!("{}:ciphertext:email", client_fingerprint());
    let config = release(&files, &[grant], "127.0.0.1:9".parse().unwrap(), SECOND);
    let e = Release::start(Some(&config), Some(&files.tls)).err().expect("refused without a key");
    assert!(e.contains("needs --pii-encryption-key"), "{e}");
    let short = with_encryption_key(&files, config);
    std::fs::write(&short.encryption_keys[0].path, [1u8; 16]).unwrap();
    let e = Release::start(Some(&short), Some(&files.tls)).err().expect("refused with a short key");
    assert!(e.contains("not 32") && !e.contains("1, 1"), "{e}");
}
