//! The release rules without a server: the one intersection, the event
//! id, tokens and key handling, and that no value or key byte reaches a
//! `Debug` string or an error.

use std::path::PathBuf;

use turbo_analysis_layers::{PiiKind, Sensitive};

use super::audit::{EventKey, event_id};
use super::caps::{Fingerprint, Form, Forms, Grant, Grants, effective};
use super::keys::{KeyRef, TokenKey, TokenKeyFile, active};
use super::*;
use crate::proto::audit::{PiiAccessKindGrant, PiiAccessMention};

const FP: &str = "aa00000000000000000000000000000000000000000000000000000000000001";
const PLANTED: &str = "alice.planted@example.com";

fn key_file(name: &str, bytes: &[u8]) -> TokenKeyFile {
    let dir = std::env::temp_dir().join(format!("turbo-release-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let path: PathBuf = dir.join(name);
    std::fs::write(&path, bytes).unwrap();
    TokenKeyFile { key: KeyRef { id: "tok".into(), version: 3 }, path }
}

fn mention(layer: &str, index: u32, form: Form) -> PiiAccessMention {
    PiiAccessMention {
        layer_instance: layer.into(),
        index,
        builtin_kind: 1,
        custom_kind: String::new(),
        form: form.audit() as i32,
    }
}

#[test]
fn effective_forms_are_the_intersection_of_request_policy_and_grant() {
    let (t, c) = (Forms::of(Form::Token), Forms::of(Form::Clear));
    let both = t.union(c);
    assert_eq!(effective(t, both, both), t);
    assert_eq!(effective(c, t, both), Forms::NONE, "the policy does not allow clear");
    assert_eq!(effective(c, both, t), Forms::NONE, "a token grant is not a clear grant");
    assert_eq!(effective(t, both, c), Forms::NONE, "a clear grant is not a token grant");
    assert_eq!(effective(both, both, Forms::NONE), Forms::NONE);
}

#[test]
fn a_grant_for_one_kind_covers_that_kind_only_and_no_service_gets_nothing() {
    let mut grants = Grants::default();
    grants.add(&format!("{FP}:clear:email")).unwrap();
    let fp = Fingerprint::parse(FP).unwrap();
    let g = grants.of(Some(&fp));
    assert_eq!(g.forms(PiiKind::Email), Forms::of(Form::Clear));
    assert_eq!(g.forms(PiiKind::Phone), Forms::NONE);
    assert_eq!(grants.of(None), Grant::default());
    let other = Fingerprint([0xbb; 32]);
    assert_eq!(grants.of(Some(&other)), Grant::default());
}

fn id(
    op: &str,
    input: &[u8],
    mentions: &[PiiAccessMention],
    effective: &[PiiAccessKindGrant],
    policy: &[u8],
) -> String {
    event_id(&EventKey { operation_key: op, input_digest: input, mentions, effective, policy_digest: policy })
}

fn kind(builtin: u32, forms: &[Form]) -> PiiAccessKindGrant {
    PiiAccessKindGrant {
        builtin_kind: builtin,
        custom_kind: String::new(),
        forms: forms.iter().map(|f| f.audit() as i32).collect(),
    }
}

#[test]
fn the_event_id_is_fixed_by_what_a_retry_keeps() {
    let a = [mention("pii", 0, Form::Token), mention("pii", 2, Form::Token)];
    let b = [mention("pii", 2, Form::Token), mention("pii", 0, Form::Token)];
    let eff = [kind(1, &[Form::Token]), kind(30, &[Form::Token])];
    let (input, policy) = ([1u8; 32], [2u8; 32]);
    let base = id("op-1", &input, &a, &eff, &policy);
    assert_eq!(base.len(), 64);
    assert!(base.bytes().all(|c| c.is_ascii_hexdigit() && !c.is_ascii_uppercase()));
    let swapped = [eff[1].clone(), eff[0].clone()];
    assert_eq!(base, id("op-1", &input, &b, &swapped, &policy), "list order does not matter");
    assert_ne!(base, id("op-2", &input, &a, &eff, &policy), "another operation");
    assert_ne!(base, id("op-1", &[3; 32], &a, &eff, &policy), "another input");
    assert_ne!(base, id("op-1", &input, &a[..1], &eff, &policy), "another mention set");
    assert_ne!(base, id("op-1", &input, &[mention("pii", 0, Form::Clear), a[1].clone()], &eff, &policy));
    assert_ne!(base, id("op-1", &input, &a, &eff[..1], &policy), "other forms in effect");
    assert_ne!(base, id("op-1", &input, &a, &eff, &[4; 32]), "another policy");
    // Length prefixes keep the fields apart.
    assert_ne!(
        id("op", &input, &[mention("1pii", 0, Form::Token)], &eff, &policy),
        id("op1", &input, &[mention("pii", 0, Form::Token)], &eff, &policy)
    );
}

#[test]
fn tokens_are_keyed_deterministic_and_name_their_key() {
    let k = TokenKey::load(&key_file("k32", &[7; 32])).unwrap();
    let t = k.token(PLANTED);
    assert_eq!(t.token.len(), 32);
    assert_eq!(t, k.token(PLANTED), "equal values give equal tokens under one key");
    assert_ne!(t.token, k.token("bob@example.com").token);
    let key = t.key.clone().unwrap();
    assert_eq!((key.id.as_str(), key.version), ("tok", 3));
    let other = TokenKey::load(&key_file("k33", &[8; 32])).unwrap();
    assert_ne!(t.token, other.token(PLANTED).token, "another key gives another token");
    // The token is a MAC over a domain prefix, not over the bare value.
    assert_ne!(t.token, turbo_pii::checksum::sha256::hmac(&[7; 32], PLANTED.as_bytes()).to_vec());
}

#[test]
fn key_bytes_never_reach_debug_or_errors() {
    let secret = b"planted-key-bytes-planted-key-bytes!";
    let k = TokenKey::load(&key_file("secret", secret)).unwrap();
    let debug = format!("{k:?}");
    assert!(!debug.contains("planted-key-bytes"), "{debug}");
    assert!(debug.contains("withheld"), "{debug}");
    let short = TokenKey::load(&key_file("short", b"planted-short")).unwrap_err();
    assert!(short.contains("not 32 to 1024") && !short.contains("planted-short"), "{short}");
    let missing = TokenKeyFile { key: KeyRef { id: "tok".into(), version: 1 }, path: "/nonexistent/key".into() };
    assert!(TokenKey::load(&missing).unwrap_err().contains("tok:1"));
}

#[test]
fn a_planned_value_prints_withheld() {
    let v = Sensitive::new(PLANTED.to_owned());
    let debug = format!("{v:?}");
    assert!(!debug.contains(PLANTED), "{debug}");
}

#[test]
fn the_active_key_is_the_one_key_or_the_named_one() {
    let a = TokenKeyFile { key: KeyRef { id: "a".into(), version: 1 }, path: "/k/a".into() };
    let b = TokenKeyFile { key: KeyRef { id: "a".into(), version: 2 }, path: "/k/b".into() };
    assert_eq!(active(&[], None).unwrap(), None);
    assert_eq!(active(std::slice::from_ref(&a), None).unwrap(), Some(&a));
    let both = [a.clone(), b.clone()];
    assert!(active(&both, None).is_err());
    assert_eq!(active(&both, Some(&b.key)).unwrap(), Some(&b));
}

#[test]
fn the_policy_digest_follows_the_policy_and_the_grant() {
    let mut grants = Grants::default();
    grants.add(&format!("{FP}:clear:email")).unwrap();
    let fp = Fingerprint::parse(FP).unwrap();
    let g = grants.of(Some(&fp));
    let p = PiiOutputPolicy { values: Forms::of(Form::Clear), ..PiiOutputPolicy::ALL };
    let d = policy_digest(p, &g);
    assert_eq!(d, policy_digest(p, &g));
    assert_ne!(d, policy_digest(PiiOutputPolicy::ALL, &g));
    assert_ne!(d, policy_digest(p, &Grant::default()));
}
