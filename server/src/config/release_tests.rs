//! The PII release flags (#1909): every flag and variable, and each
//! refusal.

use std::time::Duration;

use turbo_analysis_layers::PiiKind;

use super::super::Args;
use super::*;
use crate::release::caps::{Fingerprint, Form, Forms};
use crate::results::disclosure::PiiOutputPolicy;

const FP: &str = "00112233445566778899aabbccddeeff00112233445566778899AABBCCDDEEFF";
const MTLS: &str = "--listen 127.0.0.1:0 --admin --tls-cert c.pem --tls-key k.pem --tls-client-ca ca.pem";

fn args_env(s: &str, env: impl Fn(&str) -> Option<String>) -> Result<Args, String> {
    Args::parse_with(s.split_whitespace().map(String::from), env)
}

fn args(s: &str) -> Result<Args, String> {
    args_env(s, |_| None)
}

fn fp() -> Fingerprint {
    Fingerprint::parse(FP).unwrap()
}

#[test]
fn grants_keys_and_the_audit_endpoint_parse_from_flags() {
    let a = args(&format!(
        "{MTLS} --pii-grant {FP}:token+clear:email,phone --pii-grant={FP}:token:iban \
         --pii-token-key tok:1:/k/one --pii-token-key tok:2:/k/two:with-colon --pii-token-key-active tok:2 \
         --pii-audit-endpoint http://audit:9 --pii-audit-timeout-ms 250"
    ))
    .unwrap();
    let r = a.pii_release.expect("release configured");
    let g = r.grants.of(Some(&fp()));
    assert_eq!(g.forms(PiiKind::Email), Forms::of(Form::Token).with(Form::Clear));
    assert_eq!(g.forms(PiiKind::Iban), Forms::of(Form::Token), "grants of one service add up");
    assert_eq!(g.forms(PiiKind::Phone), Forms::of(Form::Token).with(Form::Clear));
    assert_eq!(g.forms(PiiKind::Ipv4), Forms::NONE, "a kind not named is not granted");
    assert_eq!(r.token_keys.len(), 2);
    assert_eq!(r.token_keys[1].path, std::path::PathBuf::from("/k/two:with-colon"));
    assert_eq!(r.active_token_key, Some(KeyRef { id: "tok".into(), version: 2 }));
    assert_eq!((r.audit_endpoint.as_str(), r.audit_timeout), ("http://audit:9", Duration::from_millis(250)));
    // No grant: no release and no other flag.
    assert_eq!(args(MTLS).unwrap().pii_release, None);
}

#[test]
fn the_variables_fill_in_what_the_flags_leave_out() {
    let env = |k: &str| match k {
        ENV_PII_GRANTS => Some(format!("{FP}:clear:*; {FP}:token:email")),
        ENV_PII_TOKEN_KEYS => Some("tok:7:/k/seven".into()),
        ENV_PII_AUDIT_ENDPOINT => Some("http://audit:9".into()),
        ENV_PII_AUDIT_TIMEOUT_MS => Some("40".into()),
        _ => None,
    };
    let r = args_env(MTLS, env).unwrap().pii_release.unwrap();
    let g = r.grants.of(Some(&fp()));
    assert!(g.clear.custom, "`*` grants custom kinds too");
    assert_eq!(g.forms(PiiKind::Phone), Forms::of(Form::Clear));
    assert_eq!(r.active_token_key, None, "one key is the active one");
    assert_eq!(r.audit_timeout, Duration::from_millis(40));
    // A flag wins over its variable.
    let a = args_env(&format!("{MTLS} --pii-grant {FP}:clear:email"), env).unwrap();
    assert!(!a.pii_release.unwrap().grants.of(Some(&fp())).clear.custom);
    let env = |k: &str| (k == ENV_PII_TOKEN_KEY_ACTIVE).then(|| "tok:1".to_owned());
    assert!(args_env(MTLS, env).unwrap_err().contains("need --pii-grant"));
}

#[test]
fn every_inconsistent_release_setting_is_refused() {
    let grant = format!("--pii-grant {FP}:clear:email --pii-audit-endpoint http://audit:9");
    for (cmd, want) in [
        (format!("--listen 127.0.0.1:0 --admin {grant}"), "needs mTLS"),
        (format!("--listen 127.0.0.1:0 --admin --tls-cert c --tls-key k {grant}"), "needs mTLS"),
        (format!("{MTLS} --pii-grant {FP}:clear:email"), "needs --pii-audit-endpoint"),
        (format!("{MTLS} --pii-audit-endpoint http://audit:9"), "need --pii-grant"),
        (format!("{MTLS} --pii-token-key tok:1:/k"), "need --pii-grant"),
        (format!("{MTLS} --pii-grant {FP}:token:email --pii-audit-endpoint http://a:9"), "needs --pii-token-key"),
        (format!("{MTLS} {grant} --pii-token-key a:1:/k --pii-token-key b:1:/k"), "--pii-token-key-active"),
        (format!("{MTLS} {grant} --pii-token-key a:1:/k --pii-token-key a:1:/j"), "given twice"),
        (format!("{MTLS} {grant} --pii-token-key a:1:/k --pii-token-key-active a:2"), "is not a --pii-token-key"),
        (format!("{MTLS} {grant} --pii-token-key a:0:/k"), "not a number from 1"),
        (format!("{MTLS} {grant} --pii-token-key a b:1:/k"), "unknown argument"),
        (format!("{MTLS} {grant} --pii-token-key a:1:"), "path is empty"),
        (format!("{MTLS} {grant} --pii-audit-timeout-ms 0"), "over 0"),
        (format!("{MTLS} {grant} --pii-audit-endpoint http://b:9"), "given once"),
        (format!("{MTLS} --pii-grant {FP}:ciphertext:email --pii-audit-endpoint http://a:9"), "#1885"),
        (format!("{MTLS} --pii-grant {FP}:token+token:email --pii-audit-endpoint http://a:9"), "given twice"),
        (format!("{MTLS} --pii-grant {FP}:reveal:email --pii-audit-endpoint http://a:9"), "not token or clear"),
        (format!("{MTLS} --pii-grant {FP}:clear:mail --pii-audit-endpoint http://a:9"), "not a built-in PII kind"),
        (format!("{MTLS} --pii-grant {FP}:clear:email,email --pii-audit-endpoint http://a:9"), "given twice"),
        (format!("{MTLS} --pii-grant {FP}:clear --pii-audit-endpoint http://a:9"), "FINGERPRINT:FORMS:KINDS"),
        (format!("{MTLS} --pii-grant abc:clear:email --pii-audit-endpoint http://a:9"), "64 hexadecimal"),
        (format!("{MTLS} --pii-grant {}:clear:email --pii-audit-endpoint http://a:9", "g".repeat(64)), "hexadecimal"),
    ] {
        let e = args(&cmd).unwrap_err();
        assert!(e.contains(want), "{cmd}: {e}");
    }
}

#[test]
fn the_output_policy_lists_value_forms_and_refuses_ciphertext() {
    let p = PiiOutputPolicy::parse("rewritten-text,token,clear").unwrap();
    assert!(p.rewritten_text && !p.full_text && p.values == Forms::of(Form::Token).with(Form::Clear));
    assert_eq!(PiiOutputPolicy::parse("clear").unwrap().values, Forms::of(Form::Clear));
    assert_eq!(PiiOutputPolicy::parse("none").unwrap(), PiiOutputPolicy::NONE);
    assert_eq!(PiiOutputPolicy::ALL.values, Forms::NONE, "the default allows no value form");
    assert!(PiiOutputPolicy::parse("token,token").unwrap_err().contains("twice"));
    assert!(PiiOutputPolicy::parse("ciphertext").unwrap_err().contains("#1885"));
    let a = args(&format!("{MTLS} --pii-output-allow detail,clear")).unwrap();
    assert_eq!(a.pii_output.values, Forms::of(Form::Clear));
}

#[test]
fn a_fingerprint_reads_either_case_and_prints_lowercase() {
    assert_eq!(fp().to_string(), FP.to_ascii_lowercase());
    assert_eq!(Fingerprint::parse(&FP.to_ascii_lowercase()).unwrap(), fp());
}

#[test]
fn requiring_the_recorded_policy_is_off_by_default_and_set_by_its_flag_or_variable() {
    let grant = format!("{MTLS} --pii-grant {FP}:clear:email --pii-audit-endpoint http://audit:9");
    let on = |a: Args| a.pii_release.unwrap().require_recorded_policy;
    assert!(!on(args(&grant).unwrap()), "off by default");
    assert!(on(args(&format!("{grant} --pii-require-recorded-policy")).unwrap()));
    for (v, want) in [("1", true), ("true", true), ("yes", true), ("0", false), ("false", false), ("no", false)] {
        let env = |k: &str| (k == ENV_PII_REQUIRE_RECORDED_POLICY).then(|| v.to_owned());
        assert_eq!(on(args_env(&grant, env).unwrap()), want, "{v}");
    }
    // The flag wins over a variable that turns it off.
    let off = |k: &str| (k == ENV_PII_REQUIRE_RECORDED_POLICY).then(|| "0".to_owned());
    assert!(on(args_env(&format!("{grant} --pii-require-recorded-policy"), off).unwrap()));
    for (cmd, want) in [
        (format!("{grant} --pii-require-recorded-policy=1"), "takes no value"),
        (format!("{grant} --pii-require-recorded-policy --pii-require-recorded-policy"), "given once"),
        (format!("{MTLS} --pii-require-recorded-policy"), "need --pii-grant"),
    ] {
        let e = args(&cmd).unwrap_err();
        assert!(e.contains(want), "{cmd}: {e}");
    }
    let bad = |k: &str| (k == ENV_PII_REQUIRE_RECORDED_POLICY).then(|| "on".to_owned());
    assert!(args_env(&grant, bad).unwrap_err().contains("not one of"));
    let set = |k: &str| (k == ENV_PII_REQUIRE_RECORDED_POLICY).then(|| "1".to_owned());
    assert!(args_env(MTLS, set).unwrap_err().contains("need --pii-grant"));
}

#[test]
fn encryption_keys_repeat_need_a_grant_and_a_reversible_build() {
    let grant = format!("{MTLS} --pii-grant {FP}:clear:email --pii-audit-endpoint http://audit:9");
    let cmd = format!("{grant} --pii-encryption-key vault:1:/k/one --pii-encryption-key vault:2:/k/two:x");
    if cfg!(feature = "pii-reversible") {
        let r = args(&cmd).unwrap().pii_release.unwrap();
        assert_eq!(r.encryption_keys.len(), 2);
        assert_eq!(r.encryption_keys[1].key, KeyRef { id: "vault".into(), version: 2 });
        assert_eq!(r.encryption_keys[1].path, std::path::PathBuf::from("/k/two:x"));
        assert!(r.token_keys.is_empty(), "an encryption key is not a token key");
        let env = |k: &str| (k == ENV_PII_ENCRYPTION_KEYS).then(|| "vault:3:/k/three; vault:4:/k/four".to_owned());
        assert_eq!(args_env(&grant, env).unwrap().pii_release.unwrap().encryption_keys.len(), 2);
        for (cmd, want) in [
            (format!("{grant} --pii-encryption-key a:1:/k --pii-encryption-key a:1:/j"), "given twice"),
            (format!("{grant} --pii-encryption-key a:0:/k"), "not a number from 1"),
            (format!("{grant} --pii-encryption-key a:1:"), "path is empty"),
            (format!("{MTLS} --pii-encryption-key a:1:/k"), "need --pii-grant"),
        ] {
            let e = args(&cmd).unwrap_err();
            assert!(e.contains(want), "{cmd}: {e}");
        }
    } else {
        let e = args(&cmd).unwrap_err();
        assert!(e.contains("pii-reversible"), "{e}");
    }
}
