//! The `PiiAesGcmStrategy` and `PiiFf1Strategy` arms (issue #1885). With
//! the `pii-reversible` feature they resolve an encryption key by
//! reference, convert to `turbo-pii`'s encryptors, are in the digest, and
//! the redact step writes values a holder of the key decrypts. Without it
//! the server does not know either arm.

#[path = "pii_policy/common.rs"]
mod common;
#[path = "pii_redact/fixture.rs"]
mod fixture;

use common::*;
use pb::pii_rewrite_strategy::Strategy;
use turbo_kserve::pii_policy::{PolicyError, RedactionPolicy, edit_strategy};
use turbo_kserve::proto::result as pb;

fn aes(id: &str, version: u32, context: &str) -> Strategy {
    Strategy::Aes256Gcm(pb::PiiAesGcmStrategy { key: key_ref(id, version), context: context.to_owned() })
}

fn ff1(alphabet: pb::PiiFf1Alphabet, tweak: &str) -> Strategy {
    Strategy::Ff1(pb::PiiFf1Strategy { key: key_ref("vault", 1), alphabet: alphabet as i32, tweak: tweak.to_owned() })
}

/// Email to `email`, and US SSNs to `ssn`.
#[cfg_attr(not(feature = "pii-reversible"), allow(dead_code))]
fn two_rules(email: Strategy, ssn: Strategy) -> pb::RedactionPolicy {
    policy(vec![
        rule("email", builtin(pb::PiiKindId::Email), vec![entry(email)]),
        rule("ssn", builtin(pb::PiiKindId::UsSsn), vec![entry(ssn)]),
    ])
}

/// `s` on US SSNs, whose values are digits and `-`.
#[cfg_attr(not(feature = "pii-reversible"), allow(dead_code))]
fn on_ssn(s: Strategy) -> pb::RedactionPolicy {
    policy(vec![rule("ssn", builtin(pb::PiiKindId::UsSsn), vec![entry(s)])])
}

fn refused(s: Strategy) -> PolicyError {
    match RedactionPolicy::from_proto(&one(s), &TestRegistry) {
        Ok(_) => panic!("the policy is refused"),
        Err(e) => e,
    }
}

#[test]
fn each_arm_records_its_own_strategy_value() {
    assert_eq!(edit_strategy(&aes("vault", 1, "")), pb::PiiStrategy::Aes256Gcm);
    assert_eq!(edit_strategy(&ff1(pb::PiiFf1Alphabet::Digits, "")), pb::PiiStrategy::Ff1);
}

#[cfg(feature = "pii-reversible")]
mod enabled {
    use turbo_analysis::ViewId;
    use turbo_kserve::offsets::OffsetMode;
    use turbo_kserve::pii_policy::{NoRegistry, PolicyError, RedactionPolicy, strategy_to_proto};
    use turbo_kserve::proto::result::typed_annotation::Value;
    use turbo_kserve::results::typed::document_to_typed;
    use turbo_pii::PiiStrategyKind;
    use turbo_pii::rewrite::{AesGcmEncryptor, Ff1Alphabet, Ff1Encryptor, RewriteStrategy};

    use super::fixture::*;
    use super::*;

    fn native(s: Strategy) -> RewriteStrategy {
        let p = RedactionPolicy::from_proto(&one(s), &TestRegistry).expect("the policy converts");
        p.rules[0].strategies[0].strategy.clone()
    }

    #[test]
    fn the_arms_resolve_the_encryption_key_and_convert() {
        let got = native(aes("vault", 1, "doc-7"));
        let want = AesGcmEncryptor::new(&ENCRYPTION_KEY).unwrap().with_context(b"doc-7");
        assert_eq!(got, RewriteStrategy::from(want.clone()));
        assert_eq!(got.kind(), PiiStrategyKind::Aes256Gcm);
        let RewriteStrategy::AesGcm(enc) = &got else { unreachable!() };
        let token = enc.token("email", "jane@example.com").unwrap();
        assert_eq!(want.decrypt(&token).unwrap().1, "jane@example.com", "the registered key decrypts it");
        assert_eq!(strategy_to_proto(&got), None, "a native encryptor holds key bytes, not a reference");

        let p = RedactionPolicy::from_proto(&on_ssn(ff1(pb::PiiFf1Alphabet::UpperAlphanumeric, "ssn")), &TestRegistry);
        let got = p.expect("the policy converts").rules[0].strategies[0].strategy.clone();
        let want = Ff1Encryptor::new(&ENCRYPTION_KEY, Ff1Alphabet::UpperAlphanumeric).unwrap().with_tweak(b"ssn");
        assert_eq!(got, RewriteStrategy::from(want));
        assert_eq!(got.kind(), PiiStrategyKind::Ff1);
        assert_eq!(strategy_to_proto(&got), None);
    }

    #[test]
    fn a_token_key_is_not_an_encryption_key() {
        // `tokens:1` is registered as a token key only.
        let err = refused(aes("tokens", 1, ""));
        assert!(matches!(err, PolicyError::Unresolved { field: "aes_256_gcm.key", .. }), "{err:?}");
        let err = refused(aes("vault", 2, ""));
        assert!(matches!(err, PolicyError::Unresolved { field: "aes_256_gcm.key", .. }), "{err:?}");
        let Err(err) = RedactionPolicy::from_proto(&one(aes("vault", 1, "")), &NoRegistry) else {
            panic!("refused without a registry");
        };
        assert!(matches!(err, PolicyError::Unresolved { .. }), "{err:?}");
        assert!(!err.to_string().contains("9, 9"), "{err}");
    }

    #[test]
    fn missing_keys_key_material_and_unspecified_alphabets_are_refused() {
        let no_key = Strategy::Aes256Gcm(pb::PiiAesGcmStrategy { key: None, context: String::new() });
        assert!(matches!(refused(no_key), PolicyError::Missing { field: "aes_256_gcm.key", .. }));
        let no_key = Strategy::Ff1(pb::PiiFf1Strategy {
            key: None,
            alphabet: pb::PiiFf1Alphabet::Digits as i32,
            tweak: String::new(),
        });
        assert!(matches!(refused(no_key), PolicyError::Missing { field: "ff1.key", .. }));
        let hex = "0123456789abcdef0123456789abcdef";
        assert!(matches!(refused(aes(hex, 1, "")), PolicyError::BadRef { field: "key.id", .. }));
        let err = refused(ff1(pb::PiiFf1Alphabet::Unspecified, ""));
        assert!(matches!(err, PolicyError::Unspecified { field: "ff1.alphabet", value: 0, .. }), "{err:?}");
        let unknown = Strategy::Ff1(pb::PiiFf1Strategy { alphabet: 9, ..pb::PiiFf1Strategy::default() });
        assert!(matches!(refused(unknown), PolicyError::Unspecified { field: "ff1.alphabet", value: 9, .. }));
    }

    #[test]
    fn an_ff1_alphabet_that_cannot_hold_the_kind_is_refused_at_load() {
        // Every email holds a letter (its top level domain), so digits
        // alone cannot encrypt one; every GitHub token holds a lower case
        // prefix.
        let err = refused(ff1(pb::PiiFf1Alphabet::Digits, ""));
        let PolicyError::Invalid { reason, .. } = &err else { panic!("{err:?}") };
        assert!(reason.starts_with("ff1.alphabet cannot encrypt a value of kind email"), "{reason}");
        assert!(err.to_string().contains("kind_rules[0].strategies[0]"), "{err}");
        let github = |a| policy(vec![rule("gh", builtin(pb::PiiKindId::GithubToken), vec![entry(ff1(a, ""))])]);
        let err = RedactionPolicy::from_proto(&github(pb::PiiFf1Alphabet::UpperAlphanumeric), &TestRegistry);
        assert!(matches!(err, Err(PolicyError::Invalid { .. })), "{err:?}");
        // An alphabet that holds every letter converts, and the run time
        // check still covers the rest.
        native(ff1(pb::PiiFf1Alphabet::Alphanumeric, ""));
        native(ff1(pb::PiiFf1Alphabet::UpperAlphanumeric, ""));
        RedactionPolicy::from_proto(&github(pb::PiiFf1Alphabet::Alphanumeric), &TestRegistry).unwrap();
        RedactionPolicy::from_proto(&on_ssn(ff1(pb::PiiFf1Alphabet::Digits, "")), &TestRegistry).unwrap();
        // Under `redact_and_flag_unknown` the refusal stands: the arm is
        // known, only unusable on this kind.
        let mut flagged = one(ff1(pb::PiiFf1Alphabet::Digits, ""));
        flagged.redact_and_flag_unknown = true;
        assert!(RedactionPolicy::from_proto(&flagged, &TestRegistry).is_err());
    }

    #[test]
    fn the_digest_covers_the_key_reference_context_alphabet_and_tweak() {
        let digest = |s| RedactionPolicy::from_proto(&on_ssn(s), &TestRegistry).unwrap().digest;
        let base = digest(aes("vault", 1, "a"));
        assert_eq!(base, digest(aes("vault", 1, "a")));
        assert_ne!(base, digest(aes("vault", 1, "b")));
        assert_ne!(base, digest(ff1(pb::PiiFf1Alphabet::Alphanumeric, "a")), "the arm is part of the digest");
        let f = digest(ff1(pb::PiiFf1Alphabet::Digits, "t"));
        assert_ne!(f, digest(ff1(pb::PiiFf1Alphabet::Digits, "u")));
        assert_ne!(f, digest(ff1(pb::PiiFf1Alphabet::UpperAlphanumeric, "t")));
    }

    #[test]
    fn the_redact_step_writes_values_the_key_decrypts_and_the_wire_edits_say_so() {
        let text = "Mail jane@example.com, SSN 123-45-6789, thanks.";
        let p = convert(&two_rules(aes("vault", 1, "doc-7"), ff1(pb::PiiFf1Alphabet::Digits, "ssn")), &TestRegistry);
        let doc = run_with(text, p.clone(), false);
        let (view, edits) = redacted(&doc);
        assert!(!view.text().contains("jane@") && !view.text().contains("6789"), "{}", view.text());

        let aes = AesGcmEncryptor::new(&ENCRYPTION_KEY).unwrap().with_context(b"doc-7");
        let fpe = Ff1Encryptor::new(&ENCRYPTION_KEY, Ff1Alphabet::Digits).unwrap().with_tweak(b"ssn");
        let mut kinds = Vec::new();
        for e in edits.edits() {
            let written = &view.text()[e.span.start as usize..e.span.end as usize];
            match e.strategy {
                PiiStrategyKind::Aes256Gcm => {
                    assert_eq!(aes.decrypt(written).unwrap(), ("EMAIL".to_owned(), "jane@example.com".to_owned()));
                }
                PiiStrategyKind::Ff1 => {
                    assert_eq!(written.len(), 11);
                    assert_eq!(fpe.decrypt(written).unwrap(), "123-45-6789");
                }
                other => panic!("unexpected strategy {other:?}"),
            }
            kinds.push(e.strategy);
        }
        kinds.sort_by_key(|k| k.id());
        assert_eq!(kinds, [PiiStrategyKind::Aes256Gcm, PiiStrategyKind::Ff1]);

        let typed = document_to_typed("redact", &doc, ViewId::SOURCE, OffsetMode::Utf8).unwrap();
        let layer = typed.layers.iter().find(|l| l.id == "turbo:pii-edits").expect("the edits layer");
        let mut wire: Vec<i32> = layer
            .annotations
            .iter()
            .map(|a| match &a.value {
                Some(Value::PiiEdit(e)) => {
                    assert_eq!(e.policy_digest, p.digest.as_bytes().to_vec());
                    e.strategy
                }
                _ => panic!("not an edit"),
            })
            .collect();
        wire.sort_unstable();
        assert_eq!(wire, [pb::PiiStrategy::Aes256Gcm as i32, pb::PiiStrategy::Ff1 as i32]);
    }
}

#[cfg(not(feature = "pii-reversible"))]
mod disabled {
    use turbo_kserve::pii_policy::PolicyFlag;
    use turbo_pii::rewrite::{MaskPolicy, RewriteStrategy};

    use super::*;

    #[test]
    fn a_server_without_reversible_protection_does_not_know_the_arms() {
        for s in [aes("vault", 1, ""), ff1(pb::PiiFf1Alphabet::Digits, "")] {
            let err = refused(s.clone());
            assert!(matches!(err, PolicyError::UnknownArm { field: "strategy", .. }), "{err:?}");
            let mut flagged = one(s);
            flagged.redact_and_flag_unknown = true;
            let p = RedactionPolicy::from_proto(&flagged, &TestRegistry).unwrap();
            assert_eq!(p.rules[0].strategies[0].strategy, RewriteStrategy::Mask(MaskPolicy::new('*').unwrap()));
            assert_eq!(p.flags, vec![PolicyFlag::UnknownStrategy { rule_id: "email".to_owned(), entry: 0 }]);
        }
    }
}
