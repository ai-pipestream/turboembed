//! #1885: `AesGcmEncryptor` and `Ff1Encryptor` as `RewriteStrategy` arms
//! of a `RewritePolicy`. A policy routes each kind to its encryptor, the
//! edits record AES-256-GCM and FF1, each replacement decrypts back to the
//! surface, and the policy's configuration hash follows the key, the
//! context and the tweak without carrying the key.

#![cfg(feature = "reversible")]

use turbo_analysis::{Annotator, Span, ViewId};
use turbo_analysis_layers::{Pii, PiiEvidence, PiiKind, PiiMention, PiiStrategyKind};
use turbo_pii::rewrite::{
    AesGcmEncryptor, Ff1Alphabet, Ff1Encryptor, MaskPolicy, PiiRewrite, RedactedView, RewriteError, RewritePolicy,
    RewriteStrategy, source_view,
};

const KEY: [u8; 32] = [0x5a; 32];
const OTHER_KEY: [u8; 32] = [0xa5; 32];

fn layer(text: &str, mentions: &[(usize, usize, PiiKind)]) -> Pii {
    let mut pii = Pii::new(ViewId::SOURCE);
    let detector = pii.detector("test", "1").unwrap();
    for &(start, end, kind) in mentions {
        pii.push(PiiMention {
            span: Span::new(start as u32, end as u32),
            kind,
            detector,
            evidence: &[PiiEvidence::Shape],
            score: None,
            normalized: &text[start..end],
        });
    }
    pii
}

fn replacements(r: &PiiRewrite) -> Vec<&str> {
    r.labels().mentions().map(|m| &r.text()[m.span.start as usize..m.span.end as usize]).collect()
}

fn gcm(key: &[u8], context: &[u8]) -> AesGcmEncryptor {
    AesGcmEncryptor::new(key).unwrap().with_context(context)
}

fn ff1(key: &[u8], tweak: &[u8]) -> Ff1Encryptor {
    Ff1Encryptor::new(key, Ff1Alphabet::Digits).unwrap().with_tweak(tweak)
}

fn hash(policy: RewritePolicy) -> u64 {
    RedactedView::new(policy).config_hash()
}

#[test]
fn a_policy_routes_kinds_to_the_encryptors_and_each_replacement_decrypts() {
    let text = "Mail ann@example.com, SSN 123-45-6789, call 555-0100.";
    let email = (5, 20);
    let ssn = (26, 37);
    let phone = (44, 52);
    assert_eq!(&text[email.0..email.1], "ann@example.com");
    assert_eq!(&text[ssn.0..ssn.1], "123-45-6789");
    assert_eq!(&text[phone.0..phone.1], "555-0100");
    let pii = layer(
        text,
        &[(email.0, email.1, PiiKind::Email), (ssn.0, ssn.1, PiiKind::UsSsn), (phone.0, phone.1, PiiKind::Phone)],
    );
    let aes = gcm(&KEY, b"doc-7");
    let fpe = ff1(&KEY, b"ssn");
    let policy = RewritePolicy::new(MaskPolicy::new('#').unwrap())
        .with(PiiKind::Email, aes.clone())
        .with(PiiKind::UsSsn, fpe.clone())
        .with_revision(3);
    let r = policy.rewrite(&source_view(text), &pii).unwrap();
    let got = replacements(&r);
    assert_eq!(aes.decrypt(got[0]).unwrap(), ("EMAIL".to_string(), "ann@example.com".to_string()));
    assert_ne!(got[1], "123-45-6789");
    assert_eq!(got[1].len(), 11);
    assert_eq!((&got[1][3..4], &got[1][6..7]), ("-", "-"));
    assert_eq!(fpe.decrypt(got[1]).unwrap(), "123-45-6789");
    assert_eq!(got[2], "########");
    assert_eq!(r.edits().strategy, [PiiStrategyKind::Aes256Gcm, PiiStrategyKind::Ff1, PiiStrategyKind::Mask]);
    assert!(!r.text().contains("ann@") && !r.text().contains("6789"), "{}", r.text());
}

#[test]
fn the_strategy_arms_report_their_kind_and_rewrite_alone() {
    let text = "SSN 123-45-6789";
    let pii = layer(text, &[(4, 15, PiiKind::UsSsn)]);
    for (strategy, kind) in [
        (RewriteStrategy::from(gcm(&KEY, b"")), PiiStrategyKind::Aes256Gcm),
        (RewriteStrategy::from(ff1(&KEY, b"")), PiiStrategyKind::Ff1),
    ] {
        assert_eq!(strategy.kind(), kind);
        assert!(!strategy.refuses_overlaps());
        let r = strategy.rewrite(&source_view(text), &pii).unwrap();
        assert_eq!(r.edits().strategy, [kind]);
        assert!(!r.text().contains("6789"), "{}", r.text());
    }
}

#[test]
fn a_value_ff1_cannot_take_fails_the_rewrite_and_names_no_value() {
    let text = "Mail ann@example.com";
    let pii = layer(text, &[(5, 20, PiiKind::Email)]);
    let policy = RewritePolicy::new(ff1(&KEY, b""));
    match policy.rewrite(&source_view(text), &pii) {
        Err(RewriteError::Encrypt(span, why)) => {
            assert_eq!(span, Span::new(5, 20));
            assert!(!why.contains("ann") && !why.contains("example"), "{why}");
        }
        Err(other) => panic!("expected an encryption error, got {other:?}"),
        Ok(r) => panic!("expected an encryption error, got {}", r.text()),
    }
}

#[test]
fn overlapping_mentions_encrypt_the_merged_surface_once() {
    let text = "id 123456789012";
    let pii = layer(text, &[(3, 15, PiiKind::UsSsn), (3, 12, PiiKind::Phone)]);
    let aes = gcm(&KEY, b"");
    let policy = RewritePolicy::new(aes.clone());
    let r = policy.rewrite(&source_view(text), &pii).unwrap();
    let got = replacements(&r);
    assert_eq!(got.len(), 1, "{}", r.text());
    assert_eq!(aes.decrypt(got[0]).unwrap().1, "123456789012");
}

#[test]
fn equality_and_the_config_hash_follow_key_context_and_tweak() {
    // A clone draws its own nonce seed but decrypts the same tokens, so it
    // is the same strategy.
    let a = gcm(&KEY, b"ctx");
    assert_eq!(RewriteStrategy::from(a.clone()), RewriteStrategy::from(gcm(&KEY, b"ctx")));
    assert_ne!(RewriteStrategy::from(a.clone()), RewriteStrategy::from(gcm(&OTHER_KEY, b"ctx")));
    assert_ne!(RewriteStrategy::from(a.clone()), RewriteStrategy::from(gcm(&KEY, b"ctx2")));
    assert_eq!(RewriteStrategy::from(ff1(&KEY, b"t")), RewriteStrategy::from(ff1(&KEY, b"t").clone()));
    assert_ne!(RewriteStrategy::from(ff1(&KEY, b"t")), RewriteStrategy::from(ff1(&OTHER_KEY, b"t")));

    let base = hash(RewritePolicy::new(a.clone()));
    assert_eq!(base, hash(RewritePolicy::new(gcm(&KEY, b"ctx"))));
    for other in [
        hash(RewritePolicy::new(gcm(&OTHER_KEY, b"ctx"))),
        hash(RewritePolicy::new(gcm(&KEY, b"ctx2"))),
        hash(RewritePolicy::new(gcm(&KEY, b"ctx").refusing_overlaps())),
        hash(RewritePolicy::new(ff1(&KEY, b"ctx"))),
    ] {
        assert_ne!(base, other);
    }
    let f = hash(RewritePolicy::new(ff1(&KEY, b"t")));
    assert_ne!(f, hash(RewritePolicy::new(ff1(&KEY, b"u"))));
    assert_ne!(
        f,
        hash(RewritePolicy::new(Ff1Encryptor::new(&KEY, Ff1Alphabet::Alphanumeric).unwrap().with_tweak(b"t")))
    );
    // The context and tweak are framed by their length.
    assert_ne!(
        hash(RewritePolicy::new(Ff1Encryptor::new(&KEY, Ff1Alphabet::Digits).unwrap().with_tweak(b"\x00"))),
        hash(RewritePolicy::new(Ff1Encryptor::new(&KEY, Ff1Alphabet::Digits).unwrap().with_tweak(b""))),
    );
}

#[test]
fn the_policy_debug_output_carries_no_key() {
    let policy = RewritePolicy::new(gcm(&KEY, b"ctx")).with(PiiKind::UsSsn, ff1(&KEY, b"t"));
    let debug = format!("{policy:?}");
    for b in ["5a5a", "90, 90", "[90"] {
        assert!(!debug.contains(b), "{debug}");
    }
    assert!(debug.contains("AesGcmEncryptor") && debug.contains("Ff1Encryptor"), "{debug}");
}
