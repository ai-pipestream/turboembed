//! #1885: reversible protection behind the `reversible` feature. The NIST
//! SP 800-38G FF1 samples through `Ff1Encryptor`, the round trip of
//! `AesGcmEncryptor` and `Ff1Encryptor` for every built-in kind and a
//! custom one, and tokens that fail authentication once changed. The
//! cipher's own AES-256-GCM known-answer vectors are a unit test of the
//! encryptor's module.

#![cfg(feature = "reversible")]

use turbo_analysis::{Span, ViewId};
use turbo_analysis_layers::{Pii, PiiEvidence, PiiKind, PiiMention, PiiStrategyKind};
use turbo_pii::rewrite::{AesGcmEncryptor, CryptError, Ff1Alphabet, Ff1Encryptor, PiiRewrite, source_view};

/// The AES-256 key of NIST SP 800-38G samples 7 to 9.
const NIST_KEY: [u8; 32] = [
    0x2B, 0x7E, 0x15, 0x16, 0x28, 0xAE, 0xD2, 0xA6, 0xAB, 0xF7, 0x15, 0x88, 0x09, 0xCF, 0x4F, 0x3C, 0xEF, 0x43, 0x59,
    0xD8, 0xD5, 0x80, 0xAA, 0x4F, 0x7F, 0x03, 0x6D, 0x6F, 0x04, 0xFC, 0x6A, 0x94,
];

const KEY: [u8; 32] = [0x5a; 32];

#[test]
fn ff1_meets_the_nist_aes_256_samples() {
    // Sample 9's numerals are 0-9 then a-z; this alphabet writes the
    // letters in capitals.
    for (alphabet, tweak, pt, ct) in [
        (Ff1Alphabet::Digits, &b""[..], "0123456789", "6657667009"),
        (Ff1Alphabet::Digits, b"9876543210", "0123456789", "1001623463"),
        (Ff1Alphabet::UpperAlphanumeric, b"7777pqrs777", "0123456789ABCDEFGHI", "XS8A0AZH2AVYALYZUWD"),
    ] {
        let ff1 = Ff1Encryptor::new(&NIST_KEY, alphabet).unwrap().with_tweak(tweak);
        assert_eq!(ff1.encrypt(pt).unwrap(), ct, "{pt}");
        assert_eq!(ff1.decrypt(ct).unwrap(), pt, "{ct}");
    }
}

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

/// The replacement text of each mention.
fn replacements(r: &PiiRewrite) -> Vec<&str> {
    r.labels().mentions().map(|m| &r.text()[m.span.start as usize..m.span.end as usize]).collect()
}

fn every_kind() -> (Pii, Vec<PiiKind>) {
    let mut kinds: Vec<PiiKind> = PiiKind::builtins().collect();
    let mut pii = Pii::new(ViewId::SOURCE);
    kinds.push(pii.custom_kinds.add("badge-number").unwrap());
    (pii, kinds)
}

#[test]
fn aes_gcm_tokens_decrypt_to_the_surface_for_every_kind() {
    let value = "Ab-12 345 6789 \u{e9}t\u{e9}";
    let gcm = AesGcmEncryptor::new(&KEY).unwrap().with_context(b"doc-7/body");
    let (kinds_layer, kinds) = every_kind();
    for kind in kinds {
        let text = format!("x {value} y");
        let mut pii = layer(&text, &[(2, 2 + value.len(), kind)]);
        pii.custom_kinds = kinds_layer.custom_kinds.clone();
        let r = gcm.rewrite(&source_view(&text), &pii).unwrap();
        let token = replacements(&r)[0];
        let name = pii.custom_kinds.name(kind).unwrap();
        assert_eq!(token.split_once(':').unwrap().0, name.to_ascii_uppercase(), "{name}");
        assert_eq!(gcm.decrypt(token).unwrap(), (name.to_ascii_uppercase(), value.to_string()), "{name}");
        assert_eq!(r.edits().strategy, [PiiStrategyKind::Aes256Gcm]);
        assert!(!r.text().contains("6789"), "{}", r.text());
    }
}

#[test]
fn ff1_keeps_the_separators_and_round_trips_for_every_kind() {
    let ff1 = Ff1Encryptor::new(&KEY, Ff1Alphabet::Alphanumeric).unwrap().with_tweak(b"ssn");
    let value = "123-45-6789 Ab";
    let (kinds_layer, kinds) = every_kind();
    for kind in kinds {
        let mut pii = layer(value, &[(0, value.len(), kind)]);
        pii.custom_kinds = kinds_layer.custom_kinds.clone();
        let r = ff1.rewrite(&source_view(value), &pii).unwrap();
        let out = replacements(&r)[0];
        assert_eq!(out.len(), value.len());
        for (a, b) in value.chars().zip(out.chars()) {
            assert_eq!(a.is_ascii_alphanumeric(), b.is_ascii_alphanumeric(), "{value} {out}");
            if !a.is_ascii_alphanumeric() {
                assert_eq!(a, b);
            }
        }
        assert_ne!(out, value);
        assert_eq!(ff1.decrypt(out).unwrap(), value);
        assert_eq!(r.edits().strategy, [PiiStrategyKind::Ff1]);
    }
    let digits = Ff1Encryptor::new(&KEY, Ff1Alphabet::Digits).unwrap();
    let card = "4111 1111 1111 1111";
    let out = digits.encrypt(card).unwrap();
    assert!(
        out.chars().zip(card.chars()).all(|(a, b)| a.is_ascii_digit() == b.is_ascii_digit() && (a == b || a != ' '))
    );
    assert_eq!(digits.decrypt(&out).unwrap(), card);
    assert_eq!(digits.encrypt(card).unwrap(), out, "FF1 is deterministic under one key and tweak");
    let tweaked = Ff1Encryptor::new(&KEY, Ff1Alphabet::Digits).unwrap().with_tweak(b"other");
    assert_ne!(tweaked.encrypt(card).unwrap(), out);
}

#[test]
fn a_changed_token_fails_authentication() {
    let gcm = AesGcmEncryptor::new(&KEY).unwrap().with_context(b"ctx");
    let token = gcm.token("email", "lena@example.org").unwrap();
    assert_ne!(gcm.token("email", "lena@example.org").unwrap(), token, "each token has a fresh nonce");
    let (tag, body) = token.split_once(':').unwrap();
    // Every character of the body changed in turn.
    for at in 0..body.len() {
        let mut b = body.as_bytes().to_vec();
        b[at] = if b[at] == b'A' { b'B' } else { b'A' };
        let changed = format!("{tag}:{}", String::from_utf8(b).unwrap());
        assert!(matches!(gcm.decrypt(&changed), Err(CryptError::Authentication | CryptError::Malformed)), "{changed}");
    }
    assert_eq!(gcm.decrypt(&format!("PHONE:{body}")), Err(CryptError::Authentication), "another kind");
    let other_context = AesGcmEncryptor::new(&KEY).unwrap().with_context(b"ctx2");
    assert_eq!(other_context.decrypt(&token), Err(CryptError::Authentication));
    let other_key = AesGcmEncryptor::new(&[0x5b; 32]).unwrap().with_context(b"ctx");
    assert_eq!(other_key.decrypt(&token), Err(CryptError::Authentication));
    assert_eq!(gcm.clone().decrypt(&token).unwrap(), ("EMAIL".to_string(), "lena@example.org".to_string()));
    for bad in ["", "EMAIL", "EMAIL:", "EMAIL:AAAA", "EMAIL:not base64!", &token[..token.len() - 1]] {
        assert!(matches!(gcm.decrypt(bad), Err(CryptError::Malformed | CryptError::Authentication)), "{bad}");
    }
}

#[test]
fn values_ff1_would_leave_in_the_clear_are_refused() {
    let digits = Ff1Encryptor::new(&KEY, Ff1Alphabet::Digits).unwrap();
    assert_eq!(digits.encrypt("12345"), Err(CryptError::TooFewNumerals { have: 5, need: 6 }));
    assert_eq!(digits.encrypt("AB123456"), Err(CryptError::OutsideAlphabet { at: 0 }));
    assert_eq!(digits.encrypt("123\u{0663}456"), Err(CryptError::OutsideAlphabet { at: 3 }));
    let upper = Ff1Encryptor::new(&KEY, Ff1Alphabet::UpperAlphanumeric).unwrap();
    assert_eq!(upper.encrypt("X12345a"), Err(CryptError::OutsideAlphabet { at: 6 }));
    assert_eq!(upper.encrypt("X12"), Err(CryptError::TooFewNumerals { have: 3, need: 4 }));
    assert_eq!(upper.encrypt(""), Err(CryptError::EmptyValue));
    let text = "pin 1234 here";
    let r = digits.rewrite(&source_view(text), &layer(text, &[(4, 8, PiiKind::UsSsn)]));
    assert!(matches!(r.err(), Some(turbo_pii::rewrite::RewriteError::Encrypt(s, _)) if s == Span::new(4, 8)));
    // The error of a value with a character outside the alphabet names its
    // offset, not the character.
    let text = "id Q12345678 end";
    let err = digits.rewrite(&source_view(text), &layer(text, &[(3, 12, PiiKind::UsSsn)])).err().unwrap();
    assert!(!err.to_string().contains('Q'), "{err}");
}

#[test]
fn keys_are_32_bytes_and_stay_out_of_debug() {
    assert_eq!(AesGcmEncryptor::new(&[1; 31]).err(), Some(CryptError::KeyLength(31)));
    assert_eq!(AesGcmEncryptor::new(&[1; 33]).err(), Some(CryptError::KeyLength(33)));
    assert_eq!(Ff1Encryptor::new(&[1; 16], Ff1Alphabet::Digits).err(), Some(CryptError::KeyLength(16)));
    let gcm = format!("{:?}", AesGcmEncryptor::new(&KEY).unwrap().with_context(b"secret-context"));
    let ff1 = format!("{:?}", Ff1Encryptor::new(&KEY, Ff1Alphabet::Digits).unwrap().with_tweak(b"secret-tweak"));
    for (debug, hidden) in [(&gcm, "secret-context"), (&ff1, "secret-tweak")] {
        assert!(!debug.contains("90") && !debug.contains("5a") && !debug.contains(hidden), "{debug}");
    }
}
