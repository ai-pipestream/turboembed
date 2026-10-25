//! `HmacTokenizerTest`, `HmacTokenizerIdentityTest`,
//! `PiiAuditReportTest` and `PiiUsageExampleTest`, over the typed layer.

mod support;

use support::*;
use turbo_analysis::Layer;
use turbo_pii::rewrite::{HmacTokenizer, PiiAuditReport, Pseudonymizer, TokenizerError};
use turbo_pii::{Pack, PiiDetector, PiiKind};

const KEY: &[u8] = b"0123456789abcdef0123456789abcdef";
const OTHER_KEY: &[u8] = b"fedcba9876543210fedcba9876543210";
const TEXT: &str = "Contact jane@example.com, call (555) 123-4567, or charge card 4111 1111 1111 1111.";

fn cursor() -> PiiDetector {
    PiiDetector::cursor()
}

fn tokenizer() -> HmacTokenizer {
    HmacTokenizer::new(KEY).unwrap()
}

#[test]
fn hmac_tokens() {
    let t = tokenizer();
    let token = t.token(PiiKind::Email, "jane@example.com").unwrap();
    // HMAC-SHA256 of "email", NUL, the value, from an independent implementation.
    assert_eq!(token, "EMAIL-ae13520d188358c7");
    assert_eq!(t.token(PiiKind::UsItin, "900-50-0000").unwrap(), "US-ITIN-3936bc173d71bdc2");
    assert_eq!(t.token(PiiKind::Card, "4111111111111111").unwrap(), "CARD-f04eb250c58c67b2");
    let zero = HmacTokenizer::with_length(&[0; 32], 64).unwrap();
    assert_eq!(
        zero.token(PiiKind::Email, "value").unwrap(),
        "EMAIL-bc581ed249cfacd301ceb83265bbe575acad299c802fd8e208ab051c9764b2f8"
    );
    assert_eq!(
        zero.token(PiiKind::Email, "😀𝕒").unwrap(),
        "EMAIL-8b9f32d3756cb0c8665906e65f6f902ee3862da64caa3f7634855dc537ecb7f2"
    );
    for (kind, prefix) in [
        (PiiKind::Email, "EMAIL-"),
        (PiiKind::Phone, "PHONE-"),
        (PiiKind::UsSsn, "US-SSN-"),
        (PiiKind::AwsAccessKey, "AWS-ACCESS-KEY-"),
    ] {
        assert!(t.token(kind, "value").unwrap().starts_with(prefix));
    }
    assert_eq!(
        HmacTokenizer::new(KEY).unwrap().token(PiiKind::Email, "a@b.com"),
        HmacTokenizer::new(KEY.to_vec().as_slice()).unwrap().token(PiiKind::Email, "a@b.com")
    );
    assert_ne!(
        t.token(PiiKind::Email, "a@b.com"),
        HmacTokenizer::new(OTHER_KEY).unwrap().token(PiiKind::Email, "a@b.com")
    );
    assert_ne!(t.token(PiiKind::Email, "a@b.com"), t.token(PiiKind::Email, "c@d.com"));
    assert_ne!(t.token(PiiKind::Email, "user210255@example.com"), t.token(PiiKind::Email, "user217053@example.com"));
    let phone = t.token(PiiKind::Phone, "5551234567").unwrap();
    let card = t.token(PiiKind::Card, "5551234567").unwrap();
    assert_ne!(phone.rsplit('-').next(), card.rsplit('-').next());
    let distinct: std::collections::HashSet<String> =
        (0..2000).map(|i| t.token(PiiKind::Email, &format!("user{i}@example.com")).unwrap()).collect();
    assert_eq!(distinct.len(), 2000);
}

#[test]
fn hmac_lengths_and_keys() {
    for length in [4, 5, 8, 16, 63, 64] {
        let token = HmacTokenizer::with_length(KEY, length).unwrap().token(PiiKind::Email, "jane@example.com").unwrap();
        assert_eq!(token.len(), "EMAIL-".len() + length);
        assert!(token["EMAIL-".len()..].bytes().all(|c| matches!(c, b'0'..=b'9' | b'a'..=b'f')));
    }
    let short = HmacTokenizer::with_length(KEY, 8).unwrap().token(PiiKind::Email, "a@b.com").unwrap();
    assert!(t16().token(PiiKind::Email, "a@b.com").unwrap().starts_with(&short));
    for length in [0, 1, 3, 65, 128] {
        assert_eq!(HmacTokenizer::with_length(KEY, length).map(|_| ()), Err(TokenizerError::Length(length)));
    }
    for length in [0, 1, 16, 31] {
        assert_eq!(HmacTokenizer::new(&vec![0; length]).map(|_| ()), Err(TokenizerError::ShortKey(length)));
    }
    assert!(HmacTokenizer::new(&[0; 32]).is_ok());
    let mut key = KEY.to_vec();
    let t = HmacTokenizer::new(&key).unwrap();
    let before = t.token(PiiKind::Email, "a@b.com").unwrap();
    key.fill(0);
    assert_eq!(t.token(PiiKind::Email, "a@b.com").unwrap(), before);
    assert_eq!(t.token(PiiKind::Email, ""), Err(TokenizerError::EmptyValue));
    assert!(!format!("{t:?}").contains("0123456789abcdef"));
}

fn t16() -> HmacTokenizer {
    HmacTokenizer::with_length(KEY, 16).unwrap()
}

#[test]
fn hmac_rewrites_agree_across_texts() {
    let t = tokenizer();
    let token = t.token(PiiKind::Email, "jane@example.com").unwrap();
    let text = "mail jane@example.com or jane@example.com";
    assert_eq!(t.rewrite(text, &layer(&cursor(), text)).unwrap().text(), format!("mail {token} or {token}"));
    for (text, want) in
        [("From jane@example.com", format!("From {token}")), ("To jane@example.com", format!("To {token}"))]
    {
        assert_eq!(t.rewrite(text, &layer(&cursor(), text)).unwrap().text(), want);
    }
}

const AUDIT_TEXT: &str = "jane@example.com wrote, bob@example.com replied, then jane@example.com again, card 4111111111111111, call (555) 123-4567";

fn report(text: &str) -> PiiAuditReport {
    PiiAuditReport::of(&layer(&cursor(), text), &tokenizer(), PiiAuditReport::DEFAULT_SAMPLES)
}

#[test]
fn audit_counts() {
    let r = report(AUDIT_TEXT);
    assert_eq!(r.total, 5);
    let counts: Vec<(PiiKind, usize, usize)> = r.kinds.iter().map(|k| (k.kind, k.mentions, k.distinct)).collect();
    assert_eq!(counts, [(PiiKind::Card, 1, 1), (PiiKind::Email, 3, 2), (PiiKind::Phone, 1, 1)]);
    assert!(r.kind(PiiKind::Iban).is_none());
    let t = tokenizer();
    assert_eq!(
        r.kind(PiiKind::Email).unwrap().samples,
        [t.token(PiiKind::Email, "jane@example.com").unwrap(), t.token(PiiKind::Email, "bob@example.com").unwrap()]
    );
    let rewritten = t.rewrite(AUDIT_TEXT, &layer(&cursor(), AUDIT_TEXT)).unwrap().into_text();
    for sample in &r.kind(PiiKind::Email).unwrap().samples {
        assert!(rewritten.contains(sample.as_str()));
    }
    let rendered = r.to_string();
    for value in ["jane", "example.com", "4111111111111111", "555", "123-4567", "bob"] {
        assert!(!rendered.contains(value), "{value}");
    }
    let lines: Vec<&str> = rendered.split('\n').collect();
    assert_eq!(lines.len(), 3);
    assert!(lines[1].starts_with("email: 3 mentions, 2 distinct"), "{}", lines[1]);
    assert_eq!(report(AUDIT_TEXT).to_string(), rendered);
    assert_eq!(report("nothing to see here").to_string(), "no pii found");
    assert_eq!(report("nothing to see here").total, 0);
    let other = PiiAuditReport::of(&layer(&cursor(), AUDIT_TEXT), &HmacTokenizer::new(OTHER_KEY).unwrap(), 3);
    assert_ne!(other.kind(PiiKind::Email).unwrap().samples, r.kind(PiiKind::Email).unwrap().samples);
}

#[test]
fn audit_samples_and_distinct_values() {
    let text = "a1@example.com b1@example.com c1@example.com d1@example.com e1@example.com";
    for limit in [0, 1, 2, 3, 10] {
        let r = PiiAuditReport::of(&layer(&cursor(), text), &tokenizer(), limit);
        let k = r.kind(PiiKind::Email).unwrap();
        assert_eq!((k.samples.len(), k.distinct), (limit.min(5), 5));
    }
    let short = HmacTokenizer::with_length(KEY, 4).unwrap();
    assert_eq!(short.token(PiiKind::Email, "u165@example.com"), short.token(PiiKind::Email, "u488@example.com"));
    let pii = manual(&[(0, 1, PiiKind::Email, "u165@example.com"), (2, 3, PiiKind::Email, "u488@example.com")]);
    assert_eq!(PiiAuditReport::of(&pii, &short, 3).kind(PiiKind::Email).unwrap().distinct, 2);
    let text = "key AKIAIOSFODNN7EXAMPLE host 10.0.0.7 host 10.0.0.7";
    let r = PiiAuditReport::of(&layer(&wide(), text), &tokenizer(), 3);
    assert_eq!(r.kind(PiiKind::AwsAccessKey).unwrap().mentions, 1);
    let ip = r.kind(PiiKind::Ipv4).unwrap();
    assert_eq!((ip.mentions, ip.distinct), (2, 1));
    let text = "https://john:pass@example.com/path";
    let r = PiiAuditReport::of(&layer(&wide().keeping_nested(), text), &tokenizer(), 3);
    assert_eq!(r.total, 2);
    assert_eq!(
        r.kinds.iter().map(|k| (k.kind, k.mentions)).collect::<Vec<_>>(),
        [(PiiKind::Email, 1), (PiiKind::UrlCredential, 1)]
    );
}

#[test]
fn audit_of_a_usage_example() {
    let r = PiiAuditReport::of(&layer(&cursor(), TEXT), &tokenizer(), 3);
    assert_eq!(r.total, 3);
    assert_eq!(r.kind(PiiKind::Card).unwrap().mentions, 1);
    assert!(!r.to_string().contains("jane") && !r.to_string().contains("4111"));
    let first = layer(&cursor(), "From jane@example.com");
    let stable = tokenizer().token(PiiKind::Email, "jane@example.com").unwrap();
    let r = PiiAuditReport::of(&first, &tokenizer(), 3);
    assert_eq!(r.kind(PiiKind::Email).unwrap().samples, [stable]);
}

#[test]
fn opt_in_packs_find_what_the_default_does_not() {
    let text = "Deploy from 10.0.0.7 using AKIAIOSFODNN7EXAMPLE, ask jane@example.com.";
    assert_eq!(kinds(&wide(), text), ["ipv4", "aws-access-key", "email"]);
    assert_eq!(kinds(&cursor(), text), ["email"]);
    let _ = pack(Pack::Contact);
}

#[test]
fn rewrites_on_every_thread() {
    let p = Pseudonymizer::new("[", "]");
    let t = HmacTokenizer::with_length(&[0; 32], 64).unwrap();
    std::thread::scope(|s| {
        for worker in 0..8 {
            let (p, t) = (&p, &t);
            s.spawn(move || {
                let value = format!("private-value{worker}😀");
                let pii = manual(&[(0, 1, PiiKind::Email, &value)]);
                let want = t.token(PiiKind::Email, &value).unwrap();
                for _ in 0..32 {
                    assert_eq!(t.rewrite("x", &pii).unwrap().text(), want);
                    assert_eq!(p.rewrite("x", &pii).unwrap().text(), "[EMAIL-1]");
                    assert_eq!(PiiAuditReport::of(&pii, t, 3).kinds[0].samples, std::slice::from_ref(&want));
                }
            });
        }
    });
}

/// Every text of the `pii` input set, which holds sample values of every
/// kind the packs find.
fn sample_texts() -> Vec<(String, String)> {
    let dir = turbo_goldens::data_dir().join("inputs/pii");
    let mut texts: Vec<(String, String)> = std::fs::read_dir(&dir)
        .unwrap_or_else(|e| panic!("{}: {e}", dir.display()))
        .map(|e| {
            let path = e.unwrap().path();
            let id = path.file_stem().unwrap().to_string_lossy().into_owned();
            (id, std::fs::read_to_string(&path).unwrap())
        })
        .collect();
    texts.sort();
    texts
}

#[test]
fn no_value_of_any_pack_appears_in_the_audit_and_every_default_pack_rewrites() {
    let texts = sample_texts();
    assert!(texts.len() > 50, "only {} sample texts", texts.len());
    let mut kinds = std::collections::HashSet::new();
    for p in Pack::ALL {
        let detector = pack(p);
        let mut checked = 0;
        for (id, text) in &texts {
            let pii = layer(&detector, text);
            let pseudo = Pseudonymizer::default().rewrite(text, &pii).unwrap();
            assert_eq!(pseudo.labels().check(), Ok(()));
            let hmac = tokenizer().rewrite(text, &pii).unwrap();
            assert_eq!(hmac.labels().check(), Ok(()));
            let report = PiiAuditReport::of(&pii, &tokenizer(), usize::MAX);
            let shown = [report.to_string(), format!("{report:?}")];
            for m in pii.mentions() {
                let covered = &text[m.span.start as usize..m.span.end as usize];
                for value in [covered, m.normalized] {
                    for out in &shown {
                        assert!(!out.contains(value), "{} on {id}: {value:?} is in the audit:\n{out}", p.name());
                    }
                }
                kinds.insert(m.kind);
                checked += 1;
            }
            assert_eq!(report.total, pii.mentions().count());
        }
        assert!(checked > 0, "the {} pack found nothing in the sample texts", p.name());
    }
    assert_eq!(kinds.len(), PiiKind::ALL.len(), "kinds found: {kinds:?}");
}
