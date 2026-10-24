//! Caller-defined [`CustomExtractor`]s and [`PiiKind::Custom`] kinds:
//! detection with custom checksums and confidences, composition with
//! built-in packs, in-place and view masking, pseudonymization, HMAC
//! tokenization, audit reporting, and binary graph round-tripping.

mod support;

use support::*;
use turbo_analysis::export::{DocumentGraph, export, import};
use turbo_analysis::text::Side;
use turbo_analysis::{Analyzer, Annotator, Document, Layer, Provenance, Span, ViewId};
use turbo_analysis_layers::export::codecs;
use turbo_pii::checksum::luhn;
use turbo_pii::rewrite::{
    HmacTokenizer, MASKED, MaskPolicy, MaskedView, PiiAuditReport, Pseudonymizer, mask_all, mask_by_kind,
};
use turbo_pii::text::{on_end, on_number_start, on_word_start};
use turbo_pii::{
    CustomExtractor, Extractor, Pack, Pii, PiiConfidence, PiiDetector, PiiKind, PiiKinds, PiiMention, PiiSink, Scanner,
    ScannerError, SinkError,
};

const ACCOUNT_KIND: PiiKind = PiiKind::custom("account-number");
const PERSONNUMMER_KIND: PiiKind = PiiKind::custom("se-personnummer");
const BADGE_KIND: PiiKind = PiiKind::custom("employee-badge");

const KEY: &[u8] = b"0123456789abcdef0123456789abcdef";

/// Internal account number extractor: `ACCT-dddd-dddd` where the 8 digits
/// pass a Luhn check digit (`PiiConfidence::Checksum`), normalized to
/// `ACCTdddddddd`.
struct AccountExtractor {
    prefix: &'static [u8],
}

impl CustomExtractor for AccountExtractor {
    fn name(&self) -> &'static str {
        "AccountPiiExtractor"
    }

    fn kinds(&self) -> &[PiiKind] {
        const KINDS: [PiiKind; 1] = [ACCOUNT_KIND];
        &KINDS
    }

    fn scan(&self, text: &str, sink: &mut PiiSink<'_>) {
        let b = text.as_bytes();
        let plen = self.prefix.len();
        let total = plen + 1 + 4 + 1 + 4;
        let mut i = 0;
        while i + total <= b.len() {
            if !on_word_start(text, i)
                || !b[i..i + plen].eq_ignore_ascii_case(self.prefix)
                || b[i + plen] != b'-'
                || b[i + plen + 5] != b'-'
                || !on_end(text, i + total)
            {
                i += 1;
                continue;
            }
            let d1 = &b[i + plen + 1..i + plen + 5];
            let d2 = &b[i + plen + 6..i + total];
            if !d1.iter().all(u8::is_ascii_digit) || !d2.iter().all(u8::is_ascii_digit) {
                i += 1;
                continue;
            }
            let mut digits = [0u8; 8];
            digits[..4].copy_from_slice(d1);
            digits[4..].copy_from_slice(d2);
            if luhn::valid(&digits) {
                let end = i + total;
                sink.add_with(Span::new(i as u32, end as u32), ACCOUNT_KIND, PiiConfidence::Checksum, |norm| {
                    norm.push_str("ACCT");
                    for &d in &digits {
                        norm.push(d as char);
                    }
                });
                i = end;
            } else {
                i += 1;
            }
        }
    }
}

/// Swedish national identity number (`personnummer`, `YYYYMMDD-NNNN` or
/// `YYYYMMDD+NNNN`), 12 digits whose 10-digit `YYMMDDNNNN` tail passes
/// the Skatteverket Luhn check digit and calendar bounds.
fn scan_se_personnummer(text: &str, sink: &mut PiiSink<'_>) {
    const LEN: usize = 13;
    let b = text.as_bytes();
    let mut i = 0;
    while i + LEN <= b.len() {
        if !b[i].is_ascii_digit() || !on_number_start(text, i) || !on_end(text, i + LEN) {
            i += 1;
            continue;
        }
        let sep = b[i + 8];
        let date_part = &b[i..i + 8];
        let serial_part = &b[i + 9..i + LEN];
        if !matches!(sep, b'-' | b'+')
            || !matches!(&date_part[..2], b"19" | b"20")
            || !date_part.iter().all(u8::is_ascii_digit)
            || !serial_part.iter().all(u8::is_ascii_digit)
        {
            i += 1;
            continue;
        }
        let month = (date_part[4] - b'0') * 10 + (date_part[5] - b'0');
        let day = (date_part[6] - b'0') * 10 + (date_part[7] - b'0');
        if !(1..=12).contains(&month) || !(1..=31).contains(&day) {
            i += 1;
            continue;
        }
        let mut luhn_digits = [0u8; 10];
        luhn_digits[..6].copy_from_slice(&date_part[2..8]);
        luhn_digits[6..].copy_from_slice(serial_part);
        if luhn::valid(&luhn_digits) {
            let end = i + LEN;
            sink.add_with(Span::new(i as u32, end as u32), PERSONNUMMER_KIND, PiiConfidence::Checksum, |norm| {
                for &d in date_part.iter().chain(serial_part.iter()) {
                    norm.push(d as char);
                }
            });
            i = end;
        } else {
            i += 1;
        }
    }
}

/// Labelled employee badge (`badge: B-dddd` where the 4 digits sum to 20).
fn scan_badge(text: &str, sink: &mut PiiSink<'_>) {
    let b = text.as_bytes();
    let label = b"badge: ";
    let total = label.len() + 6;
    let mut i = 0;
    while i + total <= b.len() {
        if on_word_start(text, i)
            && b[i..i + label.len()].eq_ignore_ascii_case(label)
            && b[i + label.len()] == b'B'
            && b[i + label.len() + 1] == b'-'
            && b[i + label.len() + 2..i + total].iter().all(u8::is_ascii_digit)
            && on_end(text, i + total)
        {
            let start = i + label.len();
            let end = i + total;
            let sum: u32 = b[start + 2..end].iter().map(|&d| u32::from(d - b'0')).sum();
            let confidence = if sum == 20 { PiiConfidence::LabelledChecksum } else { PiiConfidence::Format };
            sink.add_text(Span::new(start as u32, end as u32), BADGE_KIND, confidence);
            i = end;
        } else {
            i += 1;
        }
    }
}

fn custom_detector() -> PiiDetector {
    PiiDetector::pack(Pack::Contact)
        .with_custom(AccountExtractor { prefix: b"acct" })
        .with_custom_fn("SePersonnummerExtractor", PERSONNUMMER_KIND, scan_se_personnummer)
        .with_scanner(Scanner::custom_fn("BadgeExtractor", BADGE_KIND, scan_badge))
}

#[test]
fn caller_extractors_find_checksum_and_confidence_alongside_builtin_packs() {
    let d = custom_detector();
    assert!(luhn::valid(b"12345674"));
    assert!(!luhn::valid(b"12345670"));
    assert!(luhn::valid(b"8112189876"));
    assert!(!luhn::valid(b"8112189870"));

    let text = "Customer jane@example.com holds ACCT-1234-5674 (not ACCT-1234-5670), \
                personnummer 19811218-9876 (not 19811218-9870), badge: B-5555 and badge: B-1234.";
    let pii = layer(&d, text);
    assert_eq!(pii.check(), Ok(()));

    let items: Vec<_> = pii.mentions().map(|m| (m.kind, m.confidence, m.normalized, &text[m.span.range()])).collect();
    assert_eq!(
        items,
        [
            (PiiKind::Email, PiiConfidence::Format, "jane@example.com", "jane@example.com"),
            (ACCOUNT_KIND, PiiConfidence::Checksum, "ACCT12345674", "ACCT-1234-5674"),
            (PERSONNUMMER_KIND, PiiConfidence::Checksum, "198112189876", "19811218-9876"),
            (BADGE_KIND, PiiConfidence::LabelledChecksum, "B-5555", "B-5555"),
            (BADGE_KIND, PiiConfidence::Format, "B-1234", "B-1234"),
        ]
    );
}

#[test]
fn caller_kinds_mask_pseudonymize_tokenize_and_audit() {
    let d = custom_detector();
    let text = "Åsa (jane@example.com) paid from ACCT-1234-5674 and acct-1234-5674 \
                with ID 19811218-9876 and badge: B-5555.";
    let pii = layer(&d, text);
    assert_eq!(pii.check(), Ok(()));

    // In-place masking: mask_all and mask_by_kind preserve UTF-8 byte length.
    let all_masked = mask_all(text, &pii, '*').unwrap();
    assert_eq!(all_masked.len(), text.len());
    assert!(!all_masked.contains("1234-5674") && !all_masked.contains("19811218") && !all_masked.contains("B-5555"));

    let by_kind_masked = mask_by_kind(text, &pii, '*').unwrap();
    assert_eq!(by_kind_masked.len(), text.len());
    // Email keeps format (`****@*******.***`), custom kinds default to whole masking.
    assert!(by_kind_masked.contains("****@*******.***"));
    assert!(by_kind_masked.contains("paid from ************** and **************"));

    // Pipeline with MaskedView and custom per-kind policy (keep format + last 4 digits on account-number).
    let acct_policy = MaskPolicy::new('*').unwrap().keeping_format().keeping_trailing(4);
    let masked_view = MaskedView::by_kind('*').unwrap().with(ACCOUNT_KIND, acct_policy);
    assert_eq!(masked_view.policy(ACCOUNT_KIND), acct_policy);
    assert_eq!(masked_view.policy(PERSONNUMMER_KIND), MaskPolicy::new('*').unwrap());

    let analyzer = Analyzer::builder().add(d.clone()).step(masked_view.bound()).build().unwrap();
    let mut docs = vec![Document::new(text)];
    for res in analyzer.run(&mut docs) {
        res.unwrap();
    }
    let doc = &docs[0];
    let view = doc.get_key(&MASKED).unwrap();
    assert!(view.text().contains("paid from ****-****-5674 and ****-****-5674"));
    assert!(view.text().contains("with ID ************* and badge: ******."));
    // Alignment maps every mention span and multi-byte character ("Åsa") both ways.
    for m in doc.get::<Pii>().unwrap().mentions() {
        let child = doc.from_source(view.id(), m.span);
        assert_eq!(child, m.span);
        assert_eq!(doc.to_source(view.id(), child), m.span);
    }
    assert_eq!(view.alignment().to_parent(0, Side::Start), 0);
    assert_eq!(view.alignment().to_child(2, Side::End), 2);

    // Pseudonymizer numbers each custom kind independently and reuses the label for equal normalized values.
    let pseudo = Pseudonymizer::new("[", "]").rewrite(text, &pii).unwrap();
    assert_eq!(
        pseudo.text(),
        "Åsa ([EMAIL-1]) paid from [ACCOUNT-NUMBER-1] and [ACCOUNT-NUMBER-1] \
         with ID [SE-PERSONNUMMER-1] and badge: [EMPLOYEE-BADGE-1]."
    );
    assert_eq!(pseudo.labels().check(), Ok(()));
    let label_kinds: Vec<PiiKind> = pseudo.labels().mentions().map(|m| m.kind).collect();
    assert_eq!(label_kinds, [PiiKind::Email, ACCOUNT_KIND, ACCOUNT_KIND, PERSONNUMMER_KIND, BADGE_KIND]);

    // HmacTokenizer and PiiAuditReport include caller kinds and never leak raw values.
    let tokenizer = HmacTokenizer::new(KEY).unwrap();
    let acct_token = tokenizer.token(ACCOUNT_KIND, "ACCT12345674").unwrap();
    assert!(acct_token.starts_with("ACCOUNT-NUMBER-"));
    let hmac_rewrite = tokenizer.rewrite(text, &pii).unwrap();
    assert!(hmac_rewrite.text().contains(&acct_token));

    let report = PiiAuditReport::of(&pii, &tokenizer, PiiAuditReport::DEFAULT_SAMPLES);
    assert_eq!(report.total, 5);
    let acct_report = report.kind(ACCOUNT_KIND).expect("account-number in audit report");
    assert_eq!((acct_report.mentions, acct_report.distinct), (2, 1));
    assert_eq!(acct_report.samples, [acct_token]);
    let pnr_report = report.kind(PERSONNUMMER_KIND).expect("se-personnummer in audit report");
    assert_eq!((pnr_report.mentions, pnr_report.distinct), (1, 1));
    let rendered = report.to_string();
    assert!(rendered.contains("account-number: 2 mentions, 1 distinct"));
    assert!(rendered.contains("se-personnummer: 1 mentions, 1 distinct"));
    assert!(rendered.contains("employee-badge: 1 mentions, 1 distinct"));
    for raw in ["1234-5674", "12345674", "19811218", "B-5555", "jane@example.com"] {
        assert!(!rendered.contains(raw), "leaked {raw} in:\n{rendered}");
    }
}

#[test]
fn custom_kind_ordering_export_round_trip_and_sink_validation() {
    // Built-in kinds outrank custom kinds on the same span; custom kinds order by name.
    assert_eq!(PiiKind::custom("email"), PiiKind::Email);
    assert!(!PiiKind::Email.is_custom());
    assert!(ACCOUNT_KIND.is_custom());
    assert_eq!(ACCOUNT_KIND.rank(), 20);
    assert_eq!(ACCOUNT_KIND.bit(), 0);
    assert!(PiiKind::Phone < ACCOUNT_KIND);
    assert!(ACCOUNT_KIND < PERSONNUMMER_KIND);

    // Binary graph export/import preserves custom kinds.
    let mut doc = Document::new("ACCT-1234-5674 and 811218-9876");
    let mut pii = Pii::new(ViewId::SOURCE);
    pii.push(PiiMention {
        span: Span::new(0, 14),
        kind: ACCOUNT_KIND,
        confidence: PiiConfidence::Checksum,
        normalized: "ACCT12345674",
    });
    pii.push(PiiMention {
        span: Span::new(19, 30),
        kind: PERSONNUMMER_KIND,
        confidence: PiiConfidence::Checksum,
        normalized: "8112189876",
    });
    doc.add(&turbo_analysis::Key::DEFAULT, pii, Provenance::by("test")).unwrap();
    let bytes = export(&doc, &codecs()).unwrap().to_bytes().unwrap();
    let back = import(&DocumentGraph::from_bytes(&bytes).unwrap(), &codecs()).unwrap();
    let back_pii = back.get::<Pii>().unwrap();
    assert_eq!(back_pii.kind, [ACCOUNT_KIND, PERSONNUMMER_KIND]);
    assert_eq!(back_pii.check(), Ok(()));

    // SinkError validation on malformed spans, empty kinds, and empty normalized values.
    let check_sink = |text: &str, sink: &mut PiiSink<'_>| {
        assert_eq!(sink.text(), text);
        assert_eq!(
            sink.try_add(Span::new(0, 0), ACCOUNT_KIND, PiiConfidence::Format, "x"),
            Err(SinkError::Span(Span::new(0, 0)))
        );
        assert_eq!(
            sink.try_add(Span::new(0, 1), ACCOUNT_KIND, PiiConfidence::Format, "x"),
            Err(SinkError::Span(Span::new(0, 1)))
        );
        assert_eq!(
            sink.try_add(Span::new(0, 99), ACCOUNT_KIND, PiiConfidence::Format, "x"),
            Err(SinkError::Span(Span::new(0, 99)))
        );
        assert_eq!(
            sink.try_add(Span::new(0, 2), PiiKind::Custom(""), PiiConfidence::Format, "x"),
            Err(SinkError::EmptyKind)
        );
        assert_eq!(
            sink.try_add(Span::new(0, 2), ACCOUNT_KIND, PiiConfidence::Format, ""),
            Err(SinkError::EmptyValue(Span::new(0, 2)))
        );
        assert_eq!(
            sink.try_add_with(Span::new(0, 2), ACCOUNT_KIND, PiiConfidence::Format, |_| {}),
            Err(SinkError::EmptyValue(Span::new(0, 2)))
        );
    };
    let validator = PiiDetector::new(vec![Scanner::custom_fn("Validator", ACCOUNT_KIND, check_sink)]);
    let _ = validator.layer("éok", ViewId::SOURCE);

    // Scanner::with_kinds refuses custom extractors; config_hash distinguishes custom scanners.
    let custom_ext = Extractor::Custom("AccountPiiExtractor");
    assert_eq!(Scanner::with_kinds(custom_ext, PiiKinds::EMPTY), Err(ScannerError::NoKinds(custom_ext)));
    assert_eq!(
        Scanner::with_kinds(custom_ext, PiiKinds::of(&[PiiKind::Email])),
        Err(ScannerError::Foreign(custom_ext, PiiKind::Email))
    );
    let d1 = PiiDetector::cursor().with_custom_fn("A", ACCOUNT_KIND, scan_se_personnummer);
    let d2 = PiiDetector::cursor().with_custom_fn("B", ACCOUNT_KIND, scan_se_personnummer);
    assert_ne!(d1.config_hash(), PiiDetector::cursor().config_hash());
    assert_ne!(d1.config_hash(), d2.config_hash());
}
