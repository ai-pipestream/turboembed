//! `MaskerTest`, `MaskerOverlapTest`, `MaskPolicyTest`,
//! `MaskPoliciesTest`, `PseudonymizerTest`, `HmacTokenizerTest`,
//! `HmacTokenizerIdentityTest`, `PiiRewriteTest`, `PiiAuditReportTest`
//! and `PiiUsageExampleTest`, over the typed layer.

mod support;

use support::*;
use turbo_analysis::Span;
use turbo_pii::rewrite::{
    HmacTokenizer, MaskError, MaskPolicy, PiiAuditReport, Pseudonymizer, RewriteError, TokenizerError, mask, mask_all,
    mask_by_kind,
};
use turbo_pii::{Pack, Pii, PiiDetector, PiiKind};

const KEY: &[u8] = b"0123456789abcdef0123456789abcdef";
const OTHER_KEY: &[u8] = b"fedcba9876543210fedcba9876543210";
const TEXT: &str = "Contact jane@example.com, call (555) 123-4567, or charge card 4111 1111 1111 1111.";

fn cursor() -> PiiDetector {
    PiiDetector::cursor()
}

fn tokenizer() -> HmacTokenizer {
    HmacTokenizer::new(KEY).unwrap()
}

fn whole(c: char) -> MaskPolicy {
    MaskPolicy::new(c).unwrap()
}

fn pseudonyms(text: &str) -> String {
    Pseudonymizer::new("", "").rewrite(text, &layer(&cursor(), text)).unwrap().into_text()
}

#[test]
fn mentions_in_the_usage_example() {
    assert_found(
        &cursor(),
        TEXT,
        &[(8, 24, "email", "jane@example.com"), (31, 45, "phone", "5551234567"), (62, 81, "card", "4111111111111111")],
    );
}

#[test]
fn masking_keeps_the_length() {
    let pii = layer(&cursor(), TEXT);
    let masked = mask_all(TEXT, &pii, '*').unwrap();
    assert_eq!(masked, "Contact ****************, call **************, or charge card *******************.");
    let receipt = whole('*').keeping_format().keeping_trailing(4);
    assert_eq!(
        mask(TEXT, pii.mentions().map(|m| (m.span, receipt))).unwrap(),
        "Contact ****@******e.com, call (***) ***-4567, or charge card **** **** **** 1111."
    );
    assert_eq!(
        mask_by_kind(TEXT, &pii, '*').unwrap(),
        "Contact ****@*******.***, call (***) ***-****, or charge card **** **** **** 1111."
    );
    let text = "Contact jane@example.com or call +44 20 7946 0958.";
    assert_eq!(
        mask_all(text, &layer(&cursor(), text), '*').unwrap(),
        "Contact **************** or call ****************."
    );
    let text = "Card 4111111111111111 12/26";
    assert_eq!(mask_all(text, &layer(&cursor(), text), '*').unwrap(), "Card **************** 12/26");
    let text = "Nothing sensitive here.";
    assert_eq!(mask_all(text, &layer(&cursor(), text), '*').unwrap(), text);
}

#[test]
fn several_span_sources_at_once() {
    let text = "Send the Widget Press manual to jane@example.com now.";
    let pii = layer(&cursor(), text);
    let spans = pii.mentions().map(|m| m.span).chain([Span::new(9, 21)]);
    assert_eq!(
        mask(text, spans.map(|s| (s, whole('#')))).unwrap(),
        "Send the ############ manual to ################ now."
    );
}

#[test]
fn overlapping_spans_mask_in_any_order() {
    let text = "card 4111111111111111 end";
    let p = whole('*').keeping_trailing(4);
    let (full, prefix) = (Span::new(5, 21), Span::new(5, 9));
    assert_eq!(mask(text, [(full, p), (prefix, p)]).unwrap(), "card ************1111 end");
    assert_eq!(mask(text, [(prefix, p), (full, p)]).unwrap(), "card ************1111 end");
    let p = whole('*').keeping_trailing(2);
    let (left, right) = (Span::new(0, 4), Span::new(2, 6));
    assert_eq!(mask("123456", [(right, p), (left, p)]).unwrap(), "****56");
    assert_eq!(mask("123456", [(left, p), (right, p)]).unwrap(), "****56");
}

#[test]
fn per_span_policies_with_different_masks() {
    let left = (Span::new(0, 4), whole('*').keeping_trailing(1));
    let right = (Span::new(2, 6), whole('#').keeping_trailing(1));
    assert_eq!(mask("123456", [left, right]).unwrap(), "**###6");
    assert_eq!(mask("123456", [right, left]).unwrap(), "***##6");
    let left = (Span::new(0, 4), whole('*').keeping_trailing(2));
    let right = (Span::new(2, 6), whole('*').keeping_trailing(1));
    assert_eq!(mask("123456", [left, right]).unwrap(), "*****6");
    assert_eq!(mask("123456", [right, left]).unwrap(), "*****6");
}

/// What one policy over each span writes, each judged on the original
/// text, one mask byte per byte of a masked character.
fn expected_mask(text: &str, spans: &[Span], mask: u8, keep_format: bool, trailing: usize) -> String {
    let mut out = text.as_bytes().to_vec();
    for span in spans {
        let piece = &text[span.start as usize..span.end as usize];
        let mut retained = 0;
        for (at, c) in piece.char_indices().rev() {
            let keep = if turbo_unicode::java::code_point::is_letter_or_digit(c as u32) {
                retained += 1;
                retained <= trailing
            } else {
                keep_format
            };
            if !keep {
                let from = span.start as usize + at;
                out[from..from + c.len_utf8()].fill(mask);
            }
        }
    }
    String::from_utf8(out).unwrap()
}

#[test]
fn every_pair_of_spans() {
    for text in ["123456", "12-34", "a𝕒b3", "𝕒-𝕓", "*12X"] {
        let bounds: Vec<u32> = (0..=text.len()).filter(|&i| text.is_char_boundary(i)).map(|i| i as u32).collect();
        let spans: Vec<Span> = bounds
            .iter()
            .flat_map(|&s| bounds.iter().filter(move |&&e| e >= s).map(move |&e| Span::new(s, e)))
            .collect();
        for c in ['*', 'X', '7', '-'] {
            for keep_format in [false, true] {
                for trailing in [0, 1, 2, 6] {
                    let mut p = whole(c).keeping_trailing(trailing);
                    if keep_format {
                        p = p.keeping_format();
                    }
                    for &a in &spans {
                        for &b in &spans {
                            let want = expected_mask(text, &[a, b], c as u8, keep_format, trailing);
                            assert_eq!(mask(text, [(a, p), (b, p)]).unwrap(), want, "{text} {a:?} {b:?}");
                        }
                    }
                }
            }
        }
    }
}

#[test]
fn spans_off_character_boundaries_or_past_the_end_are_refused() {
    let p = whole('*');
    assert_eq!(mask("a𝕒b", [(Span::new(2, 5), p)]), Err(MaskError::Span(Span::new(2, 5))));
    assert_eq!(mask("abc", [(Span::new(1, 4), p)]), Err(MaskError::Span(Span::new(1, 4))));
    assert_eq!(MaskPolicy::new('é'), Err(MaskError::NotAscii('é')));
    assert_eq!(MaskPolicy::for_kind(PiiKind::Card, '\u{2022}'), Err(MaskError::NotAscii('\u{2022}')));
}

#[test]
fn mask_policies() {
    for (text, want) in [("4111 1111 1111 1111", "*******************"), ("jane@example.com", "****************")] {
        assert_eq!(whole('*').masked(text), want);
    }
    for (text, want) in [
        ("4111 1111 1111 1111", "**** **** **** ****"),
        ("jane@example.com", "****@*******.***"),
        ("+44 20 7946 0958", "+** ** **** ****"),
        ("(555) 123-4567", "(***) ***-****"),
    ] {
        assert_eq!(whole('*').keeping_format().masked(text), want);
    }
    let receipt = whole('*').keeping_format().keeping_trailing(4);
    assert_eq!(receipt.masked("4111 1111 1111 1111"), "**** **** **** 1111");
    assert_eq!(receipt.masked("DE89 3704 0044 0532 0130 00"), "**** **** **** **** **30 00");
    assert_eq!(whole('*').keeping_trailing(4).masked("4111 1111 1111 1111"), "***************1111");
    assert_eq!(whole('*').keeping_trailing(9).masked("123"), "123");
    // One mask byte per UTF-8 byte: 𝕒 takes four.
    let text = "user𝕒@example.com";
    assert_eq!(whole('*').masked(text), "*".repeat(text.len()));
    assert_eq!(whole('*').keeping_trailing(1).masked("ab𝕒"), "**𝕒");
    let text = "Card 4111 1111 1111 1111 and mail jane@example.com.";
    let pii = layer(&cursor(), text);
    assert_eq!(
        mask(text, pii.mentions().map(|m| (m.span, receipt))).unwrap(),
        "Card **** **** **** 1111 and mail ****@******e.com."
    );
}

fn by_kind(value: &str, kind: PiiKind) -> String {
    MaskPolicy::for_kind(kind, '*').unwrap().masked(value)
}

#[test]
fn policies_by_kind() {
    for (value, kind, want) in [
        ("4111 1111 1111 1111", PiiKind::Card, "**** **** **** 1111"),
        ("4111-1111-1111-1111", PiiKind::Card, "****-****-****-1111"),
        ("4111111111111111", PiiKind::Card, "************1111"),
        ("DE89 3704 0044 0532 0130 00", PiiKind::Iban, "**** **** **** **** **30 00"),
        ("jane@example.com", PiiKind::Email, "****@*******.***"),
        ("(555) 123-4567", PiiKind::Phone, "(***) ***-****"),
        ("192.168.1.20", PiiKind::Ipv4, "***.***.*.**"),
        ("123-45-6789", PiiKind::UsSsn, "***-**-****"),
        ("943 476 5919", PiiKind::UkNhs, "*** *** ****"),
        ("00:1b:44:11:3a:b7", PiiKind::Mac, "**:**:**:**:**:**"),
    ] {
        assert_eq!(by_kind(value, kind), want, "{kind}");
    }
    for (value, kind) in [
        ("AKIAIOSFODNN7EXAMPLE", PiiKind::AwsAccessKey),
        ("ghp_0123456789abcdefghijklmnopqrstuvwxyz", PiiKind::GithubToken),
        ("0x52908400098527886E0F7030069857D2E4169EE7", PiiKind::EthAddress),
        ("user:secret", PiiKind::UrlCredential),
        ("ab-cd", PiiKind::Jwt),
        ("ab-cd", PiiKind::BtcAddress),
    ] {
        assert_eq!(by_kind(value, kind), "*".repeat(value.len()), "{kind}");
    }
    for kind in PiiKind::ALL {
        assert_eq!(by_kind("4111 1111 1111 1111", kind).len(), 19);
    }
    assert_eq!(MaskPolicy::for_kind(PiiKind::Card, 'X').unwrap().masked("4111 1111 1111 1111"), "XXXX XXXX XXXX 1111");
    let text = "card 4111 1111 1111 1111 mail jane@example.com key AKIAIOSFODNN7EXAMPLE";
    assert_eq!(
        mask_by_kind(text, &layer(&wide(), text), '*').unwrap(),
        "card **** **** **** 1111 mail ****@*******.*** key ********************"
    );
    assert_eq!(
        mask_by_kind("card 4111111111111111", &layer(&cursor(), "card 4111111111111111"), 'X').unwrap(),
        "card XXXXXXXXXXXX1111"
    );
}

#[test]
fn pseudonyms_number_each_kind() {
    assert_eq!(pseudonyms("write to jane@example.com today"), "write to EMAIL-1 today");
    assert_eq!(
        pseudonyms("jane@example.com bob@example.com (555) 123-4567 eve@example.com"),
        "EMAIL-1 EMAIL-2 PHONE-1 EMAIL-3"
    );
    assert_eq!(
        pseudonyms("jane@example.com wrote, bob@example.com replied, then jane@example.com again"),
        "EMAIL-1 wrote, EMAIL-2 replied, then EMAIL-1 again"
    );
    assert_eq!(pseudonyms("4111 1111 1111 1111 and 4111-1111-1111-1111"), "CARD-1 and CARD-1");
    assert_eq!(pseudonyms("4111111111111111 and 5500005555555559"), "CARD-1 and CARD-2");
    assert_eq!(pseudonyms("User@example.com user@example.com"), "EMAIL-1 EMAIL-2");
    assert_eq!(pseudonyms("jane@example.com"), "EMAIL-1");
    assert_eq!(pseudonyms("jane@example.com"), "EMAIL-1");
    let text = "write to jane@example.com today";
    assert_eq!(
        Pseudonymizer::new("[", "]").rewrite(text, &layer(&cursor(), text)).unwrap().text(),
        "write to [EMAIL-1] today"
    );
    let once = pseudonyms("mail jane@example.com now");
    assert_eq!(pseudonyms(&once), once);
    let text = "Contact jane@example.com; jane@example.com replied to bob@example.com.";
    assert_eq!(pseudonyms(text), "Contact EMAIL-1; EMAIL-1 replied to EMAIL-2.");
}

#[test]
fn labels_name_the_kind_and_point_into_the_new_text() {
    for (text, want) in [
        ("mail jane@example.com", "EMAIL-1"),
        ("call (555) 123-4567", "PHONE-1"),
        ("card 4111111111111111", "CARD-1"),
        ("iban DE89370400440532013000", "IBAN-1"),
    ] {
        let r = Pseudonymizer::default().rewrite(text, &layer(&cursor(), text)).unwrap();
        assert_eq!(r.labels().mentions().next().unwrap().normalized, want);
    }
    let text = "mail jane@example.com or bob@example.com now";
    let r = Pseudonymizer::default().rewrite(text, &layer(&cursor(), text)).unwrap();
    assert_eq!(r.labels().mentions().count(), 2);
    for m in r.labels().mentions() {
        assert_eq!(&r.text()[m.span.start as usize..m.span.end as usize], m.normalized);
        assert_eq!(m.kind, PiiKind::Email);
    }
    let text = "nothing to see here";
    let r = Pseudonymizer::default().rewrite(text, &layer(&cursor(), text)).unwrap();
    assert_eq!((r.text(), r.labels().mentions().count(), r.map_offset(7)), (text, 0, 7));
}

#[test]
fn rewrites_refuse_overlaps_and_spans_off_the_text() {
    let text = "https://john:pass@example.com/path";
    let crossing = layer(&wide(), text);
    assert_found(&wide(), text, &[(8, 17, "url-credential", "john:pass"), (13, 29, "email", "pass@example.com")]);
    let overlap = Err(RewriteError::Overlap(Span::new(8, 17), Span::new(13, 29)));
    assert_eq!(Pseudonymizer::default().rewrite(text, &crossing).map(|r| r.into_text()), overlap.clone());
    assert_eq!(tokenizer().rewrite(text, &crossing).map(|r| r.into_text()), overlap);
    assert_eq!(mask_all(text, &crossing, '*').unwrap(), format!("https://{}/path", "*".repeat(21)));
    let past = manual(&[(2, 40, PiiKind::Email, "a@b.com")]);
    assert_eq!(
        Pseudonymizer::default().rewrite("short", &past).map(|r| r.into_text()),
        Err(RewriteError::Span(Span::new(2, 40)))
    );
    let inside = manual(&[(1, 3, PiiKind::Email, "a@b.com")]);
    assert!(matches!(Pseudonymizer::default().rewrite("a😀", &inside), Err(RewriteError::Span(_))));
}

#[test]
fn adjacent_and_unordered_mentions() {
    let adjacent = manual(&[(0, 5, PiiKind::Email, "a@b.com"), (5, 10, PiiKind::Email, "c@d.com")]);
    assert_eq!(Pseudonymizer::default().rewrite("0123456789", &adjacent).unwrap().text(), "EMAIL-1EMAIL-2");
    let unordered = manual(&[(6, 11, PiiKind::Phone, "5551234567"), (0, 5, PiiKind::Email, "a@b.com")]);
    assert_eq!(Pseudonymizer::default().rewrite("00000 11111", &unordered).unwrap().text(), "EMAIL-1 PHONE-1");
}

#[test]
fn offsets_map_into_the_new_text() {
    let text = "to jane@example.com now";
    let r = Pseudonymizer::default().rewrite(text, &layer(&cursor(), text)).unwrap();
    assert_eq!(r.text(), "to EMAIL-1 now");
    assert_eq!([r.map_offset(0), r.map_offset(3), r.map_offset(4), r.map_offset(7)], [0, 3, 3, 3]);
    assert_eq!(r.map_offset(20), 11);
    assert_eq!(r.map_offset(23), 14);
    assert_eq!(r.map_span(Span::new(0, 23)), Span::new(0, 14));
    assert_eq!(r.map_span(Span::new(3, 19)), Span::new(3, 10));
    assert_eq!(r.map_span(Span::new(4, 6)), Span::new(3, 10));
    let text = "a jane@example.com b (555) 123-4567 z";
    let r = Pseudonymizer::default().rewrite(text, &layer(&cursor(), text)).unwrap();
    assert_eq!(r.text(), "a EMAIL-1 b PHONE-1 z");
    for c in ['b', 'z'] {
        assert_eq!(r.map_offset(text.find(c).unwrap() as u32), r.text().find(c).unwrap() as u32);
    }
    let text = "Contact jane@example.com; jane@example.com replied to bob@example.com.";
    let r = Pseudonymizer::default().rewrite(text, &layer(&cursor(), text)).unwrap();
    assert_eq!(r.map_offset(text.find("bob").unwrap() as u32), r.text().find("EMAIL-2").unwrap() as u32);
}

#[test]
#[should_panic(expected = "past the end")]
fn an_offset_past_the_end_panics() {
    let text = "to jane@example.com now";
    Pseudonymizer::default().rewrite(text, &layer(&cursor(), text)).unwrap().map_offset(24);
}

#[test]
fn many_replacements_map_every_offset() {
    let text = "xx ".repeat(1024);
    let values: Vec<String> = (0..1024).map(|i| format!("value{}", 3 * i)).collect();
    let rows: Vec<(u32, u32, PiiKind, &str)> =
        values.iter().enumerate().map(|(i, v)| (3 * i as u32, 3 * i as u32 + 2, PiiKind::Email, v.as_str())).collect();
    let pii = manual(&rows);
    let r = Pseudonymizer::default().rewrite(&text, &pii).unwrap();
    let labels: Vec<Span> = r.labels().mentions().map(|m| m.span).collect();
    for offset in 0..=text.len() as u32 {
        assert_eq!(r.map_offset(offset), boundary(offset, false, &rows, &labels), "{offset}");
    }
}

/// Where the Java's `PiiRewriteTest.boundary` puts an offset.
fn boundary(offset: u32, end: bool, sources: &[(u32, u32, PiiKind, &str)], labels: &[Span]) -> u32 {
    let mut shift: i64 = 0;
    for (source, label) in sources.iter().zip(labels) {
        if offset <= source.0 {
            return (i64::from(offset) + shift) as u32;
        }
        if offset < source.1 {
            return if end { label.end } else { label.start };
        }
        shift += i64::from(label.end - label.start) - i64::from(source.1 - source.0);
    }
    (i64::from(offset) + shift) as u32
}

#[test]
fn every_source_span_of_every_plan() {
    for text in ["", "abcdefgh", "😀abc𝕒"] {
        let bounds: Vec<u32> = (0..=text.len()).filter(|&i| text.is_char_boundary(i)).map(|i| i as u32).collect();
        let mut plans: Vec<Vec<Span>> = vec![vec![]];
        for (i, &s) in bounds.iter().enumerate() {
            for &e in &bounds[i + 1..] {
                plans.push(vec![Span::new(s, e)]);
                for (j, &ns) in bounds.iter().enumerate().filter(|&(_, &b)| b >= e) {
                    for &ne in &bounds[j + 1..] {
                        plans.push(vec![Span::new(s, e), Span::new(ns, ne)]);
                    }
                }
            }
        }
        for plan in &plans {
            let values: Vec<String> = plan.iter().map(|s| format!("value{}", s.start)).collect();
            let rows: Vec<(u32, u32, PiiKind, &str)> =
                plan.iter().zip(&values).map(|(s, v)| (s.start, s.end, PiiKind::UsSsn, v.as_str())).collect();
            let reversed: Vec<_> = rows.iter().rev().copied().collect();
            for mode in 0..3 {
                let run = |rows: &[(u32, u32, PiiKind, &str)]| {
                    let pii: Pii = manual(rows);
                    match mode {
                        0 => Pseudonymizer::default().rewrite(text, &pii).unwrap(),
                        1 => Pseudonymizer::new("[", "]").rewrite(text, &pii).unwrap(),
                        _ => tokenizer().rewrite(text, &pii).unwrap(),
                    }
                };
                let r = run(&rows);
                let back = run(&reversed);
                assert_eq!(r.text(), back.text());
                let labels: Vec<Span> = r.labels().mentions().map(|m| m.span).collect();
                assert_eq!(labels, back.labels().mentions().map(|m| m.span).collect::<Vec<_>>());
                let mut want = String::new();
                let mut copied = 0;
                for (span, label) in plan.iter().zip(r.labels().mentions()) {
                    want.push_str(&text[copied..span.start as usize]);
                    assert_eq!(want.len() as u32, label.span.start);
                    want.push_str(label.normalized);
                    assert_eq!(want.len() as u32, label.span.end);
                    assert_eq!(label.kind, PiiKind::UsSsn);
                    copied = span.end as usize;
                }
                want.push_str(&text[copied..]);
                assert_eq!(r.text(), want);
                for &s in &bounds {
                    let start = boundary(s, false, &rows, &labels);
                    assert_eq!(r.map_offset(s), start);
                    for &e in bounds.iter().filter(|&&e| e >= s) {
                        let end = if s == e { start } else { boundary(e, true, &rows, &labels) };
                        assert_eq!(r.map_span(Span::new(s, e)), Span::new(start, end), "{text:?} {plan:?} {s}..{e}");
                    }
                }
            }
        }
    }
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
    let r = PiiAuditReport::of(&layer(&wide(), text), &tokenizer(), 3);
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
fn no_value_of_any_pack_appears_in_the_audit() {
    let texts = sample_texts();
    assert!(texts.len() > 50, "only {} sample texts", texts.len());
    let mut kinds = std::collections::HashSet::new();
    for p in Pack::ALL {
        let detector = pack(p);
        let mut checked = 0;
        for (id, text) in &texts {
            let pii = layer(&detector, text);
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

#[test]
fn empty_normalized_mention_is_refused_in_hmac_rewrite_and_skipped_in_audit_samples() {
    let pii = manual(&[(0, 5, PiiKind::Email, "")]);
    assert_eq!(
        tokenizer().rewrite("hello world", &pii).map(|r| r.into_text()),
        Err(RewriteError::EmptyValue(Span::new(0, 5)))
    );
    let report = PiiAuditReport::of(&pii, &tokenizer(), 3);
    assert_eq!(report.total, 1);
    let email = report.kind(PiiKind::Email).unwrap();
    assert_eq!((email.mentions, email.distinct), (1, 1));
    assert!(email.samples.is_empty());
}
