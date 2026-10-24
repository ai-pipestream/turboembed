//! `MaskerTest`, `MaskerOverlapTest`, `MaskPolicyTest`,
//! `MaskPoliciesTest`, `PseudonymizerTest` and `PiiRewriteTest`, over
//! the typed layer.

mod support;

use support::*;
use turbo_analysis::{Layer, Span};
use turbo_pii::rewrite::{
    HmacTokenizer, MaskError, MaskPolicy, Pseudonymizer, RewriteError, mask, mask_all, mask_by_kind,
};
use turbo_pii::{Pii, PiiDetector, PiiKind};

const KEY: &[u8] = b"0123456789abcdef0123456789abcdef";
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
fn rewrites_refuse_overlaps_empty_spans_and_spans_off_the_text() {
    let text = "https://john:pass@example.com/path";
    let disjoint = layer(&wide(), text);
    assert_found(&wide(), text, &[(8, 17, "url-credential", "john:pass")]);
    let pseudo = Pseudonymizer::default().rewrite(text, &disjoint).unwrap();
    assert_eq!(pseudo.text(), "https://URL-CREDENTIAL-1@example.com/path");
    assert_eq!(pseudo.labels().check(), Ok(()));
    let hmac = tokenizer().rewrite(text, &disjoint).unwrap();
    assert_eq!(hmac.labels().check(), Ok(()));

    for same_span in ["record 943 476 5919 today", "IMEI: 490154203237518."] {
        let pii = layer(&wide(), same_span);
        let pseudo = Pseudonymizer::default().rewrite(same_span, &pii).unwrap();
        assert_eq!(pseudo.labels().check(), Ok(()));
        let hmac = tokenizer().rewrite(same_span, &pii).unwrap();
        assert_eq!(hmac.labels().check(), Ok(()));
    }

    let crossing = layer(&wide().keeping_nested(), text);
    assert_found(
        &wide().keeping_nested(),
        text,
        &[(8, 17, "url-credential", "john:pass"), (13, 29, "email", "pass@example.com")],
    );
    let overlap = Err(RewriteError::Overlap(Span::new(8, 17), Span::new(13, 29)));
    assert_eq!(Pseudonymizer::default().rewrite(text, &crossing).map(|r| r.into_text()), overlap.clone());
    assert_eq!(tokenizer().rewrite(text, &crossing).map(|r| r.into_text()), overlap);
    assert_eq!(mask_all(text, &crossing, '*').unwrap(), format!("https://{}/path", "*".repeat(21)));

    for empty in [
        manual(&[(5, 5, PiiKind::Email, "a@b.com")]),
        manual(&[(5, 5, PiiKind::Email, "a@b.com"), (5, 10, PiiKind::Phone, "5551234567")]),
    ] {
        assert_eq!(
            Pseudonymizer::default().rewrite("0123456789", &empty).map(|r| r.into_text()),
            Err(RewriteError::Span(Span::new(5, 5)))
        );
        assert_eq!(
            tokenizer().rewrite("0123456789", &empty).map(|r| r.into_text()),
            Err(RewriteError::Span(Span::new(5, 5)))
        );
    }

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
                assert_eq!(r.labels().check(), Ok(()));
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
