//! `CursorPiiExtractorTest`: e-mail addresses, phone numbers, IBANs and
//! cards from the default extractor.

mod support;

use support::*;
use turbo_pii::{Extractor, PiiDetector, PiiKind, PiiKinds, Scanner, ScannerError};

fn cursor() -> PiiDetector {
    PiiDetector::cursor()
}

/// A domain of `length` characters ending in `tld`, labels of `a` up to 63.
fn domain_of_length(length: usize, tld: &str) -> String {
    assert!(tld.len() + 2 <= length);
    let mut domain = String::new();
    let mut remaining = length - tld.len() - 1;
    while remaining > 0 {
        if !domain.is_empty() {
            domain.push('.');
            remaining -= 1;
        }
        let mut label = remaining.min(63);
        if remaining > label && remaining - label == 1 {
            label -= 1;
        }
        assert!(label >= 1);
        domain.push_str(&"a".repeat(label));
        remaining -= label;
    }
    domain.push('.');
    domain.push_str(tld);
    assert_eq!(domain.len(), length);
    domain
}

#[test]
fn email() {
    let text = "Write to John.Doe+news@Example.co.uk today.";
    assert_found(&cursor(), text, &[(9, 36, "email", "John.Doe+news@example.co.uk")]);
}

#[test]
fn email_normalization_keeps_the_local_part_case() {
    assert_eq!(normalized(&cursor(), "User@Example.com user@example.com"), ["User@example.com", "user@example.com"]);
}

#[test]
fn email_leaves_trailing_punctuation_out() {
    assert_eq!(normalized(&cursor(), "Reach me at jane@example.com."), ["jane@example.com"]);
    assert_found(&cursor(), "Ask jane@example.com.", &[(4, 20, "email", "jane@example.com")]);
}

#[test]
fn email_refuses_invalid_forms() {
    for text in [
        "no at sign here",
        "broken @example.com address",
        "missing tld user@localhost",
        "doubled dots user..name@example.com",
        "numeric tld user@example.c1",
    ] {
        assert_none(&cursor(), text);
    }
}

#[test]
fn email_length_limits() {
    let local = "a".repeat(64);
    let mailbox = format!("{local}@{}", domain_of_length(189, "com"));
    assert_eq!(mailbox.len(), 254);
    assert_eq!(rows(&cursor(), &mailbox).len(), 1);
    assert_none(&cursor(), &format!("{}@example.com", "a".repeat(65)));
    assert_none(&cursor(), &format!("{local}@{}", domain_of_length(190, "com")));
    for length in [253, 254, 255, 300] {
        assert_none(&cursor(), &format!("u@{}", domain_of_length(length, "com")));
    }
    for (tld, length, accepted) in [
        ("com", 252, true),
        ("com", 253, false),
        ("com", 254, false),
        ("museum", 252, true),
        ("museum", 253, false),
        ("xn--p1ai", 252, true),
        ("xn--p1ai", 253, false),
    ] {
        let text = format!("u@{}", domain_of_length(length, tld));
        let want: Vec<String> = if accepted { vec![text.clone()] } else { vec![] };
        assert_eq!(normalized(&cursor(), &text), want, "{tld} {length}");
    }
}

#[test]
fn email_refuses_unregistered_and_private_use_tlds() {
    for text in [
        "user@example.notatld",
        "user@example.zz",
        "user@files.corp.internal",
        "user@host.local",
        "user@mail.corp",
        "user@box.lan",
        "user@svc.home",
        "user@node.intranet",
        "user@dev.testbed",
        "a@b.ccx",
        "admin@mail.invalidtld",
        "root@example.internal",
        "me@office.localdomain",
    ] {
        assert_none(&cursor(), text);
    }
}

#[test]
fn email_takes_punycode_tlds_with_exact_spans() {
    for (text, norm, start, end) in [
        ("user@example.xn--p1ai", "user@example.xn--p1ai", 0, 21),
        ("Contact USER@EXAMPLE.XN--P1AI please", "USER@example.xn--p1ai", 8, 29),
        ("mail@site.xn--node now", "mail@site.xn--node", 0, 18),
        ("a@b.xn--j1amh!", "a@b.xn--j1amh", 0, 13),
    ] {
        assert_found(&cursor(), text, &[(start, end, "email", norm)]);
    }
}

#[test]
fn email_takes_mixed_case_registered_tlds() {
    for (text, norm) in [
        ("user@EXAMPLE.COM", "user@example.com"),
        ("user@Example.Org", "user@example.org"),
        ("User@ExAmPle.NeT", "User@example.net"),
        ("a@B.Io", "a@b.io"),
        ("x@Y.AI", "x@y.ai"),
        ("ops@Mail.APP", "ops@mail.app"),
        ("me@Site.INFO", "me@site.info"),
        ("dev@Host.DEV", "dev@host.dev"),
    ] {
        assert_eq!(normalized(&cursor(), text), [norm], "{text}");
    }
}

#[test]
fn email_takes_registered_ascii_tlds() {
    for text in [
        "a@b.co",
        "a@b.com",
        "a@b.org",
        "a@b.net",
        "a@b.edu",
        "a@b.gov",
        "a@b.mil",
        "a@b.int",
        "a@b.io",
        "a@b.ai",
        "a@b.app",
        "a@b.dev",
        "a@b.xyz",
        "a@b.museum",
        "a@b.travel",
        "a@b.photography",
        "a@b.accountant",
        "a@b.americanexpress",
        "mail@corp.uk",
        "mail@corp.de",
        "mail@corp.fr",
        "mail@corp.jp",
        "mail@corp.cn",
        "mail@corp.br",
        "mail@corp.au",
        "mail@corp.za",
        "mail@corp.tw",
    ] {
        assert_found(&cursor(), text, &[(0, text.len() as u32, "email", text)]);
    }
}

#[test]
fn email_at_text_start_and_end() {
    assert_found(&cursor(), "jane@example.com wrote back.", &[(0, 16, "email", "jane@example.com")]);
    assert_found(&cursor(), "Please write to jane@example.com", &[(16, 32, "email", "jane@example.com")]);
}

#[test]
fn phone_international() {
    assert_found(&cursor(), "Call +44 20 7946 0958 now.", &[(5, 21, "phone", "+442079460958")]);
    assert_found(&cursor(), "Call +14155550123 now.", &[(5, 17, "phone", "+14155550123")]);
}

#[test]
fn phone_domestic_forms() {
    for (text, norm) in [
        ("Call (555) 123-4567 today.", "5551234567"),
        ("Call 555-123-4567 today.", "5551234567"),
        ("Call 1 555 123 4567 today.", "15551234567"),
    ] {
        assert_eq!(normalized(&cursor(), text)[0], norm);
    }
    assert_found(&cursor(), "Call 415-555-0123 today.", &[(5, 17, "phone", "4155550123")]);
    assert_found(&cursor(), "(415) 555-0123", &[(0, 14, "phone", "4155550123")]);
}

#[test]
fn adjacent_phones_are_both_found() {
    assert_eq!(normalized(&cursor(), "(555) 123-4567(555) 123-4568"), ["5551234567", "5551234568"]);
}

#[test]
fn phone_refused_forms() {
    for text in [
        "id 5551234567 in the table",
        "Order 4155550123 shipped.",
        "Ref 415-555-012 code.",
        "value 1234567890.25 total",
        "pi is 3.1415926535",
        "+123 is too short",
        "+1234567 is short",
    ] {
        assert_none(&cursor(), text);
    }
}

#[test]
fn international_lengths_follow_the_calling_code() {
    assert_found(&cursor(), "Call +683 4002 now.", &[(5, 14, "phone", "+6834002")]);
    assert_eq!(normalized(&cursor(), "Call +299 123456 today."), ["+299123456"]);
    assert_none(&cursor(), "Call +999 1234 5678 now.");
    assert_none(&cursor(), "Call +65 123 456 now.");
}

#[test]
fn phone_backs_off_before_a_trailing_group() {
    assert_found(&cursor(), "call 555-123-4567 99 times", &[(5, 17, "phone", "5551234567")]);
    assert_eq!(covered(&cursor(), "dial (212) 555-0199 77 now"), ["(212) 555-0199"]);
    assert_eq!(normalized(&cursor(), "dial (212) 555-0199 77 now"), ["2125550199"]);
}

#[test]
fn iban_and_the_phone_inside_it() {
    let parity = cursor().keeping_nested();
    assert!(!cursor().keeps_nested() && parity.keeps_nested());
    assert_found(
        &cursor(),
        "Wire it to DE89 3704 0044 0532 0130 00 by Friday.",
        &[(11, 38, "iban", "DE89370400440532013000")],
    );
    assert_found(
        &parity,
        "Wire it to DE89 3704 0044 0532 0130 00 by Friday.",
        &[(11, 38, "iban", "DE89370400440532013000"), (26, 38, "phone", "0532013000")],
    );
    assert_found(&cursor(), "num 3704 0044 0532 0130 00 end", &[(14, 26, "phone", "0532013000")]);
    assert_found(
        &cursor(),
        "Pay DE89 3704 0044 0532 0130 00 or call +1 555 123 4567.",
        &[(4, 31, "iban", "DE89370400440532013000"), (40, 55, "phone", "+15551234567")],
    );
    assert_found(
        &parity,
        "Pay DE89 3704 0044 0532 0130 00 or call +1 555 123 4567.",
        &[
            (4, 31, "iban", "DE89370400440532013000"),
            (19, 31, "phone", "0532013000"),
            (40, 55, "phone", "+15551234567"),
        ],
    );
}

#[test]
fn iban_forms() {
    let yemen = "YE15CBYE0001018861234567891234";
    assert_found(&cursor(), yemen, &[(0, 30, "iban", yemen)]);
    for text in ["account GB82WEST12345698765432 closed", "account GB82 WEST 1234 5698 7654 32 closed"] {
        assert_eq!(normalized(&cursor(), text)[0], "GB82WEST12345698765432");
    }
}

#[test]
fn iban_refuses_bad_check_digits() {
    let k = kinds(&cursor(), "account DE89 3704 0044 0532 0130 02 closed");
    assert!(!k.contains(&"iban") && !k.contains(&"card"));
    assert_none(&cursor(), "plain WORDS IN CAPITALS here");
    assert_found(&cursor(), "Wire to DE21 3704 0044 0532 0130 00 today.", &[(23, 35, "phone", "0532013000")]);
    assert_none(&cursor(), "Wire to DE21370400440532013000 today.");
}

#[test]
fn iban_refuses_registry_violations() {
    for text in ["wire to DE45500105170123 today", "wire to ZZ591234567890123456 today"] {
        assert!(!kinds(&cursor(), text).contains(&"iban"), "{text}");
    }
}

#[test]
fn iban_registry_length_extremes() {
    for value in [
        "NO9386011117947",
        "MT84MALT011000012345MTLCAST001S",
        "FR1420041010050500013M02606",
        "LC55HEMM000100010012001200023015",
        "RU0304452522540817810538091310419",
    ] {
        let text = format!("wire to {value} today");
        let end = 8 + value.len() as u32;
        assert_found(&cursor(), &text, &[(8, end, "iban", value)]);
    }
}

#[test]
fn card_forms() {
    assert_found(&cursor(), "Charged to 4111 1111 1111 1111 yesterday.", &[(11, 30, "card", "4111111111111111")]);
    assert_eq!(normalized(&cursor(), "card 4111-1111-1111-1111 on file"), ["4111111111111111"]);
    assert_eq!(normalized(&cursor(), "amex 378282246310005 on file"), ["378282246310005"]);
    assert_found(&cursor(), "card 4111111111111111 on file", &[(5, 21, "card", "4111111111111111")]);
}

#[test]
fn card_refuses_a_failed_checksum_and_an_unassigned_leading_digit() {
    assert_none(&cursor(), "card 4111 1111 1111 1112 on file");
    assert_none(&cursor(), "serial 1111 2222 3333 4444 here");
    assert_none(&cursor(), "card 4111111111111112 on file");
}

#[test]
fn card_backs_off_before_an_expiry() {
    assert_found(&cursor(), "Card 4111111111111111 12/26", &[(5, 21, "card", "4111111111111111")]);
    assert_none(&cursor(), "Card 4111111111111112 12/26");
}

#[test]
fn nineteen_digit_card_is_whole() {
    assert_found(&cursor(), "card 4111111111111111110 on file", &[(5, 24, "card", "4111111111111111110")]);
    assert_found(&cursor(), "card 4111111111111111110 12/26", &[(5, 24, "card", "4111111111111111110")]);
}

#[test]
fn two_cards_split_by_a_space() {
    assert_found(
        &cursor(),
        "cards 4111111111111111 5500005555555559 on file",
        &[(6, 22, "card", "4111111111111111"), (23, 39, "card", "5500005555555559")],
    );
}

#[test]
fn separated_card_back_off_keeps_the_phone_inside_in_parity() {
    assert_found(&cursor(), "Card 4111 1111 1111 1111 123 on file", &[(5, 24, "card", "4111111111111111")]);
    assert_found(
        &cursor().keeping_nested(),
        "Card 4111 1111 1111 1111 123 on file",
        &[(5, 24, "card", "4111111111111111"), (15, 28, "phone", "11111111123")],
    );
}

#[test]
fn phone_digits_inside_an_email_are_dropped_by_default_and_kept_in_parity() {
    assert_found(
        &cursor(),
        "mail +15551234567@voip.example.com please",
        &[(5, 34, "email", "+15551234567@voip.example.com")],
    );
    assert_found(
        &cursor().keeping_nested(),
        "mail +15551234567@voip.example.com please",
        &[(5, 34, "email", "+15551234567@voip.example.com"), (5, 17, "phone", "+15551234567")],
    );
}

#[test]
fn mentions_come_in_text_order() {
    assert_eq!(
        kinds(&cursor(), "jane@example.com, +44 20 7946 0958, and 4111 1111 1111 1111."),
        ["email", "phone", "card"]
    );
}

#[test]
fn supplementary_letter_neighbours_block_number_boundaries() {
    // U+10428, a letter outside the Basic Multilingual Plane: four bytes.
    let letter = "\u{10428}";
    assert_none(&cursor(), &format!("{letter}4111111111111111"));
    assert_none(&cursor(), &format!("4111111111111111{letter}"));
    assert_none(&cursor(), &format!("{letter}555-123-4567"));
    // The Java offset is 3 in UTF-16 units; it is 5 in UTF-8 bytes.
    assert_found(&cursor(), &format!("{letter} 4111111111111111"), &[(5, 21, "card", "4111111111111111")]);
}

#[test]
fn separator_neighbours_stay_outside_the_span() {
    for (text, want) in [
        ("call -555-123-4567 now", "555-123-4567"),
        ("call 555-123-4567- now", "555-123-4567"),
        ("card 4111 1111 1111 1111- end", "4111 1111 1111 1111"),
    ] {
        assert_eq!(covered(&cursor(), text), [want]);
    }
}

#[test]
fn a_kind_subset_limits_what_is_found() {
    let email = only(Extractor::Cursor, &[PiiKind::Email]);
    assert_eq!(kinds(&email, "Mail jane@example.com or call (555) 123-4567."), ["email"]);
}

#[test]
fn kind_subsets_are_checked() {
    assert_eq!(Scanner::with_kinds(Extractor::Cursor, PiiKinds::EMPTY), Err(ScannerError::NoKinds(Extractor::Cursor)));
    assert_eq!(
        Scanner::with_kinds(Extractor::Cursor, PiiKinds::of(&[PiiKind::UsSsn])),
        Err(ScannerError::Foreign(Extractor::Cursor, PiiKind::UsSsn))
    );
}

#[test]
fn empty_and_pii_free_text() {
    assert_none(&cursor(), "");
    assert_none(&cursor(), "The quick brown fox jumps over the lazy dog.");
}
