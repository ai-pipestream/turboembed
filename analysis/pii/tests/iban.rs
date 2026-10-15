//! `IbanValidationTest`: the MOD 97-10 check, registered lengths, groups
//! and boundaries of IBANs.

mod support;

use support::*;
use turbo_pii::checksum::mod97;
use turbo_pii::rewrite::{Pseudonymizer, mask_all};
use turbo_pii::{Extractor, Pack, PiiDetector, PiiKind};

const COUNTRIES: [(&str, usize); 6] = [("NO", 15), ("DE", 22), ("GB", 22), ("MT", 31), ("LC", 32), ("RU", 33)];
const EXAMPLE: &str = "DE89370400440532013000";

fn ibans() -> PiiDetector {
    only(Extractor::Cursor, &[PiiKind::Iban])
}

/// `country` and `national` with the check digits that make them valid.
fn candidate(country: &str, national: &str) -> String {
    let check = 98 - mod97::remainder(format!("{country}00{national}").as_bytes());
    format!("{country}{check:02}{national}")
}

#[test]
fn checksums_and_spans() {
    for (country, length) in COUNTRIES {
        for value in 0..97 {
            let national = format!("{}{value:02}", "0".repeat(length - 6));
            let compact = candidate(country, &national);
            for formatted in [compact.clone(), grouped(&compact, 4, ' ')] {
                let prefix = "😀 Wire: ";
                let text = format!("{prefix}{formatted}.");
                let start = prefix.len() as u32;
                assert_found(&ibans(), &text, &[(start, start + formatted.len() as u32, "iban", &compact)]);
                let pii = layer(&ibans(), &text);
                assert_eq!(mask_all(&text, &pii, '*').unwrap(), format!("{prefix}{}.", "*".repeat(formatted.len())));
            }
        }
    }
}

#[test]
fn wrong_check_digits() {
    for (country, length) in COUNTRIES {
        let compact = candidate(country, &"0".repeat(length - 4));
        for check in 2..=98 {
            let check = format!("{check:02}");
            if check != compact[2..4] {
                let wrong = format!("{country}{check}{}", &compact[4..]);
                assert_none(&ibans(), &wrong);
                assert_none(&ibans(), &grouped(&wrong, 4, ' '));
            }
        }
    }
}

#[test]
fn wrong_lengths_and_countries() {
    for (country, length) in COUNTRIES {
        for other in 14..=35 {
            if other != length {
                assert_none(&ibans(), &candidate(country, &"0".repeat(other - 4)));
            }
        }
    }
    for country in ["AA", "US", "XX", "ZZ"] {
        assert_none(&ibans(), &candidate(country, &"0".repeat(18)));
    }
}

#[test]
fn letters_expand_and_lower_case_is_refused() {
    for letter in b'A'..=b'Z' {
        let national = format!("TEST{}{}", "0".repeat(13), letter as char);
        let compact = candidate("GB", &national);
        assert_found(&ibans(), &compact, &[(0, compact.len() as u32, "iban", &compact)]);
        let lower = format!("{}{}", &compact[..compact.len() - 1], letter.to_ascii_lowercase() as char);
        assert_none(&ibans(), &lower);
    }
}

#[test]
fn word_boundaries() {
    for neighbour in ["a", "0", "é", "\u{661}", "\u{10428}", "\u{1D7D9}"] {
        assert_none(&ibans(), &format!("{neighbour}{EXAMPLE}"));
        assert_none(&ibans(), &format!("{EXAMPLE}{neighbour}"));
        let start = neighbour.len() as u32 + 1;
        assert_found(&ibans(), &format!("{neighbour} {EXAMPLE}"), &[(start, start + 22, "iban", EXAMPLE)]);
    }
}

#[test]
fn trailing_groups_and_national_groups() {
    for suffix in [" 1234567890123456789012345678901234567890", " ACCOUNT CLOSED"] {
        assert_found(&ibans(), &format!("{EXAMPLE}{suffix}"), &[(0, 22, "iban", EXAMPLE)]);
    }
    for width in 1..=6 {
        let text = format!("{} {}", &EXAMPLE[..4], grouped(&EXAMPLE[4..], width, ' '));
        assert_found(&ibans(), &text, &[(0, text.len() as u32, "iban", EXAMPLE)]);
    }
}

#[test]
fn malformed_headers_national_parts_and_short_input() {
    for header in ["D E89", "DE 89", "DE8 9", "DE89\t", "DE89-", "de89", "DE\u{ff18}\u{ff19}", "DE89\n", "DE89  "] {
        assert_none(&ibans(), &format!("{header}{}", &EXAMPLE[4..]));
    }
    for inserted in ["a", "é", "\u{660}", "\u{ff10}", "😀", "_", "\u{a0}"] {
        assert_none(&ibans(), &format!("{}{inserted}{}", &EXAMPLE[..10], &EXAMPLE[11..]));
    }
    for text in ["", "D", "DE", "DE8", "DE89"] {
        assert_none(&ibans(), text);
    }
}

#[test]
fn reference_check_digits() {
    for compact in [EXAMPLE, "NO9386011117947", "GB82WEST12345698765432"] {
        assert_eq!(candidate(&compact[..2], &compact[4..]), compact);
        assert_eq!(mod97::remainder(compact.as_bytes()), 1);
    }
}

#[test]
fn payment_pack_masks_and_replaces_the_iban() {
    let account = grouped(EXAMPLE, 4, ' ');
    let text = format!("Wire {account}; reference 1234.");
    let pii = layer(&pack(Pack::Payment), &text);
    assert_eq!(normalized(&pack(Pack::Payment), &text), [EXAMPLE]);
    assert_eq!(mask_all(&text, &pii, '*').unwrap(), format!("Wire {}; reference 1234.", "*".repeat(account.len())));
    assert_eq!(Pseudonymizer::new("", "").rewrite(&text, &pii).unwrap().text(), "Wire IBAN-1; reference 1234.");
}
