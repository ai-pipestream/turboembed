//! `CardPrefixTest`: issuer prefixes, lengths, separators and boundaries
//! of payment cards.

mod support;

use support::*;
use turbo_pii::checksum::luhn;
use turbo_pii::rewrite::{MaskPolicy, Pseudonymizer, mask, mask_all, mask_by_kind};
use turbo_pii::{Extractor, Pack, PiiDetector, PiiKind};

const NUMBERS: [&str; 7] = [
    "8100000000026",
    "81000000000028",
    "810000000000026",
    "8100000000000028",
    "81000000000000026",
    "810000000000000028",
    "8100000000000000026",
];
const EXAMPLE: &str = "8100000000000028";
const FORMATTED: &str = "8100 0000 0000 0028";

fn cards() -> PiiDetector {
    only(Extractor::Cursor, &[PiiKind::Card])
}

#[test]
fn union_pay_prefix_in_every_format() {
    for number in NUMBERS {
        for formatted in [
            number.to_owned(),
            grouped(number, 4, ' '),
            grouped(number, 4, '-'),
            grouped(number, 1, ' '),
            grouped(number, 1, '-'),
            format!("8-{}", &number[1..]),
        ] {
            for prefix in ["", "😀 Card: "] {
                let text = format!("{prefix}{formatted}.");
                let start = prefix.len() as u32;
                assert_found(&cards(), &text, &[(start, start + formatted.len() as u32, "card", number)]);
                let pii = layer(&cards(), &text);
                assert_eq!(mask_all(&text, &pii, '*').unwrap(), format!("{prefix}{}.", "*".repeat(formatted.len())));
                assert_eq!(Pseudonymizer::new("", "").rewrite(&text, &pii).unwrap().text(), format!("{prefix}CARD-1."));
            }
        }
    }
}

#[test]
fn wrong_check_digits() {
    for number in NUMBERS {
        let last = number.as_bytes()[number.len() - 1];
        for digit in b'0'..=b'9' {
            if digit != last {
                assert_none(&cards(), &format!("{}{}", &number[..number.len() - 1], digit as char));
            }
        }
    }
}

#[test]
fn wrong_lengths() {
    for number in ["810000000028", "81000000000000000028"] {
        assert!(luhn::valid(number.as_bytes()));
        assert_none(&cards(), number);
    }
}

#[test]
fn other_leading_eight_prefixes() {
    for (digit, check) in [(0, 9), (2, 7), (3, 6), (4, 5), (5, 4), (6, 3), (7, 2), (8, 1), (9, 0)] {
        let number = format!("8{digit}{}2{check}", "0".repeat(12));
        assert!(luhn::valid(number.as_bytes()));
        assert_none(&cards(), &number);
        assert_none(&cards(), &grouped(&number, 4, ' '));
    }
}

#[test]
fn other_leading_digits() {
    for (first, check, accepted) in [
        (0, 5, false),
        (1, 3, false),
        (2, 1, true),
        (3, 9, true),
        (4, 7, true),
        (5, 4, true),
        (6, 2, true),
        (7, 0, false),
        (9, 6, false),
    ] {
        let number = format!("{first}1{}2{check}", "0".repeat(12));
        assert!(luhn::valid(number.as_bytes()));
        if accepted {
            assert_found(&cards(), &number, &[(0, 16, "card", &number)]);
        } else {
            assert_none(&cards(), &number);
        }
    }
}

#[test]
fn malformed_prefixes() {
    for prefix in
        ["8", "8  1", "8--1", "8 -1", "8\t1", "8\n1", "8.1", "8,1", "8\u{a0}1", "8\u{661}", "\u{ff18}1", "81x"]
    {
        assert_none(&cards(), &format!("{prefix}{}", &EXAMPLE[2..]));
    }
}

#[test]
fn word_boundaries() {
    for neighbour in ["a", "9", "é", "\u{661}", "\u{10428}", "\u{1D7D9}"] {
        assert_none(&cards(), &format!("{neighbour}{EXAMPLE}"));
        assert_none(&cards(), &format!("{EXAMPLE}{neighbour}"));
        let start = neighbour.len() as u32 + 1;
        assert_found(&cards(), &format!("{neighbour} {EXAMPLE} {neighbour}"), &[(start, start + 16, "card", EXAMPLE)]);
    }
}

#[test]
fn a_trailing_group_is_left_out() {
    for separator in [" ", "-"] {
        assert_found(&cards(), &format!("{FORMATTED}{separator}12345678901234567890"), &[(0, 19, "card", EXAMPLE)]);
    }
}

#[test]
fn two_cards() {
    for separator in [" ", ";", "\n", " | "] {
        let text = format!("{EXAMPLE}{separator}4111111111111111");
        let at = 16 + separator.len() as u32;
        assert_found(&cards(), &text, &[(0, 16, "card", EXAMPLE), (at, at + 16, "card", "4111111111111111")]);
    }
}

#[test]
fn packs_with_cards() {
    for d in [PiiDetector::cursor(), pack(Pack::Payment), wide()] {
        assert_found(&d, EXAMPLE, &[(0, 16, "card", EXAMPLE)]);
    }
    let text = format!("Card {FORMATTED}.");
    let pii = layer(&pack(Pack::Payment), &text);
    assert_eq!(normalized(&pack(Pack::Payment), &text), [EXAMPLE]);
    assert_eq!(mask_by_kind(&text, &pii, '*').unwrap(), "Card **** **** **** 0028.");
    let keep = MaskPolicy::new('#').unwrap().keeping_format().keeping_trailing(4);
    assert_eq!(mask(&text, pii.mentions().map(|m| (m.span, keep))).unwrap(), "Card #### #### #### 0028.");
}
