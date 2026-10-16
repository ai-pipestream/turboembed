//! `BankingPiiExtractorTest` and `BankingPiiPolicyTest`: ABA routing
//! numbers by check digit and Federal Reserve prefix.

mod support;

use support::*;
use turbo_pii::rewrite::{Pseudonymizer, mask_all};
use turbo_pii::{Extractor, Pack, PiiDetector, Scanner};

fn banking() -> PiiDetector {
    PiiDetector::new(vec![Scanner::all(Extractor::Banking)])
}

const PREFIXES: [u32; 38] = [
    0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 61, 62, 63, 64, 65, 66,
    67, 68, 69, 70, 71, 72, 80,
];
const GOVERNMENT: &str = "001234561";

/// A routing number of `prefix` and `body` with its check digit.
fn routing(prefix: u32, body: &str) -> String {
    let value = format!("{prefix:02}{body}");
    let d: Vec<u32> = value.bytes().map(|c| u32::from(c - b'0')).collect();
    let (a, b, c) = (d[0] + d[3] + d[6], d[1] + d[4] + d[7], d[2] + d[5]);
    format!("{value}{}", (7 * a + 3 * b + 9 * c) % 10)
}

#[test]
fn routing_numbers() {
    for text in [
        "011000015",
        "021000021",
        "026009593",
        "111000025",
        "121000358",
        "071000013",
        "072000326",
        "124003116",
        "061000104",
        "307070115",
        "322271627",
        "120000003",
        "210000007",
        "610000005",
        "800000006",
    ] {
        assert_found(&banking(), text, &[(0, 9, "aba-routing", text)]);
    }
}

#[test]
fn broken_check_digits_reserved_prefixes_and_near_misses() {
    for text in [
        "021000022",
        "011000016",
        "121000359",
        "111000026",
        "322271628",
        "123456789",
        "111111111",
        "130000006",
        "140000009",
        "200000004",
        "330000000",
        "600000002",
        "730000008",
        "810000009",
        "900000003",
        "000000000",
        "02100002",
        "0210000210",
        "021 000 021",
        "021-000-021",
        "x021000021",
        "021000021x",
        "1021000021",
        "021000021.5",
        "021000021,5",
        "1.021000021",
        "nothing here",
        "call (555) 123-4567",
        "1234",
        "",
        "the year 2026 and the number 42",
    ] {
        assert_none(&banking(), text);
    }
}

#[test]
fn routing_numbers_in_a_sentence() {
    assert_eq!(covered(&banking(), "Wire to routing 021000021, account 12345678."), ["021000021"]);
    assert_eq!(normalized(&banking(), "from 021000021 to 121000358"), ["021000021", "121000358"]);
}

#[test]
fn prefix_allocation() {
    for prefix in 0..100 {
        for body in ["123456", "987654"] {
            let number = routing(prefix, body);
            if PREFIXES.contains(&prefix) {
                assert_found(&banking(), &number, &[(0, 9, "aba-routing", &number)]);
            } else {
                assert_none(&banking(), &number);
            }
        }
    }
}

#[test]
fn changed_digits_are_refused() {
    for prefix in PREFIXES {
        let valid = routing(prefix, "123456");
        assert_eq!(rows(&banking(), &valid).len(), 1);
        for i in 0..valid.len() {
            let mut b = valid.clone().into_bytes();
            b[i] = b'0' + (b[i] - b'0' + 1) % 10;
            assert_none(&banking(), std::str::from_utf8(&b).unwrap());
        }
    }
}

#[test]
fn government_routing_in_every_pack_that_has_it() {
    assert_eq!(routing(0, "123456"), GOVERNMENT);
    for d in [banking(), pack(Pack::Payment), wide()] {
        let text = format!("Routing {GOVERNMENT}.");
        assert_found(&d, &text, &[(8, 17, "aba-routing", GOVERNMENT)]);
        let pii = layer(&d, &text);
        assert_eq!(mask_all(&text, &pii, '*').unwrap(), "Routing *********.");
        assert_eq!(Pseudonymizer::new("", "").rewrite(&text, &pii).unwrap().text(), "Routing ABA-ROUTING-1.");
    }
}

#[test]
fn numeric_and_word_continuations_are_refused() {
    for adjacent in ["x", "é", "𐐀", "𝟙", "١", "1", "1.", "1,"] {
        assert_none(&banking(), &format!("{adjacent}{GOVERNMENT}"));
        let reversed: String = adjacent.chars().rev().collect();
        assert_none(&banking(), &format!("{GOVERNMENT}{reversed}"));
    }
}

#[test]
fn offsets_are_bytes_of_the_original() {
    for prefix in ["", " ", "(", "😀 ", "𝕒 ", "e\u{301} "] {
        let start = prefix.len() as u32;
        assert_found(&banking(), &format!("{prefix}{GOVERNMENT})"), &[(start, start + 9, "aba-routing", GOVERNMENT)]);
    }
}

#[test]
fn invalid_numeric_forms() {
    for text in [
        "000000000",
        "0000000000",
        "000 000 000",
        "00-1234561",
        "0012345610",
        "0001234561",
        "001 234 561",
        "00",
        "",
        "001234561.5",
        "1,001234561",
    ] {
        assert_none(&banking(), text);
    }
}

#[test]
fn routing_on_every_thread() {
    let d = banking();
    std::thread::scope(|s| {
        for worker in 0..8 {
            let d = &d;
            s.spawn(move || {
                let number = routing(worker, "123456");
                for _ in 0..32 {
                    assert_found(d, &format!("😀 {number}."), &[(5, 14, "aba-routing", &number)]);
                }
            });
        }
    });
}
