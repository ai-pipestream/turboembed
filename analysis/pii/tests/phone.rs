//! `PhoneSpanTest`, `PhoneMetadataLengthsTest` and
//! `InternationalPhonePrefixTest`: whole phone spans in context, lengths
//! by calling code, and no zero before the calling code.

mod support;

use support::*;
use turbo_pii::rewrite::{Pseudonymizer, mask_all};
use turbo_pii::{Extractor, Pack, PiiDetector, PiiKind};

fn phones() -> PiiDetector {
    only(Extractor::Cursor, &[PiiKind::Phone])
}

const FORMATS: [(&str, &str); 10] = [
    ("(415) 555-0123", "4155550123"),
    ("+1 (415) 555-0123", "+14155550123"),
    ("415 (5550123)", "4155550123"),
    ("+1 (4155550123)", "+14155550123"),
    ("(415 555 0123)", "4155550123"),
    ("+44 (20 7946 0958)", "+442079460958"),
    ("+683 (4002)", "+6834002"),
    ("1(4155550123)", "14155550123"),
    ("415555(0123)", "4155550123"),
    ("1 (4155550123)", "14155550123"),
];

/// The phone is found whole, masked whole and replaced whole.
#[track_caller]
fn assert_whole(d: &PiiDetector, prefix: &str, phone: &str, suffix: &str, norm: &str) {
    let text = format!("{prefix}{phone}{suffix}");
    let start = prefix.len() as u32;
    assert_found(d, &text, &[(start, start + phone.len() as u32, "phone", norm)]);
    let pii = layer(d, &text);
    assert_eq!(mask_all(&text, &pii, '*').unwrap(), format!("{prefix}{}{suffix}", "*".repeat(phone.len())));
    assert_eq!(Pseudonymizer::new("", "").rewrite(&text, &pii).unwrap().text(), format!("{prefix}PHONE-1{suffix}"));
}

#[test]
fn complete_phone_spans() {
    for (phone, norm) in FORMATS {
        for prefix in ["", "Call: ", "😀 ", "e\u{301} "] {
            for suffix in ["", ".", "]", " 😀", ")"] {
                assert_whole(&phones(), prefix, phone, suffix, norm);
            }
        }
    }
}

#[test]
fn adjacent_phones() {
    for (phone, norm) in FORMATS {
        let text = format!("{phone}(415) 555-0124");
        let at = phone.len() as u32;
        assert_found(&phones(), &text, &[(0, at, "phone", norm), (at, text.len() as u32, "phone", "4155550124")]);
    }
}

#[test]
fn a_trailing_group_keeps_the_complete_number() {
    for (phone, norm) in FORMATS {
        let text = format!("{phone} 123456789012345678901234");
        assert_found(&phones(), &text, &[(0, phone.len() as u32, "phone", norm)]);
    }
}

#[test]
fn refused_numbers() {
    for text in ["+123", "415-555-012", "4155550123", "+999 (1234)", "123(456)", "+65 (123456)"] {
        assert_none(&phones(), text);
    }
}

#[test]
fn contact_pack_masks_a_parenthesized_international_number() {
    assert_whole(&pack(Pack::Contact), "Call ", "+1 (4155550123)", ".", "+14155550123");
}

#[test]
fn national_lengths_by_calling_code() {
    for (code, lengths) in [("220", &[7, 9][..]), ("263", &[7, 9, 10][..])] {
        for length in 1..=12 {
            for separator in ["", " "] {
                let national = "2".repeat(length);
                let phone = format!("+{code}{separator}{national}");
                let text = format!("😀 Call: {phone}.");
                if lengths.contains(&length) {
                    assert_whole(&phones(), "😀 Call: ", &phone, ".", &format!("+{code}{national}"));
                } else {
                    assert_none(&phones(), &text);
                }
            }
        }
    }
}

#[test]
fn calling_code_prefixes() {
    let numbers = [
        ("1", "4155550123"),
        ("7", "1234567890"),
        ("20", "1234567890"),
        ("44", "2079460958"),
        ("49", "1234567890"),
        ("65", "12345678"),
        ("299", "123456"),
        ("683", "4002"),
    ];
    for (code, national) in numbers {
        for zeros in 0..3 {
            for separator in ["", " "] {
                for prefix in ["", "😀 Call: "] {
                    for suffix in ["", "."] {
                        let phone = format!("+{}{code}{separator}{national}", "0".repeat(zeros));
                        if zeros == 0 {
                            assert_whole(&phones(), prefix, &phone, suffix, &format!("+{code}{national}"));
                        } else {
                            assert_none(&phones(), &format!("{prefix}{phone}{suffix}"));
                        }
                    }
                }
            }
        }
    }
}

#[test]
fn national_zeros_are_kept() {
    for digits in ["390212345678", "441020304050", "10000000000"] {
        let text = format!("+{digits}");
        assert_found(&phones(), &text, &[(0, text.len() as u32, "phone", &text)]);
    }
}

#[test]
fn a_zero_before_the_calling_code_is_refused() {
    let text = "Invalid +014155550123; call +14155550123.";
    let pii = layer(&pack(Pack::Contact), text);
    assert_eq!(normalized(&pack(Pack::Contact), text), ["+14155550123"]);
    assert_eq!(mask_all(text, &pii, '*').unwrap(), "Invalid +014155550123; call ************.");
}

#[test]
fn spans_are_the_same_on_every_thread() {
    let d = phones();
    std::thread::scope(|s| {
        for worker in 0..8 {
            let d = &d;
            s.spawn(move || {
                let prefix = format!("😀 {}", " ".repeat(worker));
                for _ in 0..16 {
                    for (phone, norm) in FORMATS {
                        let start = prefix.len() as u32;
                        assert_found(
                            d,
                            &format!("{prefix}{phone}."),
                            &[(start, start + phone.len() as u32, "phone", norm)],
                        );
                    }
                }
            });
        }
    });
}
