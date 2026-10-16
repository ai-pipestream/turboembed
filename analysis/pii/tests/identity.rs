//! `UsIdentityPiiExtractorTest`, `ItinZeroSerialRegressionTest`,
//! `EuIdentityPiiExtractorTest` and `NationalIdentityPiiExtractorTest`:
//! SSNs, ITINs, NHS numbers and Steuer-IDs.

mod support;

use support::*;
use turbo_pii::rewrite::{HmacTokenizer, Pseudonymizer, mask_all};
use turbo_pii::{Extractor, Pack, PiiDetector, PiiKind};

fn us() -> PiiDetector {
    pack(Pack::UsIdentity)
}

fn eu() -> PiiDetector {
    pack(Pack::EuIdentity)
}

#[test]
fn social_security_numbers() {
    for (text, norm) in [
        ("123-45-6789", "123-45-6789"),
        ("001-01-0001", "001-01-0001"),
        ("899-99-9999", "899-99-9999"),
        ("123 45 6789", "123-45-6789"),
        ("001 01 0001", "001-01-0001"),
    ] {
        assert_found(&us(), text, &[(0, text.len() as u32, "us-ssn", norm)]);
    }
}

#[test]
fn numbers_never_issued_and_near_misses() {
    for text in [
        "000-45-6789",
        "666-45-6789",
        "123-00-6789",
        "123-45-0000",
        "000-00-0000",
        "123456789",
        "12-345-6789",
        "1234-56-789",
        "123-456789",
        "123-45-678",
        "123-45-67890",
        "123 45-6789",
        "123-45 6789",
        "123.45.6789",
        "x123-45-6789",
        "123-45-6789-0",
        "900-49-1234",
        "900-66-1234",
        "900-69-1234",
        "900-89-1234",
        "900-93-1234",
        "999-00-1234",
        "001-01-0000",
        "665-99-0000",
        "899-99-0000",
        "no identifier here",
        "call (555) 123-4567",
        "2026-08-10",
        "",
        "the ratio 10-10-10 of parts",
    ] {
        assert_none(&us(), text);
    }
}

#[test]
fn taxpayer_identification_numbers() {
    for (text, norm) in [
        ("900-70-1234", "900-70-1234"),
        ("900-70-0000", "900-70-0000"),
        ("900 70 0000", "900-70-0000"),
        ("999-88-9999", "999-88-9999"),
        ("912-50-1234", "912-50-1234"),
        ("923-99-1234", "923-99-1234"),
        ("934-94-0001", "934-94-0001"),
        ("900-65-1234", "900-65-1234"),
        ("900 92 1234", "900-92-1234"),
    ] {
        assert_found(&us(), text, &[(0, 11, "us-itin", norm)]);
    }
}

#[test]
fn us_spans_and_subsets() {
    assert_eq!(covered(&us(), "Her SSN is 123-45-6789, filed in 2026."), ["123-45-6789"]);
    let text = "SSN 123-45-6789 and ITIN 900-70-1234";
    assert_eq!(kinds(&us(), text), ["us-ssn", "us-itin"]);
    assert_eq!(kinds(&only(Extractor::UsIdentity, &[PiiKind::UsSsn]), text), ["us-ssn"]);
    assert_eq!(kinds(&only(Extractor::UsIdentity, &[PiiKind::UsItin]), text), ["us-itin"]);
    let text = "SSN 123-45-6789; ITIN 900-50-0000.";
    assert_found(&only(Extractor::UsIdentity, &[PiiKind::UsSsn]), text, &[(4, 15, "us-ssn", "123-45-6789")]);
    assert_found(&only(Extractor::UsIdentity, &[PiiKind::UsItin]), text, &[(22, 33, "us-itin", "900-50-0000")]);
}

fn itin_group(group: u32) -> bool {
    (50..=65).contains(&group) || (70..=88).contains(&group) || (90..=92).contains(&group) || group >= 94
}

#[test]
fn zero_serials_in_itin_ranges() {
    let mut accepted = Vec::new();
    for group in (0..100).filter(|&g| itin_group(g)) {
        accepted.push(format!("912-{group}-0000"));
    }
    for area in [900, 999] {
        for group in [50, 65, 70, 88, 90, 92, 94, 99] {
            accepted.push(format!("{area}-{group}-0000"));
        }
        for group in [49, 66, 69, 89, 93] {
            assert_none(&us(), &format!("{area}-{group}-0000"));
        }
    }
    for text in accepted {
        assert_found(&us(), &text, &[(0, 11, "us-itin", &text)]);
    }
    let start = "😀 ITIN ".len() as u32;
    assert_found(&us(), "😀 ITIN 900 50 0000.", &[(start, start + 11, "us-itin", "900-50-0000")]);
}

#[test]
fn us_groups() {
    for area in [899, 900, 999] {
        for group in 0..100 {
            let text = format!("{area}-{group:02}-1234");
            let kind = if area < 900 && group > 0 {
                Some("us-ssn")
            } else if area >= 900 && itin_group(group) {
                Some("us-itin")
            } else {
                None
            };
            match kind {
                Some(kind) => assert_found(&us(), &text, &[(0, 11, kind, &text)]),
                None => assert_none(&us(), &text),
            }
        }
    }
}

#[test]
fn identity_packs_mask_replace_and_tokenize_the_itin() {
    let text = "Tax ID 900-50-0000.";
    for d in [us(), wide()] {
        assert_found(&d, text, &[(7, 18, "us-itin", "900-50-0000")]);
        let pii = layer(&d, text);
        assert_eq!(mask_all(text, &pii, '*').unwrap(), "Tax ID ***********.");
        assert_eq!(Pseudonymizer::new("", "").rewrite(text, &pii).unwrap().text(), "Tax ID US-ITIN-1.");
        let tokenizer = HmacTokenizer::new(&[0; 32]).unwrap();
        let token = tokenizer.token(PiiKind::UsItin, "900-50-0000").unwrap();
        assert_eq!(tokenizer.rewrite(text, &pii).unwrap().text(), format!("Tax ID {token}."));
    }
}

#[test]
fn nhs_numbers() {
    for (text, norm) in [
        ("9434765919", "9434765919"),
        ("9999999468", "9999999468"),
        ("4010232137", "4010232137"),
        ("1234567881", "1234567881"),
        ("9876543210", "9876543210"),
        ("943 476 5919", "9434765919"),
        ("943-476-5919", "9434765919"),
        ("999 999 9468", "9999999468"),
    ] {
        assert_found(&eu(), text, &[(0, text.len() as u32, "uk-nhs", norm)]);
    }
    for text in [
        "9434765918",
        "9999999469",
        "4010232138",
        "1234567882",
        "1234567890",
        "943 476 5918",
        "0000000000",
        "1111111111",
    ] {
        assert_none(&eu(), text);
    }
    for text in
        ["943476591", "94347659190", "9434 76 5919", "943 4765919", "943-476 5919", "x9434765919", "9434765919x"]
    {
        assert!(!kinds(&eu(), text).contains(&"uk-nhs"), "{text}");
    }
}

#[test]
fn steuer_ids() {
    for (text, norm) in [
        ("65929970489", "65929970489"),
        ("81095324717", "81095324717"),
        ("23746189575", "23746189575"),
        ("50123456782", "50123456782"),
        ("65 929 970 489", "65929970489"),
        ("65-929-970-489", "65929970489"),
    ] {
        assert_found(&eu(), text, &[(0, text.len() as u32, "de-steuer-id", norm)]);
    }
    for text in [
        "65929970488",
        "81095324716",
        "23746189574",
        "50123456783",
        "02476291358",
        "12345678903",
        "11223456785",
        "11123456786",
    ] {
        assert_none(&eu(), text);
    }
}

#[test]
fn eu_spans_and_subsets() {
    let text = "NHS 943 476 5919 and IdNr 65929970489 recorded.";
    assert_eq!(covered(&eu(), text), ["943 476 5919", "65929970489"]);
    let text = "NHS 9434765919 and IdNr 65929970489";
    assert_eq!(kinds(&only(Extractor::EuIdentity, &[PiiKind::UkNhs]), text), ["uk-nhs"]);
    assert_eq!(kinds(&only(Extractor::EuIdentity, &[PiiKind::DeSteuerId]), text), ["de-steuer-id"]);
    for text in ["no identifier here", "1234", "", "the year 2026", "call (555) 123-4567"] {
        assert_none(&eu(), text);
    }
    let text = "NHS: 943 476 5919; IdNr: 65-929-970-489.";
    assert_eq!(kinds(&eu(), text), ["uk-nhs", "de-steuer-id"]);
    assert_eq!(mask_all(text, &layer(&eu(), text), '*').unwrap(), "NHS: ************; IdNr: **************.");
}

/// One national identifier with its digits and groups.
struct Family {
    kind: PiiKind,
    extractor: Extractor,
    digits: &'static str,
    normalized: &'static str,
    groups: &'static [usize],
}

const FAMILIES: [Family; 4] = [
    Family {
        kind: PiiKind::UsSsn,
        extractor: Extractor::UsIdentity,
        digits: "123456789",
        normalized: "123-45-6789",
        groups: &[3, 2, 4],
    },
    Family {
        kind: PiiKind::UsItin,
        extractor: Extractor::UsIdentity,
        digits: "900701234",
        normalized: "900-70-1234",
        groups: &[3, 2, 4],
    },
    Family {
        kind: PiiKind::UkNhs,
        extractor: Extractor::EuIdentity,
        digits: "9434765919",
        normalized: "9434765919",
        groups: &[3, 3, 4],
    },
    Family {
        kind: PiiKind::DeSteuerId,
        extractor: Extractor::EuIdentity,
        digits: "65929970489",
        normalized: "65929970489",
        groups: &[2, 3, 3, 3],
    },
];

impl Family {
    fn detector(&self) -> PiiDetector {
        only(self.extractor, &[self.kind])
    }

    fn grouped(&self, separators: &[&str]) -> String {
        let mut out = String::new();
        let mut at = 0;
        for (i, &g) in self.groups.iter().enumerate() {
            if i > 0 {
                out.push_str(separators[i - 1]);
            }
            out.push_str(&self.digits[at..at + g]);
            at += g;
        }
        out
    }

    fn with(&self, separator: &str) -> String {
        self.grouped(&vec![separator; self.groups.len() - 1])
    }

    #[track_caller]
    fn assert_form(&self, number: &str, accepted: bool, prefix: &str, suffix: &str) {
        let text = format!("{prefix}{number}{suffix}");
        let start = prefix.len() as u32;
        if accepted {
            assert_found(
                &self.detector(),
                &text,
                &[(start, start + number.len() as u32, self.kind.name(), self.normalized)],
            );
        } else {
            assert_none(&self.detector(), &text);
        }
    }
}

#[test]
fn separators() {
    for f in &FAMILIES {
        let points = (0..128u32).chain([0xa0, 0x2007, 0x202f, 0x2010, 0x2212, 0x1f600, 0x10400]);
        for cp in points {
            let c = char::from_u32(cp).unwrap().to_string();
            f.assert_form(&f.with(&c), cp == 0x20 || cp == u32::from(b'-'), "😀 ", ".");
        }
    }
}

#[test]
fn mixed_separators() {
    // The Java appends `(char) 0`, so the first separator is NUL.
    let alphabet = ["\0", " ", "-"];
    for f in &FAMILIES {
        let count = f.groups.len() - 1;
        let combinations = if count == 2 { 9 } else { 27 };
        for code in 0..combinations {
            let mut rest = code;
            let mut separators = Vec::new();
            for _ in 0..count {
                separators.push(alphabet[rest % 3]);
                rest /= 3;
            }
            let accepted = separators.iter().all(|&s| s == " ") || separators.iter().all(|&s| s == "-");
            f.assert_form(&f.grouped(&separators), accepted, "", "");
        }
    }
}

#[test]
fn boundaries() {
    for f in &FAMILIES {
        let number = f.with("-");
        for word in ["a", "9", "é", "中", "𐐀", "١", "𝟙"] {
            f.assert_form(&number, false, word, "");
            f.assert_form(&number, false, "", word);
        }
        for prefix in ["", " ", "😀", "(", "é "] {
            f.assert_form(&number, true, prefix, ".");
        }
        for continuation in [".1", ",1", "-1"] {
            f.assert_form(&number, false, "", continuation);
        }
    }
}

#[test]
fn compact_forms_and_recovery() {
    for f in &FAMILIES {
        let compact = matches!(f.kind, PiiKind::UkNhs | PiiKind::DeSteuerId);
        f.assert_form(f.digits, compact, "", "");
        let invalid = f.with("\t");
        f.assert_form(&f.with(" "), true, &format!("{invalid}; 😀 "), ".");
    }
}

#[test]
fn identities_on_every_thread() {
    std::thread::scope(|s| {
        for worker in 0..8 {
            s.spawn(move || {
                let prefix = format!("{} ", "😀".repeat(worker));
                for _ in 0..32 {
                    for f in &FAMILIES {
                        f.assert_form(&f.with("-"), true, &prefix, ".");
                    }
                }
            });
        }
    });
}
