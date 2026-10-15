//! `CaIdentityPiiExtractorTest`, `DevicePiiExtractorTest` and
//! `LabeledPiiExtractorTest`: Canadian SINs and IMEIs after a label.

mod support;

use support::*;
use turbo_pii::rewrite::mask_all;
use turbo_pii::{Extractor, Pack, PiiDetector, PiiKind, Scanner};

const SIN: &str = "046454286";
const IMEI: &str = "490154203237518";

fn sin() -> PiiDetector {
    pack(Pack::CaIdentity)
}

fn imei() -> PiiDetector {
    pack(Pack::Device)
}

#[test]
fn labelled_sins() {
    for text in [
        "SIN: 046 454 286",
        "sin=046454286",
        "Social Insurance Number: 046-454-286",
        "social insurance no. 046 454 286",
    ] {
        assert_eq!(
            rows(&sin(), text).into_iter().map(|r| (r.2, r.3)).collect::<Vec<_>>(),
            [("ca-sin", SIN.to_owned())],
            "{text}"
        );
    }
    for text in [
        "046 454 286",
        "SIN: 046 454 287",
        "SIN: 046-454 286",
        "SIN: 04645428",
        "SIN: 0464542860",
        "SIN was recorded as 046 454 286",
        "casino: 046 454 286",
    ] {
        assert_none(&sin(), text);
    }
    assert_eq!(covered(&sin(), "Employee SIN: 046 454 286."), ["046 454 286"]);
}

#[test]
fn labelled_imeis() {
    for (text, norm) in [
        ("IMEI: 490154203237518", IMEI),
        ("imei # 490154203237518", IMEI),
        ("IMEI=35-693803-564380-9", "356938035643809"),
    ] {
        assert_eq!(normalized(&imei(), text), [norm], "{text}");
        assert_eq!(kinds(&imei(), text), ["imei"]);
    }
    for text in [
        "490154203237518",
        "IMEI: 490154203237519",
        "IMEI: 49015420323751",
        "IMEI: 4901542032375180",
        "IMEI was recorded as 490154203237518",
        "xIMEI: 490154203237518",
    ] {
        assert_none(&imei(), text);
    }
    assert_eq!(covered(&imei(), "Device IMEI: 490154203237518."), [IMEI]);
}

struct Family {
    detector: fn() -> PiiDetector,
    kind: &'static str,
    digits: &'static str,
    labels: &'static [&'static str],
}

const FAMILIES: [Family; 2] = [
    Family {
        detector: sin,
        kind: "ca-sin",
        digits: SIN,
        labels: &["SIN", "Social Insurance Number", "Social Insurance No."],
    },
    Family { detector: imei, kind: "imei", digits: IMEI, labels: &["IMEI"] },
];

impl Family {
    #[track_caller]
    fn assert_number(&self, number: &str, accepted: bool) {
        let prefix = format!("{}: ", self.labels[0]);
        let text = format!("{prefix}{number}.");
        let start = prefix.len() as u32;
        if accepted {
            assert_found(&(self.detector)(), &text, &[(start, start + number.len() as u32, self.kind, self.digits)]);
        } else {
            assert_none(&(self.detector)(), &text);
        }
    }
}

#[test]
fn label_boundaries() {
    for f in &FAMILIES {
        for label in f.labels {
            for (prefix, accepted) in [
                ("x", false),
                ("9", false),
                ("é", false),
                ("中", false),
                ("𐐀", false),
                ("𝕒", false),
                ("١", false),
                ("𝟙", false),
                ("", true),
                (" ", true),
                ("(", true),
                ("😀", true),
                ("é ", true),
                ("𝕒 ", true),
            ] {
                let before = format!("{prefix}{label}: ");
                let text = format!("{before}{}.", f.digits);
                let start = before.len() as u32;
                if accepted {
                    assert_found(&(f.detector)(), &text, &[(start, start + f.digits.len() as u32, f.kind, f.digits)]);
                } else {
                    assert_none(&(f.detector)(), &text);
                }
            }
        }
    }
}

#[test]
fn sin_grouping() {
    let f = &FAMILIES[0];
    f.assert_number(SIN, true);
    for separator in [" ", "-"] {
        for first in 1..SIN.len() {
            f.assert_number(&format!("{}{separator}{}", &SIN[..first], &SIN[first..]), false);
            for second in first + 1..SIN.len() {
                let number =
                    format!("{}{separator}{}{separator}{}", &SIN[..first], &SIN[first..second], &SIN[second..]);
                f.assert_number(&number, first == 3 && second == 6);
            }
        }
    }
    for number in
        ["046 454-286", "046-454 286", "046  454 286", "046--454-286", "046\t454\t286", "046\u{a0}454\u{a0}286"]
    {
        f.assert_number(number, false);
    }
}

#[test]
fn imei_grouping() {
    let f = &FAMILIES[1];
    for number in [IMEI, "49-015420-3237518", "49015 42032 37518"] {
        f.assert_number(number, true);
    }
    for separator in [" ", "-"] {
        for at in 1..IMEI.len() {
            f.assert_number(&format!("{}{separator}{}", &IMEI[..at], &IMEI[at..]), true);
        }
    }
    for number in [
        "49-015420 3237518",
        "49 015420-3237518",
        "49  015420 3237518",
        "49--015420-3237518",
        "49\t015420\t3237518",
        "49\u{a0}015420\u{a0}3237518",
    ] {
        f.assert_number(number, false);
    }
}

#[test]
fn checksums_and_zeros() {
    for f in &FAMILIES {
        f.assert_number(f.digits, true);
        for i in 0..f.digits.len() {
            let mut b = f.digits.as_bytes().to_vec();
            b[i] = b'0' + (b[i] - b'0' + 1) % 10;
            f.assert_number(std::str::from_utf8(&b).unwrap(), false);
        }
        let zeros = "0".repeat(f.digits.len());
        assert_eq!(normalized(&(f.detector)(), &format!("{}: {zeros}", f.labels[0])), [zeros]);
    }
}

#[test]
fn confidence_is_labelled_checksum() {
    let pii = layer(&imei(), "IMEI: 490154203237518");
    assert_eq!(pii.mentions().next().unwrap().confidence.name(), "labelled-checksum");
}

#[test]
fn combined_packs() {
    let d = PiiDetector::new(vec![Scanner::all(Extractor::CaIdentity), Scanner::all(Extractor::Device)]);
    let text = "SIN: 046 454 286; IMEI: 490154203237518.";
    assert_eq!(kinds(&d, text), ["ca-sin", "imei"]);
    assert_eq!(mask_all(text, &layer(&d, text), '*').unwrap(), "SIN: ***********; IMEI: ***************.");
    assert_eq!(d.kinds().iter().collect::<Vec<_>>(), [PiiKind::Imei, PiiKind::CaSin]);
}

#[test]
fn labels_compare_ascii_case_only() {
    for text in ["😀sIn: 046454286", "iMeI: 490154203237518"] {
        assert!(kinds(&wide(), text).iter().any(|k| *k == "ca-sin" || *k == "imei"), "{text}");
    }
    for text in ["sİn: 046454286", "ımei: 490154203237518"] {
        assert!(!kinds(&wide(), text).iter().any(|k| *k == "ca-sin" || *k == "imei"), "{text}");
    }
}

#[test]
fn labelled_values_on_every_thread() {
    std::thread::scope(|s| {
        for worker in 0..8 {
            s.spawn(move || {
                let prefix = format!("{} ", "😀".repeat(worker));
                for _ in 0..32 {
                    for f in &FAMILIES {
                        let before = format!("{prefix}{}: ", f.labels[0]);
                        let start = before.len() as u32;
                        assert_found(
                            &(f.detector)(),
                            &format!("{before}{}", f.digits),
                            &[(start, start + f.digits.len() as u32, f.kind, f.digits)],
                        );
                    }
                }
            });
        }
    });
}
