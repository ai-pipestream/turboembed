//! #1878: vehicle identification numbers, parcel tracking numbers, US
//! Employer Identification Numbers and web addresses. Positive, negative
//! and check digit failure cases per scanner, single-digit mutation
//! sweeps, label boosts, masking, and precision and recall on a synthetic
//! set made here from a fixed seed. The values are made up with valid
//! check digits, the UPS sample number, or documentation domains.

mod support;

use support::*;
use turbo_pii::checksum::{luhn, parcel, vin};
use turbo_pii::rewrite::mask_by_kind;
use turbo_pii::{Extractor, Pack, PiiConfidence, PiiDetector, PiiKind};

fn logistics() -> PiiDetector {
    pack(Pack::Logistics)
}

fn found(d: &PiiDetector, text: &str) -> Vec<(&'static str, String)> {
    rows(d, text).into_iter().map(|r| (r.2, r.3)).collect()
}

fn covered_by(d: &PiiDetector, text: &str, kind: &str) -> Vec<String> {
    rows(d, text).into_iter().filter(|r| r.2 == kind).map(|r| text[r.0 as usize..r.1 as usize].to_owned()).collect()
}

fn next_digit(c: u8) -> u8 {
    if c == b'9' { b'0' } else { c + 1 }
}

/// Every one-digit change of `value` (digit to the next digit) is no
/// longer a `kind` mention.
fn digit_mutations_refused(d: &PiiDetector, kind: &str, value: &str) {
    assert_eq!(covered_by(d, value, kind), [value], "{value} itself");
    for (i, &c) in value.as_bytes().iter().enumerate() {
        if c.is_ascii_digit() {
            let mut b = value.as_bytes().to_vec();
            b[i] = next_digit(c);
            let text = String::from_utf8(b).unwrap();
            assert!(covered_by(d, &text, kind).is_empty(), "{text}");
        }
    }
}

#[test]
fn vins_with_their_check_character() {
    let d = logistics();
    for v in ["1FTFW1E51NFA00001", "3VWDB7AJ0HM000002", "1M8GDM9AXKP042788", "1HGCM82633A004352"] {
        assert_eq!(found(&d, &format!("VIN {v}.")), [("vin", v.to_owned())], "{v}");
        assert_eq!(found(&d, &format!("({v})")), [("vin", v.to_owned())], "{v}");
    }
    for not in [
        "1FTFW1E52NFA00001",  // check character wrong
        "1FTFW1E51NFAO0001",  // O is no VIN character
        "1FTFW1E51NIA00001",  // nor is I
        "11111111111111111",  // valid check, but no letter
        "1FTFW1E51NFA000011", // 18 characters
        "x1FTFW1E51NFA00001", // inside a word
        "1ftfw1e51nfa00001",  // lower case
        "part-1FTFW1E51NFA00001",
        "1FTFW1E51NFA00001-B",
    ] {
        assert!(found(&d, not).is_empty(), "{not}");
    }
    assert!(vin::valid(b"11111111111111111"));
    digit_mutations_refused(&d, "vin", "1FTFW1E51NFA00001");
}

#[test]
fn vin_model_year_character_is_never_u_z_or_zero() {
    let d = logistics();
    // A tenth character U, Z or 0 is refused even with a matching check.
    for year in *b"UZ0" {
        let mut v = *b"1FTFW1E50NFA00001";
        v[9] = year;
        v[8] = vin::check_char(&v).unwrap();
        assert!(vin::valid(&v));
        let text = String::from_utf8(v.to_vec()).unwrap();
        assert!(found(&d, &text).is_empty(), "{text}");
    }
}

#[test]
fn tracking_numbers_of_each_carrier() {
    let d = logistics();
    for (text, want) in [
        ("1Z999AA10123456784", "1Z999AA10123456784"),
        ("1Z 999 AA1 01 2345 678 4", "1Z999AA10123456784"),
        ("841654653682", "841654653682"),
        ("8416 5465 3682", "841654653682"),
        ("121513302427144", "121513302427144"),
        ("9611020508440894014924", "9611020508440894014924"),
        ("9400 2229 1175 1694 0803 50", "9400222911751694080350"),
        ("9205867800593239130527", "9205867800593239130527"),
        ("03074961150175938193", "03074961150175938193"),
        ("7012 3456 7890 1234 5676", "70123456789012345676"),
    ] {
        let sentence = format!("Sent as {text} today.");
        assert_eq!(found(&d, &sentence), [("tracking-number", want.to_owned())], "{text}");
    }
}

#[test]
fn tracking_number_near_misses_are_refused() {
    let d = logistics();
    for not in [
        "1Z999AA10123456785",          // UPS check digit
        "1Z999AA1012345678X",          // UPS check is a digit
        "1Z999aa10123456784",          // lower case serial
        "841654653683",                // FedEx Express check digit
        "121513302427145",             // GS1 check digit
        "378282246310005",             // passes Luhn: a card, not FedEx Ground
        "9400 2229 1175 1694 0803 51", // GS1 check digit
        "9700222911751694080354",      // 22 digits, no USPS or FedEx prefix
        "2024 0131 0942 1122 3344",    // 20 digits not starting 0 or 7
        "841 654 653 682",             // groups of three
        "84165 4653682",               // a first group of five
        "1234-841654653682",
        "841654653682-7",
    ] {
        assert!(covered_by(&d, not, "tracking-number").is_empty(), "{not}");
    }
    // 20-digit USPS check over the digits as written, or with the 91 prefix
    // the legacy numbers imply.
    assert_eq!(parcel::gs1_mod10(b"0307496115017593819"), Some(3));
    assert!(luhn::valid(b"378282246310005"));
    digit_mutations_refused(&d, "tracking-number", "9400222911751694080350");
    digit_mutations_refused(&d, "tracking-number", "1Z999AA10123456784");
    digit_mutations_refused(&d, "tracking-number", "121513302427144");
}

#[test]
fn eins_with_an_assigned_prefix() {
    let d = only(Extractor::UsIdentity, &[PiiKind::UsEin]);
    for v in ["12-3456789", "46-1234567", "94-7654321", "01-0000001", "99-9999999"] {
        assert_eq!(found(&d, &format!("Vendor {v}.")), [("us-ein", v.to_owned())], "{v}");
    }
    for not in [
        "07-1234567",  // prefix 07 is not assigned
        "00-1234567",  // nor 00
        "96-1234567",  // nor 96
        "20-0000000",  // all-zero serial
        "123456789",   // no hyphen
        "12 3456789",  // a space is no EIN separator
        "123-456789",  // wrong grouping
        "12-34567890", // ten serial digits
        "12-3456789-1",
        "1.12-3456789",
    ] {
        assert!(found(&d, not).is_empty(), "{not}");
    }
    // The SSN shape is not an EIN, and the EIN shape is not an SSN.
    let us = pack(Pack::UsIdentity);
    assert_eq!(kinds(&us, "123-45-6789 and 12-3456789"), ["us-ssn", "us-ein"]);
}

#[test]
fn labels_boost_vins_tracking_numbers_and_eins() {
    let d = wide();
    let tier =
        |text: &str| layer(&d, text).mentions().map(|m| (m.kind.to_string(), m.confidence())).collect::<Vec<_>>();
    use PiiConfidence::*;
    assert_eq!(tier("1FTFW1E51NFA00001"), [("vin".to_owned(), Checksum)]);
    assert_eq!(tier("VIN: 1FTFW1E51NFA00001"), [("vin".to_owned(), LabelledChecksum)]);
    assert_eq!(tier("1Z999AA10123456784"), [("tracking-number".to_owned(), Checksum)]);
    assert_eq!(tier("UPS tracking 1Z999AA10123456784"), [("tracking-number".to_owned(), LabelledChecksum)]);
    assert_eq!(tier("vendor 12-3456789"), [("us-ein".to_owned(), Format)]);
    assert_eq!(tier("EIN 12-3456789"), [("us-ein".to_owned(), Labelled)]);
    assert_eq!(tier("employer identification number: 12-3456789"), [("us-ein".to_owned(), Labelled)]);
    // A caller can require the label.
    let strict = wide().label_required(PiiKind::TrackingNumber);
    assert!(covered_by(&strict, "ref 841654653682", "tracking-number").is_empty());
    assert_eq!(covered_by(&strict, "FedEx 841654653682", "tracking-number"), ["841654653682"]);
}

#[test]
fn urls_with_scheme_or_www() {
    let d = pack(Pack::Web);
    for (text, covered, normalized) in [
        ("see https://portal.example.com/vendors?id=42.", "https://portal.example.com/vendors?id=42", None),
        ("at http://status.example.org:8080/health", "http://status.example.org:8080/health", None),
        ("(www.example.net/help)", "www.example.net/help", None),
        (
            "HTTPS://Docs.Example.COM/Guide#Intro!",
            "HTTPS://Docs.Example.COM/Guide#Intro",
            Some("https://docs.example.com/Guide#Intro"),
        ),
        ("ftp://ftp.example.org/pub/.", "ftp://ftp.example.org/pub/", None),
        ("http://192.0.2.10/admin", "http://192.0.2.10/admin", None),
        ("[link](https://example.com/a_(b))", "https://example.com/a_(b)", None),
        ("'https://example.com/x'", "https://example.com/x", None),
        ("https://example.com/user:secret", "https://example.com/user:secret", None),
    ] {
        let got = rows(&d, text);
        assert_eq!(got.len(), 1, "{text}: {got:?}");
        assert_eq!(&text[got[0].0 as usize..got[0].1 as usize], covered, "{text}");
        assert_eq!(got[0].3, normalized.unwrap_or(covered), "{text}");
    }
    for not in [
        "example.com",
        "http://mirror.corp.invalid/pkg",
        "http://localhost:8080/",
        "mailto:desk@example.com",
        "https://user:pass@example.com/x",
        "https://user@example.com/x",
        "www.example",
        "sub.www.example.com",
        "xhttps://example.com",
        "http://999.1.1.1/",
        "http://-bad.example.com/",
    ] {
        assert!(rows(&d, not).is_empty(), "{not}");
    }
}

#[test]
fn credential_urls_stay_url_credentials_in_the_wide_pack() {
    let text = "fetch https://buildbot:stage-pass99@artifacts.example.com/v2/index.json now";
    assert_eq!(kinds(&wide(), text), ["url-credential", "email"]);
    // A plain URL holds an address inside it: the url is kept, the nested
    // ipv4 is dropped by default and kept in parity mode.
    let text = "open http://192.0.2.10/admin";
    assert_eq!(kinds(&wide(), text), ["url"]);
    assert_eq!(kinds(&wide().keeping_nested(), text), ["url", "ipv4"]);
}

#[test]
fn masking_keeps_separators_of_numbers_and_hides_whole_urls() {
    let d = wide();
    let text = "EIN 12-3456789, 1Z 999 AA1 01 2345 678 4, https://example.com/x";
    let masked = mask_by_kind(text, &layer(&d, text), '*').unwrap();
    assert_eq!(masked, "EIN **-*******, ** *** *** ** **** *** *, *********************");
}
