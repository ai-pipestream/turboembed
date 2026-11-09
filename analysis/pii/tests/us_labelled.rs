//! #1878: the label-dependent US identifiers (passport, driver's licence,
//! ZIP code) and telephone extensions. Positive, negative and context
//! cases per scanner, the label window defaults, and masking. Values are
//! made up.

mod support;

use support::*;
use turbo_pii::rewrite::mask_by_kind;
use turbo_pii::{Extractor, Pack, PiiConfidence, PiiDetector, PiiKind, Scanner};

fn us() -> PiiDetector {
    pack(Pack::UsIdentity)
}

fn of_kind(d: &PiiDetector, text: &str, kind: &str) -> Vec<String> {
    rows(d, text).into_iter().filter(|r| r.2 == kind).map(|r| r.3).collect()
}

#[test]
fn passports_need_a_passport_label() {
    let d = us();
    for (text, want) in [
        ("Passport No. X38194027", "X38194027"),
        ("passport: 538194027.", "538194027"),
        ("US passport card C03005988", "C03005988"),
        ("538194027 (passport)", "538194027"),
    ] {
        assert_eq!(of_kind(&d, text, "us-passport"), [want], "{text}");
    }
    for not in [
        "booking 538194027",                          // no label
        "passport 5381940277",                        // ten digits
        "passport XY3819402",                         // two letters
        "passport x38194027",                         // lower case letter
        "passport 538-194-027",                       // a passport is not hyphenated
        "passport on file, see the ledger 538194027", // label too far
    ] {
        assert!(of_kind(&d, not, "us-passport").is_empty(), "{not}");
    }
    let pii = layer(&d, "Passport No. X38194027");
    assert_eq!(pii.mentions().next().unwrap().confidence(), PiiConfidence::Labelled);
}

#[test]
fn driver_licences_need_a_licence_label() {
    let d = us();
    for (text, want) in [
        ("driver's license D7302146", "D7302146"),
        ("Driver\u{2019}s License: D7302146", "D7302146"),
        ("drivers licence S530-4607-2861-0", "S530460728610"),
        ("DL 4813-2069-5527", "481320695527"),
        ("DLN: 123456789", "123456789"),
        ("license number 31KMB40682", "31KMB40682"),
        ("driver license WA1234567", "WA1234567"),
    ] {
        assert_eq!(of_kind(&d, text, "us-drivers-licence"), [want], "{text}");
    }
    for not in [
        "order D7302146",                     // no label
        "driver's license 123456",            // six digits
        "driver's license ABC1234",           // three leading letters
        "driver's license D730",              // too short
        "driver's license 12345678901234567", // too long
        "driver's license d7302146",          // lower case
    ] {
        assert!(of_kind(&d, not, "us-drivers-licence").is_empty(), "{not}");
    }
}

#[test]
fn zip_codes_follow_a_state_code_or_a_zip_label() {
    let d = us();
    for (text, want, tier) in [
        ("Springfield, IL 62704", "62704", PiiConfidence::Format),
        ("Austin, TX 78701-4417.", "78701-4417", PiiConfidence::Format),
        ("ZIP: 30301", "30301", PiiConfidence::Labelled),
        ("zip code 10001", "10001", PiiConfidence::Labelled),
        ("San Juan, PR 00901", "00901", PiiConfidence::Format),
        // The armed forces codes: Americas, Europe and Pacific.
        ("APO AA 34002", "34002", PiiConfidence::Format),
        ("APO AE 09012", "09012", PiiConfidence::Format),
        ("APO AP 96534", "96534", PiiConfidence::Format),
        ("FPO AP 96349-1100", "96349-1100", PiiConfidence::Format),
    ] {
        let pii = layer(&d, text);
        let zips: Vec<_> = pii.mentions().filter(|m| m.kind == PiiKind::UsZip).collect();
        assert_eq!(zips.len(), 1, "{text}");
        assert_eq!(zips[0].normalized, want, "{text}");
        assert_eq!(zips[0].confidence(), tier, "{text}");
    }
    for not in [
        "room 62704",   // no state and no label
        "il 62704",     // the state code is in capitals
        "XIL 62704",    // inside a word
        "IL 00000",     // all zeros
        "IL 627041",    // six digits
        "IL 62704-12",  // a short +4
        "IL  62704",    // two spaces
        "ZZ 62704",     // not a state code
        "ZIP: 62704.5", // a decimal
    ] {
        assert!(of_kind(&d, not, "us-zip").is_empty(), "{not}");
    }
}

#[test]
fn phone_extensions_follow_a_phone() {
    let d = PiiDetector::new(vec![Scanner::all(Extractor::PhoneExtension)]);
    for (text, want) in [
        ("+1 202-555-0143 ext. 4417", "4417"),
        ("+1 202-555-0143 Ext 4417", "4417"),
        ("+1 202-555-0143, extension 4417", "4417"),
        ("(312) 555-0199 x 2083", "2083"),
        ("(312) 555-0199 #7316", "7316"),
        ("+44 20 7946 0958 extn: 12", "12"),
    ] {
        assert_eq!(of_kind(&d, text, "phone-extension"), [want], "{text}");
    }
    for not in [
        "extension 4417",               // no phone
        "+1 202-555-0143 extra 99",     // a word, not ext
        "+1 202-555-0143 next 99",      // not after the phone
        "+1 202-555-0143 ext. 1234567", // seven digits
        "+1 202-555-0143 ext. 12-3",
        "+1 202-555-0143 Ext4417", // a word needs a gap
    ] {
        assert!(of_kind(&d, not, "phone-extension").is_empty(), "{not}");
    }
    // The contact pack has both mentions, side by side.
    assert_eq!(kinds(&pack(Pack::Contact), "call +1 202-555-0143 ext. 4417"), ["phone", "phone-extension"]);
}

#[test]
fn a_caller_can_widen_or_drop_the_label_rules() {
    // Passports without a label, when the caller turns the rule off.
    let open = us().with_label_policy(PiiKind::UsPassport, turbo_pii::LabelPolicy::Boosts);
    assert_eq!(of_kind(&open, "booking 538194027", "us-passport"), ["538194027"]);
    // A caller's own label.
    let extra = us().with_extra_label(PiiKind::UsDriversLicence, "permit");
    assert_eq!(of_kind(&extra, "permit D7302146", "us-drivers-licence"), ["D7302146"]);
    // A wider window reaches a label further off.
    let wide_window = us().with_label_window(PiiKind::UsPassport, 6);
    assert_eq!(of_kind(&wide_window, "passport on file, see the ledger 538194027", "us-passport"), ["538194027"]);
}

#[test]
fn masking_keeps_the_shape() {
    let d = wide();
    let text = "Passport No. X38194027, DL 4813-2069-5527, Austin, TX 78701-4417, +1 202-555-0143 ext. 4417";
    let masked = mask_by_kind(text, &layer(&d, text), '*').unwrap();
    assert_eq!(masked, "Passport No. *********, DL ****-****-****, Austin, TX *****-****, +* ***-***-**** ext. ****");
}

/// The label text recorded on the first mention of `text`.
fn label_of(d: &PiiDetector, text: &str) -> Option<String> {
    let pii = layer(d, text);
    let m = pii.mentions().next()?;
    m.evidence.iter().find_map(|e| match e {
        turbo_pii::PiiEvidence::Label { span, .. } => Some(text[span.start as usize..span.end as usize].to_owned()),
        _ => None,
    })
}

#[test]
fn the_longest_label_phrase_is_recorded() {
    let d = us();
    assert_eq!(label_of(&d, "Passport No. X38194027").as_deref(), Some("Passport No"));
    assert_eq!(label_of(&d, "US passport card C03005988").as_deref(), Some("passport card"));
    assert_eq!(label_of(&d, "X38194027 (passport number)").as_deref(), Some("passport number"));
    assert_eq!(label_of(&d, "ZIP code 30301").as_deref(), Some("ZIP code"));
    assert!(turbo_pii::window::default_labels(PiiKind::PhoneExtension).contains(&"extn"));
}

#[test]
fn values_joined_by_hyphens_or_underscores_are_refused() {
    let d = us();
    for (not, kind) in [
        ("DL 1234-5678-9012-3456-7", "us-drivers-licence"), // no suffix of a longer run
        ("passport X-538194027", "us-passport"),
        ("passport REF-538194027", "us-passport"),
        ("DL AB-1234567", "us-drivers-licence"),
        ("DL D-7302146", "us-drivers-licence"),
        ("passport 538194027-A", "us-passport"),
        ("DL D7302146-A", "us-drivers-licence"),
        ("DL D7302146_x", "us-drivers-licence"),
        ("passport ref_538194027", "us-passport"),
        ("Springfield, IL 62704-A", "us-zip"),
        ("ZIP: 62704-A", "us-zip"),
        ("ZIP: 62704_A", "us-zip"),
    ] {
        assert!(of_kind(&d, not, kind).is_empty(), "{not}: {:?}", rows(&d, not));
    }
    let ext = PiiDetector::new(vec![Scanner::all(Extractor::PhoneExtension)]);
    for not in ["+1 202-555-0143, x 1,000", "+1 202-555-0143, x 2.5", "+1 202-555-0143 ext 12_3"] {
        assert!(of_kind(&ext, not, "phone-extension").is_empty(), "{not}");
    }
}
