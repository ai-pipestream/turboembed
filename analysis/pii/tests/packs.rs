//! `PiiPacksTest`, `CompositePiiExtractorTest`, `PiiCompositionTest`,
//! `PiiPackCoverageTest`, `PiiTypePriorityTest`, `PiiTypeSelectionTest`,
//! `PiiAnnotatorTest` and the golden corpora of true and false positives.
//!
//! The Java composes any `PiiExtractor`, its own or a caller's. Here a
//! detector is a list of the built in scanners, and composing two is
//! joining their lists; the cases with made up extractors and type names
//! become cases of the candidate order in `analysis/pii/src/hits.rs`.

mod support;

use support::*;
use turbo_analysis::{Analyzer, Annotator, Document, Layer};
use turbo_pii::rewrite::mask_all;
use turbo_pii::{Extractor, Pack, Pii, PiiDetector, PiiKind, PiiKinds, Scanner, ScannerError};

const LEGACY: &str = "1Pjk7c5ipz7ZpbVT3VKwcFNGeqgBFpbQdw";

fn everything() -> String {
    format!(
        "mail jane@example.com call (555) 123-4567 iban DE89 3704 0044 0532 0130 00 card 4111 1111 1111 1111 routing 021000021 \
         host 10.1.2.3 peer 2001:db8::1 mac 00:1b:44:11:3a:b7 key AKIAIOSFODNN7EXAMPLE url https://u:p@example.com/ \
         github ghp_{} jwt eyJhbGciOiJIUzI1NiJ9.e30.YQ btc {LEGACY} eth 0x5aAeb6053F3E94C9b9A09f33669435E7Ef1BeAed \
         ssn 123-45-6789 itin 900-70-1234 nhs 943 476 5919 idnr 65929970489 SIN: 046 454 286 IMEI: 490154203237518",
        "a".repeat(36)
    )
}

fn kinds_found(d: &PiiDetector) -> PiiKinds {
    let mut out = PiiKinds::EMPTY;
    for m in layer(d, &everything()).mentions() {
        out = out.union(PiiKinds::of(&[m.kind]));
    }
    out
}

#[test]
fn every_pack_is_in_order_and_finds_its_kinds_only() {
    use PiiKind::*;
    let expected: [(Pack, &[PiiKind]); 10] = [
        (Pack::Payment, &[Iban, Card, AbaRouting]),
        (Pack::Contact, &[Email, Phone]),
        (Pack::Network, &[Ipv4, Ipv6, Mac]),
        (Pack::Secrets, &[AwsAccessKey, UrlCredential, GithubToken, Jwt]),
        (Pack::Crypto, &[BtcAddress, EthAddress]),
        (Pack::UsIdentity, &[UsSsn, UsItin]),
        (Pack::EuIdentity, &[UkNhs, DeSteuerId]),
        (Pack::CaIdentity, &[CaSin]),
        (Pack::Device, &[Imei]),
        (Pack::AllStructured, &PiiKind::ALL),
    ];
    for (p, kinds) in expected {
        let d = pack(p);
        let pii = layer(&d, &everything());
        assert_eq!(pii.check(), Ok(()), "{}", p.name());
        assert_eq!(kinds_found(&d), PiiKinds::of(kinds), "{}", p.name());
        assert_eq!(d.kinds(), PiiKinds::of(kinds), "{}", p.name());
        assert_none(&d, "nothing to find in this sentence");
        assert_eq!(Pack::parse(p.name()), Some(p));
    }
    assert_eq!(Pack::parse("nothing"), None);
}

#[test]
fn the_wide_pack_keeps_every_mention_of_the_narrow_ones() {
    let text = everything();
    let wide_parity = rows(&wide().keeping_nested(), &text);
    let wide_default = rows(&wide(), &text);
    for p in Pack::ALL {
        for row in rows(&pack(p).keeping_nested(), &text) {
            assert!(wide_parity.contains(&row), "{}: {row:?}", p.name());
        }
        for row in rows(&pack(p), &text) {
            let overlapped = wide_default.iter().any(|w| w.0 < row.1 && row.0 < w.1);
            assert!(wide_default.contains(&row) || overlapped, "{}: {row:?}", p.name());
        }
    }
}

#[test]
fn nested_identifiers_are_dropped_by_default_and_kept_in_parity() {
    let d = wide();
    let parity = wide().keeping_nested();
    assert!(!d.keeps_nested() && parity.keeps_nested());
    assert_ne!(d.config_hash(), parity.config_hash());

    let iban = "Please use IBAN DE56 2352 0016 5606 2427 58)";
    assert_found(&d, iban, &[(16, 43, "iban", "DE56235200165606242758")]);
    assert_found(
        &parity,
        iban,
        &[
            (16, 43, "iban", "DE56235200165606242758"),
            (21, 40, "card", "2352001656062427"),
            (31, 43, "phone", "5606242758"),
        ],
    );

    let us_phone = "Call +1 617-555-0129 today.";
    assert_found(&d, us_phone, &[(5, 20, "phone", "+16175550129")]);
    assert_found(&parity, us_phone, &[(5, 20, "phone", "+16175550129"), (8, 20, "uk-nhs", "6175550129")]);
}

#[test]
fn the_default_extractor_finds_the_four_classic_kinds() {
    use PiiKind::*;
    assert_eq!(kinds_found(&PiiDetector::cursor()), PiiKinds::of(&[Email, Phone, Iban, Card]));
    let contact = pack(Pack::Contact);
    for text in [
        "ssn 123-45-6789",
        "itin 900-70-1234",
        "idnr 65929970489",
        "SIN: 046 454 286",
        "IMEI: 490154203237518",
        "routing 021000021",
        "key AKIAIOSFODNN7EXAMPLE",
        &format!("btc {LEGACY}"),
    ] {
        assert_none(&contact, text);
    }
}

fn joined(packs: &[Pack]) -> PiiDetector {
    PiiDetector::new(packs.iter().flat_map(|p| p.scanners()).collect())
}

#[test]
fn packs_join() {
    let text = "mail jane@example.com from 10.1.2.3";
    assert_eq!(kinds(&joined(&[Pack::Contact, Pack::Network]), text), ["email", "ipv4"]);
    let text = "mail jane@example.com call (555) 123-4567";
    assert_eq!(rows(&joined(&[Pack::Contact, Pack::Contact]), text), rows(&pack(Pack::Contact), text));
    let text = "record 943 476 5919 today";
    assert_eq!(kinds(&pack(Pack::Contact), text), ["phone"]);
    assert_eq!(kinds(&wide(), text), ["uk-nhs"]);
    assert_eq!(kinds(&wide().keeping_nested(), text), ["uk-nhs", "phone"]);
}

#[test]
fn an_imei_is_also_a_card_in_either_order() {
    for order in [[Pack::Payment, Pack::Device], [Pack::Device, Pack::Payment]] {
        let text = "IMEI: 490154203237518.";
        let d = joined(&order);
        assert_found(&d, text, &[(6, 21, "imei", "490154203237518")]);
        assert_found(
            &d.clone().keeping_nested(),
            text,
            &[(6, 21, "imei", "490154203237518"), (6, 21, "card", "490154203237518")],
        );
        assert_eq!(mask_all(text, &layer(&d, text), '*').unwrap(), "IMEI: ***************.");
    }
}

#[test]
fn single_kind_scanners_compose_to_the_default() {
    let text =
        "Mail jane@example.com, call (555) 123-4567, IBAN DE89 3704 0044 0532 0130 00, card 4111 1111 1111 1111.";
    let parts = [PiiKind::Email, PiiKind::Iban, PiiKind::Card, PiiKind::Phone]
        .map(|k| Scanner::with_kinds(Extractor::Cursor, PiiKinds::of(&[k])).unwrap());
    assert_eq!(rows(&PiiDetector::new(parts.to_vec()), text), rows(&PiiDetector::cursor(), text));
    let two = PiiDetector::new(vec![
        Scanner::with_kinds(Extractor::Cursor, PiiKinds::of(&[PiiKind::Email])).unwrap(),
        Scanner::with_kinds(Extractor::Cursor, PiiKinds::of(&[PiiKind::Card])).unwrap(),
    ]);
    assert_eq!(kinds(&two, "card 4111 1111 1111 1111 for jane@example.com"), ["card", "email"]);
    assert_eq!(rows(&two, "jane@example.com paid with 4111 1111 1111 1111").len(), 2);
    for text in ["", "nothing to see here", "1 + 1 = 2"] {
        assert_none(&PiiDetector::cursor(), text);
    }
}

#[test]
fn crossing_mentions_are_kept_and_masked() {
    for prefix in ["", "Contact: ", "😀 ", "e\u{301} "] {
        for (text, address) in [
            (format!("{prefix}https://john:pass@example.com/path"), "pass@example.com"),
            (format!("{prefix}https://john:pa.ss@example.com/path"), "pa.ss@example.com"),
            (format!("{prefix}https://john:pass@example.com/path jane@example.org"), "jane@example.org"),
        ] {
            let start = text.find(address).unwrap() as u32;
            let want = (start, start + address.len() as u32, "email", address.to_owned());
            assert!(rows(&pack(Pack::Contact), &text).contains(&want), "{text}");
            assert!(rows(&wide().keeping_nested(), &text).contains(&want), "{text}");
            let masked = mask_all(&text, &layer(&wide().keeping_nested(), &text), '*').unwrap();
            assert_eq!(&masked[want.0 as usize..want.1 as usize], "*".repeat(address.len()));
        }
    }
}

#[test]
fn kind_ranks() {
    let names: Vec<&str> = PiiKind::ALL.iter().map(|k| k.name()).collect();
    assert_eq!(
        names,
        [
            "jwt",
            "aws-access-key",
            "github-token",
            "url-credential",
            "email",
            "iban",
            "imei",
            "card",
            "btc-address",
            "eth-address",
            "mac",
            "ipv6",
            "ipv4",
            "us-ssn",
            "us-itin",
            "uk-nhs",
            "de-steuer-id",
            "ca-sin",
            "aba-routing",
            "phone"
        ]
    );
    for (i, k) in PiiKind::ALL.iter().enumerate() {
        assert_eq!(k.rank() as usize, i);
        assert_eq!(PiiKind::parse(k.name()), Some(*k));
    }
    for (stronger, weaker) in [
        ("email", "phone"),
        ("iban", "card"),
        ("card", "aba-routing"),
        ("jwt", "github-token"),
        ("aws-access-key", "email"),
        ("ipv6", "ipv4"),
        ("mac", "ipv6"),
        ("btc-address", "us-ssn"),
        ("us-ssn", "phone"),
        ("de-steuer-id", "phone"),
    ] {
        assert!(PiiKind::parse(stronger).unwrap() < PiiKind::parse(weaker).unwrap());
    }
    for name in ["custom", "person", "email ", "EMAIL", "", "  "] {
        assert_eq!(PiiKind::parse(name), None);
    }
}

#[test]
fn kind_selection() {
    let samples: [(Extractor, &str); 6] = [
        (Extractor::Cursor, "jane@example.com"),
        (Extractor::Network, "10.1.2.3"),
        (Extractor::Secrets, "AKIAIOSFODNN7EXAMPLE"),
        (Extractor::Crypto, LEGACY),
        (Extractor::UsIdentity, "123-45-6789"),
        (Extractor::EuIdentity, "943 476 5919"),
    ];
    for (extractor, sample) in samples {
        let all: Vec<PiiKind> = extractor.kinds().iter().collect();
        let first = rows(&PiiDetector::new(vec![Scanner::all(extractor)]), sample)[0].2;
        for mask in 1u32..1 << all.len() {
            let chosen: Vec<PiiKind> =
                all.iter().enumerate().filter(|(i, _)| mask & 1 << i != 0).map(|(_, &k)| k).collect();
            let d = PiiDetector::new(vec![Scanner::with_kinds(extractor, PiiKinds::of(&chosen)).unwrap()]);
            let found = kinds(&d, sample);
            if chosen.iter().any(|k| k.name() == first) {
                assert_eq!(found, [first], "{} {chosen:?}", extractor.java_name());
            } else {
                assert!(found.is_empty(), "{} {chosen:?}", extractor.java_name());
            }
        }
        assert_eq!(Scanner::with_kinds(extractor, PiiKinds::EMPTY), Err(ScannerError::NoKinds(extractor)));
        let foreign = PiiKind::ALL.into_iter().find(|k| !extractor.kinds().contains(*k)).unwrap();
        assert_eq!(
            Scanner::with_kinds(extractor, PiiKinds::of(&[foreign])),
            Err(ScannerError::Foreign(extractor, foreign))
        );
    }
}

#[test]
fn the_annotator_writes_the_layer() {
    let analyzer = Analyzer::builder().add(PiiDetector::cursor()).build().expect("the analyzer");
    let mut docs = vec![
        Document::new("Invoice for jane@example.com, card 4111 1111 1111 1111."),
        Document::new("The quick brown fox jumps over the lazy dog."),
    ];
    for result in analyzer.run(&mut docs) {
        result.unwrap();
    }
    let kinds = |d: &Document| d.get::<Pii>().map(|p| p.mentions().map(|m| m.kind.name()).collect::<Vec<_>>());
    assert_eq!(kinds(&docs[0]), Some(vec!["email", "card"]));
    assert_eq!(kinds(&docs[1]), Some(vec![]));
    assert_eq!(Pii::ID, "turbo:pii");
}

#[test]
fn every_pack_on_every_thread() {
    for p in Pack::ALL {
        let d = pack(p);
        std::thread::scope(|s| {
            for worker in 0..4 {
                let d = &d;
                s.spawn(move || {
                    let text = format!("{}{}", "😀 ".repeat(worker + 1), everything());
                    let want = rows(d, &text);
                    assert!(!want.is_empty());
                    for _ in 0..16 {
                        assert_eq!(rows(d, &text), want);
                    }
                });
            }
        });
    }
}

fn data(name: &str) -> String {
    std::fs::read_to_string(std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/data").join(name)).unwrap()
}

fn fixtures(name: &str) -> Vec<String> {
    data(name).lines().filter(|l| !l.trim().is_empty() && !l.starts_with('#')).map(str::to_owned).collect()
}

#[test]
fn golden_true_positives() {
    // Rows grouped by source text, in file order.
    type Mention = (u32, u32, String, String);
    let mut groups: Vec<(String, Vec<Mention>)> = Vec::new();
    for line in fixtures("golden-true-positives.tsv") {
        let f: Vec<&str> = line.splitn(4, '\t').collect();
        let (kind, covered, norm, text) = (f[0], f[1], f[2], f[3]);
        let start = text.find(covered).expect("covered text in the source");
        assert_eq!(text.matches(covered).count(), 1, "{covered}");
        let row = (start as u32, (start + covered.len()) as u32, kind.to_owned(), norm.to_owned());
        match groups.iter_mut().find(|g| g.0 == text) {
            Some(g) => g.1.push(row),
            None => groups.push((text.to_owned(), vec![row])),
        }
    }
    assert!(groups.len() >= 15);
    for (text, want) in groups {
        let got_parity: Vec<(u32, u32, String, String)> =
            rows(&wide().keeping_nested(), &text).into_iter().map(|r| (r.0, r.1, r.2.to_owned(), r.3)).collect();
        assert_eq!(got_parity, want, "{text}");
        let mut disjoint = Vec::new();
        let mut outer_end = 0;
        for m in &want {
            if m.0 >= outer_end {
                outer_end = outer_end.max(m.1);
                disjoint.push(m.clone());
            }
        }
        let got: Vec<(u32, u32, String, String)> =
            rows(&wide(), &text).into_iter().map(|r| (r.0, r.1, r.2.to_owned(), r.3)).collect();
        assert_eq!(got, disjoint, "{text}");
    }
}

#[test]
fn golden_false_positives() {
    let lines = fixtures("golden-false-positives.txt");
    assert!(lines.len() >= 120, "{}", lines.len());
    for line in &lines {
        assert_none(&wide(), line);
        assert_none(&PiiDetector::cursor(), line);
    }
    assert_none(&wide(), &lines.join("\n"));
}

#[test]
fn layers_pass_their_check() {
    let pii = layer(&wide(), &everything());
    assert_eq!(pii.check(), Ok(()));
    assert!(pii.mentions().count() >= 19);
    let parity = layer(&wide().keeping_nested(), &everything());
    assert_eq!(parity.check(), Ok(()));
    assert!(parity.mentions().count() > 20);
}
