//! Precision and recall of the widest pack on a synthetic labelled set.
//!
//! The set is made here from a fixed seed: values of every kind built to
//! pass their checks (test card ranges, `example.com` addresses, fictional
//! phone ranges, documentation IP ranges, random hashes for the crypto
//! addresses, the NHS test range), each placed in a sentence, and near
//! misses of the checksummed kinds (one check digit or one letter's case
//! changed) with no label. The golden true positive and false positive
//! corpora of OpenNLP are added as they are. A found mention counts as
//! true when a label has its span and kind; every other found mention is
//! false, overlapping values included (a phone inside a grouped IBAN).
//!
//! The test prints a table per kind (`--nocapture`) and holds the totals
//! to the floors below, so a scanner that gets less specific or misses
//! more fails it.

mod support;

use std::collections::BTreeMap;

use support::{rows, wide};
use turbo_pii::PiiKind;
use turbo_pii::checksum::{base58, keccak, luhn, mod97};

/// SplitMix64, so the set is the same on every run and platform.
struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        z ^ (z >> 31)
    }

    fn below(&mut self, n: u64) -> u64 {
        self.next() % n
    }

    fn digits(&mut self, n: usize) -> String {
        (0..n).map(|_| char::from(b'0' + self.below(10) as u8)).collect()
    }

    fn pick<'a>(&mut self, from: &[&'a str]) -> &'a str {
        from[self.below(from.len() as u64) as usize]
    }

    fn chars(&mut self, alphabet: &[u8], n: usize) -> String {
        (0..n).map(|_| char::from(alphabet[self.below(alphabet.len() as u64) as usize])).collect()
    }
}

/// One labelled text: the text and its labels as (start, end, kind).
struct Labelled {
    text: String,
    labels: Vec<(u32, u32, &'static str)>,
}

const BEFORE: [&str; 6] = ["Please use ", "ref: ", "(", "The value ", "\u{2022} ", "Wert: "];
const AFTER: [&str; 6] = [" today.", ")", ", thanks.", " \u{2014} noted", ".", " ok"];

/// `value` in a sentence, labelled `kinds` over its whole span.
fn placed(r: &mut Rng, prefix: &str, value: &str, kinds: &[&'static str]) -> Labelled {
    let before = format!("{}{prefix}", r.pick(&BEFORE));
    let text = format!("{before}{value}{}", r.pick(&AFTER));
    let (start, end) = (before.len() as u32, (before.len() + value.len()) as u32);
    Labelled { text, labels: kinds.iter().map(|&k| (start, end, k)).collect() }
}

/// `body` with a Luhn check digit appended.
fn with_luhn(body: &str) -> String {
    (0..10).map(|d| format!("{body}{d}")).find(|v| luhn::valid(v.as_bytes())).expect("one check digit passes")
}

/// The last digit of `value` changed, which breaks a Luhn, mod 97 or
/// mod 11 check.
fn bump_last_digit(value: &str) -> String {
    let p = value.rfind(|c: char| c.is_ascii_digit()).expect("a digit");
    let d = value.as_bytes()[p] - b'0';
    format!("{}{}{}", &value[..p], (d + 1) % 10, &value[p + 1..])
}

fn grouped(value: &str, size: usize) -> String {
    value.as_bytes().chunks(size).map(|c| std::str::from_utf8(c).unwrap()).collect::<Vec<_>>().join(" ")
}

fn iban_de(r: &mut Rng) -> String {
    let bban = r.digits(18);
    let check = 98 - mod97::remainder(format!("DE00{bban}").as_bytes());
    format!("DE{check:02}{bban}")
}

fn nhs(r: &mut Rng) -> String {
    loop {
        // 999 is the range set aside for testing.
        let body = format!("999{}", r.digits(6));
        let sum: u32 = body.bytes().enumerate().map(|(i, d)| u32::from(d - b'0') * (10 - i as u32)).sum();
        let check = (11 - sum % 11) % 11;
        if check < 10 {
            return format!("{body}{check}");
        }
    }
}

fn eth(r: &mut Rng) -> String {
    let hex: String = (0..20).map(|_| format!("{:02x}", r.below(256))).collect();
    let hash = keccak::digest(hex.as_bytes());
    let cased: String = hex
        .bytes()
        .enumerate()
        .map(|(i, c)| {
            let nibble = (hash[i / 2] >> (if i % 2 == 0 { 4 } else { 0 })) & 15;
            char::from(if c.is_ascii_alphabetic() && nibble >= 8 { c.to_ascii_uppercase() } else { c })
        })
        .collect();
    format!("0x{cased}")
}

fn btc(r: &mut Rng) -> String {
    let mut payload = vec![if r.below(2) == 0 { 0x00 } else { 0x05 }];
    payload.extend((0..20).map(|_| r.below(256) as u8));
    base58::encode_checked(&payload)
}

/// The labelled texts and the near misses, from `seed`.
fn synthetic(seed: u64, per_kind: usize) -> Vec<Labelled> {
    const UPPER_DIGITS: &[u8] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZ234567";
    const ALNUM: &[u8] = b"abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789";
    let r = &mut Rng(seed);
    let mut out = Vec::new();
    for _ in 0..per_kind {
        let local = format!("{}.{}", r.chars(b"abcdefghij", 5), r.chars(b"klmnopqrst", 4));
        let domain = r.pick(&["example.com", "example.org", "example.net", "mail.example.com"]);
        out.push(placed(r, "", &format!("{local}@{domain}"), &["email"]));

        let phone = format!("+44 20 7946 0{}", r.digits(3));
        out.push(placed(r, "", &phone, &["phone"]));
        let us = format!("+1 {}-555-01{}", r.pick(&["202", "312", "415", "617"]), r.digits(2));
        out.push(placed(r, "", &us, &["phone"]));

        let card = with_luhn(&format!("411111{}", r.digits(9)));
        out.push(placed(r, "", &card, &["card"]));
        out.push(Labelled { text: format!("paid with {}", bump_last_digit(&card)), labels: vec![] });

        let iban = iban_de(r);
        out.push(placed(r, "", &iban, &["iban"]));
        out.push(placed(r, "IBAN ", &grouped(&iban, 4), &["iban"]));
        out.push(Labelled { text: format!("IBAN {} rejected", bump_last_digit(&iban)), labels: vec![] });

        let v4 = format!("{}.{}", r.pick(&["192.0.2", "198.51.100", "203.0.113"]), 1 + r.below(254));
        out.push(placed(r, "", &v4, &["ipv4"]));
        let v6 = format!("2001:db8:{:x}::{:x}", 0x100 + r.below(0xf00), 1 + r.below(0xffff));
        out.push(placed(r, "", &v6, &["ipv6"]));
        // Locally administered, so no vendor's prefix.
        let mac: Vec<String> =
            (0..6).map(|i| format!("{:02x}", if i == 0 { 0x02 } else { r.below(256) as u8 })).collect();
        out.push(placed(r, "", &mac.join(":"), &["mac"]));

        let aws = format!("AKIA{}", r.chars(UPPER_DIGITS, 16));
        out.push(placed(r, "", &aws, &["aws-access-key"]));
        let github = format!("ghp_{}", r.chars(ALNUM, 36));
        out.push(placed(r, "", &github, &["github-token"]));

        let e = eth(r);
        out.push(placed(r, "", &e, &["eth-address"]));
        if e.bytes().any(|c| c.is_ascii_uppercase()) && e[2..].bytes().any(|c| c.is_ascii_lowercase()) {
            let p = e[2..].find(|c: char| c.is_ascii_uppercase()).unwrap() + 2;
            let flipped = format!("{}{}{}", &e[..p], e[p..=p].to_ascii_lowercase(), &e[p + 1..]);
            out.push(Labelled { text: format!("send to {flipped} now"), labels: vec![] });
        }
        let b = btc(r);
        out.push(placed(r, "", &b, &["btc-address"]));

        let ssn = format!("{:03}-{:02}-{:04}", 100 + r.below(565), 1 + r.below(99), 1 + r.below(9999));
        out.push(placed(r, "SSN ", &ssn, &["us-ssn"]));
        out.push(Labelled { text: format!("SSN 666-{}", &ssn[4..]), labels: vec![] });

        let n = nhs(r);
        // Ten digits in groups are also a phone number.
        out.push(placed(r, "NHS ", &grouped3_3_4(&n), &["uk-nhs", "phone"]));
        out.push(Labelled { text: format!("NHS {}", grouped3_3_4(&bump_last_digit(&n))), labels: vec![] });

        let sin = with_luhn(&format!("0{}", r.digits(7)));
        out.push(placed(r, "SIN: ", &format!("{} {} {}", &sin[..3], &sin[3..6], &sin[6..]), &["ca-sin"]));
        out.push(Labelled { text: format!("ticket {} {} {}", &sin[..3], &sin[3..6], &sin[6..]), labels: vec![] });

        // Reporting body 00, a test range.
        let imei = with_luhn(&format!("00{}", r.digits(12)));
        out.push(placed(r, "IMEI: ", &imei, &["imei"]));
        out.push(Labelled { text: format!("IMEI: {}", bump_last_digit(&imei)), labels: vec![] });
    }
    out
}

fn grouped3_3_4(v: &str) -> String {
    format!("{} {} {}", &v[..3], &v[3..6], &v[6..])
}

fn data(name: &str) -> String {
    std::fs::read_to_string(std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/data").join(name)).unwrap()
}

fn fixtures(name: &str) -> Vec<String> {
    data(name).lines().filter(|l| !l.trim().is_empty() && !l.starts_with('#')).map(str::to_owned).collect()
}

/// The two OpenNLP corpora as labelled texts.
fn corpora() -> Vec<Labelled> {
    let mut out: Vec<Labelled> = Vec::new();
    for line in fixtures("golden-true-positives.tsv") {
        let f: Vec<&str> = line.splitn(4, '\t').collect();
        let kind = PiiKind::parse(f[0]).expect("a kind").name();
        let start = f[3].find(f[1]).expect("covered text in the source") as u32;
        let label = (start, start + f[1].len() as u32, kind);
        match out.iter_mut().find(|l| l.text == f[3]) {
            Some(l) => l.labels.push(label),
            None => out.push(Labelled { text: f[3].to_owned(), labels: vec![label] }),
        }
    }
    out.extend(fixtures("golden-false-positives.txt").into_iter().map(|text| Labelled { text, labels: vec![] }));
    out
}

#[derive(Default, Debug)]
struct Counts {
    tp: usize,
    fp: usize,
    fn_: usize,
}

fn ratio(a: usize, b: usize) -> f64 {
    if b == 0 { 1.0 } else { a as f64 / b as f64 }
}

fn measure(set: &[Labelled]) -> BTreeMap<&'static str, Counts> {
    let d = wide();
    // PII_SHOW_FALSE=1 lists each false mention.
    let show_false = std::env::var("PII_SHOW_FALSE").is_ok_and(|v| v == "1");
    let mut by_kind: BTreeMap<&'static str, Counts> = BTreeMap::new();
    for l in set {
        let found: Vec<(u32, u32, &'static str)> = rows(&d, &l.text).into_iter().map(|r| (r.0, r.1, r.2)).collect();
        for f in &found {
            let c = by_kind.entry(f.2).or_default();
            if l.labels.contains(f) {
                c.tp += 1
            } else {
                c.fp += 1;
                if show_false {
                    eprintln!("false {f:?} in {:?}", l.text);
                }
            }
        }
        for label in &l.labels {
            if !found.contains(label) {
                by_kind.entry(label.2).or_default().fn_ += 1;
            }
        }
    }
    by_kind
}

#[test]
fn precision_and_recall_on_the_synthetic_set() {
    let mut set = synthetic(225, 40);
    let labels_made = set.iter().map(|l| l.labels.len()).sum::<usize>();
    set.extend(corpora());
    let by_kind = measure(&set);
    let mut total = Counts::default();
    eprintln!("{:<16} {:>5} {:>5} {:>5} {:>9} {:>7}", "kind", "tp", "fp", "fn", "precision", "recall");
    for (kind, c) in &by_kind {
        eprintln!(
            "{kind:<16} {:>5} {:>5} {:>5} {:>9.3} {:>7.3}",
            c.tp,
            c.fp,
            c.fn_,
            ratio(c.tp, c.tp + c.fp),
            ratio(c.tp, c.tp + c.fn_)
        );
        total.tp += c.tp;
        total.fp += c.fp;
        total.fn_ += c.fn_;
    }
    let (precision, recall) = (ratio(total.tp, total.tp + total.fp), ratio(total.tp, total.tp + total.fn_));
    eprintln!(
        "{} texts, {labels_made} labels made here; all: tp {} fp {} fn {}, precision {precision:.4}, recall {recall:.4}",
        set.len(),
        total.tp,
        total.fp,
        total.fn_
    );
    assert!(labels_made > 700, "{labels_made} labels");
    assert!(precision >= PRECISION_FLOOR, "precision {precision:.4}");
    assert!(recall >= RECALL_FLOOR, "recall {recall:.4}");
}

/// The measured totals are precision 0.8944 and recall 1.0. The false
/// mentions are values inside values: phones and a card inside grouped
/// IBANs, phones in near miss NHS numbers (ten grouped digits are a phone
/// shape), and NHS numbers in US phones whose digits pass mod 11 by
/// chance. docs/pii.md has the table.
const PRECISION_FLOOR: f64 = 0.89;
const RECALL_FLOOR: f64 = 1.0;
