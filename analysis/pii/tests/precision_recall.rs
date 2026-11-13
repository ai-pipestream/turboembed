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

use support::{Counts, HUMAN_DOCS, Labelled, layer, load_labelled, measure, ratio, ratio_text, wide};
use turbo_analysis::Layer as _;
use turbo_pii::PiiKind;
use turbo_pii::checksum::{base58, es, icao9303, keccak, luhn, mod11, mod97};

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

fn es_dni_pair(r: &mut Rng) -> (String, String) {
    let n = 10_000_000 + r.below(89_000_000) as u32;
    let c = es::NIF_LETTERS[(n % 23) as usize] as char;
    let bad = es::NIF_LETTERS[((n + 1) % 23) as usize] as char;
    (format!("{n:08}-{c}"), format!("{n:08}-{bad}"))
}

fn es_nie_pair(r: &mut Rng) -> (String, String) {
    let p_idx = r.below(3) as u32;
    let p = b"XYZ"[p_idx as usize] as char;
    let body = 1_000_000 + r.below(8_000_000) as u32;
    let n = p_idx * 10_000_000 + body;
    let c = es::NIF_LETTERS[(n % 23) as usize] as char;
    let bad = es::NIF_LETTERS[((n + 1) % 23) as usize] as char;
    (format!("{p}{body:07}{c}"), format!("{p}{body:07}{bad}"))
}

fn es_cif_pair(r: &mut Rng) -> (String, String) {
    let org = r.pick(&["A", "B", "P", "Q"]);
    let body = format!("{:02}{}", 1 + r.below(50), r.digits(5));
    let c = b"0123456789JABCDEFGHI"
        .iter()
        .copied()
        .find(|&ch| {
            let s = format!("{org}{body}{}", ch as char);
            es::cif(s.as_bytes().try_into().unwrap())
        })
        .unwrap() as char;
    let bad = if c.is_ascii_digit() { char::from(b'0' + ((c as u8 - b'0' + 1) % 10)) } else { 'Z' };
    (format!("{org}-{body}-{c}"), format!("{org}-{body}-{bad}"))
}

fn de_npa_pair(r: &mut Rng) -> (String, String) {
    let first = r.chars(b"LMNPRTVWXY", 1);
    let mid = format!("{}{}", 1 + r.below(9), r.chars(b"0123456789CFGHJKLMNPRTVWXY", 7));
    let body = format!("{first}{mid}");
    let check = icao9303::check_digit(body.as_bytes()).unwrap() as char;
    let good = format!("{body}{check}");
    (good.clone(), bump_last_digit(&good))
}

fn fr_nir_pair(r: &mut Rng) -> (String, String) {
    let sex = 1 + r.below(2);
    let yy = r.below(100);
    let mm = 1 + r.below(12);
    let (dept, num_dept) = if r.below(2) == 0 { ("2A", 19u64) } else { ("2B", 18u64) };
    let com = 1 + r.below(900);
    let ord = 1 + r.below(900);
    let n = format!("{sex}{yy:02}{mm:02}{num_dept:02}{com:03}{ord:03}").parse::<u64>().unwrap();
    let key = 97 - (n % 97);
    let bad = if key == 97 { 1 } else { key + 1 };
    (
        format!("{sex} {yy:02} {mm:02} {dept} {com:03} {ord:03} {key:02}"),
        format!("{sex} {yy:02} {mm:02} {dept} {com:03} {ord:03} {bad:02}"),
    )
}

fn fr_siren_siret(r: &mut Rng) -> (String, String, String, String) {
    let s9 = with_luhn(&format!("{}{}", 90 + r.below(9), r.digits(6)));
    let siren = format!("{} {} {}", &s9[..3], &s9[3..6], &s9[6..]);
    let s14 = with_luhn(&format!("{s9}{:04}", 1 + r.below(900)));
    let siret = format!("{}-{}", &s14[..9], &s14[9..]);
    (siren.clone(), bump_last_digit(&siren), siret.clone(), bump_last_digit(&siret))
}

fn br_cpf_pair(r: &mut Rng) -> (String, String) {
    let base = format!("{}{}", 1 + r.below(8), r.digits(8));
    let cand =
        (0..100).map(|k| format!("{base}{k:02}")).find(|s| mod11::br_cpf(s.as_bytes().try_into().unwrap())).unwrap();
    let fmt = format!("{}.{}.{}-{}", &cand[..3], &cand[3..6], &cand[6..9], &cand[9..]);
    (fmt.clone(), bump_last_digit(&fmt))
}

fn br_cnpj_pair(r: &mut Rng) -> (String, String) {
    let base = format!("{}{}0001", 1 + r.below(8), r.digits(7));
    let cand =
        (0..100).map(|k| format!("{base}{k:02}")).find(|s| mod11::br_cnpj(s.as_bytes().try_into().unwrap())).unwrap();
    let fmt = format!("{}.{}.{}/{}-{}", &cand[..2], &cand[2..5], &cand[5..8], &cand[8..12], &cand[12..]);
    (fmt.clone(), bump_last_digit(&fmt))
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
    for _ in 0..per_kind {
        let (dni, dni_bad) = es_dni_pair(r);
        out.push(placed(r, "DNI ", &dni, &["es-dni"]));
        out.push(Labelled { text: format!("DNI {dni_bad}"), labels: vec![] });

        let (nie, nie_bad) = es_nie_pair(r);
        out.push(placed(r, "NIE ", &nie, &["es-nie"]));
        out.push(Labelled { text: format!("NIE {nie_bad}"), labels: vec![] });

        let (cif, cif_bad) = es_cif_pair(r);
        out.push(placed(r, "CIF ", &cif, &["es-cif"]));
        out.push(Labelled { text: format!("CIF {cif_bad}"), labels: vec![] });

        let (npa, npa_bad) = de_npa_pair(r);
        out.push(placed(r, "Ausweis ", &npa, &["de-personalausweis"]));
        out.push(Labelled { text: format!("Ausweis {npa_bad}"), labels: vec![] });

        let (nir, nir_bad) = fr_nir_pair(r);
        out.push(placed(r, "NIR ", &nir, &["fr-nir"]));
        out.push(Labelled { text: format!("NIR {nir_bad}"), labels: vec![] });

        let (siren, siren_bad, siret, siret_bad) = fr_siren_siret(r);
        out.push(placed(r, "SIREN ", &siren, &["fr-siren"]));
        out.push(Labelled { text: format!("SIREN {siren_bad}"), labels: vec![] });
        out.push(placed(r, "SIRET ", &siret, &["fr-siret"]));
        out.push(Labelled { text: format!("SIRET {siret_bad}"), labels: vec![] });

        let (cpf, cpf_bad) = br_cpf_pair(r);
        out.push(placed(r, "CPF ", &cpf, &["br-cpf"]));
        out.push(Labelled { text: format!("CPF {cpf_bad}"), labels: vec![] });

        let (cnpj, cnpj_bad) = br_cnpj_pair(r);
        out.push(placed(r, "CNPJ ", &cnpj, &["br-cnpj"]));
        out.push(Labelled { text: format!("CNPJ {cnpj_bad}"), labels: vec![] });

        let bic_code = format!(
            "{}{}{}{}",
            r.chars(b"ABCDEFGHJKLMNPQRSTUVWXYZ", 4),
            r.pick(&["DE", "FR", "ES", "IT", "NL", "BR"]),
            r.chars(b"ABCDEFGHJKLMNPQRSTUVWXYZ", 2),
            r.digits(3)
        );
        out.push(placed(r, "BIC: ", &bic_code, &["bic"]));
        out.push(Labelled { text: format!("BIC: {}ZZFF500", &bic_code[..4]), labels: vec![] });
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

/// The two OpenNLP corpora as labelled texts, with labels strictly
/// nested inside a longer label of the same text dropped to match the
/// default detector.
fn corpora() -> Vec<Labelled> {
    let mut out: Vec<Labelled> = Vec::new();
    for line in fixtures("golden-true-positives.tsv") {
        let f: Vec<&str> = line.splitn(4, '\t').collect();
        let kind = PiiKind::parse(f[0]).expect("a kind").builtin_name().unwrap();
        let start = f[3].find(f[1]).expect("covered text in the source") as u32;
        let label = (start, start + f[1].len() as u32, kind);
        match out.iter_mut().find(|l| l.text == f[3]) {
            Some(l) => l.labels.push(label),
            None => out.push(Labelled { text: f[3].to_owned(), labels: vec![label] }),
        }
    }
    for l in &mut out {
        let all = l.labels.clone();
        l.labels.retain(|m| !all.iter().any(|o| o.0 <= m.0 && m.1 <= o.1 && (o.0 < m.0 || m.1 < o.1)));
    }
    out.extend(fixtures("golden-false-positives.txt").into_iter().map(|text| Labelled { text, labels: vec![] }));
    out
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
            "{kind:<16} {:>5} {:>5} {:>5} {:>9} {:>7}",
            c.tp,
            c.fp,
            c.fn_,
            ratio_text(c.tp, c.tp + c.fp),
            ratio_text(c.tp, c.tp + c.fn_)
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

/// The measured totals are precision 0.9490 and recall 1.0. The false mentions are a phone in each near miss
/// NHS number (ten grouped digits are a phone shape); phones and cards nested inside grouped IBANs and NHS
/// numbers nested inside US phones are dropped by default. docs/pii.md has the table.
const PRECISION_FLOOR: f64 = 0.94;
const RECALL_FLOOR: f64 = 1.0;

fn human_written_set() -> Vec<Labelled> {
    load_labelled(&HUMAN_DOCS, "labels.tsv", false)
}

#[test]
fn precision_and_recall_on_human_written_text() {
    let set = human_written_set();
    let d = wide();
    for l in &set {
        let pii = layer(&d, &l.text);
        pii.check().expect("layer is ordered");
        for m in pii.mentions() {
            assert!(!l.text[m.span.range()].is_empty());
        }
    }
    let by_kind = measure(&set);
    let mut total = Counts::default();
    eprintln!("\nhuman-written evaluation set:");
    eprintln!("{:<16} {:>5} {:>5} {:>5} {:>9} {:>7}", "kind", "tp", "fp", "fn", "precision", "recall");
    for (name, c) in &by_kind {
        assert!(c.tp > 0, "{name} has no true positive in the human-written set");
        eprintln!(
            "{name:<16} {:>5} {:>5} {:>5} {:>9} {:>7}",
            c.tp,
            c.fp,
            c.fn_,
            ratio_text(c.tp, c.tp + c.fp),
            ratio_text(c.tp, c.tp + c.fn_)
        );
        total.tp += c.tp;
        total.fp += c.fp;
        total.fn_ += c.fn_;
    }
    let (precision, recall) = (ratio(total.tp, total.tp + total.fp), ratio(total.tp, total.tp + total.fn_));
    eprintln!(
        "{} documents, {} bytes; all: tp {} fp {} fn {}, precision {precision:.4}, recall {recall:.4}",
        set.len(),
        set.iter().map(|l| l.text.len()).sum::<usize>(),
        total.tp,
        total.fp,
        total.fn_
    );
    let want_by_kind: [(&str, usize, usize, usize); 30] = [
        ("jwt", 3, 0, 0),
        ("aws-access-key", 2, 0, 0),
        ("github-token", 3, 0, 0),
        ("url-credential", 2, 0, 0),
        ("email", 3, 2, 0),
        ("iban", 4, 0, 0),
        ("imei", 3, 0, 2),
        ("card", 4, 2, 0),
        ("btc-address", 8, 0, 0),
        ("eth-address", 10, 0, 0),
        ("mac", 3, 0, 0),
        ("ipv6", 3, 0, 0),
        ("ipv4", 3, 0, 0),
        ("us-ssn", 3, 0, 0),
        ("us-itin", 3, 0, 0),
        ("uk-nhs", 3, 0, 0),
        ("de-steuer-id", 3, 0, 0),
        ("ca-sin", 3, 0, 1),
        ("aba-routing", 3, 0, 0),
        ("phone", 15, 6, 2),
        ("vin", 4, 0, 0),
        ("tracking-number", 9, 0, 0),
        ("us-ein", 3, 0, 0),
        ("url", 10, 0, 0),
        ("us-passport", 2, 0, 0),
        ("us-drivers-licence", 2, 0, 0),
        ("us-zip", 10, 0, 0),
        ("phone-extension", 2, 0, 0),
        ("age", 7, 0, 0),
        ("street-address", 8, 0, 1), // missed: `P0 Box 1163` (a zero) in masc-hsus4
    ];
    for (kind, tp, fp, fn_) in want_by_kind {
        let c = by_kind.get(kind).unwrap();
        assert_eq!((c.tp, c.fp, c.fn_), (tp, fp, fn_), "{kind}");
    }
    assert_eq!((total.tp, total.fp, total.fn_), (141, 10, 6));
}
