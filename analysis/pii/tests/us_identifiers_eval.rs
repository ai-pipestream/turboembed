//! #1878: precision and recall of the widest pack for `vin`,
//! `tracking-number`, `us-ein` and `url` on a synthetic labelled set
//! made here from a fixed seed: made up values with valid check digits
//! (or an assigned EIN prefix, or a documentation domain), each placed in
//! a sentence, and near misses with one check digit changed, an
//! unassigned prefix or an unregistered top-level domain, with no label.
//! Scored against the labels only; every other mention is false.

mod support;

use support::{Counts, Labelled, measure, ratio};
use turbo_pii::checksum::{luhn, parcel, vin};

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

    fn chars(&mut self, alphabet: &[u8], n: usize) -> String {
        (0..n).map(|_| char::from(alphabet[self.below(alphabet.len() as u64) as usize])).collect()
    }

    fn pick<'a>(&mut self, from: &[&'a str]) -> &'a str {
        from[self.below(from.len() as u64) as usize]
    }
}

const BEFORE: [&str; 5] = ["Please use ", "ref ", "(", "The record shows ", "\u{2022} "];
const AFTER: [&str; 5] = [" today.", ")", ", thanks.", " \u{2014} noted", "."];

fn placed(r: &mut Rng, value: &str, kind: Option<&'static str>) -> Labelled {
    let before = r.pick(&BEFORE).to_owned();
    let text = format!("{before}{value}{}", r.pick(&AFTER));
    let labels = kind.map(|k| (before.len() as u32, (before.len() + value.len()) as u32, k)).into_iter().collect();
    Labelled { text, labels }
}

fn digit(d: u8) -> char {
    char::from(b'0' + d)
}

fn bump(c: char) -> char {
    if c == '9' { '0' } else { char::from(c as u8 + 1) }
}

/// A valid VIN (with letters, a model year that is not U, Z or 0) and the
/// same VIN with its check character changed.
fn vin_pair(r: &mut Rng) -> (String, String) {
    const ALPHABET: &[u8] = b"ABCDEFGHJKLMNPRSTUVWXYZ0123456789";
    loop {
        let mut v: [u8; 17] = r.chars(ALPHABET, 17).into_bytes().try_into().unwrap();
        v[0] = b"1245JKLSWYZ"[r.below(11) as usize];
        v[9] = b"ABCDEFGHJKLMNPRSTVWXY123456789"[r.below(30) as usize];
        v[8] = vin::check_char(&v).unwrap();
        if !v.iter().any(u8::is_ascii_digit) {
            continue;
        }
        let good = String::from_utf8(v.to_vec()).unwrap();
        v[8] = if v[8] == b'X' {
            b'0'
        } else if v[8] == b'9' {
            b'X'
        } else {
            v[8] + 1
        };
        return (good, String::from_utf8(v.to_vec()).unwrap());
    }
}

/// A valid tracking number of a random carrier form and a check digit
/// near miss.
fn tracking_pair(r: &mut Rng) -> (String, String) {
    let (body, check) = match r.below(5) {
        0 => {
            let serial = format!("{}{}", r.chars(b"ABCDEFGHJKLMNPRSTUVWXYZ0123456789", 6), r.digits(9));
            let check = parcel::ups(serial.as_bytes()).unwrap();
            (format!("1Z{serial}"), check)
        }
        1 => loop {
            let d = r.digits(11);
            let n = format!("{d}{}", digit(parcel::fedex_mod11(d.as_bytes()).unwrap()));
            if !luhn::valid(n.as_bytes()) {
                break (d.clone(), parcel::fedex_mod11(d.as_bytes()).unwrap());
            }
        },
        2 => loop {
            let d = r.digits(14);
            let c = parcel::gs1_mod10(d.as_bytes()).unwrap();
            if !luhn::valid(format!("{d}{}", digit(c)).as_bytes()) {
                break (d, c);
            }
        },
        3 => {
            let d = format!("9{}{}", r.pick(&["1", "2", "3", "4", "5"]), r.digits(19));
            let c = parcel::gs1_mod10(d.as_bytes()).unwrap();
            (d, c)
        }
        _ => {
            let d = format!("{}{}", r.pick(&["0", "7"]), r.digits(18));
            let c = parcel::gs1_mod10(d.as_bytes()).unwrap();
            (d, c)
        }
    };
    let good = format!("{body}{}", digit(check));
    (good.clone(), format!("{body}{}", bump(digit(check))))
}

fn ein_pair(r: &mut Rng) -> (String, String) {
    let assigned = ["01", "12", "20", "27", "33", "46", "52", "68", "77", "84", "91", "99"];
    let unassigned = ["00", "07", "17", "28", "49", "69", "78", "89", "96", "97"];
    let serial = format!("{}{}", r.below(9) + 1, r.digits(6));
    (format!("{}-{serial}", r.pick(&assigned)), format!("{}-{serial}", r.pick(&unassigned)))
}

fn url_pair(r: &mut Rng) -> (String, String) {
    let host = format!(
        "{}.{}",
        r.chars(b"abcdefghijklmnopqrstuvwxyz", 6),
        r.pick(&["example.com", "example.org", "example.net"])
    );
    let path = format!("/{}/{}?q={}", r.chars(b"abcdefghij", 4), r.digits(3), r.chars(b"xyz", 2));
    let scheme = r.pick(&["https://", "http://", "www."]);
    let scheme = if scheme == "www." { "https://www." } else { scheme };
    (format!("{scheme}{host}{path}"), format!("http://{}.invalid{path}", r.chars(b"klmnop", 5)))
}

fn synthetic() -> Vec<Labelled> {
    let mut r = Rng(1878);
    let mut out = Vec::new();
    for _ in 0..200 {
        for (pair, kind) in [
            (vin_pair(&mut r), "vin"),
            (tracking_pair(&mut r), "tracking-number"),
            (ein_pair(&mut r), "us-ein"),
            (url_pair(&mut r), "url"),
        ] {
            out.push(placed(&mut r, &pair.0, Some(kind)));
            out.push(placed(&mut r, &pair.1, None));
        }
    }
    out
}

#[test]
fn precision_and_recall_on_the_synthetic_us_identifier_set() {
    let set = synthetic();
    let by_kind = measure(&set);
    let mut total = Counts::default();
    for (kind, c) in &by_kind {
        eprintln!("{kind:<16} {:>5} {:>5} {:>5}", c.tp, c.fp, c.fn_);
        total.tp += c.tp;
        total.fp += c.fp;
        total.fn_ += c.fn_;
    }
    for kind in ["vin", "tracking-number", "us-ein", "url"] {
        let c = &by_kind[kind];
        assert_eq!((c.tp, c.fn_), (200, 0), "{kind}: every valid value is found");
    }
    let precision = ratio(total.tp, total.tp + total.fp);
    eprintln!("all: tp {} fp {} fn {}, precision {precision:.4}", total.tp, total.fp, total.fn_);
    // No near miss is read as one of the four kinds. The three false
    // mentions are cards: near-miss digit runs that happen to pass Luhn.
    assert_eq!(by_kind.keys().copied().collect::<Vec<_>>(), ["card", "tracking-number", "url", "us-ein", "vin"]);
    assert_eq!((by_kind["card"].tp, by_kind["card"].fp), (0, 3));
    assert_eq!((total.tp, total.fp, total.fn_), (800, 3, 0));
}
