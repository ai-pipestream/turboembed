//! Helpers shared by the integration tests.

#![allow(dead_code)]

use std::collections::BTreeMap;

use turbo_analysis::ViewId;
use turbo_goldens::{escape::unescape, hex::to_hex, sha256};
use turbo_pii::{Extractor, Pack, Pii, PiiDetector, PiiKind, PiiKinds, Scanner};

/// One mention as the tests compare it: start, end, kind name, normalized.
pub type Row = (u32, u32, &'static str, String);

pub fn layer(d: &PiiDetector, text: &str) -> Pii {
    d.layer(text, ViewId::SOURCE)
}

pub fn rows(d: &PiiDetector, text: &str) -> Vec<Row> {
    layer(d, text).mentions().map(|m| (m.span.start, m.span.end, m.kind.name(), m.normalized.to_owned())).collect()
}

/// Asserts the mentions of `text`, in order.
#[track_caller]
pub fn assert_found(d: &PiiDetector, text: &str, expected: &[(u32, u32, &str, &str)]) {
    let got = rows(d, text);
    let want: Vec<(u32, u32, &str, &str)> = expected.to_vec();
    let got_ref: Vec<(u32, u32, &str, &str)> = got.iter().map(|r| (r.0, r.1, r.2, r.3.as_str())).collect();
    assert_eq!(got_ref, want, "{text:?}");
}

#[track_caller]
pub fn assert_none(d: &PiiDetector, text: &str) {
    assert_eq!(rows(d, text), Vec::<Row>::new(), "{text:?}");
}

/// The covered text of each mention.
pub fn covered<'t>(d: &PiiDetector, text: &'t str) -> Vec<&'t str> {
    layer(d, text).mentions().map(|m| &text[m.span.start as usize..m.span.end as usize]).collect()
}

pub fn normalized(d: &PiiDetector, text: &str) -> Vec<String> {
    rows(d, text).into_iter().map(|r| r.3).collect()
}

pub fn kinds(d: &PiiDetector, text: &str) -> Vec<&'static str> {
    rows(d, text).into_iter().map(|r| r.2).collect()
}

/// One labelled text: the text and its labels as (start, end, kind).
pub struct Labelled {
    pub text: String,
    pub labels: Vec<(u32, u32, &'static str)>,
}

#[derive(Default, Debug)]
pub struct Counts {
    pub tp: usize,
    pub fp: usize,
    pub fn_: usize,
}

pub fn ratio(a: usize, b: usize) -> f64 {
    if b == 0 { 1.0 } else { a as f64 / b as f64 }
}

/// Per-kind true/false/missed counts of the widest pack over `set`.
/// PII_SHOW_FALSE=1 lists each false mention.
pub fn measure(set: &[Labelled]) -> BTreeMap<&'static str, Counts> {
    let d = wide();
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

/// Documents of `table` (id, goldens-escaped, file, pinned SHA-256)
/// under `tests/data/human-eval`, labelled from `labels_file` there
/// (five TSV columns; newlines in covered text escaped as `\n` when
/// `escaped_newlines`).
pub fn load_labelled(table: &[(&str, bool, &str, &str)], labels_file: &str, escaped_newlines: bool) -> Vec<Labelled> {
    let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/data/human-eval");
    let mut docs: Vec<(&str, Labelled)> = Vec::with_capacity(table.len());
    for &(id, escaped, file, expected_sha) in table {
        let raw = std::fs::read_to_string(dir.join(file)).expect("human-eval document on disk");
        let text = if escaped { unescape(raw.trim_end_matches('\n')).expect("valid escaped text") } else { raw };
        assert_eq!(to_hex(&sha256(text.as_bytes())), expected_sha, "{id} SHA-256");
        docs.push((id, Labelled { text, labels: Vec::new() }));
    }
    let labels_raw = std::fs::read_to_string(dir.join(labels_file)).expect("human-eval labels");
    for line in labels_raw.lines().filter(|l| !l.trim().is_empty() && !l.starts_with('#')) {
        let f: Vec<&str> = line.splitn(5, '\t').collect();
        assert_eq!(f.len(), 5, "five columns in {labels_file}: {line:?}");
        let start: u32 = f[1].parse().expect("start offset");
        let end: u32 = f[2].parse().expect("end offset");
        let kind = PiiKind::parse(f[3]).expect("a built-in kind").name();
        let (_, doc) = docs.iter_mut().find(|(id, _)| *id == f[0]).expect("known document id");
        let covered = if escaped_newlines { f[4].replace("\\n", "\n") } else { f[4].to_owned() };
        assert_eq!(&doc.text[start as usize..end as usize], covered, "{} {start}..{end}", f[0]);
        let label = (start, end, kind);
        assert!(!doc.labels.contains(&label), "duplicate label {label:?} in {}", f[0]);
        doc.labels.push(label);
    }
    docs.into_iter().map(|(_, d)| d).collect()
}

/// One extractor limited to `kinds`.
pub fn only(extractor: Extractor, kinds: &[PiiKind]) -> PiiDetector {
    PiiDetector::new(vec![Scanner::with_kinds(extractor, PiiKinds::of(kinds)).unwrap()])
}

pub fn pack(pack: Pack) -> PiiDetector {
    PiiDetector::pack(pack)
}

pub fn wide() -> PiiDetector {
    PiiDetector::pack(Pack::AllStructured)
}

/// `number` with `separator` after every `size` characters.
pub fn grouped(number: &str, size: usize, separator: char) -> String {
    let mut out = String::new();
    for (i, c) in number.chars().enumerate() {
        if i > 0 && i % size == 0 {
            out.push(separator);
        }
        out.push(c);
    }
    out
}

/// Unpadded base64url, as `Base64.getUrlEncoder().withoutPadding()`.
pub fn base64url(bytes: &[u8]) -> String {
    const A: &[u8] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-_";
    let mut out = String::new();
    for chunk in bytes.chunks(3) {
        let n = chunk.iter().enumerate().fold(0u32, |n, (i, &b)| n | u32::from(b) << (16 - 8 * i));
        for i in 0..=chunk.len() {
            out.push(A[(n >> (18 - 6 * i) & 63) as usize] as char);
        }
    }
    out
}

/// A layer of the given mentions, in the order given.
pub fn manual(mentions: &[(u32, u32, PiiKind, &str)]) -> Pii {
    let mut pii = Pii::new(ViewId::SOURCE);
    for &(start, end, kind, normalized) in mentions {
        pii.push(turbo_pii::PiiMention {
            span: turbo_analysis::Span::new(start, end),
            kind,
            confidence: turbo_pii::PiiConfidence::Format,
            normalized,
        });
    }
    pii
}
