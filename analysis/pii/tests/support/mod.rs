//! Helpers shared by the integration tests.

#![allow(dead_code)]

use turbo_analysis::ViewId;
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
