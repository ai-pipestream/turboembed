//! `PiiRewriteTest`'s offset cases: every offset of a long run of
//! replacements, and every source span of every one- and two-mention plan,
//! through the derived view's alignment.

mod support;

use support::*;
use turbo_analysis::{Layer, Side, Span};
use turbo_pii::rewrite::{HmacTokenizer, Pseudonymizer, source_view};
use turbo_pii::{Pii, PiiKind};

const KEY: &[u8] = b"0123456789abcdef0123456789abcdef";

fn tokenizer() -> HmacTokenizer {
    HmacTokenizer::new(KEY).unwrap()
}

#[test]
fn many_replacements_map_every_offset() {
    let text = "xx ".repeat(1024);
    let values: Vec<String> = (0..1024).map(|i| format!("value{}", 3 * i)).collect();
    let rows: Vec<(u32, u32, PiiKind, &str)> =
        values.iter().enumerate().map(|(i, v)| (3 * i as u32, 3 * i as u32 + 2, PiiKind::Email, v.as_str())).collect();
    let pii = manual(&rows);
    let r = Pseudonymizer::default().rewrite(&source_view(&text), &pii).unwrap();
    let labels: Vec<Span> = r.labels().mentions().map(|m| m.span).collect();
    for offset in 0..=text.len() as u32 {
        assert_eq!(r.alignment().to_child(offset, Side::Start), boundary(offset, false, &rows, &labels), "{offset}");
    }
}

/// Where the Java's `PiiRewriteTest.boundary` puts an offset.
fn boundary(offset: u32, end: bool, sources: &[(u32, u32, PiiKind, &str)], labels: &[Span]) -> u32 {
    let mut shift: i64 = 0;
    for (source, label) in sources.iter().zip(labels) {
        if offset <= source.0 {
            return (i64::from(offset) + shift) as u32;
        }
        if offset < source.1 {
            return if end { label.end } else { label.start };
        }
        shift += i64::from(label.end - label.start) - i64::from(source.1 - source.0);
    }
    (i64::from(offset) + shift) as u32
}

#[test]
fn every_source_span_of_every_plan() {
    for text in ["", "abcdefgh", "😀abc𝕒"] {
        let bounds: Vec<u32> = (0..=text.len()).filter(|&i| text.is_char_boundary(i)).map(|i| i as u32).collect();
        let mut plans: Vec<Vec<Span>> = vec![vec![]];
        for (i, &s) in bounds.iter().enumerate() {
            for &e in &bounds[i + 1..] {
                plans.push(vec![Span::new(s, e)]);
                for (j, &ns) in bounds.iter().enumerate().filter(|&(_, &b)| b >= e) {
                    for &ne in &bounds[j + 1..] {
                        plans.push(vec![Span::new(s, e), Span::new(ns, ne)]);
                    }
                }
            }
        }
        for plan in &plans {
            let values: Vec<String> = plan.iter().map(|s| format!("value{}", s.start)).collect();
            let rows: Vec<(u32, u32, PiiKind, &str)> =
                plan.iter().zip(&values).map(|(s, v)| (s.start, s.end, PiiKind::UsSsn, v.as_str())).collect();
            let reversed: Vec<_> = rows.iter().rev().copied().collect();
            for mode in 0..3 {
                let run = |rows: &[(u32, u32, PiiKind, &str)]| {
                    let pii: Pii = manual(rows);
                    match mode {
                        0 => Pseudonymizer::default().rewrite(&source_view(text), &pii).unwrap(),
                        1 => Pseudonymizer::new("[", "]").rewrite(&source_view(text), &pii).unwrap(),
                        _ => tokenizer().rewrite(&source_view(text), &pii).unwrap(),
                    }
                };
                let r = run(&rows);
                let back = run(&reversed);
                assert_eq!(r.text(), back.text());
                let labels: Vec<Span> = r.labels().mentions().map(|m| m.span).collect();
                assert_eq!(labels, back.labels().mentions().map(|m| m.span).collect::<Vec<_>>());
                let mut want = String::new();
                let mut copied = 0;
                for (span, label) in plan.iter().zip(r.labels().mentions()) {
                    want.push_str(&text[copied..span.start as usize]);
                    assert_eq!(want.len() as u32, label.span.start);
                    want.push_str(label.normalized);
                    assert_eq!(want.len() as u32, label.span.end);
                    assert_eq!(label.kind, PiiKind::UsSsn);
                    copied = span.end as usize;
                }
                want.push_str(&text[copied..]);
                assert_eq!(r.text(), want);
                assert_eq!(r.labels().check(), Ok(()));
                for &s in &bounds {
                    let start = boundary(s, false, &rows, &labels);
                    assert_eq!(r.alignment().to_child(s, Side::Start), start);
                    // An empty span strictly inside a mention takes in
                    // its whole label; back again it covers the mention.
                    let inside = plan.iter().position(|m| m.start < s && s < m.end);
                    for &e in bounds.iter().filter(|&&e| e >= s) {
                        let want = match inside {
                            Some(k) if s == e => labels[k],
                            _ if s == e => Span::new(start, start),
                            _ => Span::new(start, boundary(e, true, &rows, &labels)),
                        };
                        assert_eq!(r.to_derived(Span::new(s, e)), Ok(want), "{text:?} {plan:?} {s}..{e}");
                        assert!(r.to_original(want).unwrap().contains(Span::new(s, e)), "{text:?} {plan:?} {s}..{e}");
                    }
                }
            }
        }
    }
}
