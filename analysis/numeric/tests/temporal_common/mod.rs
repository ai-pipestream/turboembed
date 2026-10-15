//! What the temporal tests compare: each mention's span, its value as
//! OpenNLP writes it, its granularity and its origin.

#![allow(dead_code)]

use turbo_analysis::{Analyzer, Document, Layer, Span, ViewId};
use turbo_numeric::temporal::scan;
use turbo_numeric::{Date, DocumentDateSelector, Granularity, Origin, TemporalExtractor, Temporals};

/// One mention: its byte span, its value, its granularity and origin.
pub type Found = (Span, String, Granularity, Origin);

/// Every mention `CursorTemporalExtractor.extract` finds in `text`, with
/// `reference` or without one.
pub fn found(text: &str, reference: Option<Date>) -> Vec<Found> {
    let mut t = Temporals::new(ViewId::SOURCE);
    scan(text, reference, &mut t);
    assert_eq!(t.check(), Ok(()), "{text:?}");
    rows(&t)
}

/// The rows of a layer as [`Found`].
pub fn rows(t: &Temporals) -> Vec<Found> {
    t.spans.iter().map(|(i, s)| (s, t.value(i).to_string(), t.granularity[i.get()], t.origin[i.get()])).collect()
}

/// The values alone.
pub fn values(text: &str, reference: Option<Date>) -> Vec<String> {
    found(text, reference).into_iter().map(|f| f.1).collect()
}

pub fn absolute(start: usize, end: usize, value: &str, g: Granularity) -> Found {
    (Span::new(start as u32, end as u32), value.into(), g, Origin::Absolute)
}

pub fn relative(start: usize, end: usize, value: &str, g: Granularity) -> Found {
    (Span::new(start as u32, end as u32), value.into(), g, Origin::Relative)
}

pub fn day(s: &str) -> Date {
    Date::parse(s).unwrap_or_else(|| panic!("{s:?}"))
}

/// The byte span of `fragment`, which occurs in `text`.
pub fn span_of(text: &str, fragment: &str) -> Span {
    let start = text.find(fragment).unwrap_or_else(|| panic!("{fragment:?} not in {text:?}"));
    Span::new(start as u32, (start + fragment.len()) as u32)
}

/// Runs `extractor` and the document date selector over `text`:
/// OpenNLP's `TemporalAnnotator` and `DocumentDateAnnotator` in a
/// pipeline (`NumericPacks.temporal`).
pub fn analyze(extractor: TemporalExtractor, text: &str) -> Document {
    let analyzer = Analyzer::builder().add(extractor).add(DocumentDateSelector).build().expect("the analyzer");
    let mut docs = vec![Document::new(text)];
    for r in analyzer.run(&mut docs) {
        r.unwrap_or_else(|e| panic!("{text:?}: {e}"));
    }
    docs.pop().unwrap()
}
