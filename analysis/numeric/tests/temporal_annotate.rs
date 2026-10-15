//! The reference date and the document date: the cases of OpenNLP's
//! `TemporalAnnotatorTest`, `DocumentDateAnnotatorTest`,
//! `DocumentDateAnnotatorContractTest` and the temporal parts of
//! `NumericPacksTest` and `NumericManualExampleTest`.
//!
//! The Java annotators take any `TemporalExtractor` and any temporal
//! layer, and check what they are given: a result that is null, out of
//! order or out of bounds (`TemporalAnnotatorPassTest`), a value whose
//! span is not its annotation's, a day value that is no ISO date, a layer
//! already there (`TemporalExtractorContractTest` checks the interface's
//! default method). Here the extractor is one type and a `Temporals`
//! layer holds one span and a typed value per mention in text order, so
//! those cases cannot arise; the ones that can are below.

mod temporal_common;

use temporal_common::{absolute, analyze, day, relative, rows, span_of};
use turbo_analysis::{Analyzer, Annotator, Document, Ix, Layer, Span, ViewId};
use turbo_numeric::{
    DocumentDate, DocumentDateSelector, Granularity, Origin, TemporalExtractor, TemporalMention, TemporalValue,
    Temporals,
};

use Granularity::{Day, Month};

fn temporals(extractor: TemporalExtractor, text: &str) -> Vec<temporal_common::Found> {
    let mut t = Temporals::new(ViewId::SOURCE);
    extractor.extract(text, &mut t);
    rows(&t)
}

const DATELINE: &str = "Berlin, 14 July 2026. The buyer paid yesterday.";

// TemporalAnnotatorTest

/// A dateline dates the document, and a relative expression after it
/// resolves against it.
#[test]
fn a_dateline_resolves_a_later_relative_expression() {
    let all = temporals(TemporalExtractor::new(), DATELINE);
    let (d, y) = (span_of(DATELINE, "14 July 2026"), span_of(DATELINE, "yesterday"));
    assert_eq!(
        all,
        [
            absolute(d.start as usize, d.end as usize, "2026-07-14", Day),
            relative(y.start as usize, y.end as usize, "2026-07-13", Day)
        ]
    );
}

#[test]
fn the_first_day_supplies_the_reference() {
    let text = "Filed 2026-07-10. Hearing on 2026-09-02. Amended yesterday.";
    let all = temporals(TemporalExtractor::new(), text);
    assert_eq!(all.len(), 3);
    assert_eq!((all[2].0, all[2].1.as_str()), (span_of(text, "yesterday"), "2026-07-09"));
}

#[test]
fn counted_and_coarser_relatives_resolve_against_the_dateline() {
    let text = "Berlin, 14 July 2026. Shipped 3 days ago, audited last month, due tomorrow.";
    let all = temporals(TemporalExtractor::new(), text);
    assert_eq!(all.len(), 4);
    assert_eq!((all[1].0, all[1].1.as_str()), (span_of(text, "3 days ago"), "2026-07-11"));
    assert_eq!((all[2].1.as_str(), all[2].2), ("2026-06", Month));
    assert_eq!(all[3].1, "2026-07-15");
}

/// A month is too coarse to date a text, so relatives stay unread.
#[test]
fn a_text_without_a_day_leaves_relatives_unread() {
    let text = "The July 2026 report was filed yesterday.";
    let s = span_of(text, "July 2026");
    assert_eq!(
        temporals(TemporalExtractor::new(), text),
        [absolute(s.start as usize, s.end as usize, "2026-07", Month)]
    );
}

/// Nothing is resolved against the clock.
#[test]
fn a_relative_alone_gives_an_empty_layer() {
    let doc = analyze(TemporalExtractor::new(), "we shipped it yesterday");
    assert_eq!(doc.get::<Temporals>().unwrap().len(), 0);
}

#[test]
fn a_fixed_reference_resolves_without_a_dateline() {
    let text = "we shipped it yesterday";
    let s = span_of(text, "yesterday");
    assert_eq!(
        temporals(TemporalExtractor::with_reference(day("2026-07-14")), text),
        [relative(s.start as usize, s.end as usize, "2026-07-13", Day)]
    );
}

#[test]
fn a_fixed_reference_wins_over_the_dateline() {
    let all = temporals(TemporalExtractor::with_reference(day("2020-01-02")), DATELINE);
    assert_eq!(all.len(), 2);
    assert_eq!(all[1].1, "2020-01-01");
}

/// The mention that resolved the relatives is the one that dates the
/// document.
#[test]
fn the_pipeline_picks_the_date_that_resolved_the_relatives() {
    let doc = analyze(TemporalExtractor::new(), DATELINE);
    let date = doc.get::<DocumentDate>().unwrap();
    let t = doc.get::<Temporals>().unwrap();
    assert_eq!(date.date, Some(day("2026-07-14")));
    assert_eq!(t.spans.get(date.mention.unwrap()), span_of(DATELINE, "14 July 2026"));
    assert_eq!(t.value(Ix::new(1)).to_string(), "2026-07-13");
}

#[test]
fn a_relative_before_the_dateline_does_not_date_the_document() {
    let text = "Yesterday we filed. Dateline: 14 July 2026.";
    let doc = analyze(TemporalExtractor::new(), text);
    let date = doc.get::<DocumentDate>().unwrap();
    let t = doc.get::<Temporals>().unwrap();
    assert_eq!(date.date, Some(day("2026-07-14")));
    assert_eq!(t.spans.get(date.mention.unwrap()), span_of(text, "14 July 2026"));
    assert_eq!(t.value(Ix::new(0)).to_string(), "2026-07-13");
}

// DocumentDateAnnotatorTest and DocumentDateAnnotatorContractTest

fn mention(start: u32, end: u32, value: TemporalValue, origin: Origin) -> TemporalMention {
    TemporalMention { span: Span::new(start, end), value, origin }
}

/// A selector over a layer built by hand.
fn select(mentions: &[TemporalMention]) -> DocumentDate {
    let mut t = Temporals::new(ViewId::SOURCE);
    for &m in mentions {
        t.push(m);
    }
    assert_eq!(t.check(), Ok(()));
    DocumentDateSelector.select(&t)
}

#[test]
fn the_first_day_dates_the_document() {
    // "2026-07 report, filed 2026-07-10, due 2026-07-20."
    let date = select(&[
        mention(0, 7, TemporalValue::Month { year: 2026, month: 7 }, Origin::Absolute),
        mention(22, 32, TemporalValue::Day(day("2026-07-10")), Origin::Absolute),
        mention(38, 48, TemporalValue::Day(day("2026-07-20")), Origin::Absolute),
    ]);
    assert_eq!(date, DocumentDate { date: Some(day("2026-07-10")), mention: Some(Ix::new(1)) });
}

#[test]
fn months_and_quarters_never_date_the_document() {
    let date = select(&[
        mention(4, 11, TemporalValue::Month { year: 2026, month: 7 }, Origin::Absolute),
        mention(23, 30, TemporalValue::Quarter { year: 2024, quarter: 3 }, Origin::Absolute),
    ]);
    assert_eq!(date, DocumentDate::default());
}

/// A relative day before the first absolute one is passed over; the
/// contract test's text, `2026-07-14; 2026-08-01; yesterday`, with the
/// relative mention moved first.
#[test]
fn relative_days_never_date_the_document() {
    let date = select(&[
        mention(0, 9, TemporalValue::Day(day("2026-07-13")), Origin::Relative),
        mention(12, 22, TemporalValue::Day(day("2026-07-14")), Origin::Absolute),
        mention(24, 34, TemporalValue::Day(day("2026-08-01")), Origin::Absolute),
    ]);
    assert_eq!(date, DocumentDate { date: Some(day("2026-07-14")), mention: Some(Ix::new(1)) });
}

#[test]
fn the_first_of_conflicting_dates_wins_through_the_pipeline() {
    let text = "Filed 2026-07-10. Hearing on 2026-09-02.";
    let doc = analyze(TemporalExtractor::new(), text);
    let date = doc.get::<DocumentDate>().unwrap();
    assert_eq!(date.date, Some(day("2026-07-10")));
    assert_eq!(doc.get::<Temporals>().unwrap().spans.get(date.mention.unwrap()), Span::new(6, 16));
}

/// A pipeline with no step that writes the temporal layer is refused; an
/// empty temporal layer gives a document date layer with no date.
#[test]
fn the_temporal_layer_is_required() {
    let e = Analyzer::builder().add(DocumentDateSelector).build().err().expect("no temporal layer to read");
    assert!(e.to_string().contains("turbo:temporal"), "{e}");
    let doc = analyze(TemporalExtractor::new(), "no dates here");
    assert_eq!(doc.get::<DocumentDate>().unwrap(), &DocumentDate::default());
    assert_eq!(doc.get::<DocumentDate>().unwrap().len(), 0);
}

/// The layer refuses mentions out of text order, so no list order can
/// pick a later date; the Java sorts by start instead.
#[test]
fn mentions_are_in_text_order() {
    let mut t = Temporals::new(ViewId::SOURCE);
    t.push(mention(12, 22, TemporalValue::Day(day("2026-08-01")), Origin::Absolute));
    t.push(mention(0, 10, TemporalValue::Day(day("2026-07-14")), Origin::Absolute));
    assert!(t.check().is_err());
    let mut doc = Document::new("2026-07-14; 2026-08-01; yesterday");
    assert!(doc.add(&turbo_analysis::Key::<Temporals>::DEFAULT, t, turbo_analysis::Provenance::by("test")).is_err());
}

/// One selector shared between threads keeps no date of one document for
/// another.
#[test]
fn concurrent_documents() {
    let selector = DocumentDateSelector;
    std::thread::scope(|s| {
        for k in 0..4 {
            s.spawn(move || {
                for i in 0..25 {
                    let date = day("2026-07-14").plus_days(k * 25 + i).unwrap();
                    let mut t = Temporals::new(ViewId::SOURCE);
                    t.push(mention(0, 10, TemporalValue::Day(date), Origin::Absolute));
                    let (out,) = selector.annotate((&t,)).unwrap();
                    assert_eq!(out.date, Some(date));
                }
            });
        }
    });
}

// NumericPacksTest and NumericManualExampleTest, the temporal parts.

#[test]
fn the_temporal_pack_dates_the_text_and_resolves_relatives() {
    let doc = analyze(TemporalExtractor::new(), DATELINE);
    assert_eq!(doc.get::<DocumentDate>().unwrap().date, Some(day("2026-07-14")));
    let t = doc.get::<Temporals>().unwrap();
    assert_eq!(t.len(), 2);
    assert_eq!(t.spans.get(Ix::new(0)), Span::new(8, 20));
    assert_eq!(&DATELINE[t.spans.get(Ix::new(1)).range()], "yesterday");
    assert_eq!(t.value(Ix::new(1)).to_string(), "2026-07-13");
}

/// A fixed reference adds no absolute mention, so the document has no date.
#[test]
fn the_temporal_pack_with_a_fixed_reference() {
    let doc = analyze(TemporalExtractor::with_reference(day("2026-07-14")), "it shipped yesterday");
    let t = doc.get::<Temporals>().unwrap();
    assert_eq!((t.len(), t.value(Ix::new(0)).to_string()), (1, "2026-07-13".to_string()));
    assert_eq!(doc.get::<DocumentDate>().unwrap().date, None);
}

#[test]
fn a_relative_without_a_dateline_gives_nothing() {
    let doc = analyze(TemporalExtractor::new(), "The buyer paid yesterday.");
    assert_eq!(doc.get::<Temporals>().unwrap().len(), 0);
}

// Edge cases of the port.

/// The dateline is the first absolute day, not the first mention: a month
/// or a quarter before it is passed over, and a day after a relative
/// expression still resolves the expression.
#[test]
fn the_dateline_is_the_first_absolute_day() {
    let text = "Q3 2026 report: last week we filed; dated July 2026 and 2026-07-17.";
    let all = temporals(TemporalExtractor::new(), text);
    let got: Vec<&str> = all.iter().map(|f| f.1.as_str()).collect();
    assert_eq!(got, ["2026-Q3", "2026-W28", "2026-07", "2026-07-17"]);
}

#[test]
fn the_configuration_names_the_reference() {
    let dateline = TemporalExtractor::new();
    let fixed = TemporalExtractor::with_reference(day("2026-07-14"));
    let other = TemporalExtractor::with_reference(day("2026-07-15"));
    assert_ne!(dateline.config_hash(), fixed.config_hash());
    assert_ne!(fixed.config_hash(), other.config_hash());
    assert_eq!((dateline.reference(), fixed.reference()), (None, Some(day("2026-07-14"))));
    assert_eq!(TemporalExtractor::ID, "turbo-numeric:temporal");
    assert_eq!(DocumentDateSelector::ID, "turbo-numeric:document-date");
}

/// A document date known from metadata, with no mention behind it, is a
/// layer of its own.
#[test]
fn a_document_date_without_a_mention() {
    let mut doc = Document::new("undated");
    let date = DocumentDate { date: Some(day("2026-07-14")), mention: None };
    doc.add(&turbo_analysis::Key::<DocumentDate>::DEFAULT, date, turbo_analysis::Provenance::by("test")).unwrap();
    assert_eq!(doc.get::<DocumentDate>().unwrap().date, Some(day("2026-07-14")));
}
