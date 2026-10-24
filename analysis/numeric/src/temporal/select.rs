//! Picks a document's date from its temporal mentions: a port of
//! OpenNLP's `DocumentDateAnnotator`, plus [`TimesDocumentDateSelector`]
//! ([`DocumentDateSelector::from_times`]) for grammar [`Times`] pipelines.

use turbo_analysis::{Annotator, Error, Ix, Refs};
use turbo_analysis_layers::{DateTime, DocumentDate, TemporalValue, Temporals, TimeKind, Times};

use super::is_absolute_day;

/// Reads `turbo:temporal` and writes `turbo:document-date`: the earliest
/// absolute day mention, or no date when there is none.
///
/// The Java checks that each mention's own span is its annotation's and
/// that the chosen value parses as a date; a [`Temporals`] layer holds one
/// span and a typed date per mention, so neither can fail here.
///
/// For non-English or grammar-based pipelines that produce [`Times`]
/// (`turbo:times`) via `TimeAnnotator`, use
/// [`DocumentDateSelector::from_times`] ([`TimesDocumentDateSelector`]).
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq)]
pub struct DocumentDateSelector;

/// Reads `turbo:times` (with optional priority to `turbo:temporal` when
/// present) and writes `turbo:document-date`: the earliest absolute day
/// from [`Temporals`], or else the earliest resolved [`TimeKind::Date`] /
/// [`TimeKind::DateTime`] mention from [`Times`].
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq)]
pub struct TimesDocumentDateSelector;

impl DocumentDateSelector {
    /// Returns a [`TimesDocumentDateSelector`] that dates the document from
    /// the grammar [`Times`] layer (preferring [`Temporals`] if also present).
    pub const fn from_times() -> TimesDocumentDateSelector {
        TimesDocumentDateSelector
    }

    /// The document date `temporals` gives: the absolute day with the
    /// earliest start, the first in layer order among equal starts.
    pub fn select(&self, temporals: &Temporals) -> DocumentDate {
        let mut selected: Option<Ix<Temporals>> = None;
        let (start, _) = temporals.spans.columns();
        for (i, (&g, &o)) in temporals.granularity.iter().zip(&temporals.origin).enumerate() {
            if is_absolute_day(g, o) && selected.is_none_or(|s| start[i] < start[s.get()]) {
                selected = Some(Ix::new(i as u32));
            }
        }
        let Some(mention) = selected else {
            return DocumentDate::default();
        };
        let TemporalValue::Day(date) = temporals.value(mention) else {
            unreachable!("a mention of day granularity holds a day");
        };
        DocumentDate { date: Some(date), mention: Some(mention) }
    }

    /// Selects the document date from the grammar [`Times`] layer: the
    /// earliest resolved [`TimeKind::Date`] or [`TimeKind::DateTime`]
    /// mention.
    pub fn select_from_times(&self, times: &Times) -> DocumentDate {
        let (start_col, _) = times.spans.columns();
        let mut best: Option<(u32, i64)> = None;
        for (i, (&kind, &resolved_start)) in times.kind.iter().zip(&times.start).enumerate() {
            if matches!(kind, TimeKind::Date | TimeKind::DateTime)
                && let Some(seconds) = resolved_start
                && best.is_none_or(|(best_span_start, _)| start_col[i] < best_span_start)
            {
                best = Some((start_col[i], seconds));
            }
        }
        let Some((_, seconds)) = best else {
            return DocumentDate::default();
        };
        let date = DateTime::from_local_seconds(seconds).map(DateTime::date);
        DocumentDate { date, mention: None }
    }
}

impl Annotator for DocumentDateSelector {
    const ID: &'static str = "turbo-numeric:document-date";
    type Reads = (Temporals,);
    type Writes = (DocumentDate,);

    fn annotate(&self, (temporals,): Refs<'_, Self>) -> Result<(DocumentDate,), Error> {
        Ok((self.select(temporals),))
    }
}

impl Annotator for TimesDocumentDateSelector {
    const ID: &'static str = "turbo-numeric:document-date";
    type Reads = (Option<Temporals>, Times);
    type Writes = (DocumentDate,);

    fn annotate(&self, (temporals, times): Refs<'_, Self>) -> Result<(DocumentDate,), Error> {
        if let Some(t) = temporals {
            let doc_date = DocumentDateSelector.select(t);
            if doc_date.date.is_some() {
                return Ok((doc_date,));
            }
        }
        Ok((DocumentDateSelector.select_from_times(times),))
    }
}
