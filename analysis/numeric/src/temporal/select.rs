//! Picks a document's date from its temporal mentions: a port of
//! OpenNLP's `DocumentDateAnnotator`, with fallback to the grammar
//! [`Times`] layer when [`Temporals`] is absent or holds no absolute day.

use turbo_analysis::{Annotator, Error, Ix, Key, Refs};
use turbo_analysis_layers::{DateTime, DocumentDate, TemporalValue, Temporals, TimeKind, Times};

use super::is_absolute_day;

/// Reads `turbo:temporal` (when present) and falls back to `turbo:times`
/// (when present), writing `turbo:document-date`: the earliest absolute day
/// mention from `Temporals`, or else the earliest resolved `Date`/`DateTime`
/// mention from `Times`, or no date when neither yields one.
///
/// The Java checks that each mention's own span is its annotation's and
/// that the chosen value parses as a date; a [`Temporals`] layer holds one
/// span and a typed date per mention, so neither can fail here.
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq)]
pub struct DocumentDateSelector;

impl DocumentDateSelector {
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
    type Reads = (Option<Temporals>, Option<Times>);
    type Writes = (DocumentDate,);

    fn annotate(&self, (temporals, times): Refs<'_, Self>) -> Result<(DocumentDate,), Error> {
        if temporals.is_none() && times.is_none() {
            return Err(Error::Missing(Key::<Temporals>::DEFAULT.id()));
        }
        if let Some(t) = temporals {
            let doc_date = self.select(t);
            if doc_date.date.is_some() {
                return Ok((doc_date,));
            }
        }
        if let Some(times) = times {
            return Ok((self.select_from_times(times),));
        }
        Ok((DocumentDate::default(),))
    }
}
