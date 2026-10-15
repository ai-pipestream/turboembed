//! Picks a document's date from its temporal mentions: a port of
//! OpenNLP's `DocumentDateAnnotator`.

use turbo_analysis::{Annotator, Error, Ix, Refs};
use turbo_analysis_layers::{DocumentDate, TemporalValue, Temporals};

use super::is_absolute_day;

/// Reads `turbo:temporal` and writes `turbo:document-date`: the absolute
/// day mention that starts first, or no date when there is none. Relative
/// mentions and coarser ones (a month, a quarter) never date a document.
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
}

impl Annotator for DocumentDateSelector {
    const ID: &'static str = "turbo-numeric:document-date";
    type Reads = (Temporals,);
    type Writes = (DocumentDate,);

    fn annotate(&self, (temporals,): Refs<'_, Self>) -> Result<(DocumentDate,), Error> {
        Ok((self.select(temporals),))
    }
}
