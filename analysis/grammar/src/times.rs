//! Writes the time mentions a grammar finds as a [`Times`] layer, each
//! resolved against a reference date-time.
//!
//! The reference is the document's date at midnight when the pipeline has
//! a [`DocumentDate`], else the one the annotator was built with. There is
//! no clock: with neither, a mention whose value depends on a reference
//! (`tomorrow`, `next week`) is written with no start, no end and an empty
//! timex, its kind and grain still set; one that does not (`2016-11-07`,
//! `at 5`, `3 hours`) resolves as usual.

use std::sync::Mutex;

use turbo_analysis::{Annotator, Error, Refs, Span, View, ViewId};
use turbo_analysis_layers::{Date, DateTime, DocumentDate, TimeMention, Times};

use crate::engine::{Grammar, Output, Parser};
use crate::lang::en::ENGLISH_TIMES;
use crate::time::{Resolution, Resolved, Timex, depends_on_reference, resolve};

/// Resolves values that name no reference, and gives open mentions a
/// mid-range anchor with room in both directions for their kind and grain.
fn neutral() -> DateTime {
    DateTime::midnight(Date::from_epoch_day(0).unwrap())
}

/// Runs a time grammar over a text view. Parsers are pooled, so their
/// buffers are reused between documents.
pub struct TimeAnnotator {
    grammar: &'static dyn Grammar,
    reference: Option<DateTime>,
    pool: Mutex<Vec<Parser>>,
}

impl TimeAnnotator {
    pub fn new(grammar: &'static dyn Grammar) -> TimeAnnotator {
        TimeAnnotator { grammar, reference: None, pool: Mutex::new(Vec::new()) }
    }

    pub fn english() -> TimeAnnotator {
        TimeAnnotator::new(&ENGLISH_TIMES)
    }

    pub fn spanish() -> TimeAnnotator {
        TimeAnnotator::new(&crate::lang::es::SPANISH_TIMES)
    }

    /// The reference to resolve against when the pipeline has no document
    /// date.
    pub fn with_reference(mut self, reference: DateTime) -> TimeAnnotator {
        self.reference = Some(reference);
        self
    }

    /// Finds the time mentions in `text`, with spans in view `view`,
    /// resolved against `reference` when there is one.
    ///
    /// # Panics
    ///
    /// When `text` is longer than `u32::MAX` bytes.
    pub fn find(&self, text: &str, view: ViewId, reference: Option<DateTime>) -> Times {
        let mut parser = self.pool.lock().ok().and_then(|mut p| p.pop()).unwrap_or_else(|| Parser::new(self.grammar));
        parser.parse(text);
        let mut mentions = Vec::new();
        parser.select(text, Output::Time, &mut mentions);
        let mut times = Times::new(view);
        let mut timex = String::new();
        let now = reference.unwrap_or_else(neutral);
        for m in &mentions {
            let Some(value) = parser.times().get(&m.value) else { continue };
            let open = reference.is_none() && depends_on_reference(&value);
            let Some(r) = resolve(&value, now) else { continue };
            push(&mut times, Span::new(m.start, m.end), &r, open, &mut timex);
        }
        if let Ok(mut pool) = self.pool.lock() {
            pool.push(parser);
        }
        times
    }
}

/// Writes a mention's preferred reading; with `open`, only what it is.
fn push(times: &mut Times, span: Span, r: &Resolved, open: bool, timex: &mut String) {
    use std::fmt::Write as _;
    let Some(alt) = r.alternatives.get(r.preferred) else { return };
    timex.clear();
    let (mut start, mut end, mut duration_ms) = (None, None, None);
    if !open {
        let _ = write!(timex, "{}", alt.timex);
        match alt.value {
            Resolution::Point(i) => start = Some(i.seconds()),
            Resolution::Range { start: s, end: e } => {
                start = s.map(|i| i.seconds());
                end = e.map(|i| i.seconds());
            }
            Resolution::Duration { millis } => duration_ms = Some(millis),
            Resolution::NotResolved => {}
        }
        if let Timex::Range(_, _, d) = alt.timex {
            duration_ms = Some(d.millis());
        }
    }
    times.push(TimeMention {
        span,
        kind: r.kind,
        grain: r.grain,
        modifier: r.modifier,
        start,
        end,
        duration_ms,
        ambiguous: r.alternatives.len() > 1,
        timex,
    });
}

impl Annotator for TimeAnnotator {
    const VERSION: &'static str = "2";
    const ID: &'static str = "turbo-grammar:times";
    type Reads = (View, Option<DocumentDate>);
    type Writes = (Times,);

    fn annotate(&self, (view, document): Refs<'_, Self>) -> Result<(Times,), Error> {
        let text = view.text();
        if u32::try_from(text.len()).is_err() {
            return Err(Error::Annotator {
                annotator: Self::ID,
                detail: format!("a text of {} bytes is past the u32 offset range", text.len()),
            });
        }
        let reference = document.and_then(|d| d.date).map(DateTime::midnight).or(self.reference);
        Ok((self.find(text, view.id(), reference),))
    }

    /// FNV-1a over the grammar's name and the fixed reference.
    fn config_hash(&self) -> u64 {
        let fixed = self.reference.map_or(i64::MIN, DateTime::local_seconds);
        self.grammar
            .name()
            .bytes()
            .chain(fixed.to_le_bytes())
            .fold(0xcbf2_9ce4_8422_2325, |h, b| (h ^ u64::from(b)).wrapping_mul(0x0100_0000_01b3))
    }
}

#[cfg(test)]
mod tests {
    use turbo_analysis::Layer;
    use turbo_analysis_layers::{TimeGrain, TimeKind};

    use super::*;

    fn reference() -> DateTime {
        DateTime::parse("2016-11-07T16:12:00").unwrap()
    }

    #[test]
    fn mentions_resolve_against_the_reference() {
        let a = TimeAnnotator::english();
        let text = "I'll be back tomorrow at 8am, for 3 hours, every Monday.";
        let t = a.find(text, ViewId::SOURCE, Some(reference()));
        assert_eq!(t.check(), Ok(()));
        let got: Vec<_> = t.spans.iter().map(|(i, s)| (&text[s.start as usize..s.end as usize], t.get(i))).collect();
        assert_eq!(got.len(), 3, "{got:?}");
        let (s, m) = got[0];
        assert_eq!((s, m.kind, m.timex), ("tomorrow at 8am", TimeKind::DateTime, "2016-11-08T08"));
        assert_eq!(m.start, Some(DateTime::parse("2016-11-08T08:00").unwrap().local_seconds()));
        assert!(!m.ambiguous);
        let (s, m) = got[1];
        assert_eq!((s, m.kind, m.timex, m.duration_ms), ("3 hours", TimeKind::Duration, "PT3H", Some(10_800_000)));
        let (s, m) = got[2];
        assert_eq!((s, m.kind, m.timex, m.start), ("every Monday", TimeKind::Set, "XXXX-WXX-1", None));
    }

    #[test]
    fn an_open_reading_is_ambiguous_and_takes_the_next_one() {
        let t = TimeAnnotator::english().find("See you Friday", ViewId::SOURCE, Some(reference()));
        let m = t.get(t.spans.iter().next().unwrap().0);
        assert!(m.ambiguous);
        assert_eq!(m.start, Some(DateTime::parse("2016-11-11T00:00").unwrap().local_seconds()));

        let t = TimeAnnotator::english().find(
            "from 5 for 1 hour",
            ViewId::SOURCE,
            Some(DateTime::parse("2018-05-28T12:00:00").unwrap()),
        );
        assert_eq!(t.check(), Ok(()));
        assert_eq!(t.len(), 1);
        let m = t.get(t.spans.iter().next().unwrap().0);
        assert_eq!(m.kind, TimeKind::TimeRange);
        assert!(m.ambiguous);
        assert_eq!(
            (m.timex, m.start, m.end, m.duration_ms),
            ("(T05,T06:00:00,PT1H)", Some(18_000), Some(21_600), Some(3_600_000))
        );

        let late = TimeAnnotator::english().find("from 11pm for 2 hours", ViewId::SOURCE, Some(reference()));
        assert_eq!(late.check(), Ok(()));
        let m = late.get(late.spans.iter().next().unwrap().0);
        assert_eq!(
            (m.kind, m.start, m.end, m.duration_ms),
            (TimeKind::TimeRange, Some(82_800), Some(3_600), Some(7_200_000))
        );
    }

    #[test]
    fn without_a_reference_relative_mentions_stay_unresolved() {
        let text = "tomorrow, yesterday, last week, Monday or May 29, not on 2016-11-07";
        let t = TimeAnnotator::english().find(text, ViewId::SOURCE, None);
        assert_eq!(t.check(), Ok(()));
        let got: Vec<_> = t.spans.iter().map(|(i, s)| (&text[s.start as usize..s.end as usize], t.get(i))).collect();
        assert_eq!(got.len(), 6, "{got:?}");
        for &(s, m) in &got[..5] {
            assert_eq!((s, m.timex, m.start, m.end), (s, "", None, None));
        }
        assert_eq!((got[0].1.kind, got[0].1.grain), (TimeKind::Date, TimeGrain::Day));
        assert_eq!((got[1].1.kind, got[1].1.grain), (TimeKind::Date, TimeGrain::Day));
        assert_eq!((got[2].1.kind, got[2].1.grain), (TimeKind::DateRange, TimeGrain::Week));
        assert_eq!((got[3].1.kind, got[3].1.grain), (TimeKind::Date, TimeGrain::Day));
        assert_eq!((got[4].1.kind, got[4].1.grain), (TimeKind::Date, TimeGrain::Day));
        let fixed = got[5].1;
        assert_eq!(fixed.timex, "2016-11-07");
        assert!(fixed.start.is_some());
    }

    #[test]
    fn the_fixed_reference_is_used_and_hashed() {
        let a = TimeAnnotator::english().with_reference(reference());
        let b = TimeAnnotator::english();
        assert_ne!(a.config_hash(), b.config_hash());
        assert_eq!(a.config_hash(), TimeAnnotator::english().with_reference(reference()).config_hash());
    }

    #[test]
    fn spanish_mentions_resolve() {
        let text = "Nos vemos mañana a las 8 de la tarde.";
        let t = TimeAnnotator::spanish().find(text, ViewId::SOURCE, Some(reference()));
        assert_eq!(t.check(), Ok(()));
        let got: Vec<_> =
            t.spans.iter().map(|(i, s)| (&text[s.start as usize..s.end as usize], t.get(i).timex)).collect();
        assert_eq!(got, [("mañana a las 8 de la tarde", "2016-11-08T20")]);
    }

    #[test]
    fn a_second_run_gives_the_same_layer() {
        let a = TimeAnnotator::english();
        let text = "from 5 to 6pm of April 22, then next week and the last 3 days";
        let first = a.find(text, ViewId::SOURCE, Some(reference()));
        let second = a.find(text, ViewId::SOURCE, Some(reference()));
        assert_eq!(first.len(), second.len());
        for (i, _) in first.spans.iter() {
            assert_eq!(first.get(i), second.get(i));
        }
    }
}
