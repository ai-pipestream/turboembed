//! Calendar mentions written in a text view, and the date a document is
//! dated with.
//!
//! The layers hold what OpenNLP's `CursorTemporalExtractor` and
//! `DocumentDateAnnotator` find: `2026-07-14`, `July 2026`, `Q3 2024`, and
//! with a reference date `yesterday` or `3 weeks ago`. A value is kept as
//! the calendar fields OpenNLP computes, not as the ISO 8601 text it
//! writes; [`TemporalValue`] writes that text.

use std::fmt;

use turbo_analysis::{Anchor, Ix, Key, KeyId, Layer, Span, Spans, ViewId};

use crate::date::{Date, write_year};

/// How fine a mention's value is: OpenNLP's `TemporalExpression.Granularity`.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub enum Granularity {
    Day,
    /// An ISO 8601 week, in its week-based year.
    Week,
    Month,
    Quarter,
    Year,
}

impl Granularity {
    /// The Java enum constant's name: `DAY`, `WEEK`, `MONTH`, `QUARTER`
    /// or `YEAR`.
    pub fn name(self) -> &'static str {
        match self {
            Granularity::Day => "DAY",
            Granularity::Week => "WEEK",
            Granularity::Month => "MONTH",
            Granularity::Quarter => "QUARTER",
            Granularity::Year => "YEAR",
        }
    }
}

/// Whether the text states the value or the value was resolved from
/// relative text against a reference date: `TemporalExpression.Origin`.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub enum Origin {
    Absolute,
    Relative,
}

impl Origin {
    /// The Java enum constant's name: `ABSOLUTE` or `RELATIVE`.
    pub fn name(self) -> &'static str {
        match self {
            Origin::Absolute => "ABSOLUTE",
            Origin::Relative => "RELATIVE",
        }
    }
}

/// A mention's value at its granularity.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub enum TemporalValue {
    Day(Date),
    /// `week` (1 to 53) of the ISO 8601 week-based `year`.
    Week {
        year: i32,
        week: u8,
    },
    /// `month` (1 to 12) of `year`.
    Month {
        year: i32,
        month: u8,
    },
    /// `quarter` (1 to 4) of `year`.
    Quarter {
        year: i32,
        quarter: u8,
    },
    Year(i32),
}

impl TemporalValue {
    pub fn granularity(self) -> Granularity {
        match self {
            TemporalValue::Day(_) => Granularity::Day,
            TemporalValue::Week { .. } => Granularity::Week,
            TemporalValue::Month { .. } => Granularity::Month,
            TemporalValue::Quarter { .. } => Granularity::Quarter,
            TemporalValue::Year(_) => Granularity::Year,
        }
    }

    /// The value of `date` at `granularity`: what OpenNLP formats a
    /// resolved date with (`2026-W29` for the week holding 2026-07-17).
    pub fn of_date(date: Date, granularity: Granularity) -> TemporalValue {
        match granularity {
            Granularity::Day => TemporalValue::Day(date),
            Granularity::Week => {
                let (year, week) = date.iso_week();
                TemporalValue::Week { year, week }
            }
            Granularity::Month => TemporalValue::Month { year: date.year(), month: date.month() },
            Granularity::Quarter => TemporalValue::Quarter { year: date.year(), quarter: date.quarter() },
            Granularity::Year => TemporalValue::Year(date.year()),
        }
    }
}

/// Writes the value as OpenNLP's `TemporalExpression.value()` holds it:
/// `2026-07-14`, `2026-W29`, `2026-07`, `2024-Q3` or `2026`, with a year
/// written as `LocalDate.toString` writes one (`-0001`, `+10000`).
impl fmt::Display for TemporalValue {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match *self {
            TemporalValue::Day(date) => date.fmt(f),
            TemporalValue::Week { year, week } => {
                write_year(f, year)?;
                write!(f, "-W{week:02}")
            }
            TemporalValue::Month { year, month } => {
                write_year(f, year)?;
                write!(f, "-{month:02}")
            }
            TemporalValue::Quarter { year, quarter } => {
                write_year(f, year)?;
                write!(f, "-Q{quarter}")
            }
            TemporalValue::Year(year) => write_year(f, year),
        }
    }
}

/// Every temporal mention in a view, one item per mention, in text order
/// and not overlapping.
///
/// The value is stored in three integer columns whose meaning follows the
/// granularity: `year` is the calendar year (the ISO week-based year for a
/// week); `period` is the month of a day or a month, the week of a week,
/// the quarter of a quarter and 0 for a year; `day` is the day of the
/// month of a day and 0 otherwise. [`Temporals::value`] reads them back as
/// a [`TemporalValue`].
pub struct Temporals {
    pub spans: Spans<Temporals>,
    pub granularity: Vec<Granularity>,
    pub origin: Vec<Origin>,
    pub year: Vec<i32>,
    pub period: Vec<u8>,
    pub day: Vec<u8>,
}

/// One mention, as handed to [`Temporals::push`].
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct TemporalMention {
    pub span: Span,
    pub value: TemporalValue,
    pub origin: Origin,
}

impl Temporals {
    /// An empty layer over `view`.
    pub fn new(view: ViewId) -> Temporals {
        Temporals {
            spans: Spans::new(view),
            granularity: Vec::new(),
            origin: Vec::new(),
            year: Vec::new(),
            period: Vec::new(),
            day: Vec::new(),
        }
    }

    /// Appends one mention to every column.
    pub fn push(&mut self, m: TemporalMention) -> Ix<Temporals> {
        let (year, period, day) = match m.value {
            TemporalValue::Day(d) => (d.year(), d.month(), d.day()),
            TemporalValue::Week { year, week } => (year, week, 0),
            TemporalValue::Month { year, month } => (year, month, 0),
            TemporalValue::Quarter { year, quarter } => (year, quarter, 0),
            TemporalValue::Year(year) => (year, 0, 0),
        };
        self.granularity.push(m.value.granularity());
        self.origin.push(m.origin);
        self.year.push(year);
        self.period.push(period);
        self.day.push(day);
        self.spans.push(m.span)
    }

    /// Removes every mention, keeping the view.
    pub fn clear(&mut self) {
        *self = Temporals::new(self.spans.view());
    }

    /// The value of mention `i`.
    ///
    /// # Panics
    ///
    /// When the columns of `i` do not make a value, which
    /// [`Temporals::check`] refuses.
    pub fn value(&self, i: Ix<Temporals>) -> TemporalValue {
        self.try_value(i.get()).unwrap_or_else(|e| panic!("{e}"))
    }

    /// Mention `i`, column by column.
    pub fn get(&self, i: Ix<Temporals>) -> TemporalMention {
        TemporalMention { span: self.spans.get(i), value: self.value(i), origin: self.origin[i.get()] }
    }

    fn try_value(&self, i: usize) -> Result<TemporalValue, String> {
        let (year, period, day) = (self.year[i], self.period[i], self.day[i]);
        let bad = |what: &str| format!("temporal mention {i} has {what}: {year} {period} {day}");
        let blank_day = |v: TemporalValue| if day == 0 { Ok(v) } else { Err(bad("a day outside a day value")) };
        match self.granularity[i] {
            Granularity::Day => Date::new(year, period, day).map(TemporalValue::Day).ok_or_else(|| bad("no such date")),
            Granularity::Week if (1..=53).contains(&period) => blank_day(TemporalValue::Week { year, week: period }),
            Granularity::Month if (1..=12).contains(&period) => blank_day(TemporalValue::Month { year, month: period }),
            Granularity::Quarter if (1..=4).contains(&period) => {
                blank_day(TemporalValue::Quarter { year, quarter: period })
            }
            Granularity::Year if period == 0 => blank_day(TemporalValue::Year(year)),
            _ => Err(bad("a period outside its granularity's range")),
        }
    }
}

impl Layer for Temporals {
    const ID: &'static str = "turbo:temporal";
    positional!(Temporals);

    fn check(&self) -> Result<(), String> {
        let n = self.spans.len();
        let columns = [self.granularity.len(), self.origin.len(), self.year.len(), self.period.len(), self.day.len()];
        if columns.iter().any(|&l| l != n) {
            return Err("every column as long as the spans".into());
        }
        self.spans.check(Self::OVERLAP).map_err(|e| {
            if e == "spans in text order, not overlapping" {
                "temporal mentions in text order, not overlapping".into()
            } else {
                e
            }
        })?;
        for (i, span) in self.spans.iter() {
            if span.is_empty() {
                return Err(format!("temporal mention {} is empty at {}", i.get(), span.start));
            }
            self.try_value(i.get())?;
        }
        Ok(())
    }
}

/// The date a document is dated with: OpenNLP's `DocumentDateAnnotator`
/// layer, which holds at most one date. A document-scoped layer of zero
/// or one item.
///
/// `mention` is the temporal mention the date was read from, when there
/// is one: an index into the default instance of [`Temporals`] (or the
/// instance the step that wrote this layer read). A date known from
/// metadata has none.
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq)]
pub struct DocumentDate {
    pub date: Option<Date>,
    pub mention: Option<Ix<Temporals>>,
}

impl Layer for DocumentDate {
    const ID: &'static str = "turbo:document-date";

    fn anchor() -> Anchor {
        Anchor::Document
    }

    fn len(&self) -> usize {
        usize::from(self.date.is_some())
    }

    fn indexes(&self) -> Vec<(KeyId, usize)> {
        self.mention.map(|m| (Key::<Temporals>::DEFAULT.id(), m.get() + 1)).into_iter().collect()
    }

    fn check(&self) -> Result<(), String> {
        if self.date.is_none() && self.mention.is_some() {
            return Err("a mention without a date".into());
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn day(s: &str) -> Date {
        Date::parse(s).unwrap()
    }

    fn mention(start: u32, end: u32, value: TemporalValue) -> TemporalMention {
        TemporalMention { span: Span::new(start, end), value, origin: Origin::Absolute }
    }

    #[test]
    fn values_are_written_as_opennlp_writes_them() {
        let cases = [
            (TemporalValue::Day(day("2026-07-14")), "2026-07-14"),
            (TemporalValue::Week { year: 2026, week: 9 }, "2026-W09"),
            (TemporalValue::Month { year: 2026, month: 7 }, "2026-07"),
            (TemporalValue::Quarter { year: 2024, quarter: 3 }, "2024-Q3"),
            (TemporalValue::Year(2026), "2026"),
            (TemporalValue::Year(-1), "-0001"),
            (TemporalValue::Year(0), "0000"),
            (TemporalValue::Quarter { year: 10000, quarter: 1 }, "+10000-Q1"),
            (TemporalValue::Week { year: 1_000_000_000, week: 1 }, "+1000000000-W01"),
            (TemporalValue::Month { year: -999_999_999, month: 1 }, "-999999999-01"),
        ];
        for (v, s) in cases {
            assert_eq!(v.to_string(), s);
        }
    }

    #[test]
    fn a_date_reads_at_each_granularity() {
        let d = day("2021-01-01");
        assert_eq!(TemporalValue::of_date(d, Granularity::Week).to_string(), "2020-W53");
        assert_eq!(TemporalValue::of_date(d, Granularity::Quarter).to_string(), "2021-Q1");
        assert_eq!(TemporalValue::of_date(day("2026-12-31"), Granularity::Quarter).to_string(), "2026-Q4");
        assert_eq!(TemporalValue::of_date(d, Granularity::Year).granularity(), Granularity::Year);
    }

    #[test]
    fn columns_round_trip_every_granularity() {
        let values = [
            TemporalValue::Day(day("2024-02-29")),
            TemporalValue::Week { year: 2020, week: 53 },
            TemporalValue::Month { year: 2026, month: 12 },
            TemporalValue::Quarter { year: 2026, quarter: 4 },
            TemporalValue::Year(-5),
        ];
        let mut t = Temporals::new(ViewId::SOURCE);
        for (k, v) in values.into_iter().enumerate() {
            let at = 10 * k as u32;
            t.push(TemporalMention { origin: Origin::Relative, ..mention(at, at + 5, v) });
        }
        assert_eq!(t.check(), Ok(()));
        for (k, v) in values.into_iter().enumerate() {
            assert_eq!(t.value(Ix::new(k as u32)), v);
            assert_eq!(t.get(Ix::new(k as u32)).origin, Origin::Relative);
        }
        t.clear();
        assert_eq!((t.len(), t.spans.view()), (0, ViewId::SOURCE));
    }

    #[test]
    fn check_refuses_values_that_are_no_value() {
        let mut t = Temporals::new(ViewId::SOURCE);
        t.push(mention(0, 4, TemporalValue::Day(day("2026-02-28"))));
        t.day[0] = 30;
        assert!(t.check().unwrap_err().contains("no such date"));
        for (g, period, day) in [
            (Granularity::Week, 54, 0),
            (Granularity::Month, 0, 0),
            (Granularity::Quarter, 5, 0),
            (Granularity::Year, 1, 0),
            (Granularity::Month, 7, 1),
        ] {
            let mut t = Temporals::new(ViewId::SOURCE);
            t.push(mention(0, 4, TemporalValue::Year(2026)));
            (t.granularity[0], t.period[0], t.day[0]) = (g, period, day);
            assert!(t.check().is_err(), "{g:?} {period} {day}");
        }
    }

    #[test]
    fn check_wants_text_order_and_full_columns() {
        let mut t = Temporals::new(ViewId::SOURCE);
        t.push(mention(5, 9, TemporalValue::Year(2026)));
        t.push(mention(0, 4, TemporalValue::Year(2026)));
        assert!(t.check().unwrap_err().contains("text order"));
        let mut t = Temporals::new(ViewId::SOURCE);
        t.push(mention(0, 4, TemporalValue::Year(2026)));
        t.origin.pop();
        assert_eq!(t.check().unwrap_err(), "every column as long as the spans");
        let mut t = Temporals::new(ViewId::SOURCE);
        t.push(mention(3, 3, TemporalValue::Year(2026)));
        assert!(t.check().unwrap_err().contains("empty"));
        let mut bad_frags = Temporals::new(ViewId::SOURCE);
        bad_frags.push(mention(0, 10, TemporalValue::Year(2026)));
        let mut frag_spans = Spans::new(ViewId::SOURCE);
        frag_spans.push_fragments(&[Span::new(0, 4), Span::new(6, 10)]);
        bad_frags.spans = frag_spans;
        assert!(bad_frags.check().unwrap_err().contains("discontinuous"));
    }

    #[test]
    fn a_document_date_has_at_most_one_item_and_points_into_the_mentions() {
        let none = DocumentDate::default();
        assert_eq!((none.len(), none.check(), none.indexes().len()), (0, Ok(()), 0));
        let dated = DocumentDate { date: Some(day("2026-07-14")), mention: Some(Ix::new(2)) };
        assert_eq!(dated.len(), 1);
        let indexes = dated.indexes();
        assert_eq!((indexes[0].0.layer.id, indexes[0].1), ("turbo:temporal", 3));
        let metadata = DocumentDate { date: Some(day("2026-07-14")), mention: None };
        assert_eq!((metadata.len(), metadata.check()), (1, Ok(())));
        let orphan = DocumentDate { date: None, mention: Some(Ix::new(0)) };
        assert!(orphan.check().is_err());
        assert_eq!(DocumentDate::anchor(), Anchor::Document);
    }

    #[test]
    fn the_layers_are_named() {
        assert_eq!(Temporals::ID, "turbo:temporal");
        assert_eq!(DocumentDate::ID, "turbo:document-date");
        assert_eq!(Granularity::Quarter.name(), "QUARTER");
        assert_eq!(Origin::Relative.name(), "RELATIVE");
    }
}
