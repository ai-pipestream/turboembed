//! Reducing calendar dates ([`DateRewrite`]): truncate to the year, shift
//! keeping the written form, or write relative to a reference date.
//!
//! A `date` mention (`2026-07-14`, `July 14, 2026`, `14th July 2026`,
//! `July 2026`) is read again with the temporal scanner of turbo-numeric,
//! so the value is the one the detector saw:
//!
//! - [`DateMode::Year`] writes the year alone: `July 14, 2026` becomes
//!   `2026`, which HIPAA Safe Harbor (45 CFR 164.514(b)(2)(i)(C)) lets a
//!   record keep.
//! - [`DateMode::Shift`] moves the date by fixed years, months and days,
//!   plus an offset drawn from a key and a context string, and writes it
//!   in the form it was written in (`July 14, 2026` stays in month-name
//!   form). Months and years keep the day where it exists and take the
//!   month's last day where it does not, so January 31 plus one month is
//!   the end of February and a leap day plus a year is February 28. A
//!   month mention moves as its first day does, and at least one month
//!   the days' way when they move it less than that. A shift that would
//!   write a value back as it was masks it whole.
//! - [`DateMode::Relative`] writes the date relative to a reference date
//!   the caller passes (`3 months ago`), never the clock, so the same
//!   input gives the same output.
//! - [`DateMode::SafeHarbor`] writes the year alone, as [`DateMode::Year`]
//!   does, except that a year that could make someone born in it older
//!   than 89 on a reference date the caller passes (90 or more years
//!   before the reference year) is written as one category for all of
//!   them, `1936 or earlier` for a 2026 reference. Safe Harbor (C) counts
//!   such a year as an element of a date indicative of an age over 89 and
//!   lets it be aggregated into one category of age 90 or older.
//!
//! A mention of another kind than `date`, a text that is not one whole
//! absolute day or month mention (a caller's `7/14/26`), a date shifted
//! out of the years `0001` to `9999`, and a cluster of overlapping
//! mentions are masked whole, so nothing of the value stays readable.

mod form;
mod relative;

use std::fmt;

use turbo_analysis::View;
use turbo_analysis_layers::{Date, Pii, PiiKind, PiiStrategyKind};

use super::replace::EditOrigin;
use super::{PiiRewrite, RewriteError};
use form::Stated;

/// A shift the strategy cannot take.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum DateShiftError {
    /// Fewer than 32 key bytes.
    ShortKey(usize),
    /// A keyed offset of no days, or of more than
    /// [`DateShift::MAX_KEYED_DAYS`].
    MaxDays(u32),
}

impl fmt::Display for DateShiftError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            DateShiftError::ShortKey(n) => write!(f, "the key has {n} bytes, at least 32 are needed"),
            DateShiftError::MaxDays(n) => {
                write!(f, "max_days must be between 1 and {}: {n}", DateShift::MAX_KEYED_DAYS)
            }
        }
    }
}

impl std::error::Error for DateShiftError {}

/// The offset a key and a context string choose.
#[derive(Clone, PartialEq, Eq, Hash)]
struct Keyed {
    key: Vec<u8>,
    context: String,
    max_days: u32,
}

/// How far [`DateMode::Shift`] moves a date: years and months first, as one
/// count of months, then days, then the keyed offset in days.
#[derive(Clone, PartialEq, Eq, Hash, Default)]
pub struct DateShift {
    years: i32,
    months: i32,
    days: i32,
    keyed: Option<Keyed>,
}

impl fmt::Debug for DateShift {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        // The key and the context stay out of logs.
        f.debug_struct("DateShift")
            .field("years", &self.years)
            .field("months", &self.months)
            .field("days", &self.days)
            .field("keyed", &self.keyed.is_some())
            .finish()
    }
}

impl DateShift {
    pub const MIN_KEY_BYTES: usize = 32;
    /// The widest keyed offset, ten years of days.
    pub const MAX_KEYED_DAYS: u32 = 3650;

    /// A fixed shift by `years`, `months` and `days`, each negative for
    /// earlier.
    pub fn fixed(years: i32, months: i32, days: i32) -> DateShift {
        DateShift { years, months, days, keyed: None }
    }

    /// Adds an offset of 1 to `max_days` days, earlier or later, that
    /// HMAC-SHA-256 under `key` chooses from `context`. The same key and
    /// context give the same offset in every text, so every date of one
    /// document (its id as the context) or of one patient moves by the
    /// same number of days and the intervals between them stay; dates
    /// under another context move by another. Never zero, so no date
    /// stays as written: a shift whose net effect leaves a value as it was
    /// (a fixed shift cancelling the offset) masks it whole.
    pub fn keyed(mut self, key: &[u8], context: &str, max_days: u32) -> Result<DateShift, DateShiftError> {
        if key.len() < DateShift::MIN_KEY_BYTES {
            return Err(DateShiftError::ShortKey(key.len()));
        }
        if !(1..=DateShift::MAX_KEYED_DAYS).contains(&max_days) {
            return Err(DateShiftError::MaxDays(max_days));
        }
        self.keyed = Some(Keyed { key: key.to_vec(), context: context.to_owned(), max_days });
        Ok(self)
    }

    pub fn years(&self) -> i32 {
        self.years
    }

    pub fn months(&self) -> i32 {
        self.months
    }

    pub fn days(&self) -> i32 {
        self.days
    }

    /// Whether a keyed offset is added.
    pub fn is_keyed(&self) -> bool {
        self.keyed.is_some()
    }

    /// The keyed offset in days, or 0 without a key.
    pub fn keyed_days(&self) -> i64 {
        let Some(k) = &self.keyed else { return 0 };
        let mut msg = b"turbo-pii:date-shift\0".to_vec();
        msg.extend_from_slice(k.context.as_bytes());
        let mac = turbo_sha256::hmac_sha256(&k.key, &msg).0;
        let word = u64::from_be_bytes(mac[..8].try_into().expect("eight bytes"));
        let magnitude = (word >> 1) % u64::from(k.max_days) + 1;
        let magnitude = magnitude as i64;
        if word & 1 == 0 { magnitude } else { -magnitude }
    }

    /// `was` moved, or `None` when it leaves the `Date` range. A month
    /// moves as its first day does; when that lands in the same month and
    /// the days moved are not zero, it moves one month their way, so a
    /// keyed offset under a month still changes the month.
    fn moved(&self, was: Stated) -> Option<Stated> {
        match was {
            Stated::Day(d) => self.apply(d).map(Stated::Day),
            Stated::Month { year, month } => {
                let first = Date::new(year, month, 1)?;
                let mut d = self.apply(first)?;
                let months = i64::from(self.years) * 12 + i64::from(self.months);
                let days = i64::from(self.days) + self.keyed_days();
                if (d.year(), d.month()) == (year, month) && days != 0 {
                    d = first.plus_months(months + days.signum())?;
                }
                Some(Stated::Month { year: d.year(), month: d.month() })
            }
        }
    }

    /// `date` moved, or `None` when it leaves the `Date` range.
    pub fn apply(&self, date: Date) -> Option<Date> {
        let months = i64::from(self.years) * 12 + i64::from(self.months);
        date.plus_months(months)?.plus_days(i64::from(self.days))?.plus_days(self.keyed_days())
    }

    fn fold_config(&self, h: &mut turbo_analysis::Fnv1a) {
        h.u32(self.years as u32).u32(self.months as u32).u32(self.days as u32);
        match &self.keyed {
            None => {
                h.u8(0);
            }
            Some(k) => {
                // The key enters only as an HMAC under a fixed label.
                h.u8(1).bytes(&turbo_sha256::hmac_sha256(&k.key, b"turbo-pii:redacted config key").0);
                h.bytes(k.context.as_bytes()).u32(k.max_days);
            }
        }
    }
}

/// What [`DateRewrite`] writes for a date.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum DateMode {
    /// The year alone.
    Year,
    /// The date moved, in its written form.
    Shift(DateShift),
    /// The date relative to the reference date.
    Relative(Date),
    /// The year alone, or [`safe_harbor_category`] for a year 90 or more
    /// years before the reference date's.
    SafeHarbor(Date),
}

/// The latest year [`DateMode::SafeHarbor`] writes as one category under
/// `reference`: 90 years before the reference year. Someone born in that
/// year can be 90 on the reference date, and someone born later is 89 at
/// most.
pub fn safe_harbor_cutoff(reference: Date) -> i32 {
    reference.year() - 90
}

/// The text [`DateMode::SafeHarbor`] writes for every year up to
/// [`safe_harbor_cutoff`]: `1936 or earlier` for a 2026 reference.
pub fn safe_harbor_category(reference: Date) -> String {
    format!("{:04} or earlier", safe_harbor_cutoff(reference))
}

/// Truncates, shifts or relates every `date` mention ([`DateMode`]).
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct DateRewrite {
    mode: DateMode,
    mask: char,
    refuse_overlaps: bool,
}

impl DateRewrite {
    /// The default mask character (`'*'`).
    pub const DEFAULT_MASK: char = '*';

    pub fn new(mode: DateMode) -> DateRewrite {
        DateRewrite { mode, mask: DateRewrite::DEFAULT_MASK, refuse_overlaps: false }
    }

    /// [`DateMode::Year`].
    pub fn year() -> DateRewrite {
        DateRewrite::new(DateMode::Year)
    }

    /// [`DateMode::Shift`].
    pub fn shift(shift: DateShift) -> DateRewrite {
        DateRewrite::new(DateMode::Shift(shift))
    }

    /// [`DateMode::Relative`] to `reference`.
    pub fn relative(reference: Date) -> DateRewrite {
        DateRewrite::new(DateMode::Relative(reference))
    }

    /// [`DateMode::SafeHarbor`] under `reference`.
    pub fn safe_harbor(reference: Date) -> DateRewrite {
        DateRewrite::new(DateMode::SafeHarbor(reference))
    }

    /// Sets the character written for each character of what is masked.
    pub fn with_mask(mut self, mask: char) -> DateRewrite {
        self.mask = mask;
        self
    }

    /// This strategy, refusing overlapping mentions with
    /// [`RewriteError::Overlap`].
    pub fn refusing_overlaps(mut self) -> DateRewrite {
        self.refuse_overlaps = true;
        self
    }

    pub fn mode(&self) -> &DateMode {
        &self.mode
    }

    pub fn mask(&self) -> char {
        self.mask
    }

    pub fn refuses_overlaps(&self) -> bool {
        self.refuse_overlaps
    }

    /// The kind its edits record: one per mode.
    pub const fn kind(&self) -> PiiStrategyKind {
        match self.mode {
            DateMode::Year => PiiStrategyKind::DateYear,
            DateMode::Shift(_) => PiiStrategyKind::DateShift,
            DateMode::Relative(_) => PiiStrategyKind::DateRelative,
            DateMode::SafeHarbor(_) => PiiStrategyKind::DateSafeHarbor,
        }
    }

    pub(super) fn fold_config(&self, h: &mut turbo_analysis::Fnv1a) {
        match &self.mode {
            DateMode::Year => {
                h.u8(1);
            }
            DateMode::Shift(s) => {
                h.u8(2);
                s.fold_config(h);
            }
            DateMode::Relative(r) => {
                h.u8(3).u32(r.year() as u32).u8(r.month()).u8(r.day());
            }
            DateMode::SafeHarbor(r) => {
                h.u8(4).u32(r.year() as u32).u8(r.month()).u8(r.day());
            }
        }
        h.u32(u32::from(self.mask)).u8(u8::from(self.refuse_overlaps));
    }

    /// Appends `value`, the text of a `date` mention, reduced to `out`, or
    /// masked whole when it is not one whole day or month mention or a
    /// shift leaves the four-digit years.
    pub fn write_date(&self, value: &str, out: &mut String) {
        let Some(was) = form::read(value) else { return self.write_masked(value, out) };
        let written = match &self.mode {
            DateMode::Year => {
                out.push_str(&form::year_digits(was));
                true
            }
            DateMode::Shift(s) => {
                // A shift that would write the value back as it was masks
                // it instead, so no date under a shift stays readable.
                s.moved(was).is_some_and(|m| m != was && form::write_like(value, was, m, out))
            }
            DateMode::Relative(r) => {
                relative::write_relative(was, *r, out);
                true
            }
            DateMode::SafeHarbor(r) => {
                if was.year() <= safe_harbor_cutoff(*r) {
                    out.push_str(&safe_harbor_category(*r));
                } else {
                    out.push_str(&form::year_digits(was));
                }
                true
            }
        };
        if !written {
            self.write_masked(value, out);
        }
    }

    /// Appends a mention of `kind` reading `value` to `out`: reduced when
    /// `kind` is [`PiiKind::Date`], masked whole otherwise.
    pub fn write_mention(&self, kind: PiiKind, value: &str, out: &mut String) {
        if kind == PiiKind::Date {
            self.write_date(value, out);
        } else {
            self.write_masked(value, out);
        }
    }

    /// Returns `value`, a date mention's text, reduced.
    pub fn apply(&self, value: &str) -> String {
        let mut out = String::with_capacity(value.len());
        self.write_date(value, &mut out);
        out
    }

    /// Appends one mask character per character of `value`.
    pub(super) fn write_masked(&self, value: &str, out: &mut String) {
        out.extend(value.chars().map(|_| self.mask));
    }

    /// Replaces every mention of `pii`, a layer over `src`, with its
    /// reduced form, as a view derived from `src`.
    pub fn rewrite(&self, src: &View, pii: &Pii) -> Result<PiiRewrite, RewriteError> {
        let text = src.text();
        let kind = self.kind();
        PiiRewrite::derive(
            src,
            pii,
            |_| self.refuse_overlaps,
            0,
            |cluster, out| {
                let piece = &text[cluster.merged.span.range()];
                if cluster.members.len() == 1 {
                    self.write_mention(cluster.merged.kind, piece, out);
                } else {
                    self.write_masked(piece, out);
                }
                Ok(EditOrigin::only(kind))
            },
        )
    }
}
