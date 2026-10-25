//! Finds measured quantities and percentages in a text: a port of OpenNLP's
//! `CursorQuantityExtractor` and `QuantityAnnotator`
//! (`opennlp.tools.quantity`, numeric add-on).
//!
//! The Java steps over a `CharSequence` in UTF-16 units; this steps over
//! the same characters in UTF-8 bytes. The characters a mention is made
//! of (the minus, digits, separators, `%`) are ASCII, the one gap is a
//! `White_Space` character, which is always in the Basic Multilingual
//! Plane, and a unit token is read by [`word_end`], which counts UTF-16
//! units as the Java does, so both find the same mentions.

use std::fmt;

use turbo_analysis::{Annotator, Error, Refs, Span, View, ViewId};
use turbo_analysis_layers::{Money, NumberNotation, PERCENT, Quantities, QuantityMention};
use turbo_unicode::java;

use crate::money::MoneyExtractor;
use crate::scan::{
    boundary_after, boundary_before, byte_at, char_at, continues_number, parse, sign_boundary_before, whitespace_at,
    word_end,
};

/// The units OpenNLP's extractor knows unless a caller gives its own set:
/// common measurement tokens, matched exactly and case-sensitively.
/// Ambiguous English words such as `in` are left out on purpose.
pub const DEFAULT_UNITS: [&str; 37] = [
    "km", "m", "cm", "mm", "nm", "mi", "ft", "yd", //
    "kg", "g", "mg", "t", "lb", "lbs", "oz", //
    "L", "mL", "ml", //
    "ms", "ns", "min", "hr", //
    "KB", "MB", "GB", "TB", "PB", //
    "Hz", "kHz", "MHz", "GHz", //
    "kW", "MW", "GW", "kWh", //
    "mph", "kph",
];

/// The word a percentage may be written with instead of `%`, in any case.
const PERCENT_WORD: &str = "percent";

/// The longest unit token a caller may give, in UTF-16 units.
/// Fourteen admits spelled-out English units (`milliseconds`,
/// `kilocalories` are twelve) in custom sets; the default set
/// stays the short forms.
const MAX_UNIT_UNITS: usize = 14;

/// How many UTF-16 units of letters a candidate unit token may
/// have before the scan gives up. A longer run of letters is
/// no unit. The bound only sizes the scan: a token still counts
/// only when it matches a known unit exactly, so widening it
/// changes no reading, on the default document path either.
const MAX_TOKEN_UNITS: usize = 16;

/// Writes `turbo:quantities` over a text view: every number followed by a
/// percent marker or a known unit token.
///
/// The recognized forms are percentages (`50%`, `3.5 %`, `50 percent`, all
/// with the unit `%`), and unit quantities with the unit directly after
/// the number (`2.5km`) or after one `White_Space` character (`80 kg`),
/// each with an optional leading minus. A bare number is never a quantity.
/// The number is read by the number scan of docs/numbers.md without scale
/// markers, in one [`NumberNotation`], `LatinUs` by default.
///
/// Money comes first: the extractor runs [`MoneyExtractor`] with the
/// default symbol table in the same notation, and no quantity overlaps an
/// amount it finds, so `$3m` is not 3 metres.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct QuantityExtractor {
    /// The unit tokens, sorted and each once.
    units: Box<[Box<str>]>,
    notation: NumberNotation,
    money: MoneyExtractor,
}

/// Why a unit set was refused.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum UnitError {
    /// The set has no unit.
    Empty,
    /// The token is empty, longer than the unit cap in UTF-16
    /// units, or has a character that is not a letter
    /// (`Character.isLetter`).
    NotAUnitToken(Box<str>),
}

impl fmt::Display for UnitError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            UnitError::Empty => f.write_str("a unit set must not be empty"),
            UnitError::NotAUnitToken(unit) => write!(f, "not a valid unit token: {unit:?}"),
        }
    }
}

impl std::error::Error for UnitError {}

impl Default for QuantityExtractor {
    fn default() -> QuantityExtractor {
        QuantityExtractor::new(NumberNotation::LatinUs)
    }
}

impl QuantityExtractor {
    /// The default unit set, reading numbers in `notation`.
    pub fn new(notation: NumberNotation) -> QuantityExtractor {
        QuantityExtractor::with_units(DEFAULT_UNITS, notation).expect("the default units are valid")
    }

    /// A custom unit set in place of the default one, reading numbers in
    /// `notation`. Each token must be one to fourteen UTF-16 units of letters.
    /// The percent marker and the word `percent` are read with any set.
    pub fn with_units<S: AsRef<str>>(
        units: impl IntoIterator<Item = S>,
        notation: NumberNotation,
    ) -> Result<QuantityExtractor, UnitError> {
        let mut set: Vec<Box<str>> = Vec::new();
        for unit in units {
            let unit = unit.as_ref();
            if !valid_unit(unit) {
                return Err(UnitError::NotAUnitToken(unit.into()));
            }
            set.push(unit.into());
        }
        if set.is_empty() {
            return Err(UnitError::Empty);
        }
        set.sort_unstable();
        set.dedup();
        Ok(QuantityExtractor { units: set.into_boxed_slice(), notation, money: MoneyExtractor::new(notation) })
    }

    /// The unit tokens, sorted.
    pub fn units(&self) -> impl Iterator<Item = &str> {
        self.units.iter().map(|u| &**u)
    }

    pub fn notation(&self) -> NumberNotation {
        self.notation
    }

    /// Replaces the money extractor used to exclude currency amounts before
    /// scanning quantities.
    pub(crate) fn with_money(mut self, money: MoneyExtractor) -> QuantityExtractor {
        self.money = money;
        self
    }

    /// Appends every quantity mention in `text` to `out`, in text order,
    /// with byte offsets into `text`.
    ///
    /// The money amounts of the text are found first. The scan then tries
    /// a mention at each character outside them: after a mention it goes
    /// on at the mention's end, and a mention that runs into the next
    /// amount is dropped, the scan still going on at its end.
    ///
    /// # Panics
    ///
    /// When `text` is longer than `i32::MAX` bytes, as the money scan does.
    pub fn scan(&self, text: &str, out: &mut Quantities) {
        self.scan_with_fallback(text, None, out);
    }

    /// Scans `text` with `self`, falling back to `fallback` at candidate
    /// positions that fail in `self.notation` when a document's region
    /// ballot carries both notations.
    pub(crate) fn scan_with_fallback(&self, text: &str, fallback: Option<&QuantityExtractor>, out: &mut Quantities) {
        let mut money = Money::new(ViewId::SOURCE);
        self.money.scan(text, &mut money);
        let (mut money_start, mut money_end): (Vec<u32>, Vec<u32>) = {
            let (s, e) = money.spans.columns();
            (s.to_vec(), e.to_vec())
        };
        if let Some(alt) = fallback.filter(|alt| alt.notation != self.notation) {
            let mut alt_money = Money::new(ViewId::SOURCE);
            alt.money.scan(text, &mut alt_money);
            let (alt_s, alt_e) = alt_money.spans.columns();
            for (&s, &e) in alt_s.iter().zip(alt_e.iter()) {
                let overlaps = money_start.iter().zip(money_end.iter()).any(|(&ms, &me)| s < me && ms < e);
                if !overlaps {
                    let pos = money_start.partition_point(|&ms| ms < s);
                    money_start.insert(pos, s);
                    money_end.insert(pos, e);
                }
            }
        }
        let b = text.as_bytes();
        let mut digits = String::new();
        let mut amount = 0;
        let mut i = 0;
        while i < b.len() {
            // Only a minus or a digit can start a mention. Neither byte
            // occurs inside a multibyte character, so where one is found is
            // a character boundary, as the Java's step by code points would
            // reach. An amount the skipped characters fall in is passed over
            // below, as the Java passes over it.
            match b[i..].iter().position(|&x| x == b'-' || x.is_ascii_digit()) {
                Some(skip) => i += skip,
                None => break,
            }
            while amount < money_start.len() && money_end[amount] as usize <= i {
                amount += 1;
            }
            if amount < money_start.len() && money_start[amount] as usize <= i {
                i = money_end[amount] as usize;
                continue;
            }
            let found = self.match_at(text, i, &mut digits).or_else(|| {
                fallback.filter(|alt| alt.notation != self.notation).and_then(|alt| alt.match_at(text, i, &mut digits))
            });
            let Some(found) = found else {
                // A minus or a digit is one byte.
                i += 1;
                continue;
            };
            if amount == money_start.len() || found.end <= money_start[amount] as usize {
                let unit = match found.unit {
                    UnitAt::Percent => PERCENT,
                    UnitAt::Token(start) => &text[start..found.end],
                };
                out.push(QuantityMention {
                    span: Span::new(i as u32, found.end as u32),
                    negative: found.negative && digits != "0",
                    coefficient: &digits,
                    scale: found.scale,
                    unit,
                });
            }
            i = found.end;
        }
    }

    /// `matchAt`: a number at byte `start`, with an optional minus, then a
    /// percent marker or a unit. A digit that goes on with a number before
    /// it never starts a mention, so the `000` of `1,00,000 kg` is not
    /// `0 kg`.
    fn match_at(&self, text: &str, start: usize, digits: &mut String) -> Option<Found> {
        let b = text.as_bytes();
        let negative = b[start] == b'-';
        let first = start + usize::from(negative);
        if !byte_at(b, first).is_ascii_digit() || continues_number(text, first, self.notation) {
            return None;
        }
        let boundary = if negative { sign_boundary_before(text, start) } else { boundary_before(text, start) };
        if !boundary {
            return None;
        }
        let r = parse(text, first, false, self.notation, digits)?;
        let (unit, end) = self.parse_unit(text, r.end)?;
        let scale = i32::try_from(r.scale).expect("a text of at most i32::MAX bytes has a scale that fits");
        Some(Found { end, scale, negative, unit })
    }

    /// `parseUnit`: a percent marker or unit directly after the number at
    /// `number_end`, or after one `White_Space` character.
    fn parse_unit(&self, text: &str, number_end: usize) -> Option<(UnitAt, usize)> {
        if let Some(found) = self.unit_at(text, number_end) {
            return Some(found);
        }
        if !whitespace_at(text, number_end) {
            return None;
        }
        // Past the end of the text the Java reads a space and steps one
        // unit; the unit scan after it then finds nothing.
        self.unit_at(text, number_end + char_at(text, number_end).map_or(1, char::len_utf8))
    }

    /// `unitAt`: a `%` before a word boundary, the word `percent` in any
    /// case, or a known unit token, at byte `start`, and where it ends.
    fn unit_at(&self, text: &str, start: usize) -> Option<(UnitAt, usize)> {
        if byte_at(text.as_bytes(), start) == b'%' {
            return boundary_after(text, start + 1).then_some((UnitAt::Percent, start + 1));
        }
        let end = word_end(text, start, MAX_TOKEN_UNITS)?;
        let word = &text[start..end];
        // The Java's `equalsIgnoreCase` compares UTF-16 units through
        // `toUpperCase` and `toLowerCase`. No character other than an ASCII
        // letter maps to one of the letters of `percent` either way, so an
        // ASCII comparison gives the same answer.
        if word.eq_ignore_ascii_case(PERCENT_WORD) {
            return Some((UnitAt::Percent, end));
        }
        self.units.binary_search_by(|u| (**u).cmp(word)).is_ok().then_some((UnitAt::Token(start), end))
    }
}

/// `validUnit`: one to the unit cap in UTF-16 units, every character a letter.
fn valid_unit(unit: &str) -> bool {
    !unit.is_empty() && unit.encode_utf16().count() <= MAX_UNIT_UNITS && unit.chars().all(java::is_letter)
}

/// A mention found by `matchAt`. Its digits are in the scan's buffer.
struct Found {
    end: usize,
    scale: i32,
    negative: bool,
    unit: UnitAt,
}

/// The unit of a mention: a percentage, or the token that starts at a
/// byte and ends where the mention does.
enum UnitAt {
    Percent,
    Token(usize),
}

impl Annotator for QuantityExtractor {
    const ID: &'static str = "turbo-numeric:quantities";
    type Reads = (View,);
    type Writes = (Quantities,);

    fn annotate(&self, (view,): Refs<'_, Self>) -> Result<(Quantities,), Error> {
        let text = view.text();
        if i32::try_from(text.len()).is_err() {
            return Err(Error::Annotator {
                annotator: Self::ID,
                detail: format!("a text of {} bytes is past the i32 scale range", text.len()),
            });
        }
        let mut quantities = Quantities::new(view.id());
        self.scan(text, &mut quantities);
        Ok((quantities,))
    }

    /// FNV-1a over the notation and the unit tokens, fixed so that stored
    /// provenance stays comparable across builds. A zero byte ends each
    /// token, which no letter is.
    fn config_hash(&self) -> u64 {
        let notation: u64 = match self.notation {
            NumberNotation::LatinUs => 1,
            NumberNotation::LatinEu => 2,
        };
        let units = self.units.iter().flat_map(|u| u.bytes().chain([0]).map(u64::from));
        std::iter::once(notation)
            .chain(units)
            .fold(0xcbf2_9ce4_8422_2325, |h, x| (h ^ x).wrapping_mul(0x0100_0000_01b3))
    }
}
