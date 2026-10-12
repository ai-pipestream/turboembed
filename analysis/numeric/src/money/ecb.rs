//! Euro reference rates from an ECB history file: a port of OpenNLP's
//! `EcbFxRates`. The caller supplies the file; nothing is downloaded.

use std::collections::BTreeMap;
use std::fmt;
use std::path::Path;

use turbo_analysis_layers::{Currency, DecimalRef};

use super::csv::{fields, is_blank, lines};
use super::decimal::{Decimal, saturate};
use super::divide::{Quotient, decimal_quotient, divide64};
use super::natural::Natural;
use super::{Date, FxRates, ScaleRangeError};

/// The oldest a reference row may be for a lookup, in calendar days,
/// inclusive.
pub const MAX_STALENESS_DAYS: i64 = 7;

/// Euro reference rates by date, loaded from the ECB's history CSV.
///
/// The header begins with `Date,` and names unique codes of three ASCII
/// capital letters. Fields are unquoted and padded at most with ASCII
/// whitespace. Each row has a unique ISO date and a field per column; a
/// header or row may end with one more, empty field. An empty field or
/// `N/A` marks a currency with no quote that day; a quote must be a
/// positive `BigDecimal`, and an explicit `EUR` quote must be 1.
///
/// A lookup takes the latest row on or before the date asked for, at most
/// [`MAX_STALENESS_DAYS`] earlier; both currencies must be quoted in that
/// row, and an older row is not searched for a missing one. The euro is
/// always 1. A cross rate is the target quote divided by the source quote,
/// and a converted amount is the amount times the target quote divided by
/// the source quote, rounded once to 16 significant digits, half to even.
#[derive(Clone, Debug)]
pub struct EcbRates {
    /// The currency of each quote column, in header order.
    currencies: Vec<Currency>,
    /// Each row's date, sorted, with the index of its quotes.
    dates: Vec<(Date, u32)>,
    /// The quotes, row by row in file order, a column each; `None` when
    /// the field was empty or `N/A`.
    quotes: Vec<Option<Quote>>,
    /// The digits of every quote, one after another.
    digits: String,
}

/// A quote's unscaled value (a range of the shared digits) and scale.
#[derive(Copy, Clone, Debug)]
struct Quote {
    start: u32,
    end: u32,
    scale: i32,
}

/// Why a history file was refused. The messages are those of
/// `EcbFxRates`.
#[derive(Debug)]
pub enum EcbError {
    /// The file could not be read.
    Io(std::io::Error),
    /// The bytes are not UTF-8; the offset is of the first bad byte.
    Utf8 {
        offset: usize,
    },
    /// The first line is missing or does not start with `Date,`.
    NotAHistory {
        header: Option<String>,
    },
    /// A line holds a control character that is not whitespace, at this
    /// byte offset into the line.
    ControlCharacter {
        offset: usize,
    },
    /// The header names no currency.
    NoCurrency,
    /// A header field that is not three ASCII capitals.
    InvalidCurrency(String),
    DuplicateCurrency(Currency),
    /// A row whose field count is not the header's.
    ColumnCount {
        columns: usize,
        row: String,
    },
    /// A row whose first field is not an ISO date.
    InvalidDate {
        row: String,
    },
    /// A quote that is not a decimal.
    InvalidRate {
        currency: Currency,
        value: String,
        row: String,
    },
    /// A quote of zero or less.
    NonPositiveRate {
        currency: Currency,
        row: String,
    },
    /// An `EUR` quote other than 1.
    EuroNotOne {
        row: String,
    },
    DuplicateDate(Date),
    /// No row besides the header.
    NoRows,
}

impl fmt::Display for EcbError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            EcbError::Io(e) => write!(f, "reading the reference history failed: {e}"),
            EcbError::Utf8 { offset } => write!(f, "the reference history is not UTF-8 at byte {offset}"),
            EcbError::NotAHistory { header: Some(h) } => write!(f, "not a reference history CSV: header is {h}"),
            EcbError::NotAHistory { header: None } => f.write_str("not a reference history CSV: header is null"),
            EcbError::ControlCharacter { offset } => write!(f, "invalid control character at CSV offset {offset}"),
            EcbError::NoCurrency => f.write_str("reference history header must name a currency"),
            EcbError::InvalidCurrency(c) => write!(f, "invalid reference history currency code: {c}"),
            EcbError::DuplicateCurrency(c) => write!(f, "duplicate reference history currency code: {c}"),
            EcbError::ColumnCount { columns, row } => {
                write!(f, "expected {columns} reference history columns in row: {row}")
            }
            EcbError::InvalidDate { row } => write!(f, "not a reference history row: {row}"),
            EcbError::InvalidRate { currency, value, row } => {
                write!(f, "not a reference history rate for {currency}: {value} in row: {row}")
            }
            EcbError::NonPositiveRate { currency, row } => {
                write!(f, "reference history rate must be positive for {currency} in row: {row}")
            }
            EcbError::EuroNotOne { row } => write!(f, "reference history EUR rate must equal 1 in row: {row}"),
            EcbError::DuplicateDate(d) => write!(f, "duplicate reference history date: {d}"),
            EcbError::NoRows => f.write_str("the reference history contains no rows"),
        }
    }
}

impl std::error::Error for EcbError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            EcbError::Io(e) => Some(e),
            _ => None,
        }
    }
}

/// The base currency: every quote is per euro.
const EUR: Currency = match Currency::new(*b"EUR") {
    Some(c) => c,
    None => panic!("EUR is three capitals"),
};
const HEADER_PREFIX: &str = "Date,";
const NOT_AVAILABLE: &str = "N/A";

impl EcbRates {
    /// Loads a history file.
    pub fn load(path: impl AsRef<Path>) -> Result<EcbRates, EcbError> {
        EcbRates::from_bytes(&std::fs::read(path).map_err(EcbError::Io)?)
    }

    /// Loads a history from its bytes, which must be UTF-8.
    ///
    /// The bytes are checked as UTF-8 before any line is read. The Java
    /// decodes as it reads, in blocks of 8192 characters, so in a file
    /// larger than that with both a bad byte and a malformed row, the two
    /// can report different errors; both refuse the file.
    pub fn from_bytes(bytes: &[u8]) -> Result<EcbRates, EcbError> {
        let text = std::str::from_utf8(bytes).map_err(|e| EcbError::Utf8 { offset: e.valid_up_to() })?;
        EcbRates::parse(text)
    }

    /// Loads a history from its text.
    pub fn parse(text: &str) -> Result<EcbRates, EcbError> {
        let mut lines = lines(text);
        let header = lines.next();
        let header = match header {
            Some(h) if h.starts_with(HEADER_PREFIX) => h,
            _ => return Err(EcbError::NotAHistory { header: header.map(str::to_string) }),
        };
        let currencies = currency_columns(header)?;
        let columns = currencies.len() + 1;
        let mut rates = EcbRates { currencies, dates: Vec::new(), quotes: Vec::new(), digits: String::new() };
        let mut by_date = BTreeMap::new();
        let mut cells = Vec::new();
        for line in lines {
            if is_blank(line) {
                continue;
            }
            fields(line, &mut cells).map_err(|offset| EcbError::ControlCharacter { offset })?;
            let extra_empty = cells.len() == columns + 1 && cells.last() == Some(&"");
            if cells.len() != columns && !extra_empty {
                return Err(EcbError::ColumnCount { columns, row: line.to_string() });
            }
            let date = Date::parse(cells[0]).ok_or_else(|| EcbError::InvalidDate { row: line.to_string() })?;
            let row = (rates.quotes.len() / rates.currencies.len()) as u32;
            for (k, &cell) in cells[1..columns].iter().enumerate() {
                let quote = rates.quote_cell(cell, rates.currencies[k], line)?;
                rates.quotes.push(quote);
            }
            if by_date.insert(date, row).is_some() {
                return Err(EcbError::DuplicateDate(date));
            }
        }
        if by_date.is_empty() {
            return Err(EcbError::NoRows);
        }
        rates.dates = by_date.into_iter().collect();
        Ok(rates)
    }

    /// Reads one quote field of `row`.
    fn quote_cell(&mut self, value: &str, currency: Currency, row: &str) -> Result<Option<Quote>, EcbError> {
        if value.is_empty() || value == NOT_AVAILABLE {
            return Ok(None);
        }
        let rate = Decimal::parse(value).map_err(|_| EcbError::InvalidRate {
            currency,
            value: value.to_string(),
            row: row.to_string(),
        })?;
        if rate.negative || rate.is_zero() {
            return Err(EcbError::NonPositiveRate { currency, row: row.to_string() });
        }
        if currency == EUR && rate.cmp_value(&Decimal::one()).is_ne() {
            return Err(EcbError::EuroNotOne { row: row.to_string() });
        }
        let start = self.digits.len() as u32;
        self.digits.push_str(&rate.digits);
        Ok(Some(Quote { start, end: self.digits.len() as u32, scale: rate.scale }))
    }

    /// The currencies the header names, in order.
    pub fn currencies(&self) -> &[Currency] {
        &self.currencies
    }

    /// The reference dates, oldest first.
    pub fn dates(&self) -> impl Iterator<Item = Date> + '_ {
        self.dates.iter().map(|&(d, _)| d)
    }

    /// The quotes of the latest row on or before `as_of`, when it is at
    /// most [`MAX_STALENESS_DAYS`] older.
    fn usable_row(&self, as_of: Date) -> Option<&[Option<Quote>]> {
        let n = self.dates.partition_point(|&(d, _)| d <= as_of);
        let &(date, row) = self.dates.get(n.checked_sub(1)?)?;
        if as_of.epoch_day() - date.epoch_day() > MAX_STALENESS_DAYS {
            return None;
        }
        let width = self.currencies.len();
        Some(&self.quotes[row as usize * width..(row as usize + 1) * width])
    }

    /// The units of `currency` per euro in a row: 1 for the euro itself.
    fn per_euro(&self, row: &[Option<Quote>], currency: Currency) -> Option<DecimalRef<'_>> {
        if currency == EUR {
            return Some(DecimalRef { negative: false, digits: "1", scale: 0 });
        }
        let column = self.currencies.iter().position(|&c| c == currency)?;
        let q = row[column]?;
        Some(DecimalRef {
            negative: false,
            digits: &self.digits[q.start as usize..q.end as usize],
            scale: q.scale.into(),
        })
    }

    /// The quotes of both currencies on the row for `as_of`.
    fn pair(&self, from: Currency, to: Currency, as_of: Date) -> Option<(DecimalRef<'_>, DecimalRef<'_>)> {
        let row = self.usable_row(as_of)?;
        Some((self.per_euro(row, from)?, self.per_euro(row, to)?))
    }
}

impl FxRates for EcbRates {
    type Error = ScaleRangeError;

    /// The target quote divided by the source quote, to 16 digits; exactly
    /// 1 when the two currencies are the same.
    fn rate(&self, from: Currency, to: Currency, as_of: Date) -> Result<Option<Decimal>, ScaleRangeError> {
        let Some((per_from, per_to)) = self.pair(from, to, as_of) else {
            return Ok(None);
        };
        if from == to {
            return Ok(Some(Decimal::one()));
        }
        let to_digits = Natural::from_digits(per_to.digits);
        let from_digits = Natural::from_digits(per_from.digits);
        let q = match divide64(&to_digits, per_to.scale, &from_digits, per_from.scale) {
            Ok(q) => q,
            Err(ScaleRangeError) => decimal_quotient(&to_digits, &from_digits, per_to.scale - per_from.scale)?,
        };
        let mut out = Decimal::default();
        write_quotient(q, false, &mut out);
        Ok(Some(out))
    }

    /// The amount times the target quote divided by the source quote,
    /// rounded once. An amount in the target currency is kept as it is,
    /// scale and all, but still needs a quote on a usable date.
    fn convert(
        &self,
        amount: DecimalRef<'_>,
        from: Currency,
        to: Currency,
        as_of: Date,
        out: &mut Decimal,
    ) -> Result<bool, ScaleRangeError> {
        let Some((per_from, per_to)) = self.pair(from, to, as_of) else {
            return Ok(false);
        };
        if from == to {
            out.set(amount)?;
            return Ok(true);
        }
        let product_scale = amount.scale + per_to.scale;
        if amount.digits == "0" {
            // BigDecimal.multiply clamps the scale of a zero product, and
            // dividing zero gives zero at the preferred scale, clamped.
            let product_scale = i64::from(saturate(product_scale));
            *out = Decimal::zero(saturate(product_scale - per_from.scale));
            return Ok(true);
        }
        let product = Natural::from_digits(amount.digits).mul(&Natural::from_digits(per_to.digits));
        let from_digits = Natural::from_digits(per_from.digits);
        let combined = product_scale - per_from.scale;
        let divided = if i32::try_from(product_scale).is_ok() {
            divide64(&product, product_scale, &from_digits, per_from.scale)
        } else {
            // BigDecimal.multiply throws for this scale.
            Err(ScaleRangeError)
        };
        let q = match divided {
            Ok(q) => q,
            Err(ScaleRangeError) => decimal_quotient(&product, &from_digits, combined)?,
        };
        write_quotient(q, amount.negative, out);
        Ok(true)
    }
}

/// Writes a quotient into `out`.
fn write_quotient(q: Quotient, negative: bool, out: &mut Decimal) {
    out.digits.clear();
    out.digits.push_str(&q.coefficient.to_string());
    out.negative = negative && q.coefficient != 0;
    out.scale = q.scale;
}

/// `currencyColumns`: the header's codes, after the date column and
/// without one empty final field.
fn currency_columns(header: &str) -> Result<Vec<Currency>, EcbError> {
    let mut names = Vec::new();
    fields(header, &mut names).map_err(|offset| EcbError::ControlCharacter { offset })?;
    if names.last() == Some(&"") {
        names.pop();
    }
    if names.len() < 2 {
        return Err(EcbError::NoCurrency);
    }
    let mut currencies: Vec<Currency> = Vec::with_capacity(names.len() - 1);
    for name in &names[1..] {
        let code = Currency::parse(name).ok_or_else(|| EcbError::InvalidCurrency(name.to_string()))?;
        if currencies.contains(&code) {
            return Err(EcbError::DuplicateCurrency(code));
        }
        currencies.push(code);
    }
    Ok(currencies)
}
