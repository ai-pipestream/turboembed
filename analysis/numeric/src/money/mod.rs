//! Money in text and its conversion between currencies, ported from the
//! money package of the OpenNLP numeric add-on.
//!
//! - [`MoneyExtractor`] finds amounts with a currency (`$1.2M`, `100 USD`,
//!   `3 billion pounds`) and writes them as the `turbo:money` layer
//!   ([`Money`]): `CursorMoneyExtractor` and `MoneyAnnotator`.
//! - [`EcbRates`] reads the ECB's euro reference history and gives rates
//!   and converted amounts as of a date: `EcbFxRates`.
//! - [`MoneyConverter`] restates the money layer in one currency under
//!   [`CONVERTED_MONEY`] as of a fixed date, and [`DatedMoneyConverter`]
//!   as of each document's date: `MoneyConversionAnnotator`.
//!
//! Amounts are exact decimals throughout; the arithmetic follows Java's
//! `BigDecimal` digit for digit. docs/money.md describes the rules.

mod convert;
mod csv;
mod decimal;
mod divide;
mod ecb;
mod extract;
mod natural;
mod tables;

use std::fmt;

pub use convert::{DatedMoneyConverter, MoneyConverter};
pub use decimal::{Decimal, DecimalParseError, cmp_values};
pub use ecb::{EcbError, EcbRates, MAX_STALENESS_DAYS};
pub use extract::{MoneyExtractor, SymbolError};
pub use tables::{DEFAULT_SYMBOLS, is_iso_currency};
pub use turbo_analysis_layers::{CONVERTED_MONEY, Currency, Date, Money, MoneyMention};

use turbo_analysis_layers::DecimalRef;

/// A result whose scale is outside the range of a Java `int`, where
/// `BigDecimal` throws an `ArithmeticException`.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct ScaleRangeError;

impl fmt::Display for ScaleRangeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("result is outside the supported decimal scale range")
    }
}

impl std::error::Error for ScaleRangeError {}

/// Exchange rates as of a date: OpenNLP's `FxRates`.
pub trait FxRates {
    /// What a lookup can fail with, besides having no rate.
    type Error: std::error::Error + From<ScaleRangeError> + Send + Sync + 'static;

    /// Target units per source unit as of `as_of`, or `None` when there
    /// is no usable rate for the pair on that date.
    fn rate(&self, from: Currency, to: Currency, as_of: Date) -> Result<Option<Decimal>, Self::Error>;

    /// Writes `amount`, in `from`, converted into `to` as of `as_of` to
    /// `out`, and returns whether there was a usable rate; `out` is left
    /// as it was when there was none.
    ///
    /// The default multiplies the amount by [`FxRates::rate`], exactly,
    /// as the Java interface's default method does.
    fn convert(
        &self,
        amount: DecimalRef<'_>,
        from: Currency,
        to: Currency,
        as_of: Date,
        out: &mut Decimal,
    ) -> Result<bool, Self::Error> {
        let Some(rate) = self.rate(from, to, as_of)? else {
            return Ok(false);
        };
        Decimal::multiply(amount, rate.as_ref(), out)?;
        Ok(true)
    }
}

impl<T: FxRates + ?Sized> FxRates for std::sync::Arc<T> {
    type Error = T::Error;

    fn rate(&self, from: Currency, to: Currency, as_of: Date) -> Result<Option<Decimal>, Self::Error> {
        (**self).rate(from, to, as_of)
    }

    fn convert(
        &self,
        amount: DecimalRef<'_>,
        from: Currency,
        to: Currency,
        as_of: Date,
        out: &mut Decimal,
    ) -> Result<bool, Self::Error> {
        (**self).convert(amount, from, to, as_of, out)
    }
}
