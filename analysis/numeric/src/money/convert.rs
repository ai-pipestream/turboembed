//! Restates the money layer in one currency: a port of OpenNLP's
//! `MoneyConversionAnnotator`, as of a fixed date ([`MoneyConverter`]) or
//! as of each document's date ([`DatedMoneyConverter`]).

use turbo_analysis::{Annotator, Bound, Error, Refs};
use turbo_analysis_layers::{CONVERTED_MONEY, Currency, DocumentDate, Money, MoneyMention};

use super::tables::minor_units;
use super::{Date, Decimal, FxRates};

/// Reads `turbo:money` and writes, under [`CONVERTED_MONEY`], each
/// mention the rates can convert, in the target currency as of one date,
/// on its original span. A mention with no usable rate is left out; an
/// error from the rates fails the document.
///
/// Its input and output are both `Money`, so a pipeline takes it with
/// [`MoneyConverter::bound`], which names the output instance.
///
/// This is the Java annotator's fixed-date mode, which does not read the
/// document date layer. [`DatedMoneyConverter`] is its document-date mode.
#[derive(Clone, Debug)]
pub struct MoneyConverter<R> {
    rates: R,
    target: Currency,
    as_of: Date,
    minor_unit_rounding: bool,
}

impl<R: FxRates + Send + Sync + 'static> MoneyConverter<R> {
    pub fn new(rates: R, target: Currency, as_of: Date) -> MoneyConverter<R> {
        MoneyConverter { rates, target, as_of, minor_unit_rounding: false }
    }

    /// Configures whether converted values are rounded half-to-even to the
    /// target currency's ISO 4217 minor unit ([`minor_units`]).
    pub fn with_minor_unit_rounding(mut self, enabled: bool) -> Self {
        self.minor_unit_rounding = enabled;
        self
    }

    /// Whether converted values are rounded to the target currency's minor unit.
    pub fn minor_unit_rounding(&self) -> bool {
        self.minor_unit_rounding
    }

    /// The step for a pipeline: reads the default money instance, writes
    /// [`CONVERTED_MONEY`].
    pub fn bound(self) -> Bound<MoneyConverter<R>> {
        Bound::new(self).write(0, CONVERTED_MONEY.name)
    }

    pub fn target(&self) -> Currency {
        self.target
    }

    pub fn as_of(&self) -> Date {
        self.as_of
    }

    /// Appends the converted mentions of `money` to `out`, in order.
    pub fn convert(&self, money: &Money, out: &mut Money) -> Result<(), R::Error> {
        convert(&self.rates, self.target, self.as_of, self.minor_unit_rounding, money, out)
    }
}

/// Appends each mention of `money` the rates convert into `target` as of
/// `as_of` to `out`, on its original span, in order.
fn convert<R: FxRates>(
    rates: &R,
    target: Currency,
    as_of: Date,
    minor_unit_rounding: bool,
    money: &Money,
    out: &mut Money,
) -> Result<(), R::Error> {
    let mut amount = Decimal::default();
    let round_scale = if minor_unit_rounding { minor_units(target).map(i32::from) } else { None };
    for (i, span) in money.spans.iter() {
        let from = money.currency[i.get()];
        if rates.convert(money.amount(i), from, target, as_of, &mut amount)? {
            if let Some(scale) = round_scale {
                amount.round_to_scale(scale);
            }
            out.push(MoneyMention {
                span,
                negative: amount.negative,
                coefficient: &amount.digits,
                scale: amount.scale,
                currency: target,
            });
        }
    }
    Ok(())
}

/// FNV-1a over a target code, optional fixed date, and minor-unit rounding flag.
fn config_hash(target: Currency, as_of: Option<Date>, minor_unit_rounding: bool) -> u64 {
    let [a, b, c] = target.code();
    let date = as_of.map_or([0, 0], |d| [1, d.epoch_day() as u64]);
    let mut h = [u64::from(a), u64::from(b), u64::from(c), date[0], date[1]]
        .iter()
        .fold(0xcbf2_9ce4_8422_2325, |h, &x| (h ^ x).wrapping_mul(0x0100_0000_01b3));
    if minor_unit_rounding {
        h = (h ^ 1).wrapping_mul(0x0100_0000_01b3);
    }
    h
}

impl<R: FxRates + Send + Sync + 'static> Annotator for MoneyConverter<R> {
    const ID: &'static str = "turbo-numeric:money-conversion";
    type Reads = (Money,);
    type Writes = (Money,);

    fn annotate(&self, (money,): Refs<'_, Self>) -> Result<(Money,), Error> {
        let mut out = Money::new(money.spans.view());
        self.convert(money, &mut out).map_err(|e| Error::Annotator { annotator: Self::ID, detail: e.to_string() })?;
        Ok((out,))
    }

    /// FNV-1a over the target code, the date, and the rounding flag. The rates
    /// themselves are data the caller loads and are not part of it.
    fn config_hash(&self) -> u64 {
        config_hash(self.target, Some(self.as_of), self.minor_unit_rounding)
    }
}

/// Reads `turbo:money` and `turbo:document-date` and writes, under
/// [`CONVERTED_MONEY`], each mention the rates can convert, in the target
/// currency as of the document's date: the Java annotator's document-date
/// mode, `MoneyConversionAnnotator(rates, target)`. A document without a
/// date gets an empty layer, and the rates are not asked. A mention with
/// no usable rate is left out; an error from the rates fails the document.
///
/// The document date layer is required, so a pipeline without a step that
/// writes it (`DocumentDateSelector`) is refused when it is built. The
/// Java refuses a date layer with more than one date; a [`DocumentDate`]
/// holds at most one.
#[derive(Clone, Debug)]
pub struct DatedMoneyConverter<R> {
    rates: R,
    target: Currency,
    minor_unit_rounding: bool,
}

impl<R: FxRates + Send + Sync + 'static> DatedMoneyConverter<R> {
    pub fn new(rates: R, target: Currency) -> DatedMoneyConverter<R> {
        DatedMoneyConverter { rates, target, minor_unit_rounding: false }
    }

    /// Configures whether converted values are rounded half-to-even to the
    /// target currency's ISO 4217 minor unit ([`minor_units`]).
    pub fn with_minor_unit_rounding(mut self, enabled: bool) -> Self {
        self.minor_unit_rounding = enabled;
        self
    }

    /// Whether converted values are rounded to the target currency's minor unit.
    pub fn minor_unit_rounding(&self) -> bool {
        self.minor_unit_rounding
    }

    /// The step for a pipeline: reads the default money and document date
    /// instances, writes [`CONVERTED_MONEY`].
    pub fn bound(self) -> Bound<DatedMoneyConverter<R>> {
        Bound::new(self).write(0, CONVERTED_MONEY.name)
    }

    pub fn target(&self) -> Currency {
        self.target
    }

    /// Appends the mentions of `money` converted as of `date` to `out`, in
    /// order; nothing when `date` holds no date.
    pub fn convert(&self, money: &Money, date: &DocumentDate, out: &mut Money) -> Result<(), R::Error> {
        match date.date {
            Some(as_of) => convert(&self.rates, self.target, as_of, self.minor_unit_rounding, money, out),
            None => Ok(()),
        }
    }
}

impl<R: FxRates + Send + Sync + 'static> Annotator for DatedMoneyConverter<R> {
    const ID: &'static str = "turbo-numeric:money-conversion-dated";
    type Reads = (Money, DocumentDate);
    type Writes = (Money,);

    fn annotate(&self, (money, date): Refs<'_, Self>) -> Result<(Money,), Error> {
        let mut out = Money::new(money.spans.view());
        self.convert(money, date, &mut out)
            .map_err(|e| Error::Annotator { annotator: Self::ID, detail: e.to_string() })?;
        Ok((out,))
    }

    /// FNV-1a over the target code and rounding flag, with no date.
    fn config_hash(&self) -> u64 {
        config_hash(self.target, None, self.minor_unit_rounding)
    }
}
