//! Restates the money layer in one currency: a port of OpenNLP's
//! `MoneyConversionAnnotator`, as of a fixed date ([`MoneyConverter`]) or
//! as of each document's date ([`DatedMoneyConverter`]).

use turbo_analysis::{Annotator, Bound, Error, Fnv1a, Refs};
use turbo_analysis_layers::{CONVERTED_MONEY, Currency, DocumentDate, Money, MoneyMention};

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
}

impl<R: FxRates + Send + Sync + 'static> MoneyConverter<R> {
    pub fn new(rates: R, target: Currency, as_of: Date) -> MoneyConverter<R> {
        MoneyConverter { rates, target, as_of }
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
        convert(&self.rates, self.target, self.as_of, money, out)
    }
}

/// Appends each mention of `money` the rates convert into `target` as of
/// `as_of` to `out`, on its original span, in order.
fn convert<R: FxRates>(
    rates: &R,
    target: Currency,
    as_of: Date,
    money: &Money,
    out: &mut Money,
) -> Result<(), R::Error> {
    let mut amount = Decimal::default();
    for (i, span) in money.spans.iter() {
        let from = money.currency[i.get()];
        if rates.convert(money.amount(i), from, target, as_of, &mut amount)? {
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

/// FNV-1a over a target code and, for a fixed date, the date.
fn config_hash(target: Currency, as_of: Option<Date>) -> u64 {
    let [a, b, c] = target.code();
    let date = as_of.map_or([0, 0], |d| [1, d.epoch_day() as u64]);
    [u64::from(a), u64::from(b), u64::from(c), date[0], date[1]]
        .iter()
        .fold(Fnv1a::new(), |mut h, &x| {
            h.u64(x);
            h
        })
        .finish()
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

    /// FNV-1a over the target code and the date. The rates themselves are
    /// data the caller loads and are not part of it.
    fn config_hash(&self) -> u64 {
        config_hash(self.target, Some(self.as_of))
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
}

impl<R: FxRates + Send + Sync + 'static> DatedMoneyConverter<R> {
    pub fn new(rates: R, target: Currency) -> DatedMoneyConverter<R> {
        DatedMoneyConverter { rates, target }
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
            Some(as_of) => convert(&self.rates, self.target, as_of, money, out),
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

    /// FNV-1a over the target code, with no date.
    fn config_hash(&self) -> u64 {
        config_hash(self.target, None)
    }
}
