//! Restates the money layer in one currency: a port of OpenNLP's
//! `MoneyConversionAnnotator` in its fixed-date mode.

use turbo_analysis::{Annotator, Bound, Error, Refs};
use turbo_analysis_layers::{CONVERTED_MONEY, Currency, Money, MoneyMention};

use super::{Date, Decimal, FxRates};

/// Reads `turbo:money` and writes, under [`CONVERTED_MONEY`], each
/// mention the rates can convert, in the target currency as of one date,
/// on its original span. A mention with no usable rate is left out; an
/// error from the rates fails the document.
///
/// Its input and output are both `Money`, so a pipeline takes it with
/// [`MoneyConverter::bound`], which names the output instance.
///
/// The Java annotator can instead take the date from each document's
/// `DocumentDateAnnotator` layer; there is no document date layer here,
/// so the date is the caller's.
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
        let mut amount = Decimal::default();
        for (i, span) in money.spans.iter() {
            let from = money.currency[i.get()];
            if self.rates.convert(money.amount(i), from, self.target, self.as_of, &mut amount)? {
                out.push(MoneyMention {
                    span,
                    negative: amount.negative,
                    coefficient: &amount.digits,
                    scale: amount.scale,
                    currency: self.target,
                });
            }
        }
        Ok(())
    }
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
        let [a, b, c] = self.target.code();
        [u64::from(a), u64::from(b), u64::from(c), self.as_of.epoch_day() as u64]
            .iter()
            .fold(0xcbf2_9ce4_8422_2325, |h, &x| (h ^ x).wrapping_mul(0x0100_0000_01b3))
    }
}
