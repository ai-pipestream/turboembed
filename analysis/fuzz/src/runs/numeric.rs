//! The number, money and quantity runs, in each notation the generator
//! uses (`latin-us`, `latin-eu`). The money conversion run converts with
//! the checked in ECB snapshot to the target and date the goldens were
//! made with; a golden made with another configuration is skipped.

use turbo_analysis::{Analyzer, Document, Key};
use turbo_analysis_layers::{Money, NumberNotation, Numbers, Quantities};
use turbo_goldens::{Golden, Tolerance, matches_golden, matches_golden_key};
use turbo_numeric::money::{CONVERTED_MONEY, Currency, Date, EcbRates, MoneyConverter, MoneyExtractor};
use turbo_numeric::{NumberScanner, QuantityExtractor};

use super::{Outcome, Run, RustError, analyze, config, outcome};
use crate::resources::{Resources, Unavailable};

/// The conversion the money goldens were made with (the `config` records
/// of every `money-convert` golden).
pub(crate) const TARGET: &str = "EUR";
pub(crate) const AS_OF: &str = "2026-03-15";
pub(crate) const SNAPSHOT: &str = "rates/ecb-snapshot.csv";

fn broken(what: impl std::fmt::Display) -> Unavailable {
    Unavailable::Broken(what.to_string())
}

/// The notation a run name stands for and its `config notation` value.
pub(crate) fn notation(run: &str) -> Result<(NumberNotation, &'static str), Unavailable> {
    match run {
        "latin-us" => Ok((NumberNotation::LatinUs, "LATIN_US")),
        "latin-eu" => Ok((NumberNotation::LatinEu, "LATIN_EU")),
        other => Err(Unavailable::NotCovered(format!("the notation {other}"))),
    }
}

/// Which of the money layers a run compares.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum MoneyLayer {
    Extracted,
    Converted,
}

/// One of the three annotators over a text alone, and the layer compared.
pub(crate) struct Numeric {
    analyzer: Analyzer,
    config: &'static str,
    layer: Compared,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Compared {
    Numbers,
    Money(MoneyLayer),
    Quantities,
}

impl Numeric {
    pub(crate) fn open_numbers(run: &str) -> Result<Box<dyn Run>, Unavailable> {
        let (n, config) = notation(run)?;
        let analyzer =
            Analyzer::builder().add(NumberScanner::new(n).with_opennlp_parity(true)).build().map_err(broken)?;
        Ok(Box::new(Numeric { analyzer, config, layer: Compared::Numbers }))
    }

    pub(crate) fn open_quantities(run: &str) -> Result<Box<dyn Run>, Unavailable> {
        let (n, config) = notation(run)?;
        let analyzer =
            Analyzer::builder().add(QuantityExtractor::new(n).with_opennlp_parity(true)).build().map_err(broken)?;
        Ok(Box::new(Numeric { analyzer, config, layer: Compared::Quantities }))
    }

    pub(crate) fn open_money(run: &str, layer: MoneyLayer, res: &Resources) -> Result<Box<dyn Run>, Unavailable> {
        let (n, config) = notation(run)?;
        let target = Currency::parse(TARGET).ok_or_else(|| broken("the target currency"))?;
        let as_of = Date::parse(AS_OF).ok_or_else(|| broken("the as-of date"))?;
        let rates = EcbRates::load(res.rates()).map_err(|e| broken(format!("{}: {e}", res.rates().display())))?;
        let analyzer = Analyzer::builder()
            .add(MoneyExtractor::new(n).with_opennlp_parity(true))
            .step(MoneyConverter::new(rates, target, as_of).bound())
            .build()
            .map_err(broken)?;
        Ok(Box::new(Numeric { analyzer, config, layer: Compared::Money(layer) }))
    }
}

impl Run for Numeric {
    fn annotate(&self, text: &str, _golden: Option<&Golden>) -> Result<Document, RustError> {
        analyze(&self.analyzer, text)
    }

    fn compare(&self, doc: &Document, golden: &Golden) -> Outcome {
        if config(golden, "notation") != Some(self.config) {
            return Outcome::Skipped(format!("the golden was made in notation {:?}", config(golden, "notation")));
        }
        match self.layer {
            Compared::Numbers => outcome(matches_golden::<Numbers>(doc, golden, Tolerance::exact())),
            Compared::Quantities => outcome(matches_golden::<Quantities>(doc, golden, Tolerance::exact())),
            Compared::Money(MoneyLayer::Extracted) => {
                outcome(matches_golden_key(doc, &Key::<Money>::DEFAULT, golden, Tolerance::exact()))
            }
            Compared::Money(MoneyLayer::Converted) => {
                let same = config(golden, "target") == Some(TARGET)
                    && config(golden, "as_of") == Some(AS_OF)
                    && config(golden, "snapshot") == Some(SNAPSHOT);
                if !same {
                    return Outcome::Skipped("the golden converts to another target, date or snapshot".into());
                }
                outcome(matches_golden_key(doc, &CONVERTED_MONEY, golden, Tolerance::exact()))
            }
        }
    }
}
