//! The boundary steps offered to pipelines built from configuration. Each
//! option is parsed into the step's own typed settings; an unknown option
//! or a value that does not parse is refused, never ignored.
//!
//! A pipeline with drift lists three of them: "structure-boundaries" with
//! `instance` `structure`, "drift-boundaries", and "weigh-boundaries",
//! which writes the default instance the chunker reads.

use std::str::FromStr;

use turbo_analysis::annotate::Step;
use turbo_analysis::registry::{Provider, Spec};
use turbo_analysis::{Bound, Error};

use crate::boundaries::{Costs, StructureBoundaries};
use crate::drift::{Drift, DriftBoundaries, Score};
use crate::sentiment::{SentimentBoundaries, SentimentShift};
use crate::weigh::{STRUCTURE, WeighAllBoundaries, WeighBoundaries, Weight};

/// Whether every option of `spec` is one of `keys` and there is no
/// location.
fn only(spec: &Spec, keys: &[&str]) -> bool {
    spec.location.is_none() && spec.options.iter().all(|(k, _)| keys.contains(&k.as_str()))
}

/// Option `key` as a number, or `default` when it is not given.
fn number<T: FromStr + Copy>(provider: &str, spec: &Spec, key: &str, default: T) -> Result<T, Error> {
    match spec.options.iter().find(|(k, _)| k == key) {
        None => Ok(default),
        Some((_, v)) => v.parse().map_err(|_| Error::Pipeline(format!("{provider}: {key} {v:?} is not a number"))),
    }
}

/// Offers `StructureBoundaries` as "structure-boundaries", with its
/// default costs. Option `instance`: `default` (the default), when
/// structure is the only boundary signal, or `structure`, beside others.
pub struct StructureBoundariesProvider;

impl Provider<dyn Step> for StructureBoundariesProvider {
    fn name(&self) -> &'static str {
        "structure-boundaries"
    }

    fn supports(&self, spec: &Spec) -> bool {
        only(spec, &["instance"]) && matches!(spec.option("instance", "default"), "default" | "structure")
    }

    fn create(&self, spec: &Spec) -> Result<Box<dyn Step>, Error> {
        let step = Bound::new(StructureBoundaries { costs: Costs::default() });
        Ok(match spec.option("instance", "default") {
            "default" => Box::new(step),
            "structure" => Box::new(step.write(0, STRUCTURE)),
            other => return Err(Error::Pipeline(format!("structure-boundaries: instance {other:?}"))),
        })
    }
}

/// Offers `DriftBoundaries` as "drift-boundaries", writing the `drift`
/// instance. Options, each defaulting to `Drift::default()`: `window` (a
/// whole number of sentences, at least one), `smoothing` (a whole number
/// of gaps), `floor` and `shift` (whole millionths of a cosine), `scale`
/// (whole thousandths of a `Boundaries::UNIT`, what a score of one takes
/// off a gap) and `score` (`depth` or `distance`). A negative number is
/// refused, as is a window of zero.
pub struct DriftProvider;

impl Provider<dyn Step> for DriftProvider {
    fn name(&self) -> &'static str {
        "drift-boundaries"
    }

    fn supports(&self, spec: &Spec) -> bool {
        only(spec, &["window", "smoothing", "floor", "shift", "scale", "score"])
    }

    fn create(&self, spec: &Spec) -> Result<Box<dyn Step>, Error> {
        let p = self.name();
        let d = Drift::default();
        let score = match spec.option("score", "depth") {
            "depth" => Score::Depth,
            "distance" => Score::Distance,
            s => return Err(Error::Pipeline(format!("{p}: score {s:?}: depth or distance"))),
        };
        let drift = Drift {
            window: number(p, spec, "window", d.window)?,
            smoothing: number(p, spec, "smoothing", d.smoothing)?,
            score,
            floor: number(p, spec, "floor", d.floor)?,
            scale: number(p, spec, "scale", d.scale)?,
            shift: number(p, spec, "shift", d.shift)?,
        };
        Ok(Box::new(DriftBoundaries { drift }.bound()))
    }
}

/// Offers `SentimentBoundaries` as "sentiment-boundaries", writing the
/// `sentiment` instance. Options, each defaulting to
/// `SentimentShift::default()`: `window`, `polarity_weight`,
/// `aspect_weight`, `incivility_weight`, `floor`, `scale`, and `shift`.
pub struct SentimentBoundariesProvider;

impl Provider<dyn Step> for SentimentBoundariesProvider {
    fn name(&self) -> &'static str {
        "sentiment-boundaries"
    }

    fn supports(&self, spec: &Spec) -> bool {
        only(spec, &["window", "polarity_weight", "aspect_weight", "incivility_weight", "floor", "scale", "shift"])
    }

    fn create(&self, spec: &Spec) -> Result<Box<dyn Step>, Error> {
        let p = self.name();
        let d = SentimentShift::default();
        let shift = SentimentShift {
            window: number(p, spec, "window", d.window)?,
            polarity_weight: number(p, spec, "polarity_weight", d.polarity_weight)?,
            aspect_weight: number(p, spec, "aspect_weight", d.aspect_weight)?,
            incivility_weight: number(p, spec, "incivility_weight", d.incivility_weight)?,
            floor: number(p, spec, "floor", d.floor)?,
            scale: number(p, spec, "scale", d.scale)?,
            shift: number(p, spec, "shift", d.shift)?,
        };
        Ok(Box::new(SentimentBoundaries { shift }.bound()))
    }
}

/// Offers `WeighBoundaries` (or `WeighAllBoundaries` when `sentiment` is
/// given) as "weigh-boundaries". Options `structure` and `drift` default
/// to `1000`; `sentiment` defaults to `0` (off by default).
pub struct WeighProvider;

impl Provider<dyn Step> for WeighProvider {
    fn name(&self) -> &'static str {
        "weigh-boundaries"
    }

    fn supports(&self, spec: &Spec) -> bool {
        only(spec, &["structure", "drift", "sentiment"])
    }

    fn create(&self, spec: &Spec) -> Result<Box<dyn Step>, Error> {
        let p = self.name();
        let structure = Weight(number(p, spec, "structure", Weight::ONE.0)?);
        let drift = Weight(number(p, spec, "drift", Weight::ONE.0)?);
        if spec.options.iter().any(|(k, _)| k == "sentiment") {
            let sentiment = Weight(number(p, spec, "sentiment", 0)?);
            let w = WeighAllBoundaries { structure, drift, sentiment };
            Ok(Box::new(w.bound()))
        } else {
            let w = WeighBoundaries { structure, drift };
            Ok(Box::new(w.bound()))
        }
    }
}
