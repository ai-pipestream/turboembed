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
use crate::clauses::{ClauseBoundaries, ClauseCosts};
use crate::coref::{CorefBoundaries, CorefCohesion};
use crate::drift::{Drift, DriftBoundaries, Score};
use crate::entity::{EntityBoundaries, EntityContinuity, Keying};
use crate::scorer::{DistilledBoundaryScorer, DistilledWeights};
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

/// Offers `ClauseBoundaries` as "clause-boundaries", writing the default
/// `ClauseCuts` instance. Options, each defaulting to `ClauseCosts::default()`:
/// `clause`, `subordinate`, `phrase`, and `token`.
pub struct ClauseCutProvider;

impl Provider<dyn Step> for ClauseCutProvider {
    fn name(&self) -> &'static str {
        "clause-boundaries"
    }

    fn supports(&self, spec: &Spec) -> bool {
        only(spec, &["clause", "subordinate", "phrase", "token"])
    }

    fn create(&self, spec: &Spec) -> Result<Box<dyn Step>, Error> {
        let p = self.name();
        let d = ClauseCosts::default();
        let costs = ClauseCosts {
            clause: number(p, spec, "clause", d.clause)?,
            subordinate: number(p, spec, "subordinate", d.subordinate)?,
            phrase: number(p, spec, "phrase", d.phrase)?,
            token: number(p, spec, "token", d.token)?,
        };
        Ok(Box::new(Bound::new(ClauseBoundaries { costs })))
    }
}

/// Offers `CorefBoundaries` as "coref-boundaries", writing the `coref`
/// `Boundaries` instance. Options default to `CorefCohesion::default()`.
pub struct CorefBoundariesProvider;

impl Provider<dyn Step> for CorefBoundariesProvider {
    fn name(&self) -> &'static str {
        "coref-boundaries"
    }

    fn supports(&self, spec: &Spec) -> bool {
        only(
            spec,
            &[
                "chain_penalty",
                "adjacent_penalty",
                "pronoun_penalty",
                "break_reward",
                "max_sentence_distance",
                "pronoun_window",
                "shift_threshold",
            ],
        )
    }

    fn create(&self, spec: &Spec) -> Result<Box<dyn Step>, Error> {
        let p = self.name();
        let d = CorefCohesion::default();
        let cohesion = CorefCohesion {
            chain_penalty: number(p, spec, "chain_penalty", d.chain_penalty)?,
            adjacent_penalty: number(p, spec, "adjacent_penalty", d.adjacent_penalty)?,
            pronoun_penalty: number(p, spec, "pronoun_penalty", d.pronoun_penalty)?,
            break_reward: number(p, spec, "break_reward", d.break_reward)?,
            max_sentence_distance: number(p, spec, "max_sentence_distance", d.max_sentence_distance)?,
            pronoun_window: number(p, spec, "pronoun_window", d.pronoun_window)?,
            shift_threshold: number(p, spec, "shift_threshold", d.shift_threshold)?,
        };
        Ok(Box::new(CorefBoundaries { cohesion }.bound()))
    }
}

/// Offers `EntityBoundaries` as "entity-boundaries", writing the `entity`
/// `Boundaries` instance. Options default to `EntityContinuity::default()`:
/// `per_entity`, `max_gap` (at least one), `cap`, and `kind` and `chains`
/// (`true` or `false`, the entity keying).
pub struct EntityBoundariesProvider;

impl Provider<dyn Step> for EntityBoundariesProvider {
    fn name(&self) -> &'static str {
        "entity-boundaries"
    }

    fn supports(&self, spec: &Spec) -> bool {
        only(spec, &["per_entity", "max_gap", "cap", "kind", "chains"])
    }

    fn create(&self, spec: &Spec) -> Result<Box<dyn Step>, Error> {
        let p = self.name();
        let d = EntityContinuity::default();
        let continuity = EntityContinuity {
            per_entity: number(p, spec, "per_entity", d.per_entity)?,
            max_gap: number(p, spec, "max_gap", d.max_gap)?,
            cap: number(p, spec, "cap", d.cap)?,
            keying: Keying {
                kind: number(p, spec, "kind", d.keying.kind)?,
                chains: number(p, spec, "chains", d.keying.chains)?,
            },
        };
        Ok(Box::new(EntityBoundaries { continuity }.bound()))
    }
}

/// Offers `DistilledBoundaryScorer` as "distilled-boundaries", combining
/// `structure`, `drift`, `sentiment`, `coref`, `lexical` token overlap,
/// `entity` surface continuity, `bias`, and the `entity_runs` instance
/// weight into the default `Boundaries` instance.
pub struct DistilledBoundariesProvider;

impl Provider<dyn Step> for DistilledBoundariesProvider {
    fn name(&self) -> &'static str {
        "distilled-boundaries"
    }

    fn supports(&self, spec: &Spec) -> bool {
        only(spec, &["structure", "drift", "sentiment", "coref", "lexical", "entity", "bias", "entity_runs"])
    }

    fn create(&self, spec: &Spec) -> Result<Box<dyn Step>, Error> {
        let p = self.name();
        let d = DistilledWeights::default();
        let weights = DistilledWeights {
            structure: Weight(number(p, spec, "structure", d.structure.0)?),
            drift: Weight(number(p, spec, "drift", d.drift.0)?),
            sentiment: Weight(number(p, spec, "sentiment", d.sentiment.0)?),
            coref: Weight(number(p, spec, "coref", d.coref.0)?),
            lexical_overlap: number(p, spec, "lexical", d.lexical_overlap)?,
            entity_continuity: number(p, spec, "entity", d.entity_continuity)?,
            bias: number(p, spec, "bias", d.bias)?,
            entity_runs: Weight(number(p, spec, "entity_runs", d.entity_runs.0)?),
        };
        Ok(Box::new(DistilledBoundaryScorer { weights }.bound()))
    }
}

/// Offers `WeighBoundaries` (or `WeighAllBoundaries` when `sentiment` is
/// given, or `DistilledBoundaryScorer` when `coref`, `lexical`, `entity`,
/// `bias` or `entity_runs` is given) as "weigh-boundaries". Options
/// `structure` and `drift` default to `1000`; `sentiment`, `coref`,
/// `lexical`, `entity`, and `bias` default to `0` (off by default);
/// `entity_runs` defaults to `1000`, which counts the `entity` instance
/// only where a pipeline writes it.
pub struct WeighProvider;

impl Provider<dyn Step> for WeighProvider {
    fn name(&self) -> &'static str {
        "weigh-boundaries"
    }

    fn supports(&self, spec: &Spec) -> bool {
        only(spec, &["structure", "drift", "sentiment", "coref", "lexical", "entity", "bias", "entity_runs"])
    }

    fn create(&self, spec: &Spec) -> Result<Box<dyn Step>, Error> {
        let p = self.name();
        let structure = Weight(number(p, spec, "structure", Weight::ONE.0)?);
        let drift = Weight(number(p, spec, "drift", Weight::ONE.0)?);
        let has_distilled = spec
            .options
            .iter()
            .any(|(k, _)| matches!(k.as_str(), "coref" | "lexical" | "entity" | "bias" | "entity_runs"));
        if has_distilled {
            let weights = DistilledWeights {
                structure,
                drift,
                sentiment: Weight(number(p, spec, "sentiment", 0)?),
                coref: Weight(number(p, spec, "coref", 0)?),
                lexical_overlap: number(p, spec, "lexical", 0)?,
                entity_continuity: number(p, spec, "entity", 0)?,
                bias: number(p, spec, "bias", 0)?,
                entity_runs: Weight(number(p, spec, "entity_runs", Weight::ONE.0)?),
            };
            Ok(Box::new(DistilledBoundaryScorer { weights }.bound()))
        } else if spec.options.iter().any(|(k, _)| k == "sentiment") {
            let sentiment = Weight(number(p, spec, "sentiment", 0)?);
            let w = WeighAllBoundaries { structure, drift, sentiment };
            Ok(Box::new(w.bound()))
        } else {
            let w = WeighBoundaries { structure, drift };
            Ok(Box::new(w.bound()))
        }
    }
}
