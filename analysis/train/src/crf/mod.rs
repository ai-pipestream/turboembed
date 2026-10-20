//! First-order CRF sequence training using constrained forward/backward
//! gradients and the shared L-BFGS optimizer.
mod objective;
use crate::{JavaString, QnParams};
pub use objective::{CrfObjective, IndexedSequence};
use std::collections::{BTreeMap, HashMap};
use turbo_sequence::crf::{CrfConstraints, CrfSequenceModel, FeatureRow};
use turbo_sequence::namefind::NameCodec;

#[derive(Clone, Debug)]
pub struct CrfTrainingSequence {
    pub features: Vec<FeatureRow>,
    pub outcomes: Vec<String>,
}
#[derive(Clone, Debug)]
pub struct CrfTrainer {
    pub codec: NameCodec,
    pub cutoff: usize,
    pub l2: f64,
    pub iterations: usize,
}
impl Default for CrfTrainer {
    fn default() -> Self {
        Self { codec: NameCodec::Bio, cutoff: 1, l2: 1.0, iterations: 100 }
    }
}
impl CrfTrainer {
    pub fn train(&self, samples: &[CrfTrainingSequence]) -> Result<CrfSequenceModel, String> {
        if samples.is_empty()
            || self.iterations == 0
            || self.iterations > i32::MAX as usize
            || !self.l2.is_finite()
            || self.l2 < 0.0
        {
            return Err("invalid CRF training parameters".into());
        }
        let mut counts = BTreeMap::<JavaString, usize>::new();
        let mut outcomes = BTreeMap::<String, usize>::new();
        for s in samples {
            if s.features.len() != s.outcomes.len() {
                return Err("features/outcomes length mismatch".into());
            }
            for y in &s.outcomes {
                outcomes.entry(y.clone()).or_insert(0);
            }
            for row in &s.features {
                for (feature, value) in row {
                    if !value.is_finite() {
                        return Err("non-finite feature value".into());
                    }
                    *counts.entry(feature.clone()).or_default() += 1;
                }
            }
        }
        // Include all codec states for every observed entity kind, including
        // states missing from a tiny training split.
        let initial: Vec<String> = outcomes.keys().cloned().collect();
        CrfConstraints::new(&initial, self.codec).map_err(|e| e.to_string())?;
        outcomes.entry("other".into()).or_insert(0);
        for label in initial {
            if let Some((kind, _)) = label.rsplit_once('-') {
                let suffixes: &[&str] =
                    if self.codec == NameCodec::Bio { &["start", "cont"] } else { &["start", "cont", "last", "unit"] };
                for suffix in suffixes {
                    outcomes.entry(format!("{kind}-{suffix}")).or_insert(0);
                }
            }
        }
        for (i, index) in outcomes.values_mut().enumerate() {
            *index = i;
        }
        let labels: Vec<_> = outcomes.keys().cloned().collect();
        let features: Vec<_> = counts.into_iter().filter(|(_, count)| *count >= self.cutoff).map(|(f, _)| f).collect();
        let map: HashMap<_, _> = features.iter().enumerate().map(|(i, f)| (f, i)).collect();
        let indexed: Vec<_> = samples
            .iter()
            .map(|s| IndexedSequence {
                features: s
                    .features
                    .iter()
                    .map(|row| row.iter().filter_map(|(f, v)| map.get(f).map(|&i| (i, *v))).collect())
                    .collect(),
                outcomes: s.outcomes.iter().map(|s| outcomes[s]).collect(),
            })
            .collect();
        let mask = CrfConstraints::new(&labels, self.codec).map_err(|e| e.to_string())?;
        let objective = CrfObjective::new(features.len(), mask, indexed, self.l2)?;
        let params = QnParams { iterations: self.iterations as i32, l1_cost: 0.0, l2_cost: 0.0, ..QnParams::default() };
        let (weights, _) =
            crate::qn::minimize_objective(objective.dimension(), &params, &|x, g| objective.value_gradient(x, g))?;
        CrfSequenceModel::new(labels, features, self.codec, weights).map_err(|e| e.to_string())
    }
}
