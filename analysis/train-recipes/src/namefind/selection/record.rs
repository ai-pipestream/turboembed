use serde::{Deserialize, Serialize};
use turbo_formats::evaluate::{FMeasure, Fold};
use turbo_train_components::namefind::NameFold;

use super::{Setting, invalid};
use crate::Result;

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Scores {
    pub predicted: u64,
    pub reference: u64,
    pub correct: u64,
    pub precision: f64,
    pub recall: f64,
    pub f_measure: f64,
}

impl Scores {
    pub(super) fn of(f: &FMeasure) -> Self {
        Self {
            predicted: f.selected(),
            reference: f.target(),
            correct: f.true_positives(),
            precision: f.precision(),
            recall: f.recall(),
            f_measure: f.f_measure(),
        }
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FoldCard {
    pub index: usize,
    pub training_document_indices: Vec<usize>,
    pub test_document_indices: Vec<usize>,
    pub training_samples: usize,
    pub test_samples: usize,
    pub events: usize,
    pub event_checksum: u32,
    pub scores: Scores,
}

impl FoldCard {
    pub(super) fn new(f: &NameFold, partition: Fold) -> Self {
        Self {
            index: f.index,
            training_document_indices: partition.training,
            test_document_indices: partition.test,
            training_samples: f.training_samples,
            test_samples: f.test_samples,
            events: f.events,
            event_checksum: f.event_checksum,
            scores: Scores::of(&f.fmeasure),
        }
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Candidate {
    pub setting: Setting,
    pub folds: Vec<FoldCard>,
    pub merged: Scores,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SelectionCard {
    pub version: u32,
    pub model_id: String,
    pub training_samples_sha256: String,
    pub training_documents: usize,
    pub training_samples: usize,
    pub folds: usize,
    pub tie_rule: String,
    pub selected: usize,
    pub candidates: Vec<Candidate>,
}

pub(super) fn best(candidates: &[Candidate]) -> Result<usize> {
    if candidates.is_empty() || candidates.iter().any(|c| !c.merged.f_measure.is_finite()) {
        return Err(invalid("empty or nonfinite selection scores"));
    }
    let mut best = 0;
    for index in 1..candidates.len() {
        if candidates[index].merged.f_measure > candidates[best].merged.f_measure {
            best = index;
        }
    }
    Ok(best)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ties_use_grid_order_and_nan_is_rejected() {
        let empty = FMeasure::default();
        let mut candidates: Vec<_> = super::super::GRID
            .into_iter()
            .map(|setting| Candidate { setting, folds: Vec::new(), merged: Scores::of(&empty) })
            .collect();
        assert_eq!(best(&candidates).unwrap(), 0);
        candidates[2].merged.f_measure = 0.8;
        candidates[3].merged.f_measure = 0.8;
        assert_eq!(best(&candidates).unwrap(), 2);
        candidates[1].merged.f_measure = f64::NAN;
        assert!(best(&candidates).is_err());
        assert!(best(&[]).is_err());
    }
}
