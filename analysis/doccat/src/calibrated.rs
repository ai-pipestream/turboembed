//! Calibrated and conformal document categorization sidecar.

use turbo_analysis::{Annotator, Error, ModelRef, Refs};
use turbo_analysis_layers::{Calibrated, Categories};
use turbo_calibration::{Calibration, Identity, Policy, Predictor, Task};

use crate::dl::{DlSequenceScorer, DocumentCategorizerDL};

/// A single-label document categorizer whose `Categories` output probabilities
/// can be post-hoc calibrated and wrapped with conformal prediction sets.
pub trait CalibratableCategorizer: Annotator<Writes = (Categories,)> {
    fn calibration_labels(&self) -> Vec<String>;
    fn calibration_sha256(&self) -> Option<String> {
        self.model().map(|m| m.sha256.to_string())
    }
    fn calibration_inference_hash(&self) -> u64 {
        self.config_hash()
    }
}

impl<S: DlSequenceScorer + 'static> CalibratableCategorizer for DocumentCategorizerDL<S> {
    fn calibration_labels(&self) -> Vec<String> {
        self.scorer().categories().to_vec()
    }
}

pub fn identity<C: CalibratableCategorizer>(categorizer: &C, domain: impl Into<String>) -> Result<Identity, String> {
    let model = categorizer.model().ok_or("calibrated categorizer requires model provenance")?;
    let sha256 = categorizer.calibration_sha256().unwrap_or_else(|| model.sha256.to_string());
    let id = Identity {
        model: model.id.to_string(),
        revision: model.revision.to_string(),
        sha256,
        task: Task::Categorization,
        domain: domain.into(),
        labels: categorizer.calibration_labels(),
        inference_hash: categorizer.calibration_inference_hash(),
    };
    id.check()?;
    Ok(id)
}

#[derive(Clone, Debug)]
pub struct CalibratedCategorizer<C> {
    inner: C,
    predictor: Predictor,
}

impl<C: CalibratableCategorizer> CalibratedCategorizer<C> {
    pub fn new(inner: C, artifact: Calibration, domain: impl Into<String>, policy: Policy) -> Result<Self, String> {
        let expected = identity(&inner, domain)?;
        Ok(Self { inner, predictor: Predictor::new(artifact, &expected, policy)? })
    }

    pub fn inner(&self) -> &C {
        &self.inner
    }

    pub fn predictor(&self) -> &Predictor {
        &self.predictor
    }
}

impl<C: CalibratableCategorizer> Annotator for CalibratedCategorizer<C> {
    const ID: &'static str = C::ID;
    const VERSION: &'static str = C::VERSION;
    type Reads = C::Reads;
    type Writes = (Categories, Calibrated<Categories>);

    fn annotate(&self, input: Refs<'_, Self>) -> Result<Self::Writes, Error> {
        let (categories,) = self.inner.annotate(input)?;
        let mut sidecar = Calibrated::new(&self.predictor);
        for row in categories.iter() {
            sidecar
                .push(&self.predictor, row.scores)
                .map_err(|detail| Error::Annotator { annotator: Self::ID, detail })?;
        }
        Ok((categories, sidecar))
    }

    fn model(&self) -> Option<ModelRef> {
        self.inner.model()
    }

    fn config_hash(&self) -> u64 {
        self.predictor.config_hash()
    }
}
