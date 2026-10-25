//! Opt-in calibrated outputs for maxent and probabilistic SVM categorizers.
use crate::{DocumentCategorizerME, SvmCategorizer};
use turbo_analysis::{Annotator, Error, ModelRef, Refs, View};
use turbo_analysis_layers::{Calibrated, Categories, Tokens};
use turbo_calibration::{Calibration, Identity, Policy, Predictor, Task};

pub trait CalibratableCategorizer: Annotator<Reads = (View, Tokens), Writes = (Categories,)> {
    fn calibration_labels(&self) -> Vec<String>;
    /// Stable inference configuration; refuse unidentifiable custom features.
    fn calibration_config_hash(&self) -> Result<u64, String>;
}
impl CalibratableCategorizer for DocumentCategorizerME {
    fn calibration_config_hash(&self) -> Result<u64, String> {
        if !self.model().context_generator().custom().is_empty() {
            return Err("calibration cannot identify custom doccat feature generators".into());
        }
        let mut bytes = Vec::new();
        for g in self.model().feature_generators() {
            match g {
                crate::BuiltinFeatureGenerator::BagOfWords(g) => bytes.extend([
                    0,
                    u8::from(g.uses_only_all_letter_tokens()),
                    match g.pattern_mode() {
                        turbo_sequence::PatternMode::Utf16Units => 0,
                        turbo_sequence::PatternMode::CodePoints => 1,
                    },
                ]),
                crate::BuiltinFeatureGenerator::NGram(g) => {
                    bytes.push(1);
                    bytes.extend((g.min_gram() as u64).to_le_bytes());
                    bytes.extend((g.max_gram() as u64).to_le_bytes());
                }
                crate::BuiltinFeatureGenerator::Emoji(_) => bytes.push(2),
            }
        }
        Ok(turbo_analysis::fnv1a(&bytes))
    }
    fn calibration_labels(&self) -> Vec<String> {
        (0..self.number_of_categories())
            .map(|i| self.model().categories().label(turbo_analysis::Tag(i as u16)).to_owned())
            .collect()
    }
}
impl CalibratableCategorizer for SvmCategorizer {
    fn calibration_config_hash(&self) -> Result<u64, String> {
        Ok(self.config_hash())
    }
    fn calibration_labels(&self) -> Vec<String> {
        self.model().categories().to_vec()
    }
}
/// Bind using the producer's actual model and configuration, never just
/// the identity stored inside the calibration being loaded.
pub fn identity<C: CalibratableCategorizer>(categorizer: &C, domain: &str) -> Result<Identity, String> {
    let m = categorizer.model().ok_or("categorizer has no model identity")?;
    let i = Identity {
        model: m.id,
        revision: m.revision,
        sha256: m.sha256,
        task: Task::Categorization,
        domain: domain.to_owned(),
        labels: categorizer.calibration_labels(),
        inference_hash: categorizer.calibration_config_hash()?,
    };
    i.check()?;
    Ok(i)
}
pub struct CalibratedCategorizer<C> {
    categorizer: C,
    predictor: Predictor,
}
impl<C: CalibratableCategorizer> CalibratedCategorizer<C> {
    pub fn new(categorizer: C, artifact: Calibration, domain: &str, policy: Policy) -> Result<Self, String> {
        let expected = identity(&categorizer, domain)?;
        let predictor = Predictor::new(artifact, &expected, policy)?;
        Ok(Self { categorizer, predictor })
    }
    pub fn predictor(&self) -> &Predictor {
        &self.predictor
    }
}
impl<C: CalibratableCategorizer> Annotator for CalibratedCategorizer<C> {
    const ID: &'static str = "turbo-doccat:calibrated";
    const VERSION: &'static str = "1";
    type Reads = (View, Tokens);
    type Writes = (Categories, Calibrated<Categories>);
    fn annotate(&self, inputs: Refs<'_, Self>) -> Result<Self::Writes, Error> {
        let (raw,) = self.categorizer.annotate(inputs)?;
        let mut calibrated = Calibrated::new(&self.predictor);
        for row in raw.iter() {
            calibrated
                .push(&self.predictor, row.scores)
                .map_err(|detail| Error::Annotator { annotator: Self::ID, detail })?;
        }
        Ok((raw, calibrated))
    }
    fn model(&self) -> Option<ModelRef> {
        self.categorizer.model()
    }
    fn config_hash(&self) -> u64 {
        self.predictor.config_hash()
    }
}
