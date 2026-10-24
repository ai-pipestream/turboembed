use super::{DenseClassifierModel, OutputKind};
use std::{marker::PhantomData, sync::Arc};
use turbo_analysis::{Annotator, Error, ModelRef, Refs};
use turbo_analysis_layers::{Categories, CategoryScope, Vectors};

/// Scores every present vector; refuses a different encoder, missing rows,
/// invalid dimensions, or non-finite data. Softmax writes the argmax label;
/// Sigmoid applies per-label thresholds and writes the emitted label set
/// into `Categories::selected` (with `Tag::NONE` as best category when no
/// label clears its threshold).
pub struct DenseClassifierAnnotator<L: CategoryScope> {
    model: DenseClassifierModel,
    identity: ModelRef,
    hash: u64,
    _of: PhantomData<fn() -> L>,
}
impl<L: CategoryScope> DenseClassifierAnnotator<L> {
    pub fn new(model: DenseClassifierModel) -> Self {
        let bytes = model.to_bytes();
        let hash = crate::centroids::fnv1a(&bytes);
        let identity = ModelRef {
            id: "turbo:dense-classifier".into(),
            revision: "1".into(),
            sha256: turbo::bundle::sha256_hex(&bytes),
        };
        Self { model, identity, hash, _of: PhantomData }
    }
    pub fn classifier(&self) -> &DenseClassifierModel {
        &self.model
    }
}
impl<L: CategoryScope> Annotator for DenseClassifierAnnotator<L> {
    const ID: &'static str = "turbo:dense-classifier";
    type Reads = (Vectors<L>,);
    type Writes = (Categories<L>,);
    fn annotate(&self, (vectors,): Refs<'_, Self>) -> Result<Self::Writes, Error> {
        let fail = |detail| Error::Annotator { annotator: Self::ID, detail };
        self.model.check_vectors(vectors).map_err(fail)?;
        let mut out = Categories::new(Arc::clone(self.model.head().tags()));
        for x in vectors.data.chunks_exact(vectors.dim as usize) {
            let p = self.model.probabilities(x).map_err(fail)?;
            let (best, selected) = self.model.select_tags(&p).map_err(fail)?;
            match self.model.output() {
                OutputKind::Softmax => out.push(best, &p),
                OutputKind::Sigmoid => out.push_selected(best, &selected, &p),
            }
        }
        Ok((out,))
    }
    fn config_hash(&self) -> u64 {
        self.hash
    }
    fn model(&self) -> Option<ModelRef> {
        Some(self.identity.clone())
    }
}
