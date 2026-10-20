use super::DenseClassifierModel;
use std::{marker::PhantomData, sync::Arc};
use turbo_analysis::{Annotator, Error, ModelRef, Refs, Tag};
use turbo_analysis_layers::{Categories, CategoryScope, Vectors};

/// Scores every present vector; refuses a different encoder, missing rows,
/// invalid dimensions, or non-finite data. Sigmoid scores are independent;
/// Categories stores the full score row and its highest scoring label.
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
            let best = p.iter().enumerate().fold(0, |best, (i, v)| if *v > p[best] { i } else { best });
            out.push(Tag(best as u16), &p);
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
