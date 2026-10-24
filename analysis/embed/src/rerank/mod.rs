//! Cross-encoder reranking of a vector-index shortlist of Levels units.
use std::{path::Path, sync::Mutex};
use turbo::rerank::{RerankOptions, Reranker};
use turbo_analysis::{Annotator, Bound, Error, Key, Layer, ModelRef, Refs, View};
use turbo_analysis_layers::Levels;

mod layer;
pub use layer::{Reranked, Shortlist};

pub struct RerankAnnotator {
    session: Mutex<Reranker>,
    model: ModelRef,
    levels: Key<Levels>,
    options: RerankOptions,
}
impl RerankAnnotator {
    pub fn open(bundle: &Path, threads: u32, options: RerankOptions) -> Result<Self, Error> {
        let session = Reranker::open(bundle, threads).map_err(|e| failure(e.message))?;
        let id = session.identity();
        let model =
            ModelRef { id: id.model_id.clone(), revision: id.revision.clone(), sha256: id.manifest_sha256.clone() };
        Ok(Self { session: Mutex::new(session), model, levels: Key::DEFAULT, options })
    }

    /// Bind both the levels input and its explicit shortlist references.
    /// Other slots can be rebound through the returned Bound step.
    pub fn on(mut self, key: Key<Levels>) -> Bound<Self> {
        self.levels = key;
        Bound::new(self).read(1, key.name)
    }
}
fn failure(detail: impl Into<String>) -> Error {
    Error::Annotator { annotator: "turbo:rerank", detail: detail.into() }
}
impl Annotator for RerankAnnotator {
    const ID: &'static str = "turbo:rerank";
    const VERSION: &'static str = "1";
    type Reads = (View, Levels, Shortlist<Levels>);
    type Writes = (Reranked<Levels>,);
    fn model(&self) -> Option<ModelRef> {
        Some(self.model.clone())
    }
    fn config_hash(&self) -> u64 {
        turbo_analysis::Fnv1a::new()
            .u32(self.options.query_tokens)
            .u32(self.options.max_tokens)
            .u32(self.options.stride)
            .str(self.levels.name)
            .finish()
    }
    fn annotate(&self, (view, levels, shortlist): Refs<'_, Self>) -> Result<Self::Writes, Error> {
        levels.check().map_err(failure)?;
        levels.spans.check(Levels::OVERLAP).map_err(failure)?;
        shortlist.check().map_err(failure)?;
        if levels.view() != view.id() || shortlist.source.id() != self.levels.id() {
            return Err(failure("rerank view or source layer instance mismatch"));
        }
        let mut texts = Vec::with_capacity(shortlist.len());
        for &item in &shortlist.items {
            if item.get() >= levels.len() {
                return Err(failure("shortlist item exceeds levels length"));
            }
            let mut pieces = Vec::new();
            for span in levels.spans.fragments(item) {
                pieces.push(
                    view.text()
                        .get(span.start as usize..span.end as usize)
                        .ok_or_else(|| failure("level span is outside the view or splits UTF-8"))?,
                );
            }
            texts.push(pieces.join(" "));
        }
        let pairs: Vec<_> = texts.iter().map(|p| (shortlist.query.as_str(), p.as_str())).collect();
        let scores = self
            .session
            .lock()
            .map_err(|_| failure("reranker lock poisoned"))?
            .score(&pairs, self.options)
            .map_err(|e| failure(e.message))?
            .scores;
        let mut order: Vec<usize> = (0..scores.len()).collect();
        order.sort_by(|&a, &b| scores[b].total_cmp(&scores[a]).then(a.cmp(&b)));
        let result = Reranked {
            source: shortlist.source,
            items: order.iter().map(|&i| shortlist.items[i]).collect(),
            scores: order.iter().map(|&i| scores[i]).collect(),
            model: self.model.clone(),
        };
        result.check().map_err(failure)?;
        Ok((result,))
    }
}
