//! The pipeline annotators over a [`ZeroShotCategorizer`].

use std::marker::PhantomData;

use turbo_analysis::{Annotator, Error, Ix, ModelRef, Refs, Tag, View};
use turbo_analysis_layers::Categories;

use super::{NliScorer, ZeroShotCategorizer, best_index};
use crate::layer::{SpanCategorizable, validate_spans};

fn failure(annotator: &'static str, detail: String) -> Error {
    Error::Annotator { annotator, detail }
}

/// Writes the document's `Categories` from a
/// [`ZeroShotCategorizer`]: the whole view text is the premise.
pub struct ZeroShotAnnotator<S: NliScorer> {
    categorizer: ZeroShotCategorizer<S>,
}

impl<S: NliScorer> ZeroShotAnnotator<S> {
    pub fn new(categorizer: ZeroShotCategorizer<S>) -> Self {
        ZeroShotAnnotator { categorizer }
    }

    pub fn categorizer(&self) -> &ZeroShotCategorizer<S> {
        &self.categorizer
    }
}

impl<S: NliScorer + Send + Sync + 'static> Annotator for ZeroShotAnnotator<S> {
    const ID: &'static str = "turbo:zero-shot";
    const VERSION: &'static str = "1";
    type Reads = (View,);
    type Writes = (Categories,);

    fn annotate(&self, (view,): Refs<'_, Self>) -> Result<Self::Writes, Error> {
        let scores = scores_or_zero(Self::ID, &self.categorizer, view.text())?;
        Ok((Categories::document(self.categorizer.categories(), Tag(best_index(&scores) as u16), &scores),))
    }

    fn model(&self) -> Option<ModelRef> {
        Some(self.categorizer.scorer().model_ref())
    }

    fn config_hash(&self) -> u64 {
        self.categorizer.config_hash()
    }
}

/// Writes `Categories<L>` item by item from a
/// [`ZeroShotCategorizer`]: each item's fragment text is the
/// premise. An item with no text scores zero everywhere
/// instead of failing the document.
pub struct ZeroShotSpanAnnotator<S: NliScorer, L: SpanCategorizable> {
    categorizer: ZeroShotCategorizer<S>,
    _of: PhantomData<fn() -> L>,
}

impl<S: NliScorer, L: SpanCategorizable> ZeroShotSpanAnnotator<S, L> {
    pub fn new(categorizer: ZeroShotCategorizer<S>) -> Self {
        ZeroShotSpanAnnotator { categorizer, _of: PhantomData }
    }

    pub fn categorizer(&self) -> &ZeroShotCategorizer<S> {
        &self.categorizer
    }
}

impl<S: NliScorer + Send + Sync + 'static, L: SpanCategorizable> Annotator for ZeroShotSpanAnnotator<S, L> {
    const ID: &'static str = "turbo:zero-shot-spans";
    const VERSION: &'static str = "1";
    type Reads = (View, L);
    type Writes = (Categories<L>,);

    fn annotate(&self, (view, units): Refs<'_, Self>) -> Result<Self::Writes, Error> {
        if units.layer_spans().view() != view.id() {
            return Err(failure(Self::ID, "spans are not in the given view".into()));
        }
        validate_spans(Self::ID, "unit", units.layer_spans(), view.text())?;
        let mut out = Categories::<L>::with_capacity(self.categorizer.categories(), units.len());
        for i in 0..units.len() {
            let mut text = String::new();
            for frag in units.layer_spans().fragments(Ix::new(i as u32)) {
                if !text.is_empty() {
                    text.push(' ');
                }
                text.push_str(&view.text()[frag.start as usize..frag.end as usize]);
            }
            let scores = scores_or_zero(Self::ID, &self.categorizer, &text)?;
            out.push(Tag(best_index(&scores) as u16), &scores);
        }
        Ok((out,))
    }

    fn model(&self) -> Option<ModelRef> {
        Some(self.categorizer.scorer().model_ref())
    }

    fn config_hash(&self) -> u64 {
        self.categorizer.config_hash()
    }
}

/// A blank premise is not evidence for any label: in an
/// annotator it scores zero everywhere rather than failing
/// the whole document.
fn scores_or_zero<S: NliScorer>(
    annotator: &'static str,
    categorizer: &ZeroShotCategorizer<S>,
    text: &str,
) -> Result<Vec<f64>, Error> {
    if text.trim().is_empty() {
        return Ok(vec![0.0; categorizer.labels().len()]);
    }
    categorizer.categorize(text).map_err(|e| failure(annotator, e.to_string()))
}
