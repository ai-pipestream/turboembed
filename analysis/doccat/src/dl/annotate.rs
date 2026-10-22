//! Pipeline [`Annotator`] implementations for [`DocumentCategorizerDL`] and
//! [`DlSpanCategorizer`].

use std::marker::PhantomData;
use std::sync::Arc;

use turbo_analysis::{Annotator, Error, Execution, Ix, ModelRef, Refs, Spans, Tag, View};
use turbo_analysis_layers::{Categories, Subwords};

use super::{
    AverageClassificationScoringStrategy, ClassificationScoringStrategy, DlChunkingMode, DlSequenceScorer,
    DocumentCategorizerDL,
};
use crate::layer::{SpanCategorizable, validate_spans};

fn failure(annotator: &'static str, detail: impl Into<String>) -> Error {
    Error::Annotator { annotator, detail: detail.into() }
}

fn check_subwords(
    annotator: &'static str,
    view: &View,
    subwords: &Subwords,
    expected_model: &ModelRef,
) -> Result<(), Error> {
    if subwords.spans.view() != view.id() {
        return Err(failure(annotator, "subwords are not in the given view"));
    }
    if &subwords.model != expected_model {
        return Err(failure(
            annotator,
            format!(
                "subwords are from model {} (revision {}), categorizer expects {} (revision {})",
                subwords.model.id, subwords.model.revision, expected_model.id, expected_model.revision
            ),
        ));
    }
    validate_spans(annotator, "subword", &subwords.spans, view.text())
}

/// Extracts the subword piece IDs from `subwords` that belong to item `unit_ix`
/// of `unit_spans`: any piece overlapping a non-empty fragment (`p_start < frag.end && p_end > frag.start`)
/// or an empty piece inside the fragment (`p_start == p_end && p_start >= frag.start && p_end <= frag.end`).
pub(crate) fn subword_ids_in_unit<L>(subwords: &Subwords, unit_spans: &Spans<L>, unit_ix: Ix<L>) -> Vec<i32> {
    let (p_starts, p_ends) = subwords.spans.columns();
    let n_pieces = subwords.ids.len();
    let ordered = subwords.spans.is_start_ordered();
    let mut out = Vec::new();
    for frag in unit_spans.fragments(unit_ix) {
        if ordered {
            let first = p_starts.partition_point(|&s| s < frag.start);
            let scan_from = first.saturating_sub(1);
            for idx in scan_from..n_pieces {
                let ps = p_starts[idx];
                let pe = p_ends[idx];
                if ps > frag.end {
                    break;
                }
                let inside =
                    if ps == pe { ps >= frag.start && pe <= frag.end } else { ps < frag.end && pe > frag.start };
                if inside {
                    out.push(subwords.ids[idx]);
                }
            }
        } else {
            for idx in 0..n_pieces {
                let ps = p_starts[idx];
                let pe = p_ends[idx];
                let inside =
                    if ps == pe { ps >= frag.start && pe <= frag.end } else { ps < frag.end && pe > frag.start };
                if inside {
                    out.push(subwords.ids[idx]);
                }
            }
        }
    }
    out
}

impl<S, C> Annotator for DocumentCategorizerDL<S, C>
where
    S: DlSequenceScorer + 'static,
    C: ClassificationScoringStrategy + 'static,
{
    const ID: &'static str = "opennlp:doccat-dl";
    const VERSION: &'static str = "1";
    type Reads = (View, Option<Subwords>);
    type Writes = (Categories,);

    fn annotate(&self, (view, subwords): Refs<'_, Self>) -> Result<Self::Writes, Error> {
        let model_ref = self.scorer().model_ref();
        if let Some(sw) = subwords {
            check_subwords(Self::ID, view, sw, &model_ref)?;
        }
        let n_cats = self.number_of_categories();
        match self.options().chunking_mode {
            DlChunkingMode::Subwords => {
                let sw = subwords
                    .ok_or_else(|| failure(Self::ID, "DlChunkingMode::Subwords requires the opennlp:subwords layer"))?;
                if sw.ids.is_empty() || view.text().trim().is_empty() {
                    return Ok((Categories::document(Arc::clone(self.categories()), Tag::NONE, &vec![0.0; n_cats]),));
                }
                let scores = self.categorize_subword_slice(&sw.ids).map_err(|e| failure(Self::ID, e.to_string()))?;
                let best = self.best_index(&scores).map_err(|e| failure(Self::ID, e.to_string()))?;
                Ok((Categories::document(Arc::clone(self.categories()), Tag(best as u16), &scores),))
            }
            DlChunkingMode::WhitespaceWords => {
                if view.text().trim().is_empty() {
                    return Ok((Categories::document(Arc::clone(self.categories()), Tag::NONE, &vec![0.0; n_cats]),));
                }
                let scores = self.categorize_text(view.text()).map_err(|e| failure(Self::ID, e.to_string()))?;
                let best = self.best_index(&scores).map_err(|e| failure(Self::ID, e.to_string()))?;
                Ok((Categories::document(Arc::clone(self.categories()), Tag(best as u16), &scores),))
            }
        }
    }

    fn model(&self) -> Option<ModelRef> {
        Some(self.scorer().model_ref())
    }

    fn execution(&self) -> Option<Execution> {
        self.scorer().execution()
    }

    fn config_hash(&self) -> u64 {
        self.config_hash_value()
    }
}

/// Categorizes each span of a positional layer `L` (`Sentences`, `Structure`,
/// or `Chunks`) with a [`DocumentCategorizerDL`], reading `(View, Option<Subwords>, L)`
/// and writing `Categories<L>` (`turbo:categories<L>`).
pub struct DlSpanCategorizer<
    S: DlSequenceScorer,
    L: SpanCategorizable,
    C: ClassificationScoringStrategy = AverageClassificationScoringStrategy,
> {
    categorizer: Arc<DocumentCategorizerDL<S, C>>,
    _of: PhantomData<fn() -> L>,
}

impl<S: DlSequenceScorer, L: SpanCategorizable, C: ClassificationScoringStrategy> Clone for DlSpanCategorizer<S, L, C> {
    fn clone(&self) -> Self {
        Self { categorizer: Arc::clone(&self.categorizer), _of: PhantomData }
    }
}

impl<S: DlSequenceScorer, L: SpanCategorizable, C: ClassificationScoringStrategy> DlSpanCategorizer<S, L, C> {
    /// Wraps a shared [`DocumentCategorizerDL`] into a span categorizer for
    /// layer `L` (`Chunks`, `Structure`, or `Sentences`).
    pub fn new(categorizer: Arc<DocumentCategorizerDL<S, C>>) -> Self {
        Self { categorizer, _of: PhantomData }
    }

    /// Wraps an owned [`DocumentCategorizerDL`] into a span categorizer for
    /// layer `L`.
    pub fn from_categorizer(categorizer: DocumentCategorizerDL<S, C>) -> Self {
        Self::new(Arc::new(categorizer))
    }

    /// Returns a reference to the underlying [`DocumentCategorizerDL`].
    pub fn categorizer(&self) -> &DocumentCategorizerDL<S, C> {
        &self.categorizer
    }
}

impl<S, L, C> Annotator for DlSpanCategorizer<S, L, C>
where
    S: DlSequenceScorer + 'static,
    L: SpanCategorizable,
    C: ClassificationScoringStrategy + 'static,
{
    const ID: &'static str = "opennlp:doccat-dl-spans";
    const VERSION: &'static str = "1";
    type Reads = (View, Option<Subwords>, L);
    type Writes = (Categories<L>,);

    fn annotate(&self, (view, subwords, units): Refs<'_, Self>) -> Result<Self::Writes, Error> {
        if units.layer_spans().view() != view.id() {
            return Err(failure(Self::ID, "spans are not in the given view"));
        }
        validate_spans(Self::ID, "unit", units.layer_spans(), view.text())?;
        let model_ref = self.categorizer.scorer().model_ref();
        if let Some(sw) = subwords {
            check_subwords(Self::ID, view, sw, &model_ref)?;
        }
        let n_units = units.len();
        let n_cats = self.categorizer.number_of_categories();
        let zeros = vec![0.0f64; n_cats];
        let mut out = Categories::<L>::with_capacity(Arc::clone(self.categorizer.categories()), n_units);

        match self.categorizer.options().chunking_mode {
            DlChunkingMode::Subwords => {
                let sw = subwords
                    .ok_or_else(|| failure(Self::ID, "DlChunkingMode::Subwords requires the opennlp:subwords layer"))?;
                for i in 0..n_units {
                    let ix = Ix::<L>::new(i as u32);
                    let unit_ids = subword_ids_in_unit(sw, units.layer_spans(), ix);
                    if unit_ids.is_empty() {
                        out.push(Tag::NONE, &zeros);
                    } else {
                        let scores = self
                            .categorizer
                            .categorize_subword_slice(&unit_ids)
                            .map_err(|e| failure(Self::ID, e.to_string()))?;
                        let best =
                            self.categorizer.best_index(&scores).map_err(|e| failure(Self::ID, e.to_string()))?;
                        out.push(Tag(best as u16), &scores);
                    }
                }
            }
            DlChunkingMode::WhitespaceWords => {
                for i in 0..n_units {
                    let ix = Ix::<L>::new(i as u32);
                    let mut text = String::new();
                    for frag in units.layer_spans().fragments(ix) {
                        if !text.is_empty() {
                            text.push(' ');
                        }
                        text.push_str(&view.text()[frag.start as usize..frag.end as usize]);
                    }
                    if text.trim().is_empty() {
                        out.push(Tag::NONE, &zeros);
                    } else {
                        let scores =
                            self.categorizer.categorize_text(&text).map_err(|e| failure(Self::ID, e.to_string()))?;
                        let best =
                            self.categorizer.best_index(&scores).map_err(|e| failure(Self::ID, e.to_string()))?;
                        out.push(Tag(best as u16), &scores);
                    }
                }
            }
        }
        Ok((out,))
    }

    fn model(&self) -> Option<ModelRef> {
        Some(self.categorizer.scorer().model_ref())
    }

    fn execution(&self) -> Option<Execution> {
        self.categorizer.scorer().execution()
    }

    fn config_hash(&self) -> u64 {
        self.categorizer.config_hash_value()
    }
}
