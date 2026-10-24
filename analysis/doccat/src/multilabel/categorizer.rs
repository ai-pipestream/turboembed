//! The multi-label categorizer at prediction: `categorize` gives every
//! label's independent sigmoid score; the thresholded prediction is
//! every label whose score reaches its own threshold (and `min_score`),
//! capped by `max_labels`. As annotators, [`MultiLabelCategorizer`]
//! writes the document's `turbo:categories` and
//! [`MultiLabelSpanCategorizer`] each span's `Categories<L>`; an item
//! nothing reached is written with best category [`Tag::NONE`], the
//! clean abstention, never a forced argmax.

use std::collections::BTreeMap;
use std::marker::PhantomData;
use std::sync::Arc;

use turbo_analysis::{Annotator, Error, Ix, ModelRef, Refs, Tag, View};
use turbo_analysis_layers::{Categories, Tokens};

use super::model::MultiLabelDoccatModel;
use crate::layer::{SpanCategorizable, extract_tokens, validate_spans};

/// Scores documents with a [`MultiLabelDoccatModel`] and emits every
/// label clearing its threshold.
#[derive(Clone, Debug)]
pub struct MultiLabelCategorizer {
    model: MultiLabelDoccatModel,
}

impl MultiLabelCategorizer {
    /// Creates a `MultiLabelCategorizer` from a [`MultiLabelDoccatModel`].
    pub fn new(model: MultiLabelDoccatModel) -> Self {
        Self { model }
    }

    /// Returns a reference to the underlying model.
    pub fn model(&self) -> &MultiLabelDoccatModel {
        &self.model
    }

    /// Returns a [`MultiLabelSpanCategorizer`] categorizing each span of
    /// layer `L` (`Sentences`, `Structure`, or `Chunks`).
    pub fn for_spans<L: SpanCategorizable>(&self) -> MultiLabelSpanCategorizer<L> {
        MultiLabelSpanCategorizer::new(self.model.clone())
    }

    /// Categorizes tokenized `text` with empty extra information: every
    /// label's independent sigmoid score in `0.0..=1.0`, in label order.
    /// The scores do not sum to 1.0.
    pub fn categorize(&self, text: &[&str]) -> Vec<f64> {
        let extra = BTreeMap::new();
        let mut out = Vec::with_capacity(self.model.num_labels());
        let mut scratch = Vec::new();
        self.model.scores_into(text, &extra, &mut scratch, &mut out);
        out
    }

    /// Categorizes tokenized `text` and `extra_information`, returning
    /// every label's sigmoid score in label order.
    pub fn categorize_with_extra(&self, text: &[&str], extra_information: &BTreeMap<String, String>) -> Vec<f64> {
        let mut out = Vec::with_capacity(self.model.num_labels());
        let mut scratch = Vec::new();
        self.model.scores_into(text, extra_information, &mut scratch, &mut out);
        out
    }

    /// Categorizes into a pre-allocated `out` buffer
    /// (`out.len() == self.number_of_categories()`), reusing
    /// `feature_scratch` to avoid repeated vector allocations.
    pub fn categorize_into(
        &self,
        text: &[&str],
        extra_information: &BTreeMap<String, String>,
        feature_scratch: &mut Vec<String>,
        out: &mut Vec<f64>,
    ) {
        self.model.scores_into(text, extra_information, feature_scratch, out);
    }

    /// The labels `scores` emits, in label order: every label whose
    /// score reaches its threshold and the model's `min_score`, at most
    /// `max_labels` of them. Empty is clean abstention.
    pub fn predicted_tags(&self, scores: &[f64]) -> Vec<Tag> {
        self.model.predicted_tags(scores)
    }

    /// The labels `text` emits, in label order, each with its score.
    pub fn predicted_labels(&self, text: &[&str]) -> Vec<(&str, f64)> {
        let scores = self.categorize(text);
        self.model
            .predicted_tags(&scores)
            .into_iter()
            .map(|t| (self.model.label(t.0 as usize).unwrap_or(""), scores[t.0 as usize]))
            .collect()
    }

    /// The number of labels the model knows.
    pub fn number_of_categories(&self) -> usize {
        self.model.num_labels()
    }

    /// The label name at `index`, or `None` past the end.
    pub fn get_category(&self, index: usize) -> Option<&str> {
        self.model.label(index)
    }

    /// The index of `category`, or `None` when the model has no such label.
    pub fn get_index(&self, category: &str) -> Option<usize> {
        self.model.index(category)
    }
}

impl Annotator for MultiLabelCategorizer {
    const ID: &'static str = "turbo-doccat:multilabel";
    type Reads = (View, Tokens);
    type Writes = (Categories,);

    fn annotate(&self, (view, tokens): Refs<'_, Self>) -> Result<Self::Writes, Error> {
        if tokens.spans.view() != view.id() {
            return Err(Error::Annotator { annotator: Self::ID, detail: "tokens are not in the given view".into() });
        }
        let tok_strs = extract_tokens(Self::ID, tokens, view.text())?;
        let scores = self.categorize(&tok_strs);
        let predicted = self.model.predicted_tags(&scores);
        let best = best_predicted(&scores, &predicted);
        Ok((Categories::document_selected(Arc::clone(self.model.categories()), best, &predicted, &scores),))
    }

    fn model(&self) -> Option<ModelRef> {
        Some(self.model.model_ref().clone())
    }

    fn config_hash(&self) -> u64 {
        multilabel_config_hash(&self.model)
    }
}

fn multilabel_config_hash(model: &MultiLabelDoccatModel) -> u64 {
    let mut h = crate::model::context_generator_config_hash(model.context_generator());
    let mix = |h: &mut u64, bytes: &[u8]| {
        for &b in bytes {
            *h ^= b as u64;
            *h = h.wrapping_mul(0x100000001b3);
        }
    };
    mix(&mut h, model.language().as_bytes());
    mix(&mut h, &(model.num_labels() as u64).to_le_bytes());
    for i in 0..model.num_labels() {
        let label = model.categories().label(Tag(i as u16));
        mix(&mut h, &(label.len() as u64).to_le_bytes());
        mix(&mut h, label.as_bytes());
    }
    for &t in model.thresholds() {
        mix(&mut h, &t.to_bits().to_le_bytes());
    }
    mix(&mut h, &(model.max_labels().map_or(-1i64, |v| v as i64)).to_le_bytes());
    mix(&mut h, &(model.min_labels().map_or(-1i64, |v| v as i64)).to_le_bytes());
    mix(&mut h, &model.min_score().to_bits().to_le_bytes());
    h
}

/// The best category of a scored document: the highest-scoring emitted
/// label, ties toward the lower index as `getBestCategory` breaks them;
/// [`Tag::NONE`] when no label cleared its threshold, the clean
/// abstention.
fn best_predicted(scores: &[f64], predicted: &[Tag]) -> Tag {
    let mut best = Tag::NONE;
    let mut best_score = 0.0;
    for &t in predicted {
        let s = scores[t.0 as usize];
        if best == Tag::NONE || s > best_score {
            best = t;
            best_score = s;
        }
    }
    best
}

/// Categorizes each span of a positional layer `L` (`Sentences`,
/// `Structure`, or `Chunks`) with the tokens contained within that span,
/// writing `Categories<L>` (`turbo:categories<L>`): every label clearing
/// its threshold, or [`Tag::NONE`] where none does.
#[derive(Debug)]
pub struct MultiLabelSpanCategorizer<L: SpanCategorizable> {
    categorizer: MultiLabelCategorizer,
    _of: PhantomData<fn() -> L>,
}

impl<L: SpanCategorizable> Clone for MultiLabelSpanCategorizer<L> {
    fn clone(&self) -> Self {
        Self { categorizer: self.categorizer.clone(), _of: PhantomData }
    }
}

impl<L: SpanCategorizable> MultiLabelSpanCategorizer<L> {
    /// Creates a `MultiLabelSpanCategorizer<L>` from a model.
    pub fn new(model: MultiLabelDoccatModel) -> Self {
        Self { categorizer: MultiLabelCategorizer::new(model), _of: PhantomData }
    }

    /// The underlying [`MultiLabelCategorizer`].
    pub fn categorizer(&self) -> &MultiLabelCategorizer {
        &self.categorizer
    }
}

impl<L: SpanCategorizable> Annotator for MultiLabelSpanCategorizer<L> {
    const ID: &'static str = "turbo-doccat:multilabel-spans";
    type Reads = (View, Tokens, L);
    type Writes = (Categories<L>,);

    fn annotate(&self, (view, tokens, units): Refs<'_, Self>) -> Result<(Categories<L>,), Error> {
        if tokens.spans.view() != view.id() {
            return Err(Error::Annotator { annotator: Self::ID, detail: "tokens are not in the given view".into() });
        }
        if units.layer_spans().view() != view.id() {
            return Err(Error::Annotator { annotator: Self::ID, detail: "spans are not in the given view".into() });
        }
        let text = view.text();
        validate_spans(Self::ID, "token", &tokens.spans, text)?;
        validate_spans(Self::ID, "unit", units.layer_spans(), text)?;
        let model = self.categorizer.model();
        let n_units = units.len();
        let n_tokens = tokens.spans.len();
        let (tok_starts, tok_ends) = tokens.spans.columns();
        let tokens_ordered = tokens.spans.is_start_ordered();
        let mut out = Categories::<L>::with_capacity(Arc::clone(model.categories()), n_units);
        let mut scores = Vec::with_capacity(model.num_labels());
        let mut scratch = Vec::new();
        let mut tok_strs = Vec::new();
        let extra = BTreeMap::new();

        for i in 0..n_units {
            tok_strs.clear();
            for frag in units.layer_spans().fragments(Ix::new(i as u32)) {
                if tokens_ordered {
                    let a = tok_starts.partition_point(|&s| s < frag.start);
                    for t in a..n_tokens {
                        if tok_starts[t] > frag.end {
                            break;
                        }
                        if tok_ends[t] <= frag.end {
                            tok_strs.push(&text[tok_starts[t] as usize..tok_ends[t] as usize]);
                        }
                    }
                } else {
                    for t in 0..n_tokens {
                        if tok_starts[t] >= frag.start && tok_ends[t] <= frag.end {
                            tok_strs.push(&text[tok_starts[t] as usize..tok_ends[t] as usize]);
                        }
                    }
                }
            }
            model.scores_into(&tok_strs, &extra, &mut scratch, &mut scores);
            let predicted = model.predicted_tags(&scores);
            out.push_selected(best_predicted(&scores, &predicted), &predicted, &scores);
        }

        Ok((out,))
    }

    fn model(&self) -> Option<ModelRef> {
        Some(self.categorizer.model().model_ref().clone())
    }

    fn config_hash(&self) -> u64 {
        let mut h = self.categorizer.config_hash();
        for &b in L::ID.as_bytes() {
            h ^= b as u64;
            h = h.wrapping_mul(0x100000001b3);
        }
        h
    }
}
