//! Greedy left-to-right neural POS tagger and [`Annotator`] implementation
//! (`FeedforwardPOSTagger`).

use std::sync::Arc;

use turbo_analysis::{Annotator, Error, ModelRef, Refs, View};
use turbo_analysis_layers::{PosTags, Tag, TagSet, Tokens};

use super::context::SentenceFeatures;
use super::model::FeedforwardPosModel;
use crate::neural::{NeuralError, NeuralSequence};

/// Thread-safe greedy left-to-right neural POS tagger backed by a [`FeedforwardPosModel`].
#[derive(Clone, Debug)]
pub struct FeedforwardPosTagger {
    model: Arc<FeedforwardPosModel>,
    tagset: Arc<TagSet>,
    outcome_tags: Vec<Tag>,
    model_ref: ModelRef,
}

impl FeedforwardPosTagger {
    /// Creates a new `FeedforwardPosTagger` over `model` and enables its contribution cache.
    pub fn new(model: impl Into<Arc<FeedforwardPosModel>>) -> Self {
        let model = model.into();
        model.enable_scoring_cache();
        let labels: Vec<&str> = model.tags().iter().map(String::as_str).collect();
        let tagset = TagSet::new("feedforward", &labels);
        let outcome_tags = labels.iter().map(|&l| tagset.tag(l).expect("tag is in tagset")).collect();
        let model_ref = model.model_ref();
        Self { model, tagset, outcome_tags, model_ref }
    }

    /// Returns a reference to the underlying [`FeedforwardPosModel`].
    pub fn model(&self) -> &FeedforwardPosModel {
        &self.model
    }

    /// Returns the [`TagSet`] of the model's outcomes.
    pub fn tagset(&self) -> &Arc<TagSet> {
        &self.tagset
    }

    /// Tags a tokenized sentence greedily from left to right.
    pub fn tag(&self, sentence: &[impl AsRef<str>]) -> Result<Vec<String>, NeuralError> {
        let (best_indices, _) = self.best_path(sentence)?;
        let tags = self.model.tags();
        Ok(best_indices.into_iter().map(|idx| tags[idx].clone()).collect())
    }

    /// Returns a 1-element array containing the greedy tag sequence with per-token softmax probabilities.
    pub fn top_k_sequences(&self, sentence: &[impl AsRef<str>]) -> Result<Vec<NeuralSequence>, NeuralError> {
        Ok(vec![self.decode(sentence)?])
    }

    fn decode(&self, sentence: &[impl AsRef<str>]) -> Result<NeuralSequence, NeuralError> {
        let (best_indices, probs) = self.best_path(sentence)?;
        let tags = self.model.tags();
        let outcomes = best_indices.into_iter().map(|idx| tags[idx].clone()).collect();
        Ok(NeuralSequence::from_outcomes_and_probs(outcomes, probs))
    }

    fn best_path(&self, sentence: &[impl AsRef<str>]) -> Result<(Vec<usize>, Vec<f64>), NeuralError> {
        let n = sentence.len();
        if n == 0 {
            return Ok((Vec::new(), Vec::new()));
        }
        let sent_features = SentenceFeatures::new(sentence);
        let mut assigned_indices: Vec<usize> = Vec::with_capacity(n);
        let mut probs: Vec<f64> = Vec::with_capacity(n);
        let tags = self.model.tags();

        for i in 0..n {
            let prev = if i >= 1 { Some(tags[assigned_indices[i - 1]].as_str()) } else { None };
            let prev_prev = if i >= 2 { Some(tags[assigned_indices[i - 2]].as_str()) } else { None };
            let f_ids = self.model.feature_ids_at(&sent_features, i, prev, prev_prev);
            let p_rows = self.model.pretrained_rows_at(&sent_features, i);
            let scores = self.model.score_with_pretrained(&f_ids, p_rows.as_ref().map(|r| &r[..]))?;

            let mut best = 0usize;
            for (o, &s) in scores.iter().enumerate() {
                if !s.is_finite() {
                    return Err(NeuralError::NonFiniteScore("model produced a non-finite tag score".into()));
                }
                if o > 0 && s > scores[best] {
                    best = o;
                }
            }
            assigned_indices.push(best);

            let max = scores[best];
            let mut sum_exp = 0.0f64;
            for &s in &scores {
                sum_exp += turbo_jvm::strict_math::exp(s - max);
            }
            probs.push(1.0 / sum_exp);
        }
        Ok((assigned_indices, probs))
    }
}

impl Annotator for FeedforwardPosTagger {
    const ID: &'static str = "turbo-postag:feedforward";
    type Reads = (View, Tokens);
    type Writes = (PosTags,);

    fn annotate(&self, (view, tokens): Refs<'_, Self>) -> Result<(PosTags,), Error> {
        if tokens.spans.view() != view.id() {
            return Err(Error::Annotator {
                annotator: Self::ID,
                detail: format!(
                    "the tokens are in view {}, not in view {} that is tagged",
                    tokens.spans.view().0,
                    view.id().0
                ),
            });
        }
        let text = view.text();
        let (starts, ends) = tokens.spans.columns();
        let n = starts.len();
        let mut tags = Vec::with_capacity(n);
        let mut probs = Vec::with_capacity(n);
        let mut words: Vec<&str> = Vec::new();
        let mut from = 0;
        while from < n {
            let to = match &tokens.sentence {
                Some(col) => from + col[from..].partition_point(|&s| s == col[from]),
                None => n,
            };
            crate::tagger::slice_sentence_words(Self::ID, text, starts, ends, from, to, &mut words)?;
            if !words.is_empty() {
                let (best, sentence_probs) = self
                    .best_path(&words)
                    .map_err(|e| Error::Annotator { annotator: Self::ID, detail: e.to_string() })?;
                tags.extend(best.into_iter().map(|idx| self.outcome_tags[idx]));
                probs.extend(sentence_probs);
            }
            from = to;
        }
        Ok((PosTags { tagset: self.tagset.clone(), tags, prob: Some(probs) },))
    }

    fn model(&self) -> Option<ModelRef> {
        Some(self.model_ref.clone())
    }
}
