//! BiLSTM(-CRF) part-of-speech tagger (`BilstmPosTagger`), ported from `BilstmPOSTagger`.

use std::sync::Arc;

use turbo_analysis::{Annotator, Error, ModelRef, Refs, View};
use turbo_analysis_layers::{PosTags, Tag, TagSet, Tokens};

use crate::neural::bilstm_model::BilstmPosModel;
use crate::neural::crf::CrfScorer;
use crate::neural::{NeuralError, NeuralSequence};

/// Thread-safe part-of-speech tagger backed by a [`BilstmPosModel`].
#[derive(Clone, Debug)]
pub struct BilstmPosTagger {
    model: Arc<BilstmPosModel>,
    tagset: Arc<TagSet>,
    outcome_tags: Vec<Tag>,
}

impl BilstmPosTagger {
    /// Creates a new `BilstmPosTagger` over `model` and enables its per-token representation cache.
    pub fn new(model: impl Into<Arc<BilstmPosModel>>) -> Self {
        let model = model.into();
        model.enable_representation_cache();
        let labels: Vec<&str> = model.tags().iter().map(String::as_str).collect();
        let tagset = TagSet::new("bilstm", &labels);
        let outcome_tags = labels.iter().map(|&l| tagset.tag(l).expect("tag is in tagset")).collect();
        Self { model, tagset, outcome_tags }
    }

    /// Returns a reference to the underlying [`BilstmPosModel`].
    pub fn model(&self) -> &BilstmPosModel {
        &self.model
    }

    /// Returns the [`TagSet`] of the model's outcomes.
    pub fn tagset(&self) -> &Arc<TagSet> {
        &self.tagset
    }

    /// Tags a tokenized sentence, returning one tag string per token.
    pub fn tag(&self, sentence: &[impl AsRef<str>]) -> Result<Vec<String>, NeuralError> {
        if sentence.is_empty() {
            return Ok(Vec::new());
        }
        let scores = self.model.score(sentence)?;
        let tags = self.model.tags();
        if self.model.is_crf() {
            let best = CrfScorer::viterbi(
                &scores,
                self.model.transition_weights().unwrap(),
                self.model.start_weights().unwrap(),
                self.model.end_weights().unwrap(),
            )?;
            Ok(best.into_iter().map(|idx| tags[idx].clone()).collect())
        } else {
            Ok(scores.iter().map(|s| tags[argmax(s)].clone()).collect())
        }
    }

    /// Returns the single best tag sequence with per-token posterior or softmax probabilities.
    pub fn top_k_sequences(&self, sentence: &[impl AsRef<str>]) -> Result<Vec<NeuralSequence>, NeuralError> {
        if sentence.is_empty() {
            return Ok(vec![NeuralSequence::from_outcomes_and_probs(Vec::new(), Vec::new())]);
        }
        let (best, probs) = self.best_path(sentence)?;
        let tags = self.model.tags();
        let outcomes = best.into_iter().map(|idx| tags[idx].clone()).collect();
        Ok(vec![NeuralSequence::from_outcomes_and_probs(outcomes, probs)])
    }

    /// The best tag index for each token of a non-empty sentence, with its
    /// CRF posterior or softmax probability.
    fn best_path(&self, sentence: &[impl AsRef<str>]) -> Result<(Vec<usize>, Vec<f64>), NeuralError> {
        let scores = self.model.score(sentence)?;
        if self.model.is_crf() {
            let tr = self.model.transition_weights().unwrap();
            let st = self.model.start_weights().unwrap();
            let en = self.model.end_weights().unwrap();
            let best = CrfScorer::viterbi(&scores, tr, st, en)?;
            let marginals = CrfScorer::marginals(&scores, tr, st, en)?;
            let probs = best.iter().zip(&marginals).map(|(&b, m)| m[b]).collect();
            Ok((best, probs))
        } else {
            let best: Vec<usize> = scores.iter().map(|s| argmax(s)).collect();
            let probs = best.iter().zip(&scores).map(|(&b, s)| softmax_at(s, b)).collect();
            Ok((best, probs))
        }
    }
}

fn argmax(scores: &[f64]) -> usize {
    let mut best = 0usize;
    let mut best_score = scores[0];
    for (i, &s) in scores.iter().enumerate().skip(1) {
        if s > best_score {
            best_score = s;
            best = i;
        }
    }
    best
}

fn softmax_at(scores: &[f64], index: usize) -> f64 {
    let max = scores[index];
    let mut sum = 0.0f64;
    for &s in scores {
        sum += turbo_jvm::strict_math::exp(s - max);
    }
    1.0 / sum
}

impl Annotator for BilstmPosTagger {
    const ID: &'static str = "turbo-postag:bilstm";
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
        Some(self.model.model_ref().clone())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::neural::PosSample;
    use crate::neural::bilstm_trainer::{BilstmPosTrainer, BilstmSettings};
    use turbo_analysis::{Document, Span, Spans, ViewId};

    #[test]
    fn softmax_at_uses_strict_math_exp() {
        let scores = [0.5f64, -0.25f64, 1.25f64];
        let expected_sum: f64 = scores.iter().map(|&s| turbo_jvm::strict_math::exp(s - scores[2])).sum();
        assert_eq!(softmax_at(&scores, 2).to_bits(), (1.0 / expected_sum).to_bits());
    }

    #[test]
    fn bilstm_annotator_rejects_invalid_token_spans_without_panicking() {
        let samples = vec![PosSample::new(["café", "barks"], ["NN", "VBZ"]).unwrap()];
        let mut settings = BilstmSettings::defaults();
        settings.epochs = 1;
        settings.word_cutoff = 1;
        let model = BilstmPosTrainer::train(&samples, settings).unwrap();
        let tagger = BilstmPosTagger::new(model);

        let doc = Document::new("café");
        let view = doc.get::<View>().unwrap();
        for bad_span in [
            Span::new(0, 99),          // past end of text
            Span { start: 3, end: 1 }, // inverted span
            Span::new(0, 4),           // splits 2-byte UTF-8 'é' (bytes 3..5)
        ] {
            let mut spans = Spans::new(ViewId::SOURCE);
            spans.push(bad_span);
            let tokens = Tokens { spans, sentence: None };
            let err = tagger.annotate((view, &tokens)).err().unwrap();
            assert!(
                matches!(&err, Error::Annotator { annotator: BilstmPosTagger::ID, detail } if detail.contains("outside the text or off a UTF-8 boundary")),
                "unexpected error for {bad_span:?}: {err:?}"
            );
        }
    }
}
