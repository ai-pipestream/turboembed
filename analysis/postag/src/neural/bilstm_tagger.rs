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
        let outcome_tags = labels
            .iter()
            .map(|&l| tagset.tag(l).expect("tag is in tagset"))
            .collect();
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
            Ok(scores
                .iter()
                .map(|s| tags[argmax(s)].clone())
                .collect())
        }
    }

    /// Returns the single best tag sequence with per-token posterior or softmax probabilities.
    pub fn top_k_sequences(
        &self,
        sentence: &[impl AsRef<str>],
    ) -> Result<Vec<NeuralSequence>, NeuralError> {
        if sentence.is_empty() {
            return Ok(vec![NeuralSequence::from_outcomes_and_probs(
                Vec::new(),
                Vec::new(),
            )]);
        }
        let scores = self.model.score(sentence)?;
        let tags = self.model.tags();
        let n = sentence.len();
        let mut outcomes = Vec::with_capacity(n);
        let mut probs = Vec::with_capacity(n);
        if self.model.is_crf() {
            let tr = self.model.transition_weights().unwrap();
            let st = self.model.start_weights().unwrap();
            let en = self.model.end_weights().unwrap();
            let best = CrfScorer::viterbi(&scores, tr, st, en)?;
            let marginals = CrfScorer::marginals(&scores, tr, st, en)?;
            for t in 0..n {
                outcomes.push(tags[best[t]].clone());
                probs.push(marginals[t][best[t]]);
            }
        } else {
            for s in &scores {
                let best = argmax(s);
                outcomes.push(tags[best].clone());
                probs.push(softmax_at(s, best));
            }
        }
        Ok(vec![NeuralSequence::from_outcomes_and_probs(outcomes, probs)])
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
        sum += (s - max).exp();
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
            words.clear();
            words.extend((from..to).map(|i| &text[starts[i] as usize..ends[i] as usize]));
            if !words.is_empty() {
                let seqs = self.top_k_sequences(&words).map_err(|e| Error::Annotator {
                    annotator: Self::ID,
                    detail: e.to_string(),
                })?;
                let seq = &seqs[0];
                for outcome in seq.outcomes() {
                    let tag = self.tagset.tag(outcome).expect("outcome is in tagset");
                    tags.push(tag);
                }
                probs.extend_from_slice(seq.probs());
            }
            from = to;
        }
        let _ = &self.outcome_tags;
        Ok((PosTags {
            tagset: self.tagset.clone(),
            tags,
            prob: Some(probs),
        },))
    }

    fn model(&self) -> Option<ModelRef> {
        None
    }
}
