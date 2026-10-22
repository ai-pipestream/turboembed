//! BiLSTM(-CRF) part-of-speech tagger (`BilstmPosTagger`), ported from `BilstmPOSTagger`.
//!
//! Two numerics, chosen when the tagger is made ([`Numerics`]): by default
//! the forward pass runs in f32 on the CPU backend's packed kernels,
//! batched over every sentence of a call; [`BilstmPosTagger::jvm_parity`]
//! runs the f64 reference that follows Java operation by operation, for
//! the golden tests. The CRF Viterbi path and marginals (or the softmax of
//! a model without a CRF) are computed in f64 on the host in both: by
//! default with plain rounding (`crf::plain`), for parity with
//! [`CrfScorer`]'s exact sums.

use std::sync::Arc;

use turbo_analysis::{Annotator, Error, ModelRef, Refs, View};
use turbo_analysis_layers::{PosTags, Tag, TagSet, Tokens};

use crate::neural::bilstm_model::{BilstmPosModel, Packed};
use crate::neural::crf::{CrfScorer, plain};
use crate::neural::{NeuralError, NeuralSequence};

/// How a [`BilstmPosTagger`] computes its emissions.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Numerics {
    /// f32 on the CPU backend's packed matrix product (`turbo::cpu::Matrix`),
    /// batched over the sentences of a call, and the CRF decoded in plain
    /// f64. The default.
    Packed,
    /// The f64 reference in Java's order of operations, whose emissions are
    /// the JVM's within 1e-13, and the CRF decoded with exact sums as the
    /// JVM does. Opt-in, for parity with the JVM goldens.
    JvmParity,
}

/// Thread-safe part-of-speech tagger backed by a [`BilstmPosModel`].
#[derive(Clone, Debug)]
pub struct BilstmPosTagger {
    model: Arc<BilstmPosModel>,
    packed: Option<Arc<Packed>>,
    tagset: Arc<TagSet>,
    outcome_tags: Vec<Tag>,
}

/// A best tag index per token with its CRF posterior or softmax probability.
type Path = (Vec<usize>, Vec<f64>);

impl BilstmPosTagger {
    /// A tagger over `model` with the default numerics, [`Numerics::Packed`].
    ///
    /// # Panics
    /// If the packed weights cannot be allocated.
    pub fn new(model: impl Into<Arc<BilstmPosModel>>) -> Self {
        Self::with_numerics(model, Numerics::Packed).expect("packing the BiLSTM weights")
    }

    /// A tagger over `model` with [`Numerics::JvmParity`]: the f64 reference
    /// forward pass, with its per-token representation cache enabled.
    pub fn jvm_parity(model: impl Into<Arc<BilstmPosModel>>) -> Self {
        Self::with_numerics(model, Numerics::JvmParity).expect("the reference path packs nothing")
    }

    /// A tagger over `model` with `numerics`; an error if the packed
    /// weights cannot be allocated.
    pub fn with_numerics(model: impl Into<Arc<BilstmPosModel>>, numerics: Numerics) -> Result<Self, NeuralError> {
        let model = model.into();
        let packed = match numerics {
            Numerics::Packed => Some(Arc::new(Packed::new(&model)?)),
            Numerics::JvmParity => {
                model.enable_representation_cache();
                None
            }
        };
        let labels: Vec<&str> = model.tags().iter().map(String::as_str).collect();
        let tagset = TagSet::new("bilstm", &labels);
        let outcome_tags = labels.iter().map(|&l| tagset.tag(l).expect("tag is in tagset")).collect();
        Ok(Self { model, packed, tagset, outcome_tags })
    }

    /// The numerics the emissions are computed with.
    pub fn numerics(&self) -> Numerics {
        if self.packed.is_some() { Numerics::Packed } else { Numerics::JvmParity }
    }

    /// Returns a reference to the underlying [`BilstmPosModel`].
    pub fn model(&self) -> &BilstmPosModel {
        &self.model
    }

    /// Returns the [`TagSet`] of the model's outcomes.
    pub fn tagset(&self) -> &Arc<TagSet> {
        &self.tagset
    }

    /// Raw per-token tag scores `[tokens][tags]` of a non-empty sentence,
    /// with this tagger's numerics.
    pub fn emissions(&self, sentence: &[impl AsRef<str>]) -> Result<Vec<Vec<f64>>, NeuralError> {
        if self.packed.is_none() {
            return self.model.score(sentence);
        }
        let words: Vec<&str> = sentence.iter().map(AsRef::as_ref).collect();
        Ok(self.batch_emissions(&words, &[words.len()])?.pop().unwrap_or_default())
    }

    /// The emissions of each sentence, `lens` long, of `words`.
    fn batch_emissions(&self, words: &[&str], lens: &[usize]) -> Result<Vec<Vec<Vec<f64>>>, NeuralError> {
        if lens.contains(&0) {
            return Err(NeuralError::InvalidArgument("tokens must not be null or empty".into()));
        }
        let Some(packed) = &self.packed else {
            let mut at = 0;
            return lens
                .iter()
                .map(|&len| {
                    at += len;
                    self.model.score(&words[at - len..at])
                })
                .collect();
        };
        let tags = self.model.tags().len();
        let flat = packed.score(&self.model, words, lens)?;
        let mut rows = flat.chunks_exact(tags).map(|r| r.iter().map(|&v| f64::from(v)).collect());
        Ok(lens.iter().map(|&len| rows.by_ref().take(len).collect()).collect())
    }

    /// Tags a tokenized sentence, returning one tag string per token.
    pub fn tag(&self, sentence: &[impl AsRef<str>]) -> Result<Vec<String>, NeuralError> {
        if sentence.is_empty() {
            return Ok(Vec::new());
        }
        let scores = self.emissions(sentence)?;
        let tags = self.model.tags();
        if !self.model.is_crf() {
            return Ok(scores.iter().map(|s| tags[argmax(s)].clone()).collect());
        }
        let (tr, st, en) = (
            self.model.transition_weights().unwrap(),
            self.model.start_weights().unwrap(),
            self.model.end_weights().unwrap(),
        );
        let best = match self.packed {
            Some(_) => plain::viterbi(&scores, tr, st, en),
            None => CrfScorer::viterbi(&scores, tr, st, en)?,
        };
        Ok(best.into_iter().map(|idx| tags[idx].clone()).collect())
    }

    /// Returns the single best tag sequence with per-token posterior or softmax probabilities.
    pub fn top_k_sequences(&self, sentence: &[impl AsRef<str>]) -> Result<Vec<NeuralSequence>, NeuralError> {
        if sentence.is_empty() {
            return Ok(vec![NeuralSequence::from_outcomes_and_probs(Vec::new(), Vec::new())]);
        }
        let (best, probs) = self.best_path(&self.emissions(sentence)?)?;
        let tags = self.model.tags();
        let outcomes = best.into_iter().map(|idx| tags[idx].clone()).collect();
        Ok(vec![NeuralSequence::from_outcomes_and_probs(outcomes, probs)])
    }

    /// The best tag sequence of each sentence, with per-token posterior or
    /// softmax probabilities; with [`Numerics::Packed`] the sentences are
    /// one batch. An empty sentence gets an empty sequence.
    pub fn best_sequences<S: AsRef<str>>(
        &self,
        sentences: &[impl AsRef<[S]>],
    ) -> Result<Vec<NeuralSequence>, NeuralError> {
        let words: Vec<&str> = sentences.iter().flat_map(|s| s.as_ref().iter().map(AsRef::as_ref)).collect();
        let lens: Vec<usize> = sentences.iter().map(|s| s.as_ref().len()).filter(|&l| l > 0).collect();
        let mut batch = self.batch_emissions(&words, &lens)?.into_iter();
        let tags = self.model.tags();
        let mut out = Vec::with_capacity(sentences.len());
        for s in sentences {
            if s.as_ref().is_empty() {
                out.push(NeuralSequence::from_outcomes_and_probs(Vec::new(), Vec::new()));
                continue;
            }
            let scores = batch.next().expect("one emission table per non-empty sentence");
            let (best, probs) = self.best_path(&scores)?;
            out.push(NeuralSequence::from_outcomes_and_probs(
                best.into_iter().map(|i| tags[i].clone()).collect(),
                probs,
            ));
        }
        Ok(out)
    }

    /// The best tag index for each token of a sentence's emissions, with
    /// its CRF posterior or softmax probability.
    fn best_path(&self, scores: &[Vec<f64>]) -> Result<Path, NeuralError> {
        if self.model.is_crf() {
            let tr = self.model.transition_weights().unwrap();
            let st = self.model.start_weights().unwrap();
            let en = self.model.end_weights().unwrap();
            let (best, marginals) = if self.packed.is_some() {
                (plain::viterbi(scores, tr, st, en), plain::marginals(scores, tr, st, en))
            } else {
                (CrfScorer::viterbi(scores, tr, st, en)?, CrfScorer::marginals(scores, tr, st, en)?)
            };
            let probs = best.iter().zip(&marginals).map(|(&b, m)| m[b]).collect();
            Ok((best, probs))
        } else {
            let best: Vec<usize> = scores.iter().map(|s| argmax(s)).collect();
            let probs = best.iter().zip(scores).map(|(&b, s)| softmax_at(s, b)).collect();
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
        let mut words: Vec<&str> = Vec::with_capacity(n);
        let mut sentence: Vec<&str> = Vec::new();
        let mut lens = Vec::new();
        let mut from = 0;
        while from < n {
            let to = match &tokens.sentence {
                Some(col) => from + col[from..].partition_point(|&s| s == col[from]),
                None => n,
            };
            crate::tagger::slice_sentence_words(Self::ID, text, starts, ends, from, to, &mut sentence)?;
            if !sentence.is_empty() {
                words.extend_from_slice(&sentence);
                lens.push(sentence.len());
            }
            from = to;
        }
        let failed = |e: NeuralError| Error::Annotator { annotator: Self::ID, detail: e.to_string() };
        let mut tags = Vec::with_capacity(n);
        let mut probs = Vec::with_capacity(n);
        for scores in self.batch_emissions(&words, &lens).map_err(failed)? {
            let (best, sentence_probs) = self.best_path(&scores).map_err(failed)?;
            tags.extend(best.into_iter().map(|idx| self.outcome_tags[idx]));
            probs.extend(sentence_probs);
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
    use turbo_analysis::{Document, Ix, Span, Spans, ViewId};

    fn trained(crf: bool) -> BilstmPosModel {
        let samples = vec![
            PosSample::new(["the", "dog", "barks"], ["DT", "NN", "VBZ"]).unwrap(),
            PosSample::new(["a", "cat", "sleeps", "here"], ["DT", "NN", "VBZ", "RB"]).unwrap(),
            PosSample::new(["dogs", "bark"], ["NNS", "VBP"]).unwrap(),
        ];
        let mut settings = BilstmSettings::defaults();
        (settings.epochs, settings.word_cutoff, settings.crf) = (2, 1, crf);
        BilstmPosTrainer::train(&samples, settings).unwrap()
    }

    #[test]
    fn packed_batches_give_each_sentence_its_own_scores() {
        for crf in [false, true] {
            let model = trained(crf);
            let packed = BilstmPosTagger::new(model.clone());
            let parity = BilstmPosTagger::jvm_parity(model);
            assert_eq!((packed.numerics(), parity.numerics()), (Numerics::Packed, Numerics::JvmParity));
            let sentences: Vec<Vec<&str>> =
                vec![vec!["the", "cat", "barks"], vec![], vec!["dogs", "sleep", "here", "", "unseen-é"], vec!["a"]];
            let batch = packed.best_sequences(&sentences).unwrap();
            assert_eq!(batch.len(), 4);
            assert!(batch[1].outcomes().is_empty());
            for (s, b) in sentences.iter().zip(&batch).filter(|(s, _)| !s.is_empty()) {
                // Alone or in a batch, cached or not: the same bits.
                let alone = packed.top_k_sequences(s).unwrap().remove(0);
                assert_eq!(&alone, b);
                assert_eq!(packed.emissions(s).unwrap(), packed.emissions(s).unwrap());
                assert_eq!(b.outcomes(), parity.tag(s).unwrap(), "{s:?}");
                for (p, q) in b.probs().iter().zip(parity.top_k_sequences(s).unwrap()[0].probs()) {
                    assert!((p - q).abs() < 1e-5, "{s:?}: {p} against {q}");
                }
            }
            assert!(packed.emissions(&[] as &[&str]).is_err());
        }
    }

    #[test]
    fn packed_annotator_tags_every_sentence_of_a_document() {
        let tagger = BilstmPosTagger::new(trained(true));
        let text = "the dog barks a cat sleeps";
        let doc = Document::new(text);
        let view = doc.get::<View>().unwrap();
        let mut spans = Spans::new(ViewId::SOURCE);
        for (a, b) in [(0, 3), (4, 7), (8, 13), (14, 15), (16, 19), (20, 26)] {
            spans.push(Span::new(a, b));
        }
        let sentence = [0, 0, 0, 1, 1, 1].map(Ix::new).to_vec();
        let tokens = Tokens { spans, sentence: Some(sentence) };
        let (pos,) = tagger.annotate((view, &tokens)).unwrap();
        let tags: Vec<&str> = pos.tags.iter().map(|&t| tagger.tagset().label(t)).collect();
        let mut expected = tagger.tag(&["the", "dog", "barks"]).unwrap();
        expected.extend(tagger.tag(&["a", "cat", "sleeps"]).unwrap());
        assert_eq!(tags, expected);
        assert_eq!(pos.prob.as_ref().map(Vec::len), Some(6));
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
