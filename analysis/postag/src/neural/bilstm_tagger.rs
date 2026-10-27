//! BiLSTM(-CRF) part-of-speech tagger (`BilstmPosTagger`), ported from `BilstmPOSTagger`.
//!
//! Two numerics, chosen when the tagger is made ([`Numerics`]): by default
//! the forward pass runs in f32 on the CPU backend's packed kernels,
//! batched over every sentence of a call; [`BilstmPosTagger::jvm_parity`]
//! runs the f64 reference that follows Java operation by operation, for
//! the golden tests. The CRF Viterbi path and marginals (or the softmax of
//! a model without a CRF) are computed in f64 on the host in both: by
//! default with plain rounding (`crf::plain`), and with [`CrfScorer`]'s
//! exact sums for parity, or by default when a sentence's scores are too
//! large for plain rounding. A probability that is not finite is an error.

use std::sync::Arc;

use turbo::device::{Device, Precision};
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
    /// The largest absolute CRF transition, start or end weight.
    crf_weights: f64,
    tagset: Arc<TagSet>,
    outcome_tags: Vec<Tag>,
}

/// A best tag index per token with its CRF posterior or softmax probability.
type Path = (Vec<usize>, Vec<f64>);

impl BilstmPosTagger {
    /// A tagger over `model` with the default numerics, [`Numerics::Packed`],
    /// or [`Numerics::JvmParity`] if the model's weights cannot be packed
    /// (a weight past f32's range); [`numerics`](Self::numerics) says which.
    pub fn new(model: impl Into<Arc<BilstmPosModel>>) -> Self {
        let model = model.into();
        let packed = Packed::new(&model, Device::Cpu, Precision::Model).ok().map(Arc::new);
        Self::build(model, packed)
    }

    /// A tagger over `model` with [`Numerics::JvmParity`]: the f64 reference
    /// forward pass, with its per-token representation cache enabled.
    pub fn jvm_parity(model: impl Into<Arc<BilstmPosModel>>) -> Self {
        Self::build(model.into(), None)
    }

    /// A tagger over `model` with `numerics`; an error if the weights
    /// cannot be packed.
    pub fn with_numerics(model: impl Into<Arc<BilstmPosModel>>, numerics: Numerics) -> Result<Self, NeuralError> {
        let model = model.into();
        let packed = match numerics {
            Numerics::Packed => Some(Arc::new(Packed::new(&model, Device::Cpu, Precision::Model)?)),
            Numerics::JvmParity => None,
        };
        Ok(Self::build(model, packed))
    }

    /// A tagger over `model` with [`Numerics::Packed`] and its BiLSTMs on
    /// `device` (f32 weights and state, the CRF still f64 on the host); an
    /// error if the weights cannot be packed or the LSTMs cannot be made on
    /// that device. It never falls back to the CPU.
    pub fn on_device(model: impl Into<Arc<BilstmPosModel>>, device: Device) -> Result<Self, NeuralError> {
        Self::on_device_at(model, device, Precision::Model)
    }

    /// [`on_device`](Self::on_device) at `precision`: FASTEST holds the
    /// BiLSTMs' weights as F16 on the device, their sums and states F32,
    /// and is held to an accuracy bound, not to MODEL's tags. FASTEST on
    /// the CPU is an error.
    pub fn on_device_at(
        model: impl Into<Arc<BilstmPosModel>>,
        device: Device,
        precision: Precision,
    ) -> Result<Self, NeuralError> {
        let model = model.into();
        let packed = Packed::new(&model, device, precision)?;
        Ok(Self::build(model, Some(Arc::new(packed))))
    }

    /// What the BiLSTMs compute in: MODEL unless made at FASTEST.
    pub fn precision(&self) -> Precision {
        self.packed.as_ref().map_or(Precision::Model, |p| p.precision())
    }

    /// Where the BiLSTMs run: the CPU unless made [`on_device`](Self::on_device),
    /// and the CPU for [`Numerics::JvmParity`].
    pub fn device(&self) -> Device {
        self.packed.as_ref().map_or(Device::Cpu, |p| p.device())
    }

    fn build(model: Arc<BilstmPosModel>, packed: Option<Arc<Packed>>) -> Self {
        if packed.is_none() {
            model.enable_representation_cache();
        }
        let labels: Vec<&str> = model.tags().iter().map(String::as_str).collect();
        let tagset = TagSet::new("bilstm", &labels);
        let outcome_tags = labels.iter().map(|&l| tagset.tag(l).expect("tag is in tagset")).collect();
        let crf_weights = match (model.transition_weights(), model.start_weights(), model.end_weights()) {
            (Some(tr), Some(st), Some(en)) => plain::magnitude(tr.iter().map(Vec::as_slice).chain([st, en])),
            _ => 0.0,
        };
        Self { model, packed, crf_weights, tagset, outcome_tags }
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
        let reference = || {
            let mut at = 0;
            lens.iter()
                .map(|&len| {
                    at += len;
                    self.model.score(&words[at - len..at])
                })
                .collect()
        };
        let Some(packed) = &self.packed else {
            return reference();
        };
        let tags = self.model.tags().len();
        let flat = match packed.score(&self.model, words, lens) {
            Ok(flat) => flat,
            // A value past f32's range: the f64 reference holds it.
            Err(NeuralError::NonFiniteScore(_)) => return reference(),
            Err(e) => return Err(e),
        };
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
        let best = match self.plain(&scores) {
            true => plain::viterbi(&scores, tr, st, en),
            false => CrfScorer::viterbi(&scores, tr, st, en)?,
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
            let (best, marginals) = if self.plain(scores) {
                (plain::viterbi(scores, tr, st, en), plain::marginals(scores, tr, st, en))
            } else {
                (CrfScorer::viterbi(scores, tr, st, en)?, CrfScorer::marginals(scores, tr, st, en)?)
            };
            let probs = best.iter().zip(&marginals).map(|(&b, m)| m[b]).collect();
            finite((best, probs))
        } else {
            let best: Vec<usize> = scores.iter().map(|s| argmax(s)).collect();
            let probs = best.iter().zip(scores).map(|(&b, s)| softmax_at(s, b)).collect();
            finite((best, probs))
        }
    }

    /// Whether a sentence's CRF is decoded in plain f64: with the packed
    /// numerics, and only where its scores are small enough
    /// ([`plain::trusted`]).
    fn plain(&self, scores: &[Vec<f64>]) -> bool {
        self.packed.is_some() && plain::trusted(scores, self.crf_weights)
    }
}

/// `path`, unless one of its probabilities is not finite.
fn finite(path: Path) -> Result<Path, NeuralError> {
    match path.1.iter().position(|p| !p.is_finite()) {
        None => Ok(path),
        Some(i) => Err(NeuralError::NonFiniteScore(format!("the probability of token {i} is {}", path.1[i]))),
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

    fn annotate(&self, input: Refs<'_, Self>) -> Result<(PosTags,), Error> {
        self.annotate_batch(vec![input]).remove(0)
    }

    fn annotate_batch<'d>(&self, inputs: Vec<Refs<'d, Self>>) -> Vec<Result<Self::Writes, Error>> {
        let mut words = Vec::new();
        let mut lens = Vec::new();
        let mut scratch = Vec::new();
        let docs: Vec<_> = inputs
            .into_iter()
            .map(|(view, tokens)| {
                let lengths = super::batch_words::words(Self::ID, view, tokens, &mut scratch)?;
                let count = lengths.len();
                words.extend_from_slice(&scratch);
                lens.extend(lengths);
                Ok(count)
            })
            .collect();
        let failed = |e: NeuralError| Error::Annotator { annotator: Self::ID, detail: e.to_string() };
        let scores = match self.batch_emissions(&words, &lens) {
            Ok(scores) => scores,
            Err(e) => {
                let e = failed(e);
                return docs.into_iter().map(|d| d.and(Err(e.clone()))).collect();
            }
        };
        let mut paths = scores.iter().map(|s| self.best_path(s).map_err(failed));
        docs.into_iter()
            .map(|count| {
                let mut tags = Vec::new();
                let mut probs = Vec::new();
                let mut error = None;
                for path in paths.by_ref().take(count?) {
                    match path {
                        Ok((best, ps)) => {
                            tags.extend(best.into_iter().map(|i| self.outcome_tags[i]));
                            probs.extend(ps);
                        }
                        Err(e) => error = Some(e),
                    }
                }
                match error {
                    Some(e) => Err(e),
                    None => Ok((PosTags { tagset: self.tagset.clone(), tags, prob: Some(probs) },)),
                }
            })
            .collect()
    }

    fn model(&self) -> Option<ModelRef> {
        Some(self.model.model_ref().clone())
    }

    /// FNV-1a over the numerics, the device backend, its precision and the
    /// model's SHA-256: taggers of one model with other numerics, on another
    /// backend or at another precision give different probabilities, so
    /// they hash apart.
    fn config_hash(&self) -> u64 {
        let numerics: &[u8] = match (self.numerics(), self.device(), self.precision()) {
            (Numerics::Packed, Device::Cpu, _) => b"packed",
            (Numerics::Packed, Device::Cuda(_), Precision::Model) => b"packed-cuda-f32",
            (Numerics::Packed, Device::Cuda(_), Precision::Fastest) => b"packed-cuda-f16-weights",
            (Numerics::Packed, Device::LevelZero(_), Precision::Model) => b"packed-levelzero-f32",
            (Numerics::Packed, Device::LevelZero(_), Precision::Fastest) => b"packed-levelzero-f16-weights",
            (Numerics::Packed, Device::Metal(_), Precision::Model) => b"packed-metal-f32",
            (Numerics::Packed, Device::Metal(_), Precision::Fastest) => b"packed-metal-f16-weights",
            (Numerics::JvmParity, _, _) => b"jvm-parity",
        };
        turbo_analysis::Fnv1a::new()
            .bytes(numerics)
            .bytes(b"\0")
            .bytes(self.model.model_ref().sha256.as_bytes())
            .finish()
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
    fn a_non_finite_probability_is_an_error() {
        for bad in [f64::NAN, f64::INFINITY] {
            assert!(matches!(finite((vec![0, 1], vec![0.5, bad])), Err(NeuralError::NonFiniteScore(_))));
        }
        assert_eq!(finite((vec![1], vec![0.25])).unwrap(), (vec![1], vec![0.25]));
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
