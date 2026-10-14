//! The English part-of-speech tagger trained on MASC's Penn tags, and its
//! gate against OpenNLP's English Penn tagger, `en-pos-maxent.bin`.
//!
//! The recipe trains on the training documents of [`crate::masc::SPLIT`]
//! with the factory defaults (the default feature generator descriptor,
//! no tag dictionary) and the parameters the published UD 1.3 models were
//! trained with ([`turbo_train_components::ud`]: GIS, 300 iterations,
//! cutoff 5, 4 threads). Training uses no randomness, so the recipe has
//! no seed.
//!
//! The samples are what OpenNLP's `MascPOSSampleStream` reads, one per
//! sentence, its Penn tokens and their `msd` tags, with one change: a
//! sentence with a token without a tag, on which the stream throws
//! `IllegalArgumentException` and so ends OpenNLP's training run, is
//! skipped alone ([`read_samples`]), as the token recipe skips the
//! sentences its stream cannot read. A sentence of no tokens is skipped
//! too: it gives no event, and as a held-out unit it would count nothing.
//! The card records how many were skipped.
//!
//! The reference is OpenNLP's English maxent tagger from the 1.5 models
//! ([`EN_POS_MAXENT`]). OpenNLP's only English taggers trained on Penn
//! tags are that one and the 1.5 perceptron tagger, and the maxent one is
//! trained with the algorithm the recipe uses. Both models are scored in
//! the same tag set: the reference asked for Penn tags, which are its own
//! outcomes and come out unconverted, and the trained model with its
//! outcomes as they are ([`CANDIDATE_FORMAT`]). OpenNLP describes the
//! reference's training data only as "(private) English data" (the 1.5.1
//! test plan) and the zoo does not list it: a run reads it from its
//! references directory, checks it against the recorded size and SHA-256,
//! and only scores it. It is never shipped or bundled, and the trained
//! model is trained on MASC alone.
//!
//! The English UD tagger, `opennlp-en-ud-ewt-pos-1.3-2.5.4`, is not the
//! reference: it was trained on UD tags, and asked for Penn tags
//! `POSTagFormatMapper` converts each of its outcomes to one Penn tag
//! (`VERB` to `VB`, `NOUN` to `NN`, `PUNCT` to `.`), so it can never give
//! `VBD`, `NNS`, `,` or `TO`, and a gate against it would pass any tagger.
//!
//! Both models are scored on the same held-out samples with OpenNLP's
//! `POSEvaluator`, one unit per sentence, and gated by
//! [`turbo_eval::model_gate`].

use turbo_eval::model_gate::{Counts, HeldOutSet, ModelGateRecord, ModelId, PairedUnit, Task};
use turbo_formats::evaluate::PosEvaluator;
use turbo_formats::masc::{MascCorpus, MascError, MascPosReader};
use turbo_formats::sample::PosSample;
use turbo_maxent::ModelFile;
use turbo_postag::{PosModel, PosTagFormat, PosTagger};
use turbo_train::TrainingConfiguration;
use turbo_train_components::{PosTaggerFactory, Stamp, postag, ud};
use turbo_zoo::Sha256;

use crate::error::{Error, Result};
use crate::segmentation::{LANGUAGE, TIMESTAMP_MS};
use crate::{RecordedReference, Reference, Trained};

/// The format the trained model's tags are scored in: its outcomes as
/// they are, MASC's Penn tags.
pub const CANDIDATE_FORMAT: PosTagFormat = PosTagFormat::Custom;

/// OpenNLP's English maxent tagger of the 1.5 models, trained on Penn
/// tags, as published.
pub const EN_POS_MAXENT: RecordedReference = RecordedReference {
    id: "opennlp-en-pos-maxent-1.5",
    file: "en-pos-maxent.bin",
    url: "https://opennlp.sourceforge.net/models-1.5/en-pos-maxent.bin",
    size: 5_696_197,
    sha256: "645a094f45a866687a617385233fd23ae8b0f5fa8b1b76996781a50c17bdcf3d",
};

/// The tagger a recipe trains and the OpenNLP model it is gated against.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PosRecipe {
    /// The trained model's zoo id, also its file name without `.bin`.
    pub model_id: &'static str,
    pub reference: Reference,
    /// The format the reference's tags are asked for.
    pub reference_format: PosTagFormat,
}

pub const POS: PosRecipe = PosRecipe {
    model_id: "turbo-en-masc-3.0.0-pos-1",
    reference: Reference::Recorded(EN_POS_MAXENT),
    reference_format: PosTagFormat::Penn,
};

/// Reads the part-of-speech samples of `corpus` as `MascPOSSampleStream`
/// reads them, giving each to `each`, except that a sentence the stream
/// throws on for a token without a tag ([`MascError::NullTag`]) is skipped
/// alone, and so is a sentence of no tokens. Returns how many were
/// skipped. The stream has moved past a sentence when it throws on it, so
/// the next read is the next sentence.
///
/// # Errors
///
/// Any other error of the stream, which ends the read.
pub fn read_samples<'c>(corpus: &'c MascCorpus, mut each: impl FnMut(PosSample<'c>)) -> Result<usize> {
    let mut skipped = 0;
    for read in MascPosReader::new(corpus)? {
        match read {
            Ok(s) if s.sample.is_empty() => skipped += 1,
            Ok(s) => each(s.sample),
            Err(MascError::NullTag) => skipped += 1,
            Err(e) => return Err(e.into()),
        }
    }
    Ok(skipped)
}

/// The held-out part-of-speech samples, borrowed from the held-out corpus.
#[derive(Debug, Clone)]
pub struct HeldOutTags<'c> {
    samples: Vec<PosSample<'c>>,
    skipped: usize,
}

impl<'c> HeldOutTags<'c> {
    /// The samples scored.
    pub fn samples(&self) -> &[PosSample<'c>] {
        &self.samples
    }

    /// How many held-out sentences [`read_samples`] skipped.
    pub fn skipped(&self) -> usize {
        self.skipped
    }

    pub fn len(&self) -> usize {
        self.samples.len()
    }

    pub fn is_empty(&self) -> bool {
        self.samples.is_empty()
    }

    /// The SHA-256 of the samples as scored: for each, its number of
    /// tokens as a little endian `u64`, then for each token its text's
    /// length in bytes as a little endian `u64`, the text, its tag's length
    /// and the tag.
    pub fn sha256(&self) -> String {
        let mut bytes = Vec::new();
        let put = |bytes: &mut Vec<u8>, s: &str| {
            bytes.extend_from_slice(&(s.len() as u64).to_le_bytes());
            bytes.extend_from_slice(s.as_bytes());
        };
        for sample in &self.samples {
            bytes.extend_from_slice(&(sample.len() as u64).to_le_bytes());
            for (token, tag) in sample.tokens().iter().zip(sample.tags()) {
                put(&mut bytes, token.as_str());
                put(&mut bytes, tag.as_str());
            }
        }
        Sha256::of(&bytes).to_string()
    }
}

/// The factory of the recipe: the language, the default descriptor and no
/// tag dictionary, as the trainer tool builds it.
fn factory() -> Result<PosTaggerFactory> {
    Ok(PosTaggerFactory::new(LANGUAGE, None, None)?)
}

impl PosRecipe {
    /// The gate's task.
    pub fn task(&self) -> Task {
        Task::PartOfSpeechTagger
    }

    /// Trains the tagger from the training documents.
    ///
    /// # Errors
    ///
    /// What [`read_samples`] fails on, a corpus without a sample
    /// ([`Error::EmptySplit`]), or what training fails on.
    pub fn train(&self, train: &MascCorpus) -> Result<Trained> {
        let factory = factory()?;
        let mut stream = postag::PosEventStream::new(&factory)?;
        let mut events = Vec::new();
        let mut samples = 0;
        let skipped = read_samples(train, |sample| {
            stream.events(&sample, &mut events);
            samples += 1;
        })?;
        if samples == 0 {
            return Err(Error::EmptySplit { what: "training part-of-speech sample" });
        }
        let params = ud::training_parameters();
        let mut config = TrainingConfiguration::default();
        let t = postag::train_pos_model_from_events(&events, &factory, &params, &mut config, Stamp::at(TIMESTAMP_MS))?;
        Ok(Trained {
            bytes: t.package.to_bytes(),
            samples,
            skipped_sentences: skipped,
            events: events.len(),
            event_hash: t.report.event_hash.expect("trained from events"),
        })
    }

    /// The held-out samples, read as the training samples are.
    ///
    /// # Errors
    ///
    /// A sample the corpus fails on, or no sample ([`Error::EmptySplit`]).
    pub fn held_out<'c>(&self, held_out: &'c MascCorpus) -> Result<HeldOutTags<'c>> {
        let mut samples = Vec::new();
        let skipped = read_samples(held_out, |s| samples.push(s))?;
        let out = HeldOutTags { samples, skipped };
        if out.is_empty() {
            return Err(Error::EmptySplit { what: "held-out part-of-speech sample" });
        }
        Ok(out)
    }

    /// Each held-out sample's counts for the tagger in `bytes`, its tags
    /// asked for in `format`: the tokens tagged as the sample tags them,
    /// of all its tokens, as `POSEvaluator` counts them.
    ///
    /// # Errors
    ///
    /// A model file that is not a part-of-speech model, a sentence no tag
    /// sequence scores above the search's minimum for, or what the
    /// evaluator throws on.
    pub fn score(&self, bytes: &[u8], format: PosTagFormat, held_out: &HeldOutTags<'_>) -> Result<Vec<Counts>> {
        let file = ModelFile::open(bytes.to_vec()).map_err(|e| Error::Model(e.to_string()))?;
        let model = PosModel::open_with_format(file, format).map_err(|e| Error::Model(e.to_string()))?;
        let tagger = PosTagger::new(model);
        let tagset = tagger.pos_model().tagset().clone();
        let mut out = Vec::with_capacity(held_out.len());
        let (mut tags, mut probs) = (Vec::new(), Vec::new());
        let mut words: Vec<&str> = Vec::new();
        let mut labels: Vec<&str> = Vec::new();
        for sample in held_out.samples() {
            words.clear();
            words.extend(sample.tokens().iter().map(|t| t.as_str()));
            tags.clear();
            probs.clear();
            let found = tagger.tag(&words, &mut tags, &mut probs).map_err(|e| Error::Model(e.to_string()))?;
            if !found {
                return Err(Error::Model(format!("no tag sequence scored above the minimum for: {sample}")));
            }
            labels.clear();
            labels.extend(tags.iter().map(|&t| tagset.label(t)));
            let mut evaluator = PosEvaluator::default();
            evaluator.evaluate(sample, &labels)?;
            let correct = sample.tags().iter().zip(&labels).filter(|(r, p)| r.as_str() == **p).count();
            out.push(Counts::accuracy(correct as u64, evaluator.word_count()));
        }
        Ok(out)
    }

    /// Scores the trained model (in [`CANDIDATE_FORMAT`]) and the reference
    /// (in the recipe's reference format) on the same held-out samples and
    /// runs the gate. `reference_id` names the reference in the record
    /// ([`Reference::id`] of [`PosRecipe::reference`] for the published
    /// model), and
    /// `set_name` and `corpus_sha256` the held-out set.
    ///
    /// # Errors
    ///
    /// What [`PosRecipe::score`] fails on, or the gate's errors.
    pub fn gate(
        &self,
        trained: &[u8],
        reference_id: &str,
        reference: &[u8],
        held_out: &HeldOutTags<'_>,
        set_name: &str,
        corpus_sha256: &str,
    ) -> Result<ModelGateRecord> {
        let candidate = self.score(trained, CANDIDATE_FORMAT, held_out)?;
        let baseline = self.score(reference, self.reference_format, held_out)?;
        let units = candidate
            .into_iter()
            .zip(baseline)
            .map(|(candidate, reference)| PairedUnit { candidate, reference })
            .collect();
        let set = HeldOutSet {
            name: set_name.to_owned(),
            corpus_sha256: corpus_sha256.to_owned(),
            samples_sha256: held_out.sha256(),
        };
        let id = |id: &str, bytes: &[u8]| ModelId { id: id.to_owned(), sha256: Sha256::of(bytes).to_string() };
        ModelGateRecord::new(self.task(), set, id(self.model_id, trained), id(reference_id, reference), units)
            .map_err(Error::Gate)
    }
}

#[cfg(test)]
mod tests {
    use turbo_analysis::Span;
    use turbo_formats::source::Field;

    use super::*;

    fn sample<'a>(text: &'a str, tokens: &[(u32, u32)], tags: &[&'a str]) -> PosSample<'a> {
        let fields = tokens.iter().map(|&(a, b)| Field::of(text, Span::new(a, b))).collect();
        let tags = tags.iter().zip(tokens).map(|(t, &(a, b))| Field::made(t.to_string(), Span::new(a, b))).collect();
        PosSample::new(fields, tags, Span::new(0, text.len() as u32)).unwrap()
    }

    #[test]
    fn the_held_out_digest_covers_tokens_tags_and_their_boundaries() {
        let text = "Dogs ran.";
        let one =
            HeldOutTags { samples: vec![sample(text, &[(0, 4), (5, 8), (8, 9)], &["NNS", "VBD", "."])], skipped: 0 };
        let mut bytes = Vec::new();
        bytes.extend_from_slice(&3u64.to_le_bytes());
        for s in ["Dogs", "NNS", "ran", "VBD", ".", "."] {
            bytes.extend_from_slice(&(s.len() as u64).to_le_bytes());
            bytes.extend_from_slice(s.as_bytes());
        }
        assert_eq!(one.sha256(), Sha256::of(&bytes).to_string());
        // Another tag, or the same text cut into other tokens, is another set.
        let tag =
            HeldOutTags { samples: vec![sample(text, &[(0, 4), (5, 8), (8, 9)], &["NNS", "VBN", "."])], skipped: 0 };
        let cut = HeldOutTags { samples: vec![sample(text, &[(0, 4), (5, 9)], &["NNS", "VBD"])], skipped: 0 };
        assert_ne!(one.sha256(), tag.sha256());
        assert_ne!(one.sha256(), cut.sha256());
    }

    #[test]
    fn the_recipe_gates_against_the_penn_tagger_and_scores_its_own_outcomes_unchanged() {
        assert_eq!(POS.reference, Reference::Recorded(EN_POS_MAXENT));
        assert_eq!(POS.reference_format, PosTagFormat::Penn);
        assert_eq!(CANDIDATE_FORMAT, PosTagFormat::Custom);
        assert_eq!(POS.task(), Task::PartOfSpeechTagger);
        assert_eq!(POS.reference.file().unwrap(), "en-pos-maxent.bin");
        assert!(matches!(POS.reference.check(b"not the model"), Err(Error::ReferenceDigest { .. })));
    }
}
