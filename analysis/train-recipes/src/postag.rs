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
//! Held-out sentences are skipped the same way. The card records how many
//! training and held-out sentences were skipped, and the recipe fails when
//! either is more than [`crate::MAX_SKIPPED_SENTENCES`] of the sentences
//! read, so a held-out document whose tags are missing cannot drop out of
//! the gate.
//!
//! The reference is OpenNLP's English maxent tagger from the 1.5 models
//! ([`EN_POS_MAXENT`]). OpenNLP's only English taggers trained on Penn
//! tags are that one and the 1.5 perceptron tagger, and the maxent one is
//! trained with the algorithm the recipe uses. Both models are scored in
//! the reference's tag set. The reference is asked for Penn tags, which
//! are its own outcomes and come out unconverted, and the trained model
//! gives its outcomes as they are ([`CANDIDATE_FORMAT`]), MASC's tags.
//! Where MASC writes a tag the reference writes another way
//! ([`REFERENCE_SPELLINGS`]), the held-out tags and both models' tags are
//! compared in the reference's spelling. A held-out token whose tag the
//! reference cannot give even so would count against the reference
//! whatever it did, so it is left out of both models' scores
//! ([`ReferenceTags`]); the card records those tokens by tag, and the
//! recipe fails when they are more than [`crate::MAX_TOKENS_LEFT_OUT`] of
//! the held-out tokens. OpenNLP describes the
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
//! Both models are scored on the same held-out samples, one unit per
//! sentence: OpenNLP's `POSEvaluator` reads each sentence's tags, and the
//! unit counts the scored tokens tagged right. The units are gated by
//! [`turbo_eval::model_gate`].

use std::collections::{BTreeMap, HashSet};

use turbo_analysis::{Tag, TagSet};
use turbo_eval::model_gate::{Counts, HeldOutSet, ModelGateRecord, ModelId, PairedUnit, Task};
use turbo_formats::evaluate::PosEvaluator;
use turbo_formats::masc::{MascCorpus, MascError, MascPosReader};
use turbo_formats::sample::PosSample;
use turbo_maxent::ModelFile;
use turbo_postag::{PosModel, PosTagFormat, PosTagger};
use turbo_train::TrainingConfiguration;
use turbo_train_components::{PosTaggerFactory, Stamp, postag, ud};
use turbo_zoo::Sha256;

use crate::card::{HeldOutCard, SpellingCard, TagCount, TokenCoverage};
use crate::error::{Error, LeftOut, Result};
use crate::segmentation::{LANGUAGE, TIMESTAMP_MS};
use crate::{GateRun, MAX_TOKENS_LEFT_OUT, RecordedReference, Reference, Trained, check_skipped};

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

/// A tag MASC writes one way and the reference another.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Spelling {
    /// The tag as MASC, and so the trained model, writes it.
    pub masc: &'static str,
    /// The tag the reference gives the same tokens.
    pub reference: &'static str,
}

/// The MASC tags [`EN_POS_MAXENT`] writes another way. MASC 3.0.0 tags
/// brackets (`(`, `)`, `{`, `}`) `(` and `)`, where the reference has the
/// Penn Treebank's `-LRB-` and `-RRB-`, and has tags of its own for
/// hyphens and dashes (`-`), ellipses (`...`) and semicolons (`;`), tokens
/// the reference tags `:`. The tests check the reference's side when the
/// reference is present.
pub const REFERENCE_SPELLINGS: [Spelling; 5] = [
    Spelling { masc: "(", reference: "-LRB-" },
    Spelling { masc: ")", reference: "-RRB-" },
    Spelling { masc: "-", reference: ":" },
    Spelling { masc: "...", reference: ":" },
    Spelling { masc: ";", reference: ":" },
];

/// `tag` as the reference writes it: its [`REFERENCE_SPELLINGS`] entry's
/// reference spelling, or `tag` itself.
pub fn in_reference_spelling(tag: &str) -> &str {
    REFERENCE_SPELLINGS.iter().find(|s| s.masc == tag).map_or(tag, |s| s.reference)
}

/// The tags a reference tagger can give, which decide the held-out
/// tokens the gate scores: those whose tag, in the reference's spelling,
/// is one of them.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ReferenceTags {
    tags: HashSet<String>,
}

impl ReferenceTags {
    /// The tags of the reference's tag set, as it gives them.
    pub fn of(tagset: &TagSet) -> ReferenceTags {
        let tags = (0..tagset.len()).map(|i| tagset.label(Tag(i as u16)).to_owned()).collect();
        ReferenceTags { tags }
    }

    /// Whether a held-out token tagged `tag` is scored.
    pub fn scores(&self, tag: &str) -> bool {
        self.tags.contains(in_reference_spelling(tag))
    }

    /// The held-out tokens left out of the scores, by tag in the
    /// reference's spelling, sorted by tag.
    pub fn left_out(&self, held_out: &HeldOutTags<'_>) -> Vec<TagCount> {
        let mut counts: BTreeMap<&str, u64> = BTreeMap::new();
        for sample in held_out.samples() {
            for tag in sample.tags() {
                if !self.scores(tag.as_str()) {
                    *counts.entry(in_reference_spelling(tag.as_str())).or_default() += 1;
                }
            }
        }
        counts.into_iter().map(|(tag, tokens)| TagCount { tag: tag.to_owned(), tokens }).collect()
    }
}

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

    /// The tokens of the samples.
    pub fn tokens(&self) -> u64 {
        self.samples.iter().map(|s| s.len() as u64).sum()
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
    /// ([`Error::EmptySplit`]), more skipped sentences than
    /// [`crate::MAX_SKIPPED_SENTENCES`] allows ([`Error::TooManyLeftOut`]),
    /// or what training fails on.
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
        check_skipped(LeftOut::TrainingSentences, skipped, samples)?;
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
    /// A sample the corpus fails on, no sample ([`Error::EmptySplit`]), or
    /// more skipped sentences than [`crate::MAX_SKIPPED_SENTENCES`] allows
    /// ([`Error::TooManyLeftOut`]).
    pub fn held_out<'c>(&self, held_out: &'c MascCorpus) -> Result<HeldOutTags<'c>> {
        let mut samples = Vec::new();
        let skipped = read_samples(held_out, |s| samples.push(s))?;
        let out = HeldOutTags { samples, skipped };
        if out.is_empty() {
            return Err(Error::EmptySplit { what: "held-out part-of-speech sample" });
        }
        check_skipped(LeftOut::HeldOutSentences, out.skipped, out.len())?;
        Ok(out)
    }

    /// Each held-out sample's counts for the tagger in `bytes`, its tags
    /// asked for in `format`: of the tokens `reference` scores, those
    /// tagged as the sample tags them, compared in the reference's
    /// spelling.
    ///
    /// # Errors
    ///
    /// A model file that is not a part-of-speech model, a sentence no tag
    /// sequence scores above the search's minimum for, or what the
    /// evaluator throws on.
    pub fn score(
        &self,
        bytes: &[u8],
        format: PosTagFormat,
        held_out: &HeldOutTags<'_>,
        reference: &ReferenceTags,
    ) -> Result<Vec<Counts>> {
        score_with(&open(bytes, format)?, held_out, reference)
    }

    /// Scores the trained model (in [`CANDIDATE_FORMAT`]) and the reference
    /// (in the recipe's reference format) on the same held-out tokens, those
    /// whose tag the reference can give ([`ReferenceTags`]), and runs the
    /// gate. `reference_id` names the reference in the record
    /// ([`Reference::id`] of [`PosRecipe::reference`] for the published
    /// model), and `set_name` and `corpus_sha256` the held-out set.
    ///
    /// # Errors
    ///
    /// What [`PosRecipe::score`] fails on, a reference that cannot give the
    /// tags of more than [`MAX_TOKENS_LEFT_OUT`] of the held-out tokens
    /// ([`Error::TagsOutsideReference`]), or the gate's errors.
    pub fn gate(
        &self,
        trained: &[u8],
        reference_id: &str,
        reference: &[u8],
        held_out: &HeldOutTags<'_>,
        set_name: &str,
        corpus_sha256: &str,
    ) -> Result<GateRun> {
        let reference_tagger = open(reference, self.reference_format)?;
        let tags = ReferenceTags::of(reference_tagger.pos_model().tagset());
        let left_out = tags.left_out(held_out);
        let tokens = held_out.tokens();
        if !MAX_TOKENS_LEFT_OUT.allows(left_out.iter().map(|t| t.tokens).sum(), tokens) {
            return Err(Error::TagsOutsideReference {
                reference: reference_id.to_owned(),
                left_out,
                of: tokens,
                limit: MAX_TOKENS_LEFT_OUT,
            });
        }
        let candidate = self.score(trained, CANDIDATE_FORMAT, held_out, &tags)?;
        let baseline = score_with(&reference_tagger, held_out, &tags)?;
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
        let record =
            ModelGateRecord::new(self.task(), set, id(self.model_id, trained), id(reference_id, reference), units)
                .map_err(Error::Gate)?;
        let spellings = REFERENCE_SPELLINGS
            .iter()
            .map(|s| SpellingCard { masc: s.masc.to_owned(), reference: s.reference.to_owned() })
            .collect();
        let held_out = HeldOutCard {
            samples: held_out.len(),
            skipped_sentences: held_out.skipped(),
            tokens: Some(TokenCoverage { tokens, spellings, left_out }),
        };
        Ok(GateRun { record, held_out })
    }
}

/// The tagger in `bytes`, its tags asked for in `format`.
fn open(bytes: &[u8], format: PosTagFormat) -> Result<PosTagger> {
    let file = ModelFile::open(bytes.to_vec()).map_err(|e| Error::Model(e.to_string()))?;
    let model = PosModel::open_with_format(file, format).map_err(|e| Error::Model(e.to_string()))?;
    Ok(PosTagger::new(model))
}

/// [`PosRecipe::score`] with a tagger already open.
fn score_with(tagger: &PosTagger, held_out: &HeldOutTags<'_>, reference: &ReferenceTags) -> Result<Vec<Counts>> {
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
        let (mut scored, mut right, mut right_as_written) = (0u64, 0u64, 0u64);
        for (expected, &given) in sample.tags().iter().zip(&labels) {
            right_as_written += u64::from(expected.as_str() == given);
            if reference.scores(expected.as_str()) {
                scored += 1;
                right += u64::from(in_reference_spelling(expected.as_str()) == in_reference_spelling(given));
            }
        }
        // The tags as written, counted over every token, are what
        // `POSEvaluator` counts.
        debug_assert_eq!(evaluator.word_count(), sample.len() as u64);
        debug_assert_eq!(right_as_written as f64 / sample.len() as f64, evaluator.word_accuracy());
        out.push(Counts::accuracy(right, scored));
    }
    Ok(out)
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
    fn held_out_tags_are_scored_in_the_reference_s_spelling_when_it_can_give_them() {
        assert_eq!(in_reference_spelling("("), "-LRB-");
        assert_eq!(in_reference_spelling(")"), "-RRB-");
        for colon in ["-", "...", ";", ":"] {
            assert_eq!(in_reference_spelling(colon), ":");
        }
        assert_eq!(in_reference_spelling("-LRB-"), "-LRB-");
        assert_eq!(in_reference_spelling("NN"), "NN");

        let tags = ReferenceTags::of(&TagSet::new("reference", &["-LRB-", ":", "NN", "UH"]));
        for scored in ["(", "-LRB-", "-", "...", ";", ":", "NN"] {
            assert!(tags.scores(scored), "{scored}");
        }
        for left_out in [")", "-RRB-", "PRN", "UH`", "nn"] {
            assert!(!tags.scores(left_out), "{left_out}");
        }
        let text = "( Agh ) they ; Agh";
        let held_out = HeldOutTags {
            samples: vec![
                sample(text, &[(0, 1), (2, 5), (6, 7)], &["(", "UH`", ")"]),
                sample(text, &[(8, 12), (13, 14), (15, 18)], &["PRN", ";", "UH`"]),
            ],
            skipped: 0,
        };
        assert_eq!(held_out.tokens(), 6);
        let count = |tag: &str, tokens| TagCount { tag: tag.to_owned(), tokens };
        // Sorted by tag, a bracket under the reference's spelling.
        assert_eq!(tags.left_out(&held_out), [count("-RRB-", 1), count("PRN", 1), count("UH`", 2)]);
    }
}
