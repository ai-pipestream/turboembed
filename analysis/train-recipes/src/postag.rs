//! The English part-of-speech tagger trained on MASC's Penn tags, and its
//! gate against OpenNLP's English Penn tagger, `en-pos-maxent.bin`.
//!
//! The recipe trains on the training documents of [`crate::masc::SPLIT`]
//! with the factory defaults (the default feature generator descriptor,
//! no tag dictionary) and the parameters the published UD 1.3 models were
//! trained with ([`turbo_train_components::ud`]: GIS, 300 iterations,
//! cutoff 5, 4 threads). Training uses no randomness, so the recipe has
//! no seed. The reference was trained otherwise: its manifest records
//! OpenNLP 1.5.0, 100 iterations and cutoff 5, and it holds a tag
//! dictionary (`tags.tagdict`) and no feature generator descriptor, so
//! the gate compares the two models' data and settings together.
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
//! trained with the algorithm the recipe uses. The reference is asked for
//! Penn tags, which are its own outcomes and come out unconverted, and the
//! trained model gives its outcomes as they are ([`CANDIDATE_FORMAT`]),
//! MASC's tags. The trained model's tag is compared with the held-out tag
//! as written. The reference's tag is compared with it through a mapping
//! of the reference's tags ([`REFERENCE_MAPPINGS`]): where MASC and the
//! reference follow different conventions (brackets, hyphens, ellipses,
//! semicolons, quotation marks), the reference's tag for the convention
//! is counted right too. The mapping is by tag, not by token, so it is
//! lenient: the reference's ``` `` ```, its tag for an opening quotation
//! mark, counts right on every held-out `''`, closing quotation marks
//! too. It only ever favours the reference. A held-out token whose tag the reference cannot
//! give, as written or through the mapping, would count against it
//! whatever it did, so it is left out of both models' scores
//! ([`ReferenceTags`]); the card records those tokens by tag, and the
//! recipe fails when they are more than [`crate::MAX_TOKENS_LEFT_OUT`] of
//! the held-out tokens. The card also lists the reference's most frequent
//! mismatches on the scored tokens ([`MISMATCHES_LISTED`]), so a gap of
//! convention the mapping does not cover shows. The reference's training
//! data is not published, and the zoo does not list it: a run reads it
//! from its references directory, checks it against the recorded size
//! and SHA-256, and only scores it. It is never shipped or bundled, and
//! the trained model is trained on MASC alone.
//!
//! The English UD tagger, `opennlp-en-ud-ewt-pos-1.3-2.5.4`, is not the
//! reference: it was trained on UD tags, and asked for Penn tags
//! `POSTagFormatMapper` converts each of its outcomes to one Penn tag
//! (`VERB` to `VB`, `NOUN` to `NN`, `PUNCT` to `.`), so it can never give
//! `VBD`, `NNS`, `,` or `TO`, and a gate against it would pass any tagger.
//!
//! Both models are scored on the same held-out samples, one unit per
//! sentence: OpenNLP's `POSEvaluator` reads each sentence's tags, and the
//! unit counts the scored tokens each model tags right. The units are
//! gated by [`turbo_eval::model_gate`].

use std::collections::{BTreeMap, HashSet};

use turbo_analysis::{Tag, TagSet};
use turbo_eval::model_gate::{
    Counts, HeldOutSet, ModelGateRecord, ModelId, PairedUnit, Scoring, TagMapping, TagScoring, Task,
};
use turbo_formats::evaluate::PosEvaluator;
use turbo_formats::masc::{MascCorpus, MascError, MascPosReader};
use turbo_formats::sample::PosSample;
use turbo_maxent::ModelFile;
use turbo_postag::{PosModel, PosTagFormat, PosTagger};
use turbo_train::TrainingConfiguration;
use turbo_train_components::{PosTaggerFactory, Stamp, postag, ud};
use turbo_zoo::Sha256;

use crate::card::{HeldOutCard, Mismatch, TagCount, TokenCoverage};
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

/// A held-out tag for which the reference's tag is another: a convention
/// in which MASC 3.0.0 and the reference differ. Only the reference's
/// tag is compared through it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ReferenceMapping {
    /// The held-out tag, as MASC writes it.
    pub held_out: &'static str,
    /// The reference's tag counted right for it.
    pub reference: &'static str,
}

/// The reference's tags counted right for another held-out tag
/// ([`EN_POS_MAXENT`] against MASC 3.0.0). Counted with the recipe's
/// reader over the 390 documents the split reads from the recorded
/// release, the Twitter genre left out (corpus digest
/// `445449f535cac62e74b2712ad975d7b727e72b8276720dd27492266565ae0afd`):
///
/// - Brackets: MASC tags `(` and `{` as `(`, and `)` and `}` as `)`,
///   where the reference has the Penn Treebank's `-LRB-` and `-RRB-`.
/// - Hyphens: MASC tags a `-` token `-` (1,108 times) or `:` (1,243
///   times), where the reference tags it `:`. MASC tags an en dash `–`
///   `NN` (91 times) or `-` (9 times), and the reference tags none of the
///   25 held-out en dashes `:`, so the mapping is for hyphens, not dashes.
/// - Ellipses: MASC tags `...` (and its spaced forms) `...`, 425 tokens;
///   the reference `:`.
/// - Semicolons: MASC tags them `:` as the reference does (725 tokens),
///   and twice `;`.
/// - Double quotation marks: MASC tags a `"` token `''` whether it opens
///   or closes a quotation. Of the 4,785 `"` tokens, 4,777 are tagged
///   `''` (2,054 opening, 2,389 closing and 334 neither) and 8 are tagged
///   `"`, a tag the reference cannot give. A mark counts as opening when
///   the character before it is none, whitespace, `(`, `[` or `{` and the
///   one after it is not whitespace; as closing when the character after
///   it is none, whitespace or one of `.,;:!?)]}` and the one before it
///   is not whitespace; and as neither when both or none of these hold.
///   MASC gives ``` `` ``` only to a token that is a backtick (80 tokens,
///   in 13 documents). The reference tags an opening quotation mark
///   ``` `` ``` and a closing one `''`. The mapping is by tag, so it also
///   counts the reference's ``` `` ``` right on a closing mark: of the 336
///   held-out `"` tokens it credits, 272 open a quotation, 34 close one
///   and 30 are neither. It only ever favours the reference, and the 64
///   that do not open a quotation are at most 0.08 points of its score.
///
/// The tests check the reference's side when the reference is present.
pub const REFERENCE_MAPPINGS: [ReferenceMapping; 6] = [
    ReferenceMapping { held_out: "(", reference: "-LRB-" },
    ReferenceMapping { held_out: ")", reference: "-RRB-" },
    ReferenceMapping { held_out: "-", reference: ":" },
    ReferenceMapping { held_out: "...", reference: ":" },
    ReferenceMapping { held_out: ";", reference: ":" },
    ReferenceMapping { held_out: "''", reference: "``" },
];

/// Whether the reference's tag `given` is right for the held-out tag
/// `held_out`: the same tag, or the tag [`REFERENCE_MAPPINGS`] maps it to.
pub fn reference_is_right(held_out: &str, given: &str) -> bool {
    given == held_out || REFERENCE_MAPPINGS.iter().any(|m| m.held_out == held_out && m.reference == given)
}

/// [`REFERENCE_MAPPINGS`] as the gate record and the card write them.
pub fn tag_mappings() -> Vec<TagMapping> {
    REFERENCE_MAPPINGS
        .iter()
        .map(|m| TagMapping { held_out: m.held_out.to_owned(), reference: m.reference.to_owned() })
        .collect()
}

/// Which model a tagger's tags are scored as, which decides the format
/// its tags are asked for and how they are compared with the held-out
/// tags.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Scored {
    /// The trained model: its outcomes as they are ([`CANDIDATE_FORMAT`]),
    /// compared with the held-out tag as written.
    Candidate,
    /// The reference: its tags in the recipe's reference format, compared
    /// through [`REFERENCE_MAPPINGS`] ([`reference_is_right`]).
    Reference,
}

impl Scored {
    /// Whether `given` is right for the held-out tag `held_out`.
    pub fn is_right(self, held_out: &str, given: &str) -> bool {
        match self {
            Scored::Candidate => given == held_out,
            Scored::Reference => reference_is_right(held_out, given),
        }
    }
}

/// How many of the reference's most frequent mismatches the card lists.
pub const MISMATCHES_LISTED: usize = 10;

/// The tags a reference tagger can give, which decide the held-out
/// tokens the gate scores: those whose tag is one of them, as written or
/// through [`REFERENCE_MAPPINGS`].
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
        self.tags.contains(tag)
            || REFERENCE_MAPPINGS.iter().any(|m| m.held_out == tag && self.tags.contains(m.reference))
    }

    /// The held-out tokens left out of the scores, by their tag as
    /// written, sorted by tag.
    pub fn left_out(&self, held_out: &HeldOutTags<'_>) -> Vec<TagCount> {
        let mut counts: BTreeMap<&str, u64> = BTreeMap::new();
        for sample in held_out.samples() {
            for tag in sample.tags() {
                if !self.scores(tag.as_str()) {
                    *counts.entry(tag.as_str()).or_default() += 1;
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

    /// The SHA-256 of the samples as scored with a reference of the tags
    /// `reference`: for each sample, its number of tokens as a little
    /// endian `u64`, then for each token its text's length in bytes as a
    /// little endian `u64`, the text, its tag's length and the tag, and a
    /// byte, 1 when the token is scored ([`ReferenceTags::scores`]) and 0
    /// when it is left out; after the samples, the number of
    /// [`REFERENCE_MAPPINGS`] as a `u64` and each one's held-out tag and
    /// reference tag, each as its length and its bytes. Two gates whose
    /// references leave out other tokens, or that map other tags, have
    /// other digests.
    pub fn sha256(&self, reference: &ReferenceTags) -> String {
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
                bytes.push(u8::from(reference.scores(tag.as_str())));
            }
        }
        bytes.extend_from_slice(&(REFERENCE_MAPPINGS.len() as u64).to_le_bytes());
        for m in REFERENCE_MAPPINGS {
            put(&mut bytes, m.held_out);
            put(&mut bytes, m.reference);
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

    /// Reads the training samples as [`PosRecipe::train`] does, without
    /// making events, and checks them as it does.
    ///
    /// # Errors
    ///
    /// As [`PosRecipe::train`], but for training.
    pub fn check_training(&self, train: &MascCorpus) -> Result<()> {
        let mut samples = 0;
        let skipped = read_samples(train, |_| samples += 1)?;
        if samples == 0 {
            return Err(Error::EmptySplit { what: "training part-of-speech sample" });
        }
        check_skipped(LeftOut::TrainingSentences, skipped, samples)
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

    /// Each held-out sample's counts for the tagger in `bytes`, scored as
    /// `scored` ([`Scored`]: its tags asked for in that model's format, and
    /// compared with the held-out tags as that model's are), over the
    /// tokens `reference` scores.
    ///
    /// # Errors
    ///
    /// A model file that is not a part-of-speech model, a sentence no tag
    /// sequence scores above the search's minimum for, or what the
    /// evaluator throws on.
    pub fn score(
        &self,
        bytes: &[u8],
        scored: Scored,
        held_out: &HeldOutTags<'_>,
        reference: &ReferenceTags,
    ) -> Result<Vec<Counts>> {
        Ok(score_with(&open(bytes, self.format(scored))?, scored, held_out, reference)?.units)
    }

    /// The format a model scored as `scored` is asked for its tags in.
    pub fn format(&self, scored: Scored) -> PosTagFormat {
        match scored {
            Scored::Candidate => CANDIDATE_FORMAT,
            Scored::Reference => self.reference_format,
        }
    }

    /// The held-out tokens the reference, of the tags `tags`, cannot give
    /// the tag of, by tag ([`ReferenceTags::left_out`]).
    ///
    /// # Errors
    ///
    /// More of them than [`MAX_TOKENS_LEFT_OUT`] allows
    /// ([`Error::TagsOutsideReference`], naming the reference
    /// `reference_id`).
    pub fn check_coverage(
        &self,
        reference_id: &str,
        tags: &ReferenceTags,
        held_out: &HeldOutTags<'_>,
    ) -> Result<Vec<TagCount>> {
        let left_out = tags.left_out(held_out);
        let tokens = held_out.tokens();
        if MAX_TOKENS_LEFT_OUT.allows(left_out.iter().map(|t| t.tokens).sum(), tokens) {
            Ok(left_out)
        } else {
            Err(Error::TagsOutsideReference {
                reference: reference_id.to_owned(),
                left_out,
                of: tokens,
                limit: MAX_TOKENS_LEFT_OUT,
            })
        }
    }

    /// The tags of the reference model in `reference`, asked for in the
    /// recipe's reference format.
    ///
    /// # Errors
    ///
    /// A model file that is not a part-of-speech model.
    pub fn reference_tags(&self, reference: &[u8]) -> Result<ReferenceTags> {
        Ok(ReferenceTags::of(open(reference, self.reference_format)?.pos_model().tagset()))
    }

    /// Scores the trained model ([`Scored::Candidate`]) and the reference
    /// ([`Scored::Reference`]) on the same held-out tokens, those whose tag
    /// the reference can give ([`ReferenceTags`]), and runs the gate.
    /// `reference_id` names the reference in the record ([`Reference::id`]
    /// of [`PosRecipe::reference`] for the published model), and
    /// `set_name` and `corpus_sha256` the held-out set.
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
        let left_out = self.check_coverage(reference_id, &tags, held_out)?;
        let candidate = self.score(trained, Scored::Candidate, held_out, &tags)?;
        let baseline = score_with(&reference_tagger, Scored::Reference, held_out, &tags)?;
        let units = candidate
            .into_iter()
            .zip(baseline.units)
            .map(|(candidate, reference)| PairedUnit { candidate, reference })
            .collect();
        let reference_mappings = tag_mappings();
        let set = HeldOutSet {
            name: set_name.to_owned(),
            corpus_sha256: corpus_sha256.to_owned(),
            samples_sha256: held_out.sha256(&tags),
            skipped_sentences: held_out.skipped() as u64,
            scoring: Scoring::Tags(TagScoring {
                tokens: held_out.tokens(),
                left_out: left_out.iter().map(|t| t.tokens).sum(),
                reference_mappings: reference_mappings.clone(),
            }),
        };
        let id = |id: &str, bytes: &[u8]| ModelId { id: id.to_owned(), sha256: Sha256::of(bytes).to_string() };
        let record =
            ModelGateRecord::new(self.task(), set, id(self.model_id, trained), id(reference_id, reference), units)
                .map_err(Error::Gate)?;
        let held_out = HeldOutCard {
            samples: held_out.len(),
            skipped_sentences: held_out.skipped(),
            tokens: Some(TokenCoverage {
                tokens: held_out.tokens(),
                reference_mappings,
                left_out,
                reference_mismatches: baseline.mismatches,
            }),
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

/// A tagger's counts on the held-out samples, and its most frequent
/// mismatches on the scored tokens.
struct Scores {
    units: Vec<Counts>,
    /// At most [`MISMATCHES_LISTED`], most frequent first.
    mismatches: Vec<Mismatch>,
}

/// [`PosRecipe::score`] with a tagger already open.
fn score_with(
    tagger: &PosTagger,
    scored_as: Scored,
    held_out: &HeldOutTags<'_>,
    reference: &ReferenceTags,
) -> Result<Scores> {
    let tagset = tagger.pos_model().tagset().clone();
    let mut units = Vec::with_capacity(held_out.len());
    let mut mismatches: BTreeMap<(&str, &str), u64> = BTreeMap::new();
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
                if scored_as.is_right(expected.as_str(), given) {
                    right += 1;
                } else {
                    *mismatches.entry((expected.as_str(), given)).or_default() += 1;
                }
            }
        }
        // The tags as written, counted over every token, are what
        // `POSEvaluator` counts.
        debug_assert_eq!(evaluator.word_count(), sample.len() as u64);
        debug_assert_eq!(right_as_written as f64 / sample.len() as f64, evaluator.word_accuracy());
        units.push(Counts::accuracy(right, scored));
    }
    Ok(Scores { units, mismatches: most_frequent(mismatches) })
}

/// The [`MISMATCHES_LISTED`] most frequent mismatches, most frequent
/// first, then by held-out tag and given tag.
fn most_frequent(mismatches: BTreeMap<(&str, &str), u64>) -> Vec<Mismatch> {
    let mut all: Vec<((&str, &str), u64)> = mismatches.into_iter().collect();
    // Stable, so equal counts keep the map's order of the tags.
    all.sort_by_key(|a| std::cmp::Reverse(a.1));
    all.into_iter()
        .take(MISMATCHES_LISTED)
        .map(|((held_out, given), tokens)| Mismatch { held_out: held_out.to_owned(), given: given.to_owned(), tokens })
        .collect()
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
    fn the_held_out_digest_covers_tokens_tags_what_is_scored_and_the_mapping() {
        let text = "Dogs ran.";
        let reference = ReferenceTags::of(&TagSet::new("reference", &["NNS", "VBD", "VBN", "."]));
        let one =
            HeldOutTags { samples: vec![sample(text, &[(0, 4), (5, 8), (8, 9)], &["NNS", "VBD", "."])], skipped: 0 };
        let mut bytes = Vec::new();
        let put = |bytes: &mut Vec<u8>, s: &str| {
            bytes.extend_from_slice(&(s.len() as u64).to_le_bytes());
            bytes.extend_from_slice(s.as_bytes());
        };
        bytes.extend_from_slice(&3u64.to_le_bytes());
        for (token, tag) in [("Dogs", "NNS"), ("ran", "VBD"), (".", ".")] {
            put(&mut bytes, token);
            put(&mut bytes, tag);
            bytes.push(1);
        }
        bytes.extend_from_slice(&(REFERENCE_MAPPINGS.len() as u64).to_le_bytes());
        for m in REFERENCE_MAPPINGS {
            put(&mut bytes, m.held_out);
            put(&mut bytes, m.reference);
        }
        assert_eq!(one.sha256(&reference), Sha256::of(&bytes).to_string());
        // Another tag, the same text cut into other tokens, or a reference
        // that leaves another token out, is another set.
        let tag =
            HeldOutTags { samples: vec![sample(text, &[(0, 4), (5, 8), (8, 9)], &["NNS", "VBN", "."])], skipped: 0 };
        let cut = HeldOutTags { samples: vec![sample(text, &[(0, 4), (5, 9)], &["NNS", "VBD"])], skipped: 0 };
        let no_dot = ReferenceTags::of(&TagSet::new("reference", &["NNS", "VBD", "VBN"]));
        assert_ne!(one.sha256(&reference), tag.sha256(&reference));
        assert_ne!(one.sha256(&reference), cut.sha256(&reference));
        assert_ne!(one.sha256(&reference), one.sha256(&no_dot));
    }

    #[test]
    fn only_the_reference_s_tags_are_compared_through_the_mapping() {
        for (held_out, given) in [("(", "-LRB-"), (")", "-RRB-"), ("-", ":"), ("...", ":"), (";", ":"), ("''", "``")] {
            assert!(Scored::Reference.is_right(held_out, given), "{held_out} {given}");
            assert!(!Scored::Candidate.is_right(held_out, given), "{held_out} {given}");
        }
        for same in ["(", ":", "''", "``", "NN"] {
            assert!(Scored::Reference.is_right(same, same) && Scored::Candidate.is_right(same, same));
        }
        // The mapping goes one way, from the held-out tag to the reference's.
        for (held_out, given) in [("-LRB-", "("), (":", "-"), (":", "..."), ("``", "''"), ("-", "..."), ("NN", "NNS")] {
            assert!(!Scored::Reference.is_right(held_out, given), "{held_out} {given}");
        }
        assert_eq!(MISMATCHES_LISTED, 10);
    }

    #[test]
    fn held_out_tokens_are_scored_when_the_reference_can_give_their_tag_as_written_or_mapped() {
        let tags = ReferenceTags::of(&TagSet::new("reference", &["-LRB-", ":", "NN", "UH", "``"]));
        for scored in ["(", "-LRB-", "-", "...", ";", ":", "NN", "``", "''"] {
            assert!(tags.scores(scored), "{scored}");
        }
        for left_out in [")", "-RRB-", "PRN", "UH`", "nn", "\""] {
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
        // Sorted by the tag as written.
        assert_eq!(tags.left_out(&held_out), [count(")", 1), count("PRN", 1), count("UH`", 2)]);
    }

    #[test]
    fn the_most_frequent_mismatches_come_first_and_are_at_most_ten() {
        let mut counts = BTreeMap::new();
        let held: Vec<String> = (0..12).map(|i| format!("T{i:02}")).collect();
        for (i, tag) in held.iter().enumerate() {
            counts.insert((tag.as_str(), "NN"), (i % 4) as u64 + 1);
        }
        let top = most_frequent(counts);
        assert_eq!(top.len(), MISMATCHES_LISTED);
        let got: Vec<(&str, u64)> = top.iter().map(|m| (m.held_out.as_str(), m.tokens)).collect();
        assert_eq!(
            got,
            [
                ("T03", 4),
                ("T07", 4),
                ("T11", 4),
                ("T02", 3),
                ("T06", 3),
                ("T10", 3),
                ("T01", 2),
                ("T05", 2),
                ("T09", 2),
                ("T00", 1)
            ]
        );
    }
}
