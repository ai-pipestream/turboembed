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
//!
//! The UD taggers ([`crate::ud`]) are scored and gated by the same code,
//! each as a [`PosRecipe`] of its own with no mapping and the UD limit on
//! the tokens left out ([`PosRecipe::reference_mappings`],
//! [`PosRecipe::max_left_out`]).

mod scoring;
#[cfg(test)]
mod tests;

use turbo_eval::model_gate::{Counts, HeldOutSet, ModelGateRecord, ModelId, Scoring, TagMapping, TagScoring, Task};
use turbo_formats::masc::{MascCorpus, MascError, MascPosReader};
use turbo_formats::sample::PosSample;
use turbo_postag::PosTagFormat;
use turbo_train::TrainingConfiguration;
use turbo_train_components::{PosTaggerFactory, Stamp, postag, ud};
use turbo_zoo::Sha256;

use crate::card::{HeldOutCard, TagCount, TokenCoverage};
use crate::error::{Error, LeftOut, Result};
use crate::segmentation::{LANGUAGE, TIMESTAMP_MS};
use crate::{GateRun, Limit, MAX_TOKENS_LEFT_OUT, RecordedReference, Reference, Trained, check_skipped, spans};

pub(crate) use self::scoring::most_frequent;
pub use self::scoring::{HeldOutTags, ReferenceTags};
use self::scoring::{open, score_with};

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
/// `held_out` under `mappings`: the same tag, or the tag a mapping maps it
/// to ([`REFERENCE_MAPPINGS`] for the MASC tagger).
pub fn reference_is_right(mappings: &[ReferenceMapping], held_out: &str, given: &str) -> bool {
    given == held_out || mappings.iter().any(|m| m.held_out == held_out && m.reference == given)
}

/// `mappings` as the gate record and the card write them.
pub fn tag_mappings(mappings: &[ReferenceMapping]) -> Vec<TagMapping> {
    mappings.iter().map(|m| TagMapping { held_out: m.held_out.to_owned(), reference: m.reference.to_owned() }).collect()
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
    /// through the recipe's mappings ([`reference_is_right`]).
    Reference,
}

impl Scored {
    /// Whether `given` is right for the held-out tag `held_out`, the
    /// reference's tags compared through `mappings`.
    pub fn is_right(self, mappings: &[ReferenceMapping], held_out: &str, given: &str) -> bool {
        match self {
            Scored::Candidate => given == held_out,
            Scored::Reference => reference_is_right(mappings, held_out, given),
        }
    }
}

/// How many of the reference's most frequent mismatches the card lists.
pub const MISMATCHES_LISTED: usize = 10;

/// The tagger a recipe trains and the OpenNLP model it is gated against.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PosRecipe {
    /// The trained model's zoo id, also its file name without `.bin`.
    pub model_id: &'static str,
    pub reference: Reference,
    /// The format the reference's tags are asked for.
    pub reference_format: PosTagFormat,
    /// The reference's tags counted right for another held-out tag.
    pub reference_mappings: &'static [ReferenceMapping],
    /// The most held-out tokens the gate may leave out because the
    /// reference cannot give their tag.
    pub max_left_out: Limit,
}

pub const POS: PosRecipe = PosRecipe {
    model_id: "turbo-en-masc-3.0.0-pos-1",
    reference: Reference::Recorded(EN_POS_MAXENT),
    reference_format: PosTagFormat::Penn,
    reference_mappings: &REFERENCE_MAPPINGS,
    max_left_out: MAX_TOKENS_LEFT_OUT,
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
            names: None,
            chunks: None,
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
        let out = HeldOutTags::new(samples, skipped);
        if out.is_empty() {
            return Err(Error::EmptySplit { what: "held-out part-of-speech sample" });
        }
        check_skipped(LeftOut::HeldOutSentences, out.skipped(), out.len())?;
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
    /// More of them than [`PosRecipe::max_left_out`] allows
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
        if self.max_left_out.allows(left_out.iter().map(|t| t.tokens).sum(), tokens) {
            Ok(left_out)
        } else {
            Err(Error::TagsOutsideReference {
                reference: reference_id.to_owned(),
                left_out,
                of: tokens,
                limit: self.max_left_out,
            })
        }
    }

    /// The tags of the reference model in `reference`, asked for in the
    /// recipe's reference format, compared through the recipe's mappings.
    ///
    /// # Errors
    ///
    /// A model file that is not a part-of-speech model.
    pub fn reference_tags(&self, reference: &[u8]) -> Result<ReferenceTags> {
        Ok(ReferenceTags::of(open(reference, self.reference_format)?.pos_model().tagset(), self.reference_mappings))
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
    /// tags of more than [`PosRecipe::max_left_out`] of the held-out tokens
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
        let tags = ReferenceTags::of(reference_tagger.pos_model().tagset(), self.reference_mappings);
        let left_out = self.check_coverage(reference_id, &tags, held_out)?;
        let candidate = self.score(trained, Scored::Candidate, held_out, &tags)?;
        let baseline = score_with(&reference_tagger, Scored::Reference, held_out, &tags)?;
        let units = spans::pair(candidate, baseline.units);
        let reference_mappings = tag_mappings(self.reference_mappings);
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
            names: None,
            chunks: None,
        };
        Ok(GateRun { record, held_out })
    }
}
