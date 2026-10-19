//! The English chunker trained on MASC's noun and verb chunks, and its
//! gate against OpenNLP's English chunker, `en-chunker.bin`.
//!
//! The recipe trains on the training documents of [`crate::masc::SPLIT`]
//! as `ChunkerME.train` trains with its defaults: the `ChunkerFactory` and
//! its default context generator, and `TrainingParameters.defaultParams()`
//! ([`training_parameters`]: GIS, 100 iterations, cutoff 5, one thread,
//! beam size 10). Training uses no randomness, so the recipe has no seed.
//! The reference's manifest records OpenNLP 1.5.0 and the same iterations
//! and cutoff.
//!
//! The samples are [`MascChunkReader`]'s, one per sentence: its Penn
//! tokens, their `msd` tags, and chunk tags of MASC's noun chunks (`NP`)
//! and verb chunks (`VP`). OpenNLP has no MASC chunk reader. A sentence
//! with chunks that chunk tags cannot express is skipped ([`ChunkSkip`]):
//! it has part of a chunk that crosses into another sentence or names a
//! token no sentence has, a chunk with a gap, or two chunks that share a
//! token. MASC's chunk layers are marked `automatic-validated` in its
//! resource header, and they hold such chunks. Counted with the recipe's
//! reader over the documents the split reads from the recorded release,
//! in the sentences that hold no part of a chunk, the pair of chunks that
//! share a token is most often a verb chunk inside a noun chunk (1,532
//! pairs: 1,325 in the training documents, 207 in the held-out ones),
//! mostly a participle or gerund inside a noun phrase (`the wounded
//! sailor` with a verb chunk `wounded`, `running footsteps` with a verb
//! chunk `running`); then a noun chunk and a verb chunk over the same
//! tokens (295), a noun chunk inside a verb chunk (207), chunks that
//! cross (65), a noun chunk inside a noun chunk (23) and a verb chunk
//! inside a verb chunk (2). A sentence with a token without a tag, or of
//! no tokens, is skipped as the tagger recipe skips it. Held-out
//! sentences are skipped the same way. The card records how many
//! training and held-out sentences were skipped. On the recorded release
//! the recipe fails unless they are the counts it records
//! ([`RECORDED_SKIPPED_SENTENCES`]); on any corpus it fails when either
//! is more than [`MAX_SKIPPED_SENTENCES`] of the sentences read, a bound
//! of its own, wider than the other recipes' because MASC's chunk layers
//! have more such sentences.
//!
//! Some chunks are in no sentence the reader reads, so no sample has
//! them: MASC's sentence layer does not cover all of the text, and a
//! sentence region that does not end before the next one starts is
//! dropped, as OpenNLP's MASC reader drops it, so a chunk can name a
//! token no kept sentence has. The card counts these chunks on each side,
//! with the chunks whose tokens are in more than one sentence and those
//! with no token ([`ChunksCard`], [`chunk_counts`]). In the recorded
//! release 16,597 of the 221,931 chunks of the documents the split reads
//! are outside every kept sentence: 13,509 of 183,681 in the training
//! documents and 3,088 of 38,250 in the held-out ones. The other MASC
//! recipes read the same sentences, so their samples do not have that
//! text either.
//!
//! The reference is OpenNLP's English chunker from the 1.5 models
//! ([`EN_CHUNKER`]), OpenNLP's only English chunker. Its training data is
//! not published (its chunk types are CoNLL-2000's, which is drawn from
//! the Wall Street Journal part of the Penn Treebank, an LDC corpus), and
//! the zoo does not list it: a run reads it from its references
//! directory, checks it against the recorded size and SHA-256, and only
//! scores it. It is never shipped or bundled, and the trained model is
//! trained on MASC alone.
//!
//! Both models are scored on the same held-out samples, one unit per
//! sentence, each with its own `ChunkerEvaluator` given the sample's
//! Penn tags, as OpenNLP's evaluator gives them. Only the chunk types
//! both models give are scored ([`SPAN_TYPES`]: `NP` and `VP`). MASC's
//! chunks are of those two types only, so no held-out chunk is renamed or
//! left out. The reference also gives CoNLL-2000's other types (`PP`,
//! `ADVP`, `ADJP`, `SBAR`, `PRT`, `CONJP`, `INTJ`, `LST`, `UCP`), where
//! MASC has no chunk; its tags of those types are made `O` before it is
//! scored ([`scored_tag`]), which leaves its `NP` and `VP` phrases as they
//! were, since a phrase ends at any tag that does not continue it. The
//! trained model gives only `NP` and `VP`, so this changes nothing on its
//! side. The reference reads three tags MASC writes and its training
//! data does not have (it has no feature for them) as the tag its data
//! gives the same tokens ([`REFERENCE_TAGS`]); the trained model reads the
//! tags as MASC writes them. The gate record lists that mapping
//! (`reference_inputs`).
//!
//! How MASC's chunks differ from CoNLL-2000's, and which way that leans
//! the gate, is in the crate's README.

mod samples;
#[cfg(test)]
mod tests;

use turbo_eval::model_gate::{Counts, HeldOutSet, ModelGateRecord, ModelId, TagMapping, Task};
use turbo_formats::evaluate::ChunkerEvaluator;
use turbo_formats::masc::MascCorpus;
#[cfg(doc)]
use turbo_formats::masc::{ChunkSkip, MascChunkReader};
use turbo_maxent::ModelFile;
use turbo_phrase_chunker::{ChunkerModel, PhraseChunker};
use turbo_sequence::chunker::TokenTag;
use turbo_train::{TrainingConfiguration, TrainingParameters};
use turbo_train_components::{ChunkerFactory, Stamp, chunk};
use turbo_zoo::Sha256;

#[cfg(doc)]
use crate::card::ChunksCard;
use crate::card::HeldOutCard;
use crate::error::{Error, LeftOut, Result};
use crate::postag::ReferenceMapping;
use crate::segmentation::{LANGUAGE, TIMESTAMP_MS};
use crate::spans::{self, SpanTypes};
use crate::{GateRun, Limit, RecordedReference, Reference, Trained};

use self::samples::check_skipped;
pub use self::samples::{HeldOutChunks, Skipped, chunk_counts, read_samples, scored_tag};

/// OpenNLP's English chunker of the 1.5 models, as published.
pub const EN_CHUNKER: RecordedReference = RecordedReference {
    id: "opennlp-en-chunker-1.5",
    file: "en-chunker.bin",
    url: "https://opennlp.sourceforge.net/models-1.5/en-chunker.bin",
    size: 2_560_304,
    sha256: "7861a0c2f134d9c12a022a1ba501e88bc7039f6db72b4140e1bafd1fb5ef76cc",
};

/// The chunker the recipe trains and the OpenNLP model it is gated
/// against.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ChunkRecipe {
    /// The trained model's zoo id, also its file name without `.bin`.
    pub model_id: &'static str,
    pub reference: Reference,
}

pub const CHUNKS: ChunkRecipe =
    ChunkRecipe { model_id: "turbo-en-masc-3.0.0-chunks-1", reference: Reference::Recorded(EN_CHUNKER) };

/// The chunk types the gate scores: the two MASC annotates, which the
/// reference gives too. No held-out type is renamed.
pub const SPAN_TYPES: SpanTypes = SpanTypes { types: &["NP", "VP"], renamed: &[] };

/// The tags the reference is given for three tags MASC writes and the
/// reference's training data does not have: its features (with the
/// cutoff of 5 its manifest records) name the tags `#`, `$`, `''`, `(`,
/// `)`, `,`, `.`, `:`, `?`, `CC`, `CD`, `DT`, `EX`, `FW`, `IN`, `JJ`,
/// `JJR`, `JJS`, `MD`, `NN`, `NNP`, `NNPS`, `NNS`, `PDT`, `POS`, `PRP`,
/// `PRP$`, `RB`, `RBR`, `RBS`, `RP`, `SYM`, `TO`, `UH`, `VB`, `VBD`,
/// `VBG`, `VBN`, `VBP`, `VBZ`, `WDT`, `WP`, `WP$`, `WRB` and ``` `` ```.
/// MASC tags some hyphens `-`, ellipses `...` and two semicolons `;`,
/// which the Penn Treebank's tagged text, and so the reference's data,
/// tags `:` (the tagger recipe's [`crate::postag::REFERENCE_MAPPINGS`]
/// has the counts). The held-out samples have 155 `-` and 173 `...`
/// tokens and no `;`. MASC's brackets are `(` and `)`, as the
/// reference's are, so they are not mapped. Two more held-out tags have
/// no feature in the reference and are given as written: `LS` (9 tokens)
/// and MASC's ``UH` `` (3 tokens, a tag written with a backtick). The
/// mapping is applied to the reference's input only, and
/// the gate record lists it (`reference_inputs`). On the held-out set it
/// is worth 0.02 points to the reference: 82.46 with it, 82.44 without.
pub const REFERENCE_TAGS: [ReferenceMapping; 3] = [
    ReferenceMapping { held_out: "-", reference: ":" },
    ReferenceMapping { held_out: "...", reference: ":" },
    ReferenceMapping { held_out: ";", reference: ":" },
];

/// The tag the reference is given for the held-out tag `tag`
/// ([`REFERENCE_TAGS`]).
pub fn reference_tag(tag: &str) -> &str {
    REFERENCE_TAGS.iter().find(|m| m.held_out == tag).map_or(tag, |m| m.reference)
}

/// [`REFERENCE_TAGS`] as the gate record writes them
/// (`reference_inputs`).
pub fn reference_inputs() -> Vec<TagMapping> {
    REFERENCE_TAGS
        .iter()
        .map(|m| TagMapping { held_out: m.held_out.to_owned(), reference: m.reference.to_owned() })
        .collect()
}

/// The most sentences the recipe's reader may skip, of those it reads
/// from the training documents and, apart, of those it reads from the
/// held-out documents: one in ten. The other recipes allow one in a
/// thousand ([`crate::MAX_SKIPPED_SENTENCES`]); MASC's chunk layers have
/// more sentences with chunks that chunk tags cannot express (in the
/// recorded release about one in thirteen of the training sentences and
/// one in eighteen of the held-out ones; the README has the counts), and
/// the bound is there to stop a reader that loses far more than that on
/// any corpus. On the recorded release the counts themselves are checked
/// ([`RECORDED_SKIPPED_SENTENCES`]).
pub const MAX_SKIPPED_SENTENCES: Limit = Limit { count: 1, of: 10 };

/// The sentences the recipe's reader skips in the recorded release
/// (corpus digest `445449f5...`, the Twitter genre left out): 1,872 of
/// the 24,958 training sentences (1,537 with chunks that share a token
/// and 335 with part of a chunk) and 349 of the 6,385 held-out ones (280,
/// 68 and the one sentence of no tokens). A run on the recorded release
/// fails on any other count ([`check_recorded_skips`]), so a change to the
/// reader that skips more or fewer sentences of it is not missed.
pub const RECORDED_SKIPPED_SENTENCES: RecordedSkipped = RecordedSkipped { training: 1_872, held_out: 349 };

/// The sentences of each side of the split the reader skips.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RecordedSkipped {
    pub training: usize,
    pub held_out: usize,
}

/// Checks `skipped`, the sentences of `what` the reader skipped in the
/// recorded release, against [`RECORDED_SKIPPED_SENTENCES`].
///
/// # Errors
///
/// [`Error::SkippedNotRecorded`] for another count.
pub fn check_recorded_skips(what: LeftOut, skipped: usize) -> Result<()> {
    let recorded = match what {
        LeftOut::TrainingSentences => RECORDED_SKIPPED_SENTENCES.training,
        LeftOut::HeldOutSentences => RECORDED_SKIPPED_SENTENCES.held_out,
    };
    if skipped == recorded {
        Ok(())
    } else {
        Err(Error::SkippedNotRecorded { what, count: skipped as u64, recorded: recorded as u64 })
    }
}

/// `Iterations` of `TrainingParameters.defaultParams()`.
pub const ITERATIONS: i32 = turbo_train::params::DEFAULT_ITERATIONS;
/// `Cutoff` of `TrainingParameters.defaultParams()`.
pub const CUTOFF: i32 = turbo_train::params::DEFAULT_CUTOFF;
/// The threads GIS trains with when no `Threads` parameter is given.
pub const THREADS: i32 = 1;

/// The training parameters: `TrainingParameters.defaultParams()`, which
/// the trainer tool gives `ChunkerME.train` when no parameters file is
/// named (GIS, [`ITERATIONS`], [`CUTOFF`]; [`THREADS`] and beam size 10,
/// GIS's and `ChunkerME`'s defaults).
pub fn training_parameters() -> TrainingParameters {
    TrainingParameters::defaults()
}

/// Which model a chunker is scored as, which decides the tags it is
/// given.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Scored {
    /// The trained model: the held-out tags as MASC writes them.
    Candidate,
    /// The reference: the held-out tags through [`REFERENCE_TAGS`].
    Reference,
}

impl Scored {
    /// The tag the model is given for the held-out tag `tag`.
    pub fn tag(self, tag: &str) -> &str {
        match self {
            Scored::Candidate => tag,
            Scored::Reference => reference_tag(tag),
        }
    }
}

/// The factory of the recipe: the language and the default context
/// generator, as `ChunkerTrainerTool` builds it.
fn factory() -> ChunkerFactory {
    ChunkerFactory::new(LANGUAGE)
}

impl ChunkRecipe {
    /// The gate's task.
    pub fn task(&self) -> Task {
        Task::Chunker
    }

    /// Trains the chunker from the training documents.
    ///
    /// # Errors
    ///
    /// What [`read_samples`] fails on, a corpus without a sample
    /// ([`Error::EmptySplit`]), more skipped sentences than
    /// [`MAX_SKIPPED_SENTENCES`] allows ([`Error::TooManyLeftOut`]), or
    /// what training fails on.
    pub fn train(&self, train: &MascCorpus) -> Result<Trained> {
        let factory = factory();
        let mut stream = chunk::ChunkerEventStream::new(&factory);
        let mut events = Vec::new();
        let mut samples = 0;
        let skipped = read_samples(train, |sample| {
            stream.events(&sample, &mut events);
            samples += 1;
        })?;
        if samples == 0 {
            return Err(Error::EmptySplit { what: "training chunk sample" });
        }
        check_skipped(LeftOut::TrainingSentences, skipped.total(), samples)?;
        let params = training_parameters();
        let mut config = TrainingConfiguration::default();
        let t =
            chunk::train_chunker_model_from_events(&events, &factory, &params, &mut config, Stamp::at(TIMESTAMP_MS))?;
        Ok(Trained {
            bytes: t.package.to_bytes(),
            samples,
            skipped_sentences: skipped.total(),
            events: events.len(),
            event_hash: t.report.event_hash.expect("trained from events"),
            names: None,
            chunks: Some(chunk_counts(train)),
        })
    }

    /// Reads the training samples as [`ChunkRecipe::train`] does, without
    /// making events, checks them as it does, and returns the sentences
    /// the reader skipped.
    ///
    /// # Errors
    ///
    /// As [`ChunkRecipe::train`], but for training.
    pub fn check_training(&self, train: &MascCorpus) -> Result<Skipped> {
        let mut samples = 0;
        let skipped = read_samples(train, |_| samples += 1)?;
        if samples == 0 {
            return Err(Error::EmptySplit { what: "training chunk sample" });
        }
        check_skipped(LeftOut::TrainingSentences, skipped.total(), samples)?;
        Ok(skipped)
    }

    /// The held-out samples, read as the training samples are, their
    /// chunk tags of the types scored, with the held-out documents' chunks
    /// in no sentence the reader reads ([`HeldOutChunks::new`]).
    ///
    /// # Errors
    ///
    /// A sample the corpus fails on, no sample ([`Error::EmptySplit`]), or
    /// more skipped sentences than [`MAX_SKIPPED_SENTENCES`] allows
    /// ([`Error::TooManyLeftOut`]).
    pub fn held_out<'c>(&self, held_out: &'c MascCorpus) -> Result<HeldOutChunks<'c>> {
        let mut samples = Vec::new();
        let skipped = read_samples(held_out, |s| samples.push(s))?;
        if samples.is_empty() {
            return Err(Error::EmptySplit { what: "held-out chunk sample" });
        }
        check_skipped(LeftOut::HeldOutSentences, skipped.total(), samples.len())?;
        HeldOutChunks::new(samples, skipped, chunk_counts(held_out), &SPAN_TYPES)
    }

    /// Each held-out sample's counts for the chunker in `bytes`, given the
    /// tags `scored` gives it ([`Scored`]), its chunk tags of types
    /// [`SPAN_TYPES`] does not score made `O` ([`scored_tag`]).
    ///
    /// # Errors
    ///
    /// A model file that is not a chunker, a sentence no chunk tag
    /// sequence scores above the search's minimum for, or what the
    /// evaluator throws on.
    pub fn score(&self, bytes: &[u8], scored: Scored, held_out: &HeldOutChunks<'_>) -> Result<Vec<Counts>> {
        let chunker = open(bytes)?;
        let mut units = Vec::with_capacity(held_out.len());
        let (mut outcomes, mut probs) = (Vec::new(), Vec::new());
        let mut words: Vec<TokenTag<'_>> = Vec::new();
        let mut predicted: Vec<&str> = Vec::new();
        for sample in held_out.samples() {
            words.clear();
            words.extend(
                sample
                    .tokens()
                    .iter()
                    .zip(sample.tags())
                    .map(|(token, tag)| TokenTag { token: token.as_str(), tag: scored.tag(tag.as_str()) }),
            );
            outcomes.clear();
            probs.clear();
            if !chunker.outcomes(&words, &mut outcomes, &mut probs) {
                return Err(Error::Model(format!("no chunk tag sequence scored above the minimum for: {sample}")));
            }
            predicted.clear();
            predicted.extend(outcomes.iter().map(|o| scored_tag(o, &SPAN_TYPES)));
            let mut evaluator = ChunkerEvaluator::default();
            evaluator.evaluate(sample, &predicted)?;
            units.push(spans::counts(evaluator.fmeasure()));
        }
        Ok(units)
    }

    /// Scores the trained model ([`Scored::Candidate`]) and the reference
    /// ([`Scored::Reference`]) on the same held-out samples, over the
    /// chunk types both give ([`SPAN_TYPES`]), and runs the gate.
    /// `reference_id` names the reference in the record ([`Reference::id`]
    /// of [`ChunkRecipe::reference`] for the published model), and
    /// `set_name` and `corpus_sha256` the held-out set.
    ///
    /// # Errors
    ///
    /// What [`ChunkRecipe::score`] fails on, or the gate's errors.
    pub fn gate(
        &self,
        trained: &[u8],
        reference_id: &str,
        reference: &[u8],
        held_out: &HeldOutChunks<'_>,
        set_name: &str,
        corpus_sha256: &str,
    ) -> Result<GateRun> {
        let candidate = self.score(trained, Scored::Candidate, held_out)?;
        let baseline = self.score(reference, Scored::Reference, held_out)?;
        let units = spans::pair(candidate, baseline);
        let set = HeldOutSet {
            name: set_name.to_owned(),
            corpus_sha256: corpus_sha256.to_owned(),
            samples_sha256: held_out.sha256(&SPAN_TYPES),
            skipped_sentences: held_out.skipped().total() as u64,
            scoring: SPAN_TYPES.scoring(held_out.spans(), held_out.left_out(), reference_inputs()),
        };
        let id = |id: &str, bytes: &[u8]| ModelId { id: id.to_owned(), sha256: Sha256::of(bytes).to_string() };
        let record =
            ModelGateRecord::new(self.task(), set, id(self.model_id, trained), id(reference_id, reference), units)
                .map_err(Error::Gate)?;
        let held_out = HeldOutCard {
            samples: held_out.len(),
            skipped_sentences: held_out.skipped().total(),
            tokens: None,
            names: None,
            chunks: Some(held_out.chunks().clone()),
        };
        Ok(GateRun { record, held_out })
    }
}

/// The chunker in `bytes`.
fn open(bytes: &[u8]) -> Result<PhraseChunker> {
    let file = ModelFile::open(bytes.to_vec()).map_err(|e| Error::Model(e.to_string()))?;
    let model = ChunkerModel::open(file).map_err(|e| Error::Model(e.to_string()))?;
    Ok(PhraseChunker::new(model))
}
