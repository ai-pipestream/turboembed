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
//! token. MASC's chunk layers
//! are marked `automatic-validated` in its resource header, and they hold
//! such chunks: counted with the recipe's reader over the documents the
//! split reads from the recorded release (the README has the counts), a
//! noun chunk inside a verb chunk is the most common. A sentence with a
//! token without a tag, or of no tokens, is skipped as the tagger recipe
//! skips it. Held-out sentences are skipped the same way. The card
//! records how many training and held-out sentences were skipped, and the
//! recipe fails when either is more than [`MAX_SKIPPED_SENTENCES`] of the
//! sentences read, a bound of its own, wider than the other recipes'
//! because MASC's chunk layers have more such sentences.
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
//! tags as MASC writes them.
//!
//! How MASC's chunks differ from CoNLL-2000's, and which way that leans
//! the gate, is in the crate's README.

use turbo_eval::model_gate::{Counts, HeldOutSet, ModelGateRecord, ModelId, Task};
use turbo_formats::evaluate::ChunkerEvaluator;
use turbo_formats::masc::{ChunkSkip, MascChunkRead, MascChunkReader, MascCorpus, MascError};
use turbo_formats::sample::ChunkSample;
use turbo_formats::sample::chunk::phrase_spans;
use turbo_formats::source::Field;
use turbo_maxent::ModelFile;
use turbo_phrase_chunker::{ChunkerModel, PhraseChunker};
use turbo_sequence::chunker::TokenTag;
use turbo_train::{TrainingConfiguration, TrainingParameters};
use turbo_train_components::{ChunkerFactory, Stamp, chunk};
use turbo_zoo::Sha256;

use crate::card::HeldOutCard;
use crate::error::{Error, LeftOut, Result};
use crate::postag::ReferenceMapping;
use crate::segmentation::{LANGUAGE, TIMESTAMP_MS};
use crate::spans::{self, SpanTypes};
use crate::{GateRun, Limit, RecordedReference, Reference, Trained};

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
/// has the counts). MASC's brackets are `(` and `)`, as the reference's
/// are, so they are not mapped. Only the reference's input goes through
/// this; it only ever helps the reference.
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

/// The most sentences the recipe's reader may skip, of those it reads
/// from the training documents and, apart, of those it reads from the
/// held-out documents: one in ten. The other recipes allow one in a
/// thousand ([`crate::MAX_SKIPPED_SENTENCES`]); MASC's chunk layers have
/// more sentences with chunks that chunk tags cannot express (in the
/// recorded release about one in thirteen of the training sentences and
/// one in eighteen of the held-out ones; the README has the counts), and
/// the bound is there to stop a reader that loses far more than that.
pub const MAX_SKIPPED_SENTENCES: Limit = Limit { count: 1, of: 10 };

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

/// The sentences [`read_samples`] skipped, by why.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct Skipped {
    pub no_tokens: usize,
    pub null_tag: usize,
    pub partial_chunk: usize,
    pub overlapping_chunks: usize,
    pub chunk_with_gap: usize,
}

impl Skipped {
    fn add(&mut self, reason: ChunkSkip) {
        let count = match reason {
            ChunkSkip::NoTokens => &mut self.no_tokens,
            ChunkSkip::NullTag => &mut self.null_tag,
            ChunkSkip::PartialChunk => &mut self.partial_chunk,
            ChunkSkip::OverlappingChunks => &mut self.overlapping_chunks,
            ChunkSkip::ChunkWithGap => &mut self.chunk_with_gap,
        };
        *count += 1;
    }

    /// Every sentence skipped.
    pub fn total(&self) -> usize {
        self.no_tokens + self.null_tag + self.partial_chunk + self.overlapping_chunks + self.chunk_with_gap
    }
}

/// Reads the chunk samples of `corpus` ([`MascChunkReader`]), giving each
/// to `each`, and returns the sentences it skipped.
///
/// # Errors
///
/// A corpus read without its chunk files, or a token the text cannot
/// give.
pub fn read_samples<'c>(corpus: &'c MascCorpus, mut each: impl FnMut(ChunkSample<'c>)) -> Result<Skipped> {
    let mut skipped = Skipped::default();
    for read in MascChunkReader::new(corpus)? {
        match read? {
            MascChunkRead::Sample(s) => each(s.sample),
            MascChunkRead::Skipped { reason, .. } => skipped.add(reason),
        }
    }
    Ok(skipped)
}

/// Checks the sentences the reader skipped against
/// [`MAX_SKIPPED_SENTENCES`]: `skipped` of `skipped + read`.
fn check_skipped(what: LeftOut, skipped: usize, read: usize) -> Result<()> {
    MAX_SKIPPED_SENTENCES.check(what, skipped as u64, (skipped + read) as u64)
}

/// The tag scored for the chunk tag `tag` of a type `types` does not
/// score: `O`; any other tag as it is. A tag's type follows its first two
/// characters (`B-`, `I-`), as `ChunkSample` reads it.
pub fn scored_tag<'t>(tag: &'t str, types: &SpanTypes) -> &'t str {
    match tag.get(2..) {
        Some(kind) if tag != "O" && !types.predicted(kind) => "O",
        _ => tag,
    }
}

/// The held-out chunk samples, their chunk tags of the types scored
/// ([`SPAN_TYPES`]: renamed where it renames them, `O` where it leaves
/// them out), borrowed from the held-out corpus.
#[derive(Debug, Clone)]
pub struct HeldOutChunks<'c> {
    samples: Vec<ChunkSample<'c>>,
    skipped: Skipped,
    /// The held-out phrases of the types scored.
    spans: u64,
    /// The held-out phrases of other types, made `O`.
    left_out: u64,
}

impl<'c> HeldOutChunks<'c> {
    /// The held-out samples of `samples`, their phrases of other types
    /// than `types` scores made `O` and the renamed ones renamed.
    ///
    /// # Errors
    ///
    /// A chunk tag `ChunkSample` cannot read a phrase from
    /// ([`MascError::Sample`]).
    pub fn new(samples: Vec<ChunkSample<'c>>, skipped: Skipped, types: &SpanTypes) -> Result<HeldOutChunks<'c>> {
        let mut out = HeldOutChunks { samples: Vec::with_capacity(samples.len()), skipped, spans: 0, left_out: 0 };
        for sample in samples {
            let tags: Vec<&str> = sample.chunks().iter().map(Field::as_str).collect();
            let phrases = phrase_spans(&tags).map_err(MascError::from)?;
            let mut chunks: Vec<Field<'c>> = sample.chunks().to_vec();
            for &(start, end, cut) in &phrases {
                let kind = &tags[start as usize][cut..];
                match types.held_out(kind) {
                    Some(scored) => {
                        out.spans += 1;
                        if scored != kind {
                            for (i, field) in chunks[start as usize..end as usize].iter_mut().enumerate() {
                                let prefix = if i == 0 { "B-" } else { "I-" };
                                *field = Field::made(format!("{prefix}{scored}"), field.span);
                            }
                        }
                    }
                    None => {
                        out.left_out += 1;
                        for field in &mut chunks[start as usize..end as usize] {
                            *field = Field::made("O".to_owned(), field.span);
                        }
                    }
                }
            }
            let source = sample.source();
            let sample = ChunkSample::new(sample.tokens().to_vec(), sample.tags().to_vec(), chunks, source);
            out.samples.push(sample.map_err(MascError::from)?);
        }
        Ok(out)
    }

    /// The samples scored.
    pub fn samples(&self) -> &[ChunkSample<'c>] {
        &self.samples
    }

    /// The held-out sentences [`read_samples`] skipped.
    pub fn skipped(&self) -> Skipped {
        self.skipped
    }

    pub fn len(&self) -> usize {
        self.samples.len()
    }

    pub fn is_empty(&self) -> bool {
        self.samples.is_empty()
    }

    /// The held-out phrases scored.
    pub fn spans(&self) -> u64 {
        self.spans
    }

    /// The held-out phrases of other types, left out of both models'
    /// scores.
    pub fn left_out(&self) -> u64 {
        self.left_out
    }

    /// The tokens of the samples.
    pub fn tokens(&self) -> u64 {
        self.samples.iter().map(|s| s.tokens().len() as u64).sum()
    }

    /// The SHA-256 of the samples as scored: for each sample, its number
    /// of tokens as a little endian `u64`, then for each token its text,
    /// its tag and its chunk tag, each as its length in bytes as a little
    /// endian `u64` and its bytes; after the samples, the types scored,
    /// the renamed types (each held-out type and the type it is scored
    /// as) and [`REFERENCE_TAGS`] (each held-out tag and the reference's),
    /// each list as its length and its strings.
    pub fn sha256(&self, types: &SpanTypes) -> String {
        let mut bytes = Vec::new();
        let put = |bytes: &mut Vec<u8>, s: &str| {
            bytes.extend_from_slice(&(s.len() as u64).to_le_bytes());
            bytes.extend_from_slice(s.as_bytes());
        };
        for sample in &self.samples {
            bytes.extend_from_slice(&(sample.tokens().len() as u64).to_le_bytes());
            for ((token, tag), chunk) in sample.tokens().iter().zip(sample.tags()).zip(sample.chunks()) {
                put(&mut bytes, token.as_str());
                put(&mut bytes, tag.as_str());
                put(&mut bytes, chunk.as_str());
            }
        }
        bytes.extend_from_slice(&(types.types.len() as u64).to_le_bytes());
        for t in types.types {
            put(&mut bytes, t);
        }
        bytes.extend_from_slice(&(types.renamed.len() as u64).to_le_bytes());
        for (held_out, scored) in types.renamed {
            put(&mut bytes, held_out);
            put(&mut bytes, scored);
        }
        bytes.extend_from_slice(&(REFERENCE_TAGS.len() as u64).to_le_bytes());
        for m in REFERENCE_TAGS {
            put(&mut bytes, m.held_out);
            put(&mut bytes, m.reference);
        }
        Sha256::of(&bytes).to_string()
    }
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
        })
    }

    /// Reads the training samples as [`ChunkRecipe::train`] does, without
    /// making events, and checks them as it does.
    ///
    /// # Errors
    ///
    /// As [`ChunkRecipe::train`], but for training.
    pub fn check_training(&self, train: &MascCorpus) -> Result<()> {
        let mut samples = 0;
        let skipped = read_samples(train, |_| samples += 1)?;
        if samples == 0 {
            return Err(Error::EmptySplit { what: "training chunk sample" });
        }
        check_skipped(LeftOut::TrainingSentences, skipped.total(), samples)
    }

    /// The held-out samples, read as the training samples are, their
    /// chunk tags of the types scored ([`HeldOutChunks::new`]).
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
        HeldOutChunks::new(samples, skipped, &SPAN_TYPES)
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
            scoring: SPAN_TYPES.scoring(held_out.spans(), held_out.left_out()),
        };
        let id = |id: &str, bytes: &[u8]| ModelId { id: id.to_owned(), sha256: Sha256::of(bytes).to_string() };
        let record =
            ModelGateRecord::new(self.task(), set, id(self.model_id, trained), id(reference_id, reference), units)
                .map_err(Error::Gate)?;
        let held_out =
            HeldOutCard { samples: held_out.len(), skipped_sentences: held_out.skipped().total(), tokens: None };
        Ok(GateRun { record, held_out })
    }
}

/// The chunker in `bytes`.
fn open(bytes: &[u8]) -> Result<PhraseChunker> {
    let file = ModelFile::open(bytes.to_vec()).map_err(|e| Error::Model(e.to_string()))?;
    let model = ChunkerModel::open(file).map_err(|e| Error::Model(e.to_string()))?;
    Ok(PhraseChunker::new(model))
}

#[cfg(test)]
mod tests {
    use turbo_analysis::Span;

    use super::*;

    fn sample<'a>(text: &'a str, tokens: &[(u32, u32)], tags: &[&str], chunks: &[&str]) -> ChunkSample<'a> {
        let fields: Vec<Field<'a>> = tokens.iter().map(|&(a, b)| Field::of(text, Span::new(a, b))).collect();
        let made = |values: &[&str]| -> Vec<Field<'a>> {
            values.iter().zip(&fields).map(|(v, f)| Field::made((*v).to_owned(), f.span)).collect()
        };
        let (tags, chunks) = (made(tags), made(chunks));
        ChunkSample::new(fields, tags, chunks, Span::new(0, text.len() as u32)).unwrap()
    }

    #[test]
    fn tags_of_types_not_scored_are_made_o_and_their_phrases_left_out() {
        for (tag, scored) in
            [("B-NP", "B-NP"), ("I-VP", "I-VP"), ("O", "O"), ("B-PP", "O"), ("I-ADJP", "O"), ("B-SBAR", "O")]
        {
            assert_eq!(scored_tag(tag, &SPAN_TYPES), scored, "{tag}");
        }
        // making them O leaves the phrases of the types scored as they were
        let tags = ["B-NP", "I-PP", "I-NP", "B-PP", "B-NP", "I-NP", "I-VP", "I-ADVP", "B-VP"];
        let mapped: Vec<&str> = tags.iter().map(|t| scored_tag(t, &SPAN_TYPES)).collect();
        let kept = |tags: &[&str]| -> Vec<(u32, u32, String)> {
            phrase_spans(tags)
                .unwrap()
                .into_iter()
                .map(|(s, e, cut)| (s, e, tags[s as usize][cut..].to_owned()))
                .filter(|(_, _, k)| SPAN_TYPES.predicted(k))
                .collect()
        };
        assert_eq!(kept(&tags), kept(&mapped));
        assert_eq!(
            kept(&mapped),
            phrase_spans(&mapped)
                .unwrap()
                .into_iter()
                .map(|(s, e, cut)| (s, e, mapped[s as usize][cut..].to_owned()))
                .collect::<Vec<_>>()
        );
    }

    #[test]
    fn held_out_phrases_are_renamed_or_left_out_as_the_types_say() {
        let text = "a b c d";
        let s = sample(
            text,
            &[(0, 1), (2, 3), (4, 5), (6, 7)],
            &["DT", "NN", "IN", "NN"],
            &["B-NP", "I-NP", "B-PP", "B-X"],
        );
        let types = SpanTypes { types: &["NP", "Y"], renamed: &[("X", "Y")] };
        let held = HeldOutChunks::new(vec![s], Skipped::default(), &types).unwrap();
        let chunks: Vec<&str> = held.samples()[0].chunks().iter().map(Field::as_str).collect();
        assert_eq!(chunks, ["B-NP", "I-NP", "O", "B-Y"]);
        assert_eq!((held.spans(), held.left_out(), held.tokens()), (2, 1, 4));
        // the spans of the fields are their tokens'
        assert_eq!(held.samples()[0].chunks()[3].span, Span::new(6, 7));
    }

    #[test]
    fn the_held_out_digest_covers_tokens_tags_chunk_tags_types_and_the_reference_tags() {
        let text = "Dogs ran";
        let one = |chunk: &str| {
            let s = sample(text, &[(0, 4), (5, 8)], &["NNS", "VBD"], &["B-NP", chunk]);
            HeldOutChunks::new(vec![s], Skipped::default(), &SPAN_TYPES).unwrap()
        };
        let mut bytes = Vec::new();
        let put = |bytes: &mut Vec<u8>, s: &str| {
            bytes.extend_from_slice(&(s.len() as u64).to_le_bytes());
            bytes.extend_from_slice(s.as_bytes());
        };
        bytes.extend_from_slice(&2u64.to_le_bytes());
        for (token, tag, chunk) in [("Dogs", "NNS", "B-NP"), ("ran", "VBD", "B-VP")] {
            put(&mut bytes, token);
            put(&mut bytes, tag);
            put(&mut bytes, chunk);
        }
        bytes.extend_from_slice(&2u64.to_le_bytes());
        put(&mut bytes, "NP");
        put(&mut bytes, "VP");
        bytes.extend_from_slice(&0u64.to_le_bytes());
        bytes.extend_from_slice(&3u64.to_le_bytes());
        for (held_out, reference) in [("-", ":"), ("...", ":"), (";", ":")] {
            put(&mut bytes, held_out);
            put(&mut bytes, reference);
        }
        assert_eq!(one("B-VP").sha256(&SPAN_TYPES), Sha256::of(&bytes).to_string());
        assert_ne!(one("B-VP").sha256(&SPAN_TYPES), one("O").sha256(&SPAN_TYPES));
        let other = SpanTypes { types: &["NP"], renamed: &[] };
        assert_ne!(one("B-VP").sha256(&SPAN_TYPES), one("B-VP").sha256(&other));
    }

    #[test]
    fn only_the_reference_is_given_mapped_tags() {
        for (masc, reference) in
            [("-", ":"), ("...", ":"), (";", ":"), ("(", "("), (")", ")"), ("''", "''"), ("NN", "NN")]
        {
            assert_eq!(Scored::Reference.tag(masc), reference, "{masc}");
            assert_eq!(Scored::Candidate.tag(masc), masc);
        }
    }

    #[test]
    fn skipped_sentences_are_counted_by_why() {
        let mut s = Skipped::default();
        for reason in [
            ChunkSkip::NoTokens,
            ChunkSkip::OverlappingChunks,
            ChunkSkip::OverlappingChunks,
            ChunkSkip::PartialChunk,
            ChunkSkip::NullTag,
            ChunkSkip::ChunkWithGap,
        ] {
            s.add(reason);
        }
        assert_eq!(
            s,
            Skipped { no_tokens: 1, null_tag: 1, partial_chunk: 1, overlapping_chunks: 2, chunk_with_gap: 1 }
        );
        assert_eq!(s.total(), 6);
        assert!(MAX_SKIPPED_SENTENCES.allows(10, 100) && !MAX_SKIPPED_SENTENCES.allows(11, 100));
        use turbo_train::Key;
        let p = training_parameters();
        assert_eq!(p.get_str(Key::ALGORITHM).unwrap(), Some("MAXENT"));
        assert_eq!(p.get_int(Key::ITERATIONS).unwrap(), Some(ITERATIONS));
        assert_eq!(p.get_int(Key::CUTOFF).unwrap(), Some(CUTOFF));
        assert_eq!(p.get_int(Key::THREADS).unwrap(), None);
        assert_eq!(chunk::beam_size(&p).unwrap(), chunk::DEFAULT_BEAM_SIZE);
        assert_eq!((ITERATIONS, CUTOFF, THREADS, chunk::DEFAULT_BEAM_SIZE), (100, 5, 1, 10));
        assert_eq!(CHUNKS.task(), Task::Chunker);
    }
}
