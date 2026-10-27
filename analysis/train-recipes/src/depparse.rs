//! Recipes that train the MaxEnt (`DependencyParserME`) and Feedforward
//! (`FeedforwardDependencyParser`) arc-standard dependency parsers on the
//! CC BY 4.0 Universal Dependencies 2.16 treebanks of [`crate::treebank`]
//! (issue #922), and their UAS and LAS gates on the held-out `test` split.
//!
//! Non-projective trees are retained in training via pseudo-projective
//! encoding ([`DependencyGraph::projectivize`]) and decoded at inference
//! time ([`DependencyGraph::deprojectivize`]). Each candidate model is
//! evaluated on the treebank's `test` split for both Unlabeled Attachment
//! Score (UAS, [`DependencyAttachment::Unlabeled`]) and Labeled Attachment
//! Score (LAS, [`DependencyAttachment::Labeled`]) and gated against its
//! single-step projective-only baseline (`<model_id>-projective-baseline`).

pub mod card;

use turbo_analysis::Ix;
use turbo_analysis_layers::Words;
use turbo_depparse::{
    Activation, DependencyEvaluator, DependencyEventStream, DependencyModel, DependencyParser, DependencyParserME,
    DependencySample, FeatureLayout, FeedforwardDependencyModel, FeedforwardDependencyParser,
    FeedforwardDependencyTrainer, FeedforwardSettings,
};
use turbo_eval::model_gate::{
    Counts, DependencyAttachment, DependencyScoring, HeldOutSet, ModelGateRecord, ModelId, PairedUnit, Scoring, Task,
};
use turbo_formats::FormatError;
use turbo_formats::conllu::ConlluDependencyReader;
use turbo_train::{Key, TrainingParameters};
use turbo_unicode::WhitespaceMode;
use turbo_zoo::Sha256;

pub use card::{
    DEPPARSE_CARD_VERSION, DepparseHeldOutCard, DepparseModelCard, DepparseParametersCard, DepparseScoresCard,
    DepparseTrainingCard, UAS_GATE_FILE, write_depparse_model_dir,
};

use crate::check_skipped;
use crate::error::{Error, LeftOut, Result};
use crate::treebank::{
    CATALAN_ANCORA, FINNISH_FTB, ITALIAN_MARKIT, PORTUGUESE_PORTTINARI, RELEASE, SPANISH_ANCORA, Treebank, TreebankText,
};
use crate::ud::TAGSET;

/// GIS iterations for the UD MaxEnt dependency parser recipes.
pub const MAXENT_ITERATIONS: i32 = 100;
/// Predicate cutoff for the UD MaxEnt dependency parser recipes.
pub const MAXENT_CUTOFF: i32 = 5;
/// GIS partitions for the UD MaxEnt dependency parser recipes.
pub const MAXENT_THREADS: i32 = 4;

/// Deterministic training parameters for the MaxEnt dependency parser.
pub fn maxent_parameters(iterations: i32) -> TrainingParameters {
    let mut params = crate::ud::training_parameters();
    params.put_str(Key::ITERATIONS, &iterations.to_string());
    params.put_str(Key::CUTOFF, &MAXENT_CUTOFF.to_string());
    params.put_str(Key::THREADS, &MAXENT_THREADS.to_string());
    params
}

/// Deterministic training settings for the Feedforward dependency parser.
pub fn feedforward_settings(epochs: usize) -> FeedforwardSettings {
    FeedforwardSettings {
        embedding_size: 16,
        hidden_size: 64,
        epochs,
        batch_size: 512,
        learning_rate: 0.02,
        l2: 1e-8,
        dropout: 0.5,
        word_cutoff: 5,
        seed: 17,
        layout: FeatureLayout::ChenManning18,
        activation: Activation::Cube,
    }
}

/// Which dependency parser architecture a recipe trains.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DepparseKind {
    Maxent,
    Feedforward,
}

impl DepparseKind {
    /// Short lowercase identifier used in model IDs and cards.
    pub const fn name(self) -> &'static str {
        match self {
            DepparseKind::Maxent => "maxent",
            DepparseKind::Feedforward => "feedforward",
        }
    }
}

/// One dependency parser recipe on a UD 2.16 treebank.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DepparseRecipe {
    pub model_id: &'static str,
    pub treebank: Treebank,
    pub kind: DepparseKind,
}

const fn recipe(model_id: &'static str, treebank: Treebank, kind: DepparseKind) -> DepparseRecipe {
    DepparseRecipe { model_id, treebank, kind }
}

use DepparseKind::{Feedforward, Maxent};

/// All ten UD 2.16 dependency parser recipes (both parsers across all five
/// treebanks), in the order the binary runs them.
pub const DEPPARSE_RECIPES: [DepparseRecipe; 10] = [
    recipe("turbo-es-ud-ancora-2.16-depparse-maxent-1", SPANISH_ANCORA, Maxent),
    recipe("turbo-es-ud-ancora-2.16-depparse-feedforward-1", SPANISH_ANCORA, Feedforward),
    recipe("turbo-ca-ud-ancora-2.16-depparse-maxent-1", CATALAN_ANCORA, Maxent),
    recipe("turbo-ca-ud-ancora-2.16-depparse-feedforward-1", CATALAN_ANCORA, Feedforward),
    recipe("turbo-pt-ud-porttinari-2.16-depparse-maxent-1", PORTUGUESE_PORTTINARI, Maxent),
    recipe("turbo-pt-ud-porttinari-2.16-depparse-feedforward-1", PORTUGUESE_PORTTINARI, Feedforward),
    recipe("turbo-fi-ud-ftb-2.16-depparse-maxent-1", FINNISH_FTB, Maxent),
    recipe("turbo-fi-ud-ftb-2.16-depparse-feedforward-1", FINNISH_FTB, Feedforward),
    recipe("turbo-it-ud-markit-2.16-depparse-maxent-1", ITALIAN_MARKIT, Maxent),
    recipe("turbo-it-ud-markit-2.16-depparse-feedforward-1", ITALIAN_MARKIT, Feedforward),
];

/// Dependency samples read from a CoNLL-U split, with projectivity and token
/// counts.
#[derive(Clone)]
pub struct ReadDepparseSamples {
    pub samples: Vec<DependencySample<Words>>,
    pub projective_samples: usize,
    pub skipped: usize,
    pub tokens: u64,
    pub punctuation_tokens: u64,
}

/// Reads the dependency samples of `conllu` using [`ConlluDependencyReader`].
pub fn read_samples(conllu: &str, treebank: &'static str, file: &'static str) -> Result<ReadDepparseSamples> {
    let conllu_err = |error: FormatError| Error::Conllu { treebank, file, error: Box::new(error) };
    let mut reader = ConlluDependencyReader::new(conllu, TAGSET, WhitespaceMode::Unicode).map_err(conllu_err)?;
    let mut samples = Vec::new();
    let mut projective_samples = 0;
    let mut tokens = 0u64;
    let mut punctuation_tokens = 0u64;
    for item in &mut reader {
        let raw = item.map_err(conllu_err)?;
        let sample = DependencySample::<Words>::from_format(&raw).map_err(|e| Error::Model(e.to_string()))?;
        if sample.graph().is_projective() {
            projective_samples += 1;
        }
        tokens += sample.tokens().len() as u64;
        punctuation_tokens += sample.tags().iter().filter(|t| t.as_str() == "PUNCT").count() as u64;
        samples.push(sample);
    }
    if samples.is_empty() {
        return Err(Error::EmptySplit { what: "dependency sample of the treebank file" });
    }
    Ok(ReadDepparseSamples { samples, projective_samples, skipped: reader.skipped().len(), tokens, punctuation_tokens })
}

/// SHA-256 of the held-out dependency samples as scored.
pub fn samples_sha256(samples: &[DependencySample<Words>]) -> String {
    let mut bytes = Vec::new();
    let put = |bytes: &mut Vec<u8>, s: &str| {
        bytes.extend_from_slice(&(s.len() as u64).to_le_bytes());
        bytes.extend_from_slice(s.as_bytes());
    };
    for s in samples {
        bytes.extend_from_slice(&(s.tokens().len() as u64).to_le_bytes());
        for i in 0..s.tokens().len() {
            put(&mut bytes, &s.tokens()[i]);
            put(&mut bytes, &s.tags()[i]);
            let arc = s.graph().arc(Ix::new(i as u32)).expect("checked sample index");
            let head = arc.head.map_or(0u32, |h| (h.get() + 1) as u32);
            bytes.extend_from_slice(&head.to_le_bytes());
            put(&mut bytes, arc.relation);
        }
    }
    Sha256::of(&bytes).to_string()
}

/// A trained dependency model and its single-step projective-only baseline.
#[derive(Debug, Clone)]
pub struct TrainedDepparse {
    pub bytes: Vec<u8>,
    pub reference_bytes: Vec<u8>,
    pub samples: usize,
    pub projective_samples: usize,
    pub skipped_sentences: usize,
    pub tokens: u64,
    pub events: usize,
}

/// What one dependency parser recipe produced.
#[derive(Debug)]
pub struct DepparseOutcome {
    pub trained: TrainedDepparse,
    /// Labeled Attachment Score (LAS) gate record (`gate.json`).
    pub gate: ModelGateRecord,
    /// Unlabeled Attachment Score (UAS) gate record (`gate-uas.json`).
    pub uas_gate: ModelGateRecord,
    pub card: DepparseModelCard,
}

impl DepparseRecipe {
    /// The recipe of [`DEPPARSE_RECIPES`] that trains `id`.
    pub fn trains(id: &str) -> Option<DepparseRecipe> {
        DEPPARSE_RECIPES.into_iter().find(|r| r.model_id == id)
    }

    /// Identifier of the single-step projective-only baseline reference model.
    pub fn reference_id(&self) -> String {
        format!("{}-projective-baseline", self.model_id)
    }

    /// Name of the held-out `test` split in gate records.
    pub fn held_out_name(&self) -> String {
        format!("{} {RELEASE} test ({})", self.treebank.repository, self.treebank.test.name)
    }

    /// Checks that both `train` and `test` splits have valid dependency trees
    /// and skip at most [`crate::MAX_SKIPPED_SENTENCES`].
    pub fn check_data(&self, text: &TreebankText) -> Result<()> {
        let train = read_samples(&text.train, self.treebank.repository, self.treebank.train.name)?;
        check_skipped(LeftOut::TrainingSentences, train.skipped, train.samples.len())?;
        if train.projective_samples == 0 {
            return Err(Error::EmptySplit { what: "projective training sentence" });
        }
        let held_out = read_samples(&text.test, self.treebank.repository, self.treebank.test.name)?;
        check_skipped(LeftOut::HeldOutSentences, held_out.skipped, held_out.samples.len())?;
        Ok(())
    }

    /// Trains the candidate model on all trees (pseudo-projectivizing
    /// non-projective trees) and the single-step baseline reference on only
    /// the projective trees.
    pub fn train(&self, train: &str) -> Result<TrainedDepparse> {
        let read = read_samples(train, self.treebank.repository, self.treebank.train.name)?;
        check_skipped(LeftOut::TrainingSentences, read.skipped, read.samples.len())?;
        let projective: Vec<DependencySample<Words>> =
            read.samples.iter().filter(|s| s.graph().is_projective()).cloned().collect();
        if projective.is_empty() {
            return Err(Error::EmptySplit { what: "projective training sentence" });
        }
        let stream = DependencyEventStream::generate(&read.samples).map_err(|e| Error::Model(e.to_string()))?;
        let events = stream.events.len();
        let lang = self.treebank.language;
        let (bytes, reference_bytes) = match self.kind {
            DepparseKind::Maxent => {
                let model = DependencyModel::train(lang, &read.samples, &maxent_parameters(MAXENT_ITERATIONS))
                    .map_err(|e| Error::Model(e.to_string()))?;
                let base = DependencyModel::train(lang, &projective, &maxent_parameters(1))
                    .map_err(|e| Error::Model(e.to_string()))?;
                let bytes = model.bytes().ok_or_else(|| Error::Model("missing linear model bytes".into()))?.to_vec();
                let ref_bytes =
                    base.bytes().ok_or_else(|| Error::Model("missing baseline model bytes".into()))?.to_vec();
                (bytes, ref_bytes)
            }
            DepparseKind::Feedforward => {
                let model = FeedforwardDependencyTrainer::train(&read.samples, &feedforward_settings(5))
                    .map_err(|e| Error::Model(e.to_string()))?;
                let base = FeedforwardDependencyTrainer::train(&projective, &feedforward_settings(1))
                    .map_err(|e| Error::Model(e.to_string()))?;
                let bytes = model.to_bytes().map_err(|e| Error::Model(e.to_string()))?;
                let ref_bytes = base.to_bytes().map_err(|e| Error::Model(e.to_string()))?;
                (bytes, ref_bytes)
            }
        };
        Ok(TrainedDepparse {
            bytes,
            reference_bytes,
            samples: read.samples.len(),
            projective_samples: read.projective_samples,
            skipped_sentences: read.skipped,
            tokens: read.tokens,
            events,
        })
    }

    /// Evaluates `trained` and `reference` on `held_out` and builds the LAS
    /// (`gate.json`) and UAS (`gate-uas.json`) gate records.
    pub fn gate(
        &self,
        trained: &[u8],
        reference: &[u8],
        held_out: &ReadDepparseSamples,
    ) -> Result<(ModelGateRecord, ModelGateRecord, DepparseHeldOutCard)> {
        let (cand_uas, cand_las, cand_scores) = score_model(self.kind, trained, &held_out.samples)?;
        let (ref_uas, ref_las, ref_scores) = score_model(self.kind, reference, &held_out.samples)?;
        let uas_units: Vec<PairedUnit> = cand_uas
            .into_iter()
            .zip(ref_uas)
            .map(|(candidate, reference)| PairedUnit { candidate, reference })
            .collect();
        let las_units: Vec<PairedUnit> = cand_las
            .into_iter()
            .zip(ref_las)
            .map(|(candidate, reference)| PairedUnit { candidate, reference })
            .collect();
        let ref_id = self.reference_id();
        let cand_model = ModelId { id: self.model_id.to_owned(), sha256: Sha256::of(trained).to_string() };
        let ref_model = ModelId { id: ref_id, sha256: Sha256::of(reference).to_string() };
        let samples_hash = samples_sha256(&held_out.samples);
        let make_set = |attachment: DependencyAttachment| HeldOutSet {
            name: self.held_out_name(),
            corpus_sha256: self.treebank.test.sha256.to_owned(),
            samples_sha256: samples_hash.clone(),
            skipped_sentences: held_out.skipped as u64,
            scoring: Scoring::Dependencies(DependencyScoring {
                attachment,
                without_punctuation: false,
                tokens: held_out.tokens,
                left_out: 0,
            }),
        };
        let gate = ModelGateRecord::new(
            Task::DependencyParser,
            make_set(DependencyAttachment::Labeled),
            cand_model.clone(),
            ref_model.clone(),
            las_units,
        )?;
        let uas_gate = ModelGateRecord::new(
            Task::DependencyParser,
            make_set(DependencyAttachment::Unlabeled),
            cand_model,
            ref_model,
            uas_units,
        )?;
        let card = DepparseHeldOutCard {
            samples: held_out.samples.len(),
            projective_samples: held_out.projective_samples,
            nonprojective_samples: held_out.samples.len() - held_out.projective_samples,
            skipped_sentences: held_out.skipped,
            tokens: held_out.tokens,
            punctuation_tokens: held_out.punctuation_tokens,
            candidate: cand_scores,
            reference: ref_scores,
        };
        Ok((gate, uas_gate, card))
    }
}

fn score_model(
    kind: DepparseKind,
    bytes: &[u8],
    samples: &[DependencySample<Words>],
) -> Result<(Vec<Counts>, Vec<Counts>, DepparseScoresCard)> {
    match kind {
        DepparseKind::Maxent => {
            let model = DependencyModel::open(bytes.to_vec()).map_err(|e| Error::Model(e.to_string()))?;
            score_with_parser(&DependencyParserME::new(model), samples)
        }
        DepparseKind::Feedforward => {
            let model = FeedforwardDependencyModel::from_bytes(bytes).map_err(|e| Error::Model(e.to_string()))?;
            let beam = model.default_beam_size();
            let parser = FeedforwardDependencyParser::new(model, beam).map_err(|e| Error::Model(e.to_string()))?;
            score_with_parser(&parser, samples)
        }
    }
}

fn score_with_parser<P: DependencyParser<Words>>(
    parser: &P,
    samples: &[DependencySample<Words>],
) -> Result<(Vec<Counts>, Vec<Counts>, DepparseScoresCard)> {
    let mut evaluator = DependencyEvaluator::default();
    let mut uas_counts = Vec::with_capacity(samples.len());
    let mut las_counts = Vec::with_capacity(samples.len());
    for sample in samples {
        let mut unit_eval = DependencyEvaluator::default();
        unit_eval.evaluate(parser, sample).map_err(|e| Error::Model(e.to_string()))?;
        evaluator.all.words += unit_eval.all.words;
        evaluator.all.heads += unit_eval.all.heads;
        evaluator.all.labels += unit_eval.all.labels;
        evaluator.without_punctuation.words += unit_eval.without_punctuation.words;
        evaluator.without_punctuation.heads += unit_eval.without_punctuation.heads;
        evaluator.without_punctuation.labels += unit_eval.without_punctuation.labels;
        uas_counts.push(Counts::accuracy(unit_eval.all.heads, unit_eval.all.words));
        las_counts.push(Counts::accuracy(unit_eval.all.labels, unit_eval.all.words));
    }
    let scores = DepparseScoresCard {
        uas: evaluator.all.uas() * 100.0,
        las: evaluator.all.las() * 100.0,
        uas_without_punctuation: evaluator.without_punctuation.uas() * 100.0,
        las_without_punctuation: evaluator.without_punctuation.las() * 100.0,
    };
    Ok((uas_counts, las_counts, scores))
}

/// Trains `recipe` on `text.train`, evaluates and gates it on `text.test`
/// for both LAS and UAS, and builds its model card.
pub fn run_depparse(recipe: &DepparseRecipe, text: &TreebankText) -> Result<DepparseOutcome> {
    let trained = recipe.train(&text.train)?;
    let held_out = read_samples(&text.test, recipe.treebank.repository, recipe.treebank.test.name)?;
    check_skipped(LeftOut::HeldOutSentences, held_out.skipped, held_out.samples.len())?;
    let (gate, uas_gate, held_out_card) = recipe.gate(&trained.bytes, &trained.reference_bytes, &held_out)?;
    let card = DepparseModelCard::new(recipe, &trained, held_out_card, &gate, &uas_gate);
    Ok(DepparseOutcome { trained, gate, uas_gate, card })
}

#[cfg(test)]
mod tests;
