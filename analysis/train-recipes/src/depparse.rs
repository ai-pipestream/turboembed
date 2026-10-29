//! Historical UD 2.16 dependency parser recipe metadata and CoNLL-U sample
//! reading. Classical MaxEnt (`DependencyParserME`) and Feedforward
//! (`FeedforwardDependencyParser`) arc-standard dependency parser models are
//! retired under #1529 and replaced by the biaffine parser head on the shared
//! encoder (`turbo-core` / `turbo-tagger`, #1514).

pub mod card;

use turbo_analysis::Ix;
use turbo_analysis_layers::Words;
use turbo_depparse::DependencySample;
use turbo_eval::model_gate::ModelGateRecord;
use turbo_formats::FormatError;
use turbo_formats::conllu::ConlluDependencyReader;
use turbo_train::{Key, TrainingParameters};
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
    let mut reader = ConlluDependencyReader::new(conllu, TAGSET).map_err(conllu_err)?;
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

    /// Classical MaxEnt and Feedforward dependency parser training is retired
    /// under #1529 and replaced by the biaffine parser head (#1514).
    pub fn train(&self, _train: &str) -> Result<TrainedDepparse> {
        Err(Error::Model("classical dependency parser training is retired (#1529); use biaffine parser (#1514)".into()))
    }

    /// Classical MaxEnt and Feedforward dependency parser gating is retired
    /// under #1529 and replaced by the biaffine parser head (#1514).
    pub fn gate(
        &self,
        _trained: &[u8],
        _reference: &[u8],
        _held_out: &ReadDepparseSamples,
    ) -> Result<(ModelGateRecord, ModelGateRecord, DepparseHeldOutCard)> {
        Err(Error::Model("classical dependency parser gating is retired (#1529); use biaffine parser (#1514)".into()))
    }
}

/// Classical MaxEnt and Feedforward dependency parser recipes are retired
/// under #1529 and replaced by the biaffine parser head (#1514).
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
