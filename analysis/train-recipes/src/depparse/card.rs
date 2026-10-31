//! Model card and directory layout for UD dependency parser models.

use std::path::Path;

use serde::{Deserialize, Serialize};
use turbo_eval::model_gate::ModelGateRecord;
use turbo_zoo::{Artifact, File, Kind, Licence, LicenceClass, Sha256, Sha512, TrainingData, class_of, index::Format};

use super::{
    DepparseKind, DepparseOutcome, DepparseRecipe, MAXENT_CUTOFF, MAXENT_ITERATIONS, MAXENT_THREADS, TrainedDepparse,
    feedforward_settings,
};
use crate::card::{
    GateCard, MODEL_LICENCE, MODEL_LICENCE_TERMS, MODEL_VERSION, ModelFileCard, publishable_class, read_card, write_dir,
};
use crate::error::{Error, Result};
use crate::segmentation::TIMESTAMP_MS;
use crate::treebank::TextRights;
use crate::ud_card::TreebankCard;

/// Schema version of [`DepparseModelCard`].
pub const DEPPARSE_CARD_VERSION: u32 = 1;

/// File name of the Unlabeled Attachment Score (UAS) gate record kept beside
/// `gate.json` (LAS) in each dependency parser model directory.
pub const UAS_GATE_FILE: &str = "gate-uas.json";

/// Training hyperparameters recorded in a [`DepparseModelCard`].
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum DepparseParametersCard {
    Maxent {
        algorithm: String,
        iterations: i32,
        cutoff: i32,
        threads: i32,
    },
    Feedforward {
        embedding_size: usize,
        hidden_size: usize,
        epochs: usize,
        batch_size: usize,
        learning_rate: f64,
        l2: f64,
        dropout: f64,
        word_cutoff: usize,
        seed: i64,
        layout: String,
        activation: String,
    },
}

impl DepparseParametersCard {
    pub fn of(kind: DepparseKind) -> Self {
        match kind {
            DepparseKind::Maxent => DepparseParametersCard::Maxent {
                algorithm: "MAXENT".to_owned(),
                iterations: MAXENT_ITERATIONS,
                cutoff: MAXENT_CUTOFF,
                threads: MAXENT_THREADS,
            },
            DepparseKind::Feedforward => {
                let s = feedforward_settings(5);
                DepparseParametersCard::Feedforward {
                    embedding_size: s.embedding_size,
                    hidden_size: s.hidden_size,
                    epochs: s.epochs,
                    batch_size: s.batch_size,
                    learning_rate: s.learning_rate,
                    l2: s.l2,
                    dropout: s.dropout,
                    word_cutoff: s.word_cutoff,
                    seed: s.seed,
                    layout: "ChenManning18".to_owned(),
                    activation: "cube".to_owned(),
                }
            }
        }
    }
}

/// Training split statistics and hyperparameters for a dependency parser.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DepparseTrainingCard {
    pub parameters: DepparseParametersCard,
    pub pseudo_projective: bool,
    pub timestamp_ms: i64,
    pub samples: usize,
    pub projective_samples: usize,
    pub nonprojective_samples: usize,
    pub skipped_sentences: usize,
    pub tokens: u64,
    pub events: usize,
}

/// Attachment scores (in points, `0..=100`) on the held-out `test` split.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DepparseScoresCard {
    pub uas: f64,
    pub las: f64,
    pub uas_without_punctuation: f64,
    pub las_without_punctuation: f64,
}

/// Held-out `test` split statistics and attachment scores.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DepparseHeldOutCard {
    pub samples: usize,
    pub projective_samples: usize,
    pub nonprojective_samples: usize,
    pub skipped_sentences: usize,
    pub tokens: u64,
    pub punctuation_tokens: u64,
    pub candidate: DepparseScoresCard,
    pub reference: DepparseScoresCard,
}

/// Model card for a trained UD dependency parser.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DepparseModelCard {
    pub card_version: u32,
    pub id: String,
    pub component: turbo_zoo::Component,
    pub parser_kind: String,
    pub language: String,
    pub file: ModelFileCard,
    pub license: Licence,
    pub license_terms: String,
    pub trained_on: Vec<TrainingData>,
    pub class: LicenceClass,
    pub attribution: String,
    pub citation: String,
    pub treebank: TreebankCard,
    pub training: DepparseTrainingCard,
    pub held_out: DepparseHeldOutCard,
    /// Labeled Attachment Score (LAS) gate (`gate.json`).
    pub gate: GateCard,
    /// Unlabeled Attachment Score (UAS) gate (`gate-uas.json`).
    pub uas_gate: GateCard,
}

impl DepparseModelCard {
    /// Constructs the card of a model trained by `recipe`.
    pub fn new(
        recipe: &DepparseRecipe,
        trained: &TrainedDepparse,
        held_out: DepparseHeldOutCard,
        gate: &ModelGateRecord,
        uas_gate: &ModelGateRecord,
    ) -> Self {
        let t = &recipe.treebank;
        let trained_on = vec![t.training_data()];
        let class = class_of(std::iter::once(MODEL_LICENCE).chain(trained_on.iter().map(|d| d.license)));
        let mut uas_gate_card = GateCard::of(uas_gate, None, None);
        uas_gate_card.record = UAS_GATE_FILE.to_owned();
        DepparseModelCard {
            card_version: DEPPARSE_CARD_VERSION,
            id: recipe.model_id.to_owned(),
            component: turbo_zoo::Component::DependencyParser,
            parser_kind: recipe.kind.name().to_owned(),
            language: t.language.to_owned(),
            file: ModelFileCard {
                name: format!("{}.bin", recipe.model_id),
                size: trained.bytes.len() as u64,
                sha256: Sha256::of(&trained.bytes).to_string(),
                sha512: Sha512::of(&trained.bytes).to_string(),
            },
            license: MODEL_LICENCE,
            license_terms: MODEL_LICENCE_TERMS.to_owned(),
            trained_on,
            class,
            attribution: t.attribution(),
            citation: t.citation.to_owned(),
            treebank: TreebankCard::of(t),
            training: DepparseTrainingCard {
                parameters: DepparseParametersCard::of(recipe.kind),
                pseudo_projective: true,
                timestamp_ms: TIMESTAMP_MS,
                samples: trained.samples,
                projective_samples: trained.projective_samples,
                nonprojective_samples: trained.samples - trained.projective_samples,
                skipped_sentences: trained.skipped_sentences,
                tokens: trained.tokens,
                events: trained.events,
            },
            held_out,
            gate: GateCard::of(gate, None, None),
            uas_gate: uas_gate_card,
        }
    }

    /// The model's zoo entry once the `.bin` file is published at `url`.
    pub fn artifact(&self, url: &str) -> Result<Artifact> {
        let class = class_of(std::iter::once(self.license).chain(self.trained_on.iter().map(|t| t.license)));
        if class != self.class {
            return Err(Error::InvalidCard {
                field: "class",
                message: format!("{} is not {class}, the class its licences give", self.class),
            });
        }
        publishable_class(&self.id, self.license, &self.trained_on)?;
        let not = |reason| Error::NotShippable { id: self.id.clone(), reason };
        let recipe = DepparseRecipe::trains(&self.id).ok_or(not("no recipe trains it"))?;
        if let TextRights::NotSettled { .. } = recipe.treebank.text_rights {
            return Err(not("the rights to its treebank's text are not settled"));
        }
        if self.gate.reference != recipe.reference_id() || self.uas_gate.reference != recipe.reference_id() {
            return Err(not("it was not gated against its recipe's reference"));
        }
        if !self.gate.pass || !self.uas_gate.pass {
            return Err(not("it failed its gate"));
        }
        let t = &recipe.treebank;
        if self.trained_on != [t.training_data()] || self.treebank != TreebankCard::of(t) {
            return Err(not("it was not trained and gated on its recipe's treebank"));
        }
        let bad = |field: &'static str, message: String| Error::InvalidCard { field, message };
        if self.component != turbo_zoo::Component::DependencyParser {
            return Err(bad(
                "component",
                format!("{} is not {}, its recipe's", self.component, turbo_zoo::Component::DependencyParser),
            ));
        }
        if self.parser_kind != recipe.kind.name() {
            return Err(bad("parser_kind", format!("{} is not {}", self.parser_kind, recipe.kind.name())));
        }
        if self.language != t.language {
            return Err(bad("language", format!("{} is not {}, its recipe's", self.language, t.language)));
        }
        if self.attribution.trim().is_empty() || self.attribution.chars().any(char::is_control) {
            return Err(bad("attribution", format!("{:?} is empty or holds a control character", self.attribution)));
        }
        Ok(Artifact {
            id: self.id.parse().map_err(|e| bad("id", e))?,
            kind: Kind::Model,
            component: Some(self.component),
            language: Some(self.language.parse().map_err(|e| bad("language", e))?),
            format: Format::OpennlpModel,
            version: MODEL_VERSION.to_owned(),
            license: self.license,
            license_terms: self.license_terms.parse().map_err(|e| bad("licence terms", e))?,
            trained_on: self.trained_on.clone(),
            class: self.class,
            attribution: Some(self.attribution.clone()),
            origin: None,
            files: vec![File {
                path: self.file.name.parse().map_err(|e| bad("file name", e))?,
                url: url.parse().map_err(|e| bad("url", e))?,
                size: self.file.size,
                sha256: self.file.sha256.parse().map_err(|e| bad("sha256", e))?,
                sha512: self.file.sha512.parse().map_err(|e| bad("sha512", e))?,
                download: None,
            }],
        })
    }

    /// Serializes the card as pretty JSON with a trailing newline.
    pub fn to_json(&self) -> String {
        let mut s = serde_json::to_string_pretty(self).expect("card serializes");
        s.push('\n');
        s
    }

    /// Reads and validates a [`DepparseModelCard`] from `path`.
    pub fn read(path: &Path) -> Result<DepparseModelCard> {
        read_card(path, DEPPARSE_CARD_VERSION)
    }

    /// Human-readable one-line summary of the trained model and its test UAS/LAS.
    pub fn summary(&self) -> String {
        format!(
            "{} ({}): {} bytes, SHA-256 {}; train {} sentences ({} projective, {} non-projective, {} tokens, {} events); test {} sentences ({} tokens): UAS {:.2} (no-punct {:.2}), LAS {:.2} (no-punct {:.2})",
            self.id,
            self.parser_kind,
            self.file.size,
            self.file.sha256,
            self.training.samples,
            self.training.projective_samples,
            self.training.nonprojective_samples,
            self.training.tokens,
            self.training.events,
            self.held_out.samples,
            self.held_out.tokens,
            self.held_out.candidate.uas,
            self.held_out.candidate.uas_without_punctuation,
            self.held_out.candidate.las,
            self.held_out.candidate.las_without_punctuation,
        )
    }
}

/// Writes `<dir>/<model_id>/` containing `<model_id>.bin`, `card.json`,
/// `gate.json` (LAS), and `gate-uas.json` (UAS).
pub fn write_depparse_model_dir(dir: &Path, outcome: &DepparseOutcome) -> Result<()> {
    let card = &outcome.card;
    write_dir(dir, &card.id, (&card.file.name, &outcome.trained.bytes), &card.to_json(), &outcome.gate)?;
    outcome.uas_gate.write(&dir.join(&card.id).join(UAS_GATE_FILE))?;
    Ok(())
}
