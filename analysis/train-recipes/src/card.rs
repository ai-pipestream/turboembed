//! The model card of a trained model, and its entry in the zoo.
//!
//! A trained model's directory holds the model file, `card.json` (this
//! card) and `gate.json` (the gate record, [`ModelGateRecord`]), as the
//! chunker's evaluation records are kept beside each other under
//! `analysis/eval/records`. The card says what the model was trained
//! from and with, its licence and the attribution its data asks for, and
//! whether it passed its gate. Like the gate record it is strict JSON: every
//! field is required and an unknown field is an error.

use std::path::Path;

use serde::{Deserialize, Serialize};
use turbo_eval::model_gate::ModelGateRecord;
use turbo_postag::PosTagFormat;
use turbo_train_components::ud;
use turbo_zoo::{Artifact, File, Kind, Licence, LicenceClass, Sha256, Sha512, TrainingData, class_of, index::Format};

use crate::error::{Error, Result, io};
use crate::masc::{self, Split};
use crate::segmentation::{Component, LANGUAGE, SENTENCES_PER_SAMPLE, TIMESTAMP_MS};
use crate::{Recipe, Trained, postag};

/// The card's schema version.
pub const CARD_VERSION: u32 = 2;
/// The card's file name in the model's directory.
pub const CARD_FILE: &str = "card.json";
/// The gate record's file name in the model's directory.
pub const GATE_FILE: &str = "gate.json";
/// The trained models' own licence: the repository's.
pub const MODEL_LICENCE: Licence = Licence::Apache2;
/// Where the repository states it.
pub const MODEL_LICENCE_TERMS: &str = "https://github.com/ai-pipestream/turboembed/blob/main/LICENSE";
/// The version the zoo lists a model under: the recipe's.
pub const MODEL_VERSION: &str = "1";

/// The model file.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ModelFileCard {
    pub name: String,
    pub size: u64,
    pub sha256: String,
    pub sha512: String,
}

/// The corpus and the split the model was trained and gated on.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CorpusCard {
    pub name: String,
    /// [`masc::corpus_digest`] of the directory read.
    pub corpus_sha256: String,
    /// Whether that digest is the recorded release's ([`masc::RELEASE`]).
    /// A model trained on an unrecorded copy is not a release.
    pub release_recorded: bool,
    pub split: Split,
    pub training_documents: usize,
    /// The held-out documents, relative to the corpus directory.
    pub held_out_documents: Vec<String>,
    /// Documents the corpus read logged and left out.
    pub skipped_documents: Vec<String>,
}

/// The trainer's parameters, as the parameters file gives them.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ParametersCard {
    pub algorithm: String,
    pub iterations: i32,
    pub cutoff: i32,
    pub threads: i32,
}

/// What training read and made.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TrainingCard {
    pub parameters: ParametersCard,
    /// Sentences per sentence detector sample; `null` for the other
    /// components.
    pub sentences_per_sample: Option<usize>,
    /// `legacy` (OpenNLP 2.5.4's `Character.isWhitespace` or `Zs`) for the
    /// sentence detector and the tokenizer; `null` for the tagger, which
    /// reads tokens.
    pub whitespace: Option<String>,
    /// The manifest's `Timestamp`.
    pub timestamp_ms: i64,
    /// Training uses no randomness, so there is no seed.
    pub seed: Option<i64>,
    pub samples: usize,
    /// Sentences left out of training ([`Trained::skipped_sentences`]).
    pub skipped_sentences: usize,
    pub events: usize,
    pub event_hash: u32,
}

/// What the gate read from the held-out documents.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct HeldOutCard {
    /// The samples scored, a unit of the gate each.
    pub samples: usize,
    /// Held-out sentences the recipe's reader skipped, for the reasons it
    /// skips training sentences ([`TrainingCard::skipped_sentences`]);
    /// at most [`crate::MAX_SKIPPED_SENTENCES`] of those read.
    pub skipped_sentences: usize,
}

/// The tags a tagger was scored in, as `POSTaggerME` returns them for a
/// `POSTagFormat`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TagFormat {
    /// Universal Dependencies tags: the outcomes of a UD model, or a Penn
    /// model's converted by `POSTagFormatMapper`.
    Ud,
    /// Penn Treebank tags: the outcomes of a Penn model, or a UD model's
    /// converted by `POSTagFormatMapper`, one Penn tag per UD tag.
    Penn,
    /// The model's outcomes as they are (`POSTagFormat.CUSTOM`).
    Custom,
}

impl TagFormat {
    /// The format of a tagger asked for `format`; `None` for
    /// `PosTagFormat::Unknown`, which no recipe asks for.
    pub fn of(format: PosTagFormat) -> Option<TagFormat> {
        match format {
            PosTagFormat::Ud => Some(TagFormat::Ud),
            PosTagFormat::Penn => Some(TagFormat::Penn),
            PosTagFormat::Custom => Some(TagFormat::Custom),
            PosTagFormat::Unknown => None,
        }
    }
}

/// The gate's outcome; the record itself is [`GATE_FILE`].
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct GateCard {
    pub record: String,
    pub reference: String,
    /// For a tagger, the tags the trained model was scored in; `null`
    /// for the other components.
    pub candidate_tag_format: Option<TagFormat>,
    /// For a tagger, the tags the reference was scored in; `null` for the
    /// other components.
    pub reference_tag_format: Option<TagFormat>,
    pub candidate_score: f64,
    pub reference_score: f64,
    pub difference_low: f64,
    pub difference_high: f64,
    pub pass: bool,
}

/// The model card.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ModelCard {
    pub card_version: u32,
    pub id: String,
    pub component: turbo_zoo::Component,
    pub language: String,
    pub file: ModelFileCard,
    pub license: Licence,
    pub license_terms: String,
    pub trained_on: Vec<TrainingData>,
    /// The class of the licence and the training data's licences.
    pub class: LicenceClass,
    /// The credit CC BY asks for, to be shipped with the model.
    pub attribution: String,
    pub corpus: CorpusCard,
    pub training: TrainingCard,
    pub held_out: HeldOutCard,
    pub gate: GateCard,
}

impl ModelCard {
    /// The card of a model `recipe` trained, gated by `gate` on what
    /// `held_out` describes.
    pub fn new(
        recipe: &Recipe,
        trained: &Trained,
        corpus: CorpusCard,
        held_out: HeldOutCard,
        gate: &ModelGateRecord,
    ) -> ModelCard {
        let trained_on = vec![masc::training_data()];
        let class = class_of(std::iter::once(MODEL_LICENCE).chain(trained_on.iter().map(|t| t.license)));
        let r = &gate.result;
        let (sentences_per_sample, whitespace, candidate_tag_format, reference_tag_format) = match recipe {
            Recipe::Segmentation(s) => (
                (s.component == Component::Sentence).then_some(SENTENCES_PER_SAMPLE),
                Some("legacy".to_owned()),
                None,
                None,
            ),
            Recipe::Pos(p) => (None, None, TagFormat::of(postag::CANDIDATE_FORMAT), TagFormat::of(p.reference_format)),
        };
        ModelCard {
            card_version: CARD_VERSION,
            id: recipe.model_id().to_owned(),
            component: recipe.component(),
            language: LANGUAGE.to_owned(),
            file: ModelFileCard {
                name: format!("{}.bin", recipe.model_id()),
                size: trained.bytes.len() as u64,
                sha256: Sha256::of(&trained.bytes).to_string(),
                sha512: Sha512::of(&trained.bytes).to_string(),
            },
            license: MODEL_LICENCE,
            license_terms: MODEL_LICENCE_TERMS.to_owned(),
            trained_on,
            class,
            attribution: masc::ATTRIBUTION.to_owned(),
            corpus,
            training: TrainingCard {
                parameters: ParametersCard {
                    algorithm: "MAXENT".to_owned(),
                    iterations: ud::ITERATIONS,
                    cutoff: ud::CUTOFF,
                    threads: ud::THREADS,
                },
                sentences_per_sample,
                whitespace,
                timestamp_ms: TIMESTAMP_MS,
                seed: None,
                samples: trained.samples,
                skipped_sentences: trained.skipped_sentences,
                events: trained.events,
                event_hash: trained.event_hash,
            },
            held_out,
            gate: GateCard {
                record: GATE_FILE.to_owned(),
                reference: gate.reference.id.clone(),
                candidate_tag_format,
                reference_tag_format,
                candidate_score: r.candidate.score,
                reference_score: r.reference.score,
                difference_low: r.difference.low,
                difference_high: r.difference.high,
                pass: r.pass,
            },
        }
    }

    /// The model's zoo entry once the file is published at `url`.
    ///
    /// # Errors
    ///
    /// [`Error::NotShippable`] when the gate failed or the corpus is not
    /// the recorded release, or an address or field the index refuses.
    pub fn artifact(&self, url: &str) -> Result<Artifact> {
        if !self.gate.pass {
            return Err(Error::NotShippable { id: self.id.clone(), reason: "it failed its gate" });
        }
        if !self.corpus.release_recorded {
            return Err(Error::NotShippable {
                id: self.id.clone(),
                reason: "it was trained on a corpus that is not the recorded release",
            });
        }
        let bad = |field: &'static str, message: String| Error::InvalidCard { field, message };
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
            origin: None,
            files: vec![File {
                path: self.file.name.parse().map_err(|e| bad("file name", e))?,
                url: url.parse().map_err(|e| bad("url", e))?,
                size: self.file.size,
                sha256: self.file.sha256.parse().map_err(|e| bad("sha256", e))?,
                sha512: self.file.sha512.parse().map_err(|e| bad("sha512", e))?,
            }],
        })
    }

    /// The card as written: pretty JSON with a final newline.
    pub fn to_json(&self) -> String {
        let mut s = serde_json::to_string_pretty(self).expect("a card serializes");
        s.push('\n');
        s
    }

    pub fn read(path: &Path) -> Result<ModelCard> {
        let bytes = std::fs::read(path).map_err(io(path))?;
        let card: ModelCard = serde_json::from_slice(&bytes)
            .map_err(|e| Error::Json { path: path.to_owned(), message: e.to_string() })?;
        if card.card_version != CARD_VERSION {
            return Err(Error::CardVersion { path: path.to_owned(), found: card.card_version });
        }
        Ok(card)
    }
}

/// Writes a trained model's directory `dir/<model id>/`: the model file,
/// [`CARD_FILE`] and [`GATE_FILE`]. The directory must not exist yet, so
/// nothing is replaced.
///
/// # Errors
///
/// An existing directory, or a write that fails.
pub fn write_model_dir(dir: &Path, card: &ModelCard, model: &[u8], gate: &ModelGateRecord) -> Result<()> {
    let out = dir.join(&card.id);
    std::fs::create_dir_all(dir).map_err(io(dir))?;
    std::fs::create_dir(&out).map_err(io(&out))?;
    let model_path = out.join(&card.file.name);
    std::fs::write(&model_path, model).map_err(io(&model_path))?;
    let card_path = out.join(CARD_FILE);
    std::fs::write(&card_path, card.to_json()).map_err(io(&card_path))?;
    Ok(gate.write(&out.join(GATE_FILE))?)
}
