//! The model card of a model a UD recipe trained, and its zoo entry.
//!
//! A trained model's directory holds the model file, `card.json` (this
//! card) and `gate.json`: the gate record ([`ModelGateRecord`]), or for a
//! model with no reference the record of the model scored alone
//! ([`UngatedRecord`]). The card says what the model was trained from and
//! with, the treebank's licence, the attribution and citation it asks
//! for, and how the gate went. It is strict JSON: every field is required
//! and an unknown field is an error.
//!
//! [`ModelGateRecord`]: turbo_eval::model_gate::ModelGateRecord
//! [`UngatedRecord`]: turbo_eval::model_gate::UngatedRecord

use std::path::Path;

use serde::{Deserialize, Serialize};
use turbo_train_components::ud as published;
use turbo_zoo::{Artifact, File, Kind, Licence, LicenceClass, Sha256, Sha512, TrainingData, class_of, index::Format};

use crate::Trained;
use crate::card::{
    CARD_FILE, GATE_FILE, MODEL_LICENCE, MODEL_LICENCE_TERMS, MODEL_VERSION, ModelFileCard, ParametersCard, TagCount,
    TagFormat, TrainingCard,
};
use crate::error::{Error, Result, io};
use crate::segmentation::TIMESTAMP_MS;
use crate::treebank::{Grant, RELEASE, TreebankFile};
use crate::ud::{SENTENCES_PER_SAMPLE, TAG_FORMAT, UdGate, UdRecipe};

/// The schema version of [`UdModelCard`].
pub const UD_CARD_VERSION: u32 = 1;
/// The version the zoo lists a model scored alone under: the recipe's,
/// marked experimental.
pub const EXPERIMENTAL_VERSION: &str = "1-experimental";

/// One file of the treebank as read.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TreebankFileCard {
    pub name: String,
    pub size: u64,
    pub sha256: String,
}

impl TreebankFileCard {
    fn of(f: &TreebankFile) -> TreebankFileCard {
        TreebankFileCard { name: f.name.to_owned(), size: f.size, sha256: f.sha256.to_owned() }
    }
}

/// The treebank the model was trained and gated on.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TreebankCard {
    /// The repository under `github.com/UniversalDependencies`.
    pub repository: String,
    /// The Universal Dependencies release tag.
    pub release: String,
    pub license: Licence,
    /// The treebank's `LICENSE.txt` at the release.
    pub license_terms: String,
    /// Another licence `LICENSE.txt` offers beside CC BY 4.0, which the
    /// recipe did not take; `null` when it offers CC BY 4.0 alone.
    pub other_licence_offered: Option<String>,
    pub licence_file: TreebankFileCard,
    pub readme: TreebankFileCard,
    /// UD's own split: the model is trained on `train` and gated on
    /// `test`; `dev` is checked against its digest and not read.
    pub train: TreebankFileCard,
    pub dev: TreebankFileCard,
    pub test: TreebankFileCard,
    /// The samples read from `test` and scored.
    pub held_out_samples: usize,
    /// Sentences of `test` left out of the held-out samples: those with no
    /// word once empty nodes are left out (0 for the sentence detector and
    /// the tokenizer).
    pub held_out_skipped_sentences: usize,
}

/// The gate's outcome; the record itself is [`GATE_FILE`].
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", deny_unknown_fields)]
pub enum UdGateCard {
    /// Gated against the OpenNLP UD 1.3 model.
    Gated {
        record: String,
        reference: String,
        reference_sha256: String,
        /// For a tagger, the tags both were scored in; `null` for the other
        /// components.
        tag_format: Option<TagFormat>,
        /// For a tagger, the held-out tags the reference has no outcome
        /// for, whose tokens were left out of both models' counts.
        tags_left_out: Vec<TagCount>,
        candidate_score: f64,
        reference_score: f64,
        difference_low: f64,
        difference_high: f64,
        pass: bool,
    },
    /// Scored alone: no reference the recipe may use.
    Ungated { record: String, no_reference: String, candidate_score: f64, candidate_low: f64, candidate_high: f64 },
}

/// The model card of a UD recipe's model.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct UdModelCard {
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
    /// The work the treebank's README asks to be cited.
    pub citation: String,
    pub treebank: TreebankCard,
    pub training: TrainingCard,
    pub gate: UdGateCard,
}

impl UdModelCard {
    /// The card of a model `recipe` trained, scored on `held_out_samples`
    /// held-out samples with `held_out_skipped` sentences left out, and
    /// gated (or scored alone) by `gate`.
    pub fn new(
        recipe: &UdRecipe,
        trained: &Trained,
        held_out_samples: usize,
        held_out_skipped: usize,
        gate: &UdGate,
    ) -> UdModelCard {
        let t = &recipe.treebank;
        let trained_on = vec![t.training_data()];
        let class = class_of(std::iter::once(MODEL_LICENCE).chain(trained_on.iter().map(|d| d.license)));
        let segmentation = matches!(recipe.component, published::Component::Sentence | published::Component::Tokens);
        let gate = match gate {
            UdGate::Gated { record, tags_left_out } => {
                let r = &record.result;
                UdGateCard::Gated {
                    record: GATE_FILE.to_owned(),
                    reference: record.reference.id.clone(),
                    reference_sha256: record.reference.sha256.clone(),
                    tag_format: (recipe.component == published::Component::Pos)
                        .then(|| TagFormat::of(TAG_FORMAT))
                        .flatten(),
                    tags_left_out: tags_left_out.clone(),
                    candidate_score: r.candidate.score,
                    reference_score: r.reference.score,
                    difference_low: r.difference.low,
                    difference_high: r.difference.high,
                    pass: r.pass,
                }
            }
            UdGate::Ungated(record) => UdGateCard::Ungated {
                record: GATE_FILE.to_owned(),
                no_reference: record.no_reference.clone(),
                candidate_score: record.result.score,
                candidate_low: record.result.low,
                candidate_high: record.result.high,
            },
        };
        UdModelCard {
            card_version: UD_CARD_VERSION,
            id: recipe.model_id.to_owned(),
            component: recipe.zoo_component(),
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
            treebank: TreebankCard {
                repository: t.repository.to_owned(),
                release: RELEASE.to_owned(),
                license: crate::treebank::LICENCE,
                license_terms: t.licence_terms(),
                other_licence_offered: match t.grant {
                    Grant::CcBy40 => None,
                    Grant::CcBy40OrOther { other } => Some(other.to_owned()),
                },
                licence_file: TreebankFileCard::of(&t.licence_file),
                readme: TreebankFileCard::of(&t.readme),
                train: TreebankFileCard::of(&t.train),
                dev: TreebankFileCard::of(&t.dev),
                test: TreebankFileCard::of(&t.test),
                held_out_samples,
                held_out_skipped_sentences: held_out_skipped,
            },
            training: TrainingCard {
                parameters: ParametersCard {
                    algorithm: "MAXENT".to_owned(),
                    iterations: published::ITERATIONS,
                    cutoff: published::CUTOFF,
                    threads: published::THREADS,
                },
                sentences_per_sample: (recipe.component == published::Component::Sentence)
                    .then_some(SENTENCES_PER_SAMPLE),
                whitespace: segmentation.then(|| "legacy".to_owned()),
                timestamp_ms: TIMESTAMP_MS,
                seed: None,
                samples: trained.samples,
                skipped_sentences: trained.skipped_sentences,
                events: trained.events,
                event_hash: trained.event_hash,
            },
            gate,
        }
    }

    /// The model's zoo entry once the file is published at `url`. A model
    /// scored alone is listed with the version [`EXPERIMENTAL_VERSION`].
    ///
    /// # Errors
    ///
    /// [`Error::NotShippable`] when the card is not `recipe`'s, the gate
    /// failed, the gate's reference is not the recipe's (by id, and by the
    /// SHA-256 the zoo lists or the recipe records), or the model was
    /// scored alone while the recipe has a reference; or an address or
    /// field the index refuses.
    pub fn artifact(&self, recipe: &UdRecipe, url: &str) -> Result<Artifact> {
        let refuse = |reason: &'static str| Error::NotShippable { id: self.id.clone(), reason };
        if self.id != recipe.model_id {
            return Err(refuse("the card is not the recipe's"));
        }
        let version = match (&self.gate, recipe.reference_model()) {
            (UdGateCard::Gated { pass: false, .. }, _) => return Err(refuse("it failed its gate")),
            (UdGateCard::Gated { reference, reference_sha256, .. }, Some(expected)) => {
                if reference != expected.id() || *reference_sha256 != expected.sha256()? {
                    return Err(refuse("its gate's reference is not the recipe's"));
                }
                MODEL_VERSION
            }
            (UdGateCard::Ungated { .. }, None) => EXPERIMENTAL_VERSION,
            (UdGateCard::Gated { .. }, None) => {
                return Err(refuse("it was gated though the recipe has no reference"));
            }
            (UdGateCard::Ungated { .. }, Some(_)) => {
                return Err(refuse("it was not gated though the recipe has a reference"));
            }
        };
        let bad = |field: &'static str, message: String| Error::InvalidCard { field, message };
        Ok(Artifact {
            id: self.id.parse().map_err(|e| bad("id", e))?,
            kind: Kind::Model,
            component: Some(self.component),
            language: Some(self.language.parse().map_err(|e| bad("language", e))?),
            format: Format::OpennlpModel,
            version: version.to_owned(),
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

    pub fn read(path: &Path) -> Result<UdModelCard> {
        let bytes = std::fs::read(path).map_err(io(path))?;
        let card: UdModelCard = serde_json::from_slice(&bytes)
            .map_err(|e| Error::Json { path: path.to_owned(), message: e.to_string() })?;
        if card.card_version != UD_CARD_VERSION {
            return Err(Error::CardVersion {
                path: path.to_owned(),
                found: card.card_version,
                expected: UD_CARD_VERSION,
            });
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
pub fn write_ud_model_dir(dir: &Path, card: &UdModelCard, model: &[u8], gate: &UdGate) -> Result<()> {
    let out = dir.join(&card.id);
    std::fs::create_dir_all(dir).map_err(io(dir))?;
    std::fs::create_dir(&out).map_err(io(&out))?;
    for (name, bytes) in [
        (card.file.name.as_str(), model),
        (CARD_FILE, card.to_json().as_bytes()),
        (GATE_FILE, gate.to_json().as_bytes()),
    ] {
        let path = out.join(name);
        std::fs::write(&path, bytes).map_err(io(&path))?;
    }
    Ok(())
}
