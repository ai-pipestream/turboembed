use std::path::Path;

use serde::{Deserialize, Serialize};
use turbo_eval::model_gate::{LabelMapping, ModelGateRecord};
use turbo_train::Algorithm;
use turbo_train_components::langdetect::ONE_PASS;
use turbo_zoo::{Artifact, File, Kind, Licence, LicenceClass, Sha256, Sha512, TrainingData, class_of, index::Format};

use super::{
    ACCEPTED_LICENCES, CARD_FILE, CARD_VERSION, ExcludedLanguage, LANGDETECT, LanguageCount, LanguageOutcome,
    LanguageRecipe, Plan, SENTENCE_URL, SENTENCES_FILE, Split, TrainedLanguages, ZOO_LANGUAGE,
};
use crate::card::{
    GATE_FILE, GateCard, MODEL_LICENCE, MODEL_LICENCE_TERMS, MODEL_VERSION, ModelFileCard, publishable_class,
};
use crate::error::{Error, Result, io};
use crate::segmentation::TIMESTAMP_MS;
use crate::tatoeba::{self, Export, ExportFile, SentenceLicence};

/// The export the model was trained and gated on.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ExportCard {
    pub detailed: ExportFile,
    pub cc0: ExportFile,
    /// [`Export::digest`].
    pub digest: String,
    /// Whether it is [`EXPORT`](super::EXPORT). A model trained on another
    /// export is not a release.
    pub recorded: bool,
    /// The records of the detailed export.
    pub records: usize,
    pub no_language: usize,
    pub other_languages: usize,
    /// The records of the CC0 export.
    pub cc0_records: usize,
    /// Its records whose id the detailed export holds with another
    /// language or text: sentences changed between the two files' writes,
    /// still left out as CC0.
    pub cc0_differing: usize,
    /// Its records whose id the detailed export does not hold, ignored.
    pub cc0_not_in_detailed: usize,
    pub accepted_licences: Vec<SentenceLicence>,
}

/// The training parameters as the card records them.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct LanguageTrainingCard {
    pub algorithm: String,
    pub iterations: i32,
    pub cutoff: i32,
    pub threads: i32,
    pub data_indexer: String,
    /// The manifest's `Timestamp`.
    pub timestamp_ms: i64,
    pub samples: usize,
    pub events: usize,
    pub event_hash: u32,
    pub inexact_events: usize,
}

/// [`SENTENCES_FILE`] as the card lists it.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SentencesFileCard {
    pub name: String,
    /// The ids it lists, one per training sentence.
    pub sentences: usize,
    pub size: u64,
    pub sha256: String,
    pub sha512: String,
}

/// What the gate scored.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct LanguageHeldOutCard {
    /// The held-out sentences, a unit each.
    pub samples: usize,
    pub reference_mappings: Vec<LabelMapping>,
}

/// The language detector's model card. Strict JSON, as the gate record:
/// every field is required and an unknown field is an error.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct LanguageCard {
    pub card_version: u32,
    pub id: String,
    pub component: turbo_zoo::Component,
    pub language: String,
    /// The model's outcomes, Tatoeba's codes.
    pub languages: Vec<String>,
    pub file: ModelFileCard,
    pub license: Licence,
    pub license_terms: String,
    pub trained_on: Vec<TrainingData>,
    pub class: LicenceClass,
    /// The credit CC BY 2.0 FR and Tatoeba's terms ask for, shipped with
    /// the model: it points at [`SENTENCES_FILE`].
    pub attribution: String,
    pub training_sentences: SentencesFileCard,
    pub export: ExportCard,
    pub split: Split,
    pub per_language: Vec<LanguageCount>,
    pub excluded_languages: Vec<ExcludedLanguage>,
    pub training: LanguageTrainingCard,
    pub held_out: LanguageHeldOutCard,
    pub gate: GateCard,
}

/// The card's record of `export` and what `plan` read of it; `recorded`
/// is whether it is [`EXPORT`](super::EXPORT).
pub fn export_card(export: &Export, recorded: bool, plan: &Plan<'_>) -> ExportCard {
    ExportCard {
        detailed: export.detailed.clone(),
        cc0: export.cc0.clone(),
        digest: export.digest(),
        recorded,
        records: plan.records,
        no_language: plan.no_language,
        other_languages: plan.other_languages,
        cc0_records: plan.cc0_records,
        cc0_differing: plan.cc0_differing,
        cc0_not_in_detailed: plan.cc0_not_in_detailed,
        accepted_licences: ACCEPTED_LICENCES.to_vec(),
    }
}

/// The data the model is trained on, as the zoo lists it: the export of
/// digest `digest` ([`Export::digest`]).
pub fn training_data(digest: &str) -> TrainingData {
    TrainingData {
        name: format!("Tatoeba export {}", &digest[..16.min(digest.len())]),
        license: Licence::CcBy20Fr,
        license_terms: tatoeba::TERMS_URL.parse().expect("a valid address"),
    }
}

impl LanguageCard {
    pub fn new(
        recipe: &LanguageRecipe,
        export: &Export,
        recorded: bool,
        plan: &Plan<'_>,
        trained: &TrainedLanguages,
        gate: &ModelGateRecord,
    ) -> LanguageCard {
        let digest = export.digest();
        let trained_on = vec![training_data(&digest)];
        let class = class_of(std::iter::once(MODEL_LICENCE).chain(trained_on.iter().map(|t| t.license)));
        let r = &gate.result;
        LanguageCard {
            card_version: CARD_VERSION,
            id: recipe.model_id.to_owned(),
            component: turbo_zoo::Component::LanguageDetector,
            language: ZOO_LANGUAGE.to_owned(),
            languages: plan.languages.iter().map(|l| l.tatoeba.clone()).collect(),
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
            attribution: format!(
                "Trained on {} sentences from Tatoeba (https://tatoeba.org), licensed under CC BY 2.0 FR \
                 (https://creativecommons.org/licenses/by/2.0/fr/), from its export of digest {digest}. \
                 {SENTENCES_FILE}, shipped with the model, lists their ids: the sentence of id N, with its author, \
                 is at {SENTENCE_URL}N.",
                trained.samples
            ),
            training_sentences: SentencesFileCard {
                name: SENTENCES_FILE.to_owned(),
                sentences: trained.samples,
                size: trained.sentences.len() as u64,
                sha256: Sha256::of(&trained.sentences).to_string(),
                sha512: Sha512::of(&trained.sentences).to_string(),
            },
            export: export_card(export, recorded, plan),
            split: recipe.split,
            per_language: plan.languages.clone(),
            excluded_languages: plan.excluded.clone(),
            training: LanguageTrainingCard {
                algorithm: Algorithm::Maxent.name().to_owned(),
                iterations: recipe.iterations,
                cutoff: recipe.cutoff,
                threads: 1,
                data_indexer: ONE_PASS.to_owned(),
                timestamp_ms: TIMESTAMP_MS,
                samples: trained.samples,
                events: trained.events,
                event_hash: trained.event_hash,
                inexact_events: trained.inexact_events,
            },
            held_out: LanguageHeldOutCard {
                samples: plan.held_out.len(),
                reference_mappings: recipe.reference_mappings(plan),
            },
            gate: GateCard {
                record: GATE_FILE.to_owned(),
                reference: gate.reference.id.clone(),
                reference_sha256: gate.reference.sha256.clone(),
                candidate_tag_format: None,
                reference_tag_format: None,
                candidate_score: r.candidate.score,
                reference_score: r.reference.score,
                difference_low: r.difference.low,
                difference_high: r.difference.high,
                pass: r.pass,
            },
        }
    }

    /// The model's zoo entry once the model file is published at
    /// `model_url` and [`SENTENCES_FILE`] at `sentences_url`.
    ///
    /// # Errors
    ///
    /// [`Error::NotShippable`] when the model may not leave the machine it
    /// was trained on, it is not [`LANGDETECT`]'s model, it was not gated
    /// against the recipe's reference, the gate failed, or the export is
    /// not the recorded one; [`Error::InvalidCard`] for a `class` other
    /// than its licences give, a component or language other than its
    /// recipe's, or a field the index refuses.
    pub fn artifact(&self, model_url: &str, sentences_url: &str) -> Result<Artifact> {
        let class = class_of(std::iter::once(self.license).chain(self.trained_on.iter().map(|t| t.license)));
        if class != self.class {
            return Err(Error::InvalidCard {
                field: "class",
                message: format!("{} is not {class}, the class its licences give", self.class),
            });
        }
        publishable_class(&self.id, self.license, &self.trained_on)?;
        let not = |reason| Error::NotShippable { id: self.id.clone(), reason };
        if self.id != LANGDETECT.model_id {
            return Err(not("no recipe trains it"));
        }
        let reference = LANGDETECT.reference;
        if self.gate.reference != reference.id() || self.gate.reference_sha256 != reference.sha256()? {
            return Err(not("it was not gated against its recipe's reference"));
        }
        if !self.gate.pass {
            return Err(not("it failed its gate"));
        }
        if !self.export.recorded {
            return Err(not("it was trained on a Tatoeba export that is not the recorded one"));
        }
        let bad = |field: &'static str, message: String| Error::InvalidCard { field, message };
        if self.component != turbo_zoo::Component::LanguageDetector {
            return Err(bad(
                "component",
                format!("{} is not {}, its recipe's", self.component, turbo_zoo::Component::LanguageDetector),
            ));
        }
        if self.language != ZOO_LANGUAGE {
            return Err(bad("language", format!("{} is not {ZOO_LANGUAGE}, its recipe's", self.language)));
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
            files: vec![
                File {
                    path: self.file.name.parse().map_err(|e| bad("file name", e))?,
                    url: model_url.parse().map_err(|e| bad("url", e))?,
                    size: self.file.size,
                    sha256: self.file.sha256.parse().map_err(|e| bad("sha256", e))?,
                    sha512: self.file.sha512.parse().map_err(|e| bad("sha512", e))?,
                    download: None,
                },
                File {
                    path: self.training_sentences.name.parse().map_err(|e| bad("sentences file name", e))?,
                    url: sentences_url.parse().map_err(|e| bad("sentences url", e))?,
                    size: self.training_sentences.size,
                    sha256: self.training_sentences.sha256.parse().map_err(|e| bad("sentences sha256", e))?,
                    sha512: self.training_sentences.sha512.parse().map_err(|e| bad("sentences sha512", e))?,
                    download: None,
                },
            ],
        })
    }

    /// The card as written: pretty JSON with a final newline.
    pub fn to_json(&self) -> String {
        let mut s = serde_json::to_string_pretty(self).expect("a card serializes");
        s.push('\n');
        s
    }

    /// Reads a card written by [`write_language_dir`].
    ///
    /// # Errors
    ///
    /// A file that cannot be read or is not a card of this version.
    pub fn read(path: &Path) -> Result<LanguageCard> {
        let bytes = std::fs::read(path).map_err(io(path))?;
        let card: LanguageCard = serde_json::from_slice(&bytes)
            .map_err(|e| Error::Json { path: path.to_owned(), message: e.to_string() })?;
        if card.card_version != CARD_VERSION {
            return Err(Error::Json {
                path: path.to_owned(),
                message: format!("card version {}, this build reads {CARD_VERSION}", card.card_version),
            });
        }
        Ok(card)
    }

    /// A line of what was trained on and held out, and a line of the
    /// languages left out.
    pub fn summary(&self) -> String {
        let excluded: Vec<String> =
            self.excluded_languages.iter().map(|e| format!("{} ({})", e.tatoeba, e.usable)).collect();
        format!(
            "{}: {} languages, trained on {} sentences ({} events), held out {}; {} languages left out with fewer \
             than {} sentences: {}",
            self.id,
            self.languages.len(),
            self.training.samples,
            self.training.events,
            self.held_out.samples,
            self.excluded_languages.len(),
            self.split.min_sentences,
            if excluded.is_empty() { "none".to_owned() } else { excluded.join(", ") }
        )
    }
}

/// Writes the model's directory `dir/<model id>/`: the model file,
/// [`SENTENCES_FILE`], [`CARD_FILE`] and [`GATE_FILE`]. The directory
/// must not exist.
///
/// # Errors
///
/// An existing directory, or a write that fails.
pub fn write_language_dir(dir: &Path, outcome: &LanguageOutcome) -> Result<()> {
    let card = &outcome.card;
    let out = dir.join(&card.id);
    std::fs::create_dir_all(dir).map_err(io(dir))?;
    std::fs::create_dir(&out).map_err(io(&out))?;
    let model_path = out.join(&card.file.name);
    std::fs::write(&model_path, &outcome.trained.bytes).map_err(io(&model_path))?;
    let sentences_path = out.join(&card.training_sentences.name);
    std::fs::write(&sentences_path, &outcome.trained.sentences).map_err(io(&sentences_path))?;
    let card_path = out.join(CARD_FILE);
    std::fs::write(&card_path, card.to_json()).map_err(io(&card_path))?;
    Ok(outcome.gate.write(&out.join(GATE_FILE))?)
}
