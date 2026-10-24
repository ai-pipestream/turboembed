use std::path::Path;

use serde::{Deserialize, Serialize};
use turbo_eval::model_gate::ModelGateRecord;
use turbo_train::Algorithm;
use turbo_train_components::langdetect::ONE_PASS;
use turbo_zoo::{Artifact, File, Kind, Licence, LicenceClass, Sha256, Sha512, TrainingData, class_of, index::Format};

use super::{
    CARD_VERSION, COMPARE_TATOEBA_FILE, COMPARE_WEB_FILE, DOCUMENTS_FILE, LANGDETECT_WEB, Scored, TrainedWeb,
    WEB_GATE_FILE, WebInput, WebLanguageCount, WebLanguageRecipe,
};
use crate::card::{
    GATE_FILE, GateCard, MODEL_LICENCE, MODEL_LICENCE_TERMS, MODEL_VERSION, ModelFileCard, publishable_class,
};
use crate::error::{Error, Result, io};
use crate::fineweb::{self, COMMONCRAWL_TERMS, Dataset, FINEWEB, FINEWEB_2, FileChoice, WebFile, WebPlan, WebSplit};
use crate::langdetect::{
    self, ExcludedLanguage, ExportCard, LanguageCount, LanguageTrainingCard, Plan, SENTENCE_URL, SENTENCES_FILE,
    SentencesFileCard, Split, ZOO_LANGUAGE,
};
use crate::segmentation::TIMESTAMP_MS;
use crate::tatoeba::Export;

/// A dataset the web windows come from, as the card lists it.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DatasetCard {
    pub name: String,
    pub repo: String,
    pub revision: String,
    pub license: Licence,
    pub license_terms: String,
}

impl DatasetCard {
    fn of(d: Dataset) -> DatasetCard {
        DatasetCard {
            name: d.name.to_owned(),
            repo: d.repo.to_owned(),
            revision: d.revision.to_owned(),
            license: d.licence,
            license_terms: d.licence_terms.to_owned(),
        }
    }
}

/// The web data the model was trained and gated on.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct WebDataCard {
    pub datasets: Vec<DatasetCard>,
    /// [`fineweb::lists_digest`] of the files and ranges lists.
    pub lists_sha256: String,
    /// Whether the lists are the recorded ones. A model trained on others
    /// is not a release.
    pub recorded: bool,
    pub split: WebSplit,
    /// Each file read and how its documents were chosen.
    pub files: Vec<FileChoice>,
    pub files_listed: Vec<WebFile>,
}

/// [`DOCUMENTS_FILE`] as the card lists it.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DocumentsFileCard {
    pub name: String,
    /// The documents it lists, one per training window.
    pub documents: usize,
    pub size: u64,
    pub sha256: String,
    pub sha512: String,
}

/// The training as the card records it: the Tatoeba recipe's
/// parameters, and the samples of each kind.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct WebTrainingCard {
    #[serde(flatten)]
    pub parameters: LanguageTrainingCard,
    pub sentences: usize,
    pub windows: usize,
}

/// A comparison with [`COMPARED`](super::COMPARED) as the card records it:
/// the record's units, both scores and the interval of their difference. It
/// is not a gate, so the record's pass is left out.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ComparisonCard {
    pub record: String,
    pub units: usize,
    pub candidate_score: f64,
    pub compared_score: f64,
    pub difference_low: f64,
    pub difference_high: f64,
}

impl ComparisonCard {
    /// The card of the comparison `record`, written as `file`.
    pub(crate) fn of(record: &ModelGateRecord, file: &str) -> ComparisonCard {
        let r = &record.result;
        ComparisonCard {
            record: file.to_owned(),
            units: record.units.len(),
            candidate_score: r.candidate.score,
            compared_score: r.reference.score,
            difference_low: r.difference.low,
            difference_high: r.difference.high,
        }
    }
}

/// One language's held-out sentences the compared model got right.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ComparedSentences {
    pub tatoeba: String,
    pub compared_correct: usize,
}

/// The web language detector's model card. Strict JSON, as the gate
/// record: every field is required and an unknown field is an error.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct WebLanguageCard {
    pub card_version: u32,
    pub id: String,
    pub component: turbo_zoo::Component,
    pub language: String,
    /// The model's outcomes, Tatoeba's codes: the languages it was trained
    /// on sentences or windows of, in the recipe's order.
    pub languages: Vec<String>,
    pub file: ModelFileCard,
    pub license: Licence,
    pub license_terms: String,
    pub trained_on: Vec<TrainingData>,
    pub class: LicenceClass,
    /// The credit CC BY 2.0 FR, Tatoeba's terms and ODC-By 1.0 ask for,
    /// shipped with the model: it points at [`SENTENCES_FILE`] and
    /// [`DOCUMENTS_FILE`].
    pub attribution: String,
    pub training_sentences: SentencesFileCard,
    pub training_documents: DocumentsFileCard,
    pub export: ExportCard,
    pub split: Split,
    pub per_language: Vec<LanguageCount>,
    pub excluded_languages: Vec<ExcludedLanguage>,
    pub web: WebDataCard,
    pub web_per_language: Vec<WebLanguageCount>,
    pub compared: String,
    pub compared_sha256: String,
    pub compared_per_language: Vec<ComparedSentences>,
    pub training: WebTrainingCard,
    /// The held-out sentences and windows, a unit each.
    pub held_out_sentences: usize,
    pub held_out_windows: usize,
    /// The gate on the held-out sentences.
    pub gate: GateCard,
    /// The gate on the held-out windows.
    pub web_gate: GateCard,
    /// The comparison on the held-out sentences.
    pub compared_on_sentences: ComparisonCard,
    /// The comparison on the held-out windows of the languages
    /// [`COMPARED`](super::COMPARED) has.
    pub compared_on_windows: ComparisonCard,
}

/// The card of the record `record`, written as `file`.
fn gate_card(record: &ModelGateRecord, file: &str) -> GateCard {
    GateCard { record: file.to_owned(), ..GateCard::of(record, None, None) }
}

/// The datasets the model is trained on, as the zoo lists them: the
/// Tatoeba export of digest `export_digest`, FineWeb-2 and FineWeb at
/// their revisions.
pub fn training_data(export_digest: &str) -> Vec<TrainingData> {
    let web = |d: Dataset| TrainingData {
        name: format!("{} {}", d.name, &d.revision[..12]),
        license: d.licence,
        license_terms: d.licence_terms.parse().expect("a valid address"),
    };
    vec![langdetect::training_data(export_digest), web(FINEWEB_2), web(FINEWEB)]
}

impl WebLanguageCard {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        recipe: &WebLanguageRecipe,
        export: &Export,
        recorded_export: bool,
        web: &WebInput<'_>,
        lists_digest: &str,
        plan: &Plan<'_>,
        web_plan: &WebPlan,
        trained: &TrainedWeb,
        scored: &Scored,
    ) -> WebLanguageCard {
        let digest = export.digest();
        let trained_on = training_data(&digest);
        let class = class_of(std::iter::once(MODEL_LICENCE).chain(trained_on.iter().map(|t| t.license)));
        let t = WebTrainingCard {
            parameters: LanguageTrainingCard {
                algorithm: Algorithm::Maxent.name().to_owned(),
                iterations: recipe.tatoeba.iterations,
                cutoff: recipe.tatoeba.cutoff,
                threads: 1,
                data_indexer: ONE_PASS.to_owned(),
                timestamp_ms: TIMESTAMP_MS,
                samples: trained.sentences + trained.windows,
                events: trained.events,
                event_hash: trained.event_hash,
                inexact_events: trained.inexact_events,
            },
            sentences: trained.sentences,
            windows: trained.windows,
        };
        WebLanguageCard {
            card_version: CARD_VERSION,
            id: recipe.model_id.to_owned(),
            component: turbo_zoo::Component::LanguageDetector,
            language: ZOO_LANGUAGE.to_owned(),
            languages: recipe
                .tatoeba
                .languages
                .iter()
                .zip(&scored.web_counts)
                .filter(|(p, w)| w.training > 0 || plan.languages.iter().any(|l| l.tatoeba == p.tatoeba))
                .map(|(p, _)| p.tatoeba.to_owned())
                .collect(),
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
                 (https://creativecommons.org/licenses/by/2.0/fr/), from its export of digest {digest}; \
                 {SENTENCES_FILE}, shipped with the model, lists their ids: the sentence of id N, with its author, \
                 is at {SENTENCE_URL}N. And on {} documents from FineWeb-2 (https://huggingface.co/datasets/{}, \
                 revision {}) and FineWeb (https://huggingface.co/datasets/{}, revision {}), made available under \
                 the Open Data Commons Attribution License (ODC-By) 1.0 (https://opendatacommons.org/licenses/by/1-0/), \
                 their use also subject to CommonCrawl's terms of use ({COMMONCRAWL_TERMS}); {DOCUMENTS_FILE}, \
                 shipped with the model, lists each document's dataset, file, row and id.",
                trained.sentences, trained.windows, FINEWEB_2.repo, FINEWEB_2.revision, FINEWEB.repo, FINEWEB.revision
            ),
            training_sentences: SentencesFileCard {
                name: SENTENCES_FILE.to_owned(),
                sentences: trained.sentences,
                size: trained.sentences_file.len() as u64,
                sha256: Sha256::of(&trained.sentences_file).to_string(),
                sha512: Sha512::of(&trained.sentences_file).to_string(),
            },
            training_documents: DocumentsFileCard {
                name: DOCUMENTS_FILE.to_owned(),
                documents: trained.windows,
                size: trained.documents_file.len() as u64,
                sha256: Sha256::of(&trained.documents_file).to_string(),
                sha512: Sha512::of(&trained.documents_file).to_string(),
            },
            export: langdetect::export_card(export, recorded_export, plan),
            split: recipe.tatoeba.split,
            per_language: plan.languages.clone(),
            excluded_languages: plan.excluded.clone(),
            web: WebDataCard {
                datasets: vec![DatasetCard::of(FINEWEB_2), DatasetCard::of(FINEWEB)],
                lists_sha256: lists_digest.to_owned(),
                recorded: web.recorded,
                split: recipe.web.split,
                files: web_plan.files.clone(),
                files_listed: web.files.to_vec(),
            },
            web_per_language: scored.web_counts.clone(),
            compared: scored.compare_tatoeba.reference.id.clone(),
            compared_sha256: scored.compare_tatoeba.reference.sha256.clone(),
            compared_per_language: plan
                .languages
                .iter()
                .zip(&scored.compared_sentences)
                .map(|(l, &n)| ComparedSentences { tatoeba: l.tatoeba.clone(), compared_correct: n })
                .collect(),
            training: t,
            held_out_sentences: plan.held_out.len(),
            held_out_windows: web_plan.held_out.len(),
            gate: gate_card(&scored.tatoeba_gate, GATE_FILE),
            web_gate: gate_card(&scored.web_gate, WEB_GATE_FILE),
            compared_on_sentences: ComparisonCard::of(&scored.compare_tatoeba, COMPARE_TATOEBA_FILE),
            compared_on_windows: ComparisonCard::of(&scored.compare_web, COMPARE_WEB_FILE),
        }
    }

    /// The model's zoo entry once the model file is published at
    /// `model_url`, [`SENTENCES_FILE`] at `sentences_url` and
    /// [`DOCUMENTS_FILE`] at `documents_url`.
    ///
    /// # Errors
    ///
    /// [`Error::NotShippable`] when the model may not leave the machine it
    /// was trained on, it is not [`LANGDETECT_WEB`]'s model, either gate
    /// was not run against the recipe's reference or failed, or the export
    /// or the web lists are not the recorded ones; [`Error::InvalidCard`]
    /// for a `class` other than its licences give, a component or language
    /// other than its recipe's, or a field the index refuses.
    pub fn artifact(&self, model_url: &str, sentences_url: &str, documents_url: &str) -> Result<Artifact> {
        let class = class_of(std::iter::once(self.license).chain(self.trained_on.iter().map(|t| t.license)));
        if class != self.class {
            return Err(Error::InvalidCard {
                field: "class",
                message: format!("{} is not {class}, the class its licences give", self.class),
            });
        }
        publishable_class(&self.id, self.license, &self.trained_on)?;
        let not = |reason| Error::NotShippable { id: self.id.clone(), reason };
        if self.id != LANGDETECT_WEB.model_id {
            return Err(not("no recipe trains it"));
        }
        let reference = LANGDETECT_WEB.tatoeba.reference;
        for gate in [&self.gate, &self.web_gate] {
            if gate.reference != reference.id() || gate.reference_sha256 != reference.sha256()? {
                return Err(not("it was not gated against its recipe's reference"));
            }
            if !gate.pass {
                return Err(not("it failed a gate"));
            }
        }
        if !self.export.recorded {
            return Err(not("it was trained on a Tatoeba export that is not the recorded one"));
        }
        if !self.web.recorded {
            return Err(not("it was trained on web ranges that are not the recorded ones"));
        }
        let files = fineweb::parse_files(fineweb::FILES_NAME, fineweb::FILES)?;
        let ranges = fineweb::parse_ranges(fineweb::RANGES_NAME, fineweb::RANGES)?;
        if self.web.lists_sha256 != fineweb::lists_digest(&files, &ranges) {
            return Err(not("its card names other web lists than the recorded ones"));
        }
        let compared = LANGDETECT_WEB.compared;
        if self.compared != compared.id() || self.compared_sha256 != compared.sha256()? {
            return Err(not("it was not compared with its recipe's model"));
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
        let file = |name: &str, url: &str, size: u64, sha256: &str, sha512: &str| -> Result<File> {
            Ok(File {
                path: name.parse().map_err(|e| bad("file name", e))?,
                url: url.parse().map_err(|e| bad("url", e))?,
                size,
                sha256: sha256.parse().map_err(|e| bad("sha256", e))?,
                sha512: sha512.parse().map_err(|e| bad("sha512", e))?,
            })
        };
        let (s, d) = (&self.training_sentences, &self.training_documents);
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
                file(&self.file.name, model_url, self.file.size, &self.file.sha256, &self.file.sha512)?,
                file(&s.name, sentences_url, s.size, &s.sha256, &s.sha512)?,
                file(&d.name, documents_url, d.size, &d.sha256, &d.sha512)?,
            ],
        })
    }

    /// The card as written: pretty JSON with a final newline.
    pub fn to_json(&self) -> String {
        let mut s = serde_json::to_string_pretty(self).expect("a card serializes");
        s.push('\n');
        s
    }

    /// Reads a card written by [`write_web_language_dir`].
    ///
    /// # Errors
    ///
    /// A file that cannot be read or is not a card of this version.
    pub fn read(path: &Path) -> Result<WebLanguageCard> {
        let bytes = std::fs::read(path).map_err(io(path))?;
        let card: WebLanguageCard = serde_json::from_slice(&bytes)
            .map_err(|e| Error::Json { path: path.to_owned(), message: e.to_string() })?;
        if card.card_version != CARD_VERSION {
            return Err(Error::Json {
                path: path.to_owned(),
                message: format!("card version {}, this build reads {CARD_VERSION}", card.card_version),
            });
        }
        Ok(card)
    }

    /// A line of what was trained on and held out, a line per gate and
    /// comparison, and the files that gave fewer windows than wanted.
    pub fn summary(&self) -> String {
        let gate = |g: &GateCard| {
            format!(
                "{:.4} against {} {:.4} (difference {:+.4} to {:+.4}){}",
                g.candidate_score,
                g.reference,
                g.reference_score,
                g.difference_low,
                g.difference_high,
                if g.pass { ": pass" } else { ": below the margin" }
            )
        };
        let compared = |c: &ComparisonCard| {
            format!(
                "{:.4} beside {:.4} over {} units (difference {:+.4} to {:+.4})",
                c.candidate_score, c.compared_score, c.units, c.difference_low, c.difference_high
            )
        };
        let short: Vec<String> = self
            .web
            .files
            .iter()
            .filter(|f| f.chosen < f.wanted)
            .map(|f| {
                format!(
                    "{} {} {} of {}",
                    f.path,
                    if f.side == fineweb::Side::Training { "training" } else { "held out" },
                    f.chosen,
                    f.wanted
                )
            })
            .collect();
        format!(
            "{}: {} languages, trained on {} sentences and {} web windows ({} events), held out {} sentences and \
             {} windows; {} languages left out of the sentences\n  sentences: {}\n  windows: {}\n  sentences, \
             beside the Tatoeba model: {}\n  windows of its languages, beside the Tatoeba model: {}\n  files short \
             of windows: {}",
            self.id,
            self.languages.len(),
            self.training.sentences,
            self.training.windows,
            self.training.parameters.events,
            self.held_out_sentences,
            self.held_out_windows,
            self.excluded_languages.len(),
            gate(&self.gate),
            gate(&self.web_gate),
            compared(&self.compared_on_sentences),
            compared(&self.compared_on_windows),
            if short.is_empty() { "none".to_owned() } else { short.join(", ") },
        )
    }
}
