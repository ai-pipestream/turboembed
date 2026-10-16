//! The language detector trained on Tatoeba's sentences and on web text
//! from FineWeb-2 and FineWeb (issue #429), gated against OpenNLP's
//! `langdetect-183.bin` on held-out sentences and on held-out web text.
//!
//! - Data: the training sentences of [`crate::langdetect::LANGDETECT`],
//!   read, checked and split exactly as that recipe does, and the
//!   training windows of [`crate::fineweb`] ([`fineweb::WEB`]): at most
//!   1,000 web windows of at most 300 characters per language beside at
//!   most 5,000 sentences. No training window holds a held-out sentence of
//!   its language, and no held-out window a training sentence
//!   ([`fineweb::Avoid`]). One event each, the sentences first, then the windows,
//!   each language by language in the order of
//!   [`crate::langdetect::LANGUAGES`].
//! - Training: the Tatoeba recipe's parameters (`MAXENT`, 100
//!   iterations, cutoff 5, one thread, the `OnePass` indexer) and its
//!   fixed manifest `Timestamp`, so a run gives the same bytes.
//! - Gates: the Tatoeba recipe's held-out sentences, the set its model
//!   was gated on ([`GATE_FILE`]), and the held-out web windows
//!   ([`WEB_GATE_FILE`]). A unit is right when the model's best language
//!   is its language, as written for the trained model and through
//!   [`crate::langdetect::LANGUAGES`] for the reference. The model is a
//!   release only when it passes both.
//! - Comparison: the held-out sentences, and the held-out windows of the
//!   languages it has, scored again with the Tatoeba recipe's model as the
//!   zoo lists it ([`COMPARED`]) in the reference's place
//!   ([`COMPARE_TATOEBA_FILE`], [`COMPARE_WEB_FILE`]), so the card says
//!   what the web text changed. It is reported, not a gate.
//!
//! The web windows' languages are the labels FineWeb-2's and FineWeb's
//! classifiers gave their documents, so the web gate measures agreement
//! with those classifiers, on text much like the text trained on. The
//! reference was trained on other text (the Leipzig corpora) and is
//! scored against the same labels.

use std::collections::HashMap;
use std::path::Path;

use serde::{Deserialize, Serialize};
use turbo_eval::model_gate::{
    Counts, HeldOutSet, LabelMapping, LabelScoring, ModelGateRecord, ModelId, PairedUnit, Scoring, Task,
};
use turbo_langdetect::{LanguageDetector, LanguageModel, Scratch};
use turbo_maxent::ModelFile;
use turbo_train::{Algorithm, TrainingConfiguration};
use turbo_train_components::Stamp;
use turbo_train_components::langdetect::{
    LanguageDetectorFactory, LanguageEventStream, ONE_PASS, train_language_model_from_events,
};
use turbo_zoo::{Artifact, File, Kind, Licence, LicenceClass, Sha256, Sha512, TrainingData, class_of, index::Format};

use crate::Reference;
use crate::card::{
    GATE_FILE, GateCard, MODEL_LICENCE, MODEL_LICENCE_TERMS, MODEL_VERSION, ModelFileCard, publishable_class,
};
use crate::error::{Error, Result, io};
use crate::fineweb::{
    self, Avoid, COMMONCRAWL_TERMS, Dataset, FINEWEB, FINEWEB_2, FileChoice, Overlap, Store, WEB, WebFile, WebPlan,
    WebRange, WebRecipe, WebSample, WebSplit,
};
use crate::langdetect::{
    self, CARD_FILE, ExcludedLanguage, ExportCard, LANGDETECT, LanguageCount, LanguageRecipe, LanguageTrainingCard,
    Plan, SENTENCE_URL, SENTENCES_FILE, SentencesFileCard, Split, ZOO_LANGUAGE,
};
use crate::segmentation::TIMESTAMP_MS;
use crate::tatoeba::Export;

pub const MODEL_ID: &str = "turbo-mul-tatoeba-fineweb-langdetect-1";

/// The version of [`WebLanguageCard`] this build writes and reads.
pub const CARD_VERSION: u32 = 1;

/// The list of web documents trained on, shipped with the model.
pub const DOCUMENTS_FILE: &str = "training-documents.txt";

/// The gate record on the held-out web windows.
pub const WEB_GATE_FILE: &str = "gate-web.json";

/// The comparison with [`COMPARED`] on the held-out sentences.
pub const COMPARE_TATOEBA_FILE: &str = "compare-tatoeba.json";

/// The comparison with [`COMPARED`] on the held-out web windows of the
/// languages it has.
pub const COMPARE_WEB_FILE: &str = "compare-web.json";

/// The Tatoeba recipe's model as the zoo lists it, which both held-out
/// sets are scored with beside the trained model.
pub const COMPARED: Reference = Reference::Zoo(langdetect::MODEL_ID);

/// The recipe: the Tatoeba recipe's languages, split, parameters and
/// reference under this model's id, the web side for the same languages
/// in the same order, and the model scored beside it.
#[derive(Copy, Clone, Debug)]
pub struct WebLanguageRecipe {
    pub model_id: &'static str,
    pub tatoeba: LanguageRecipe,
    pub web: WebRecipe,
    pub compared: Reference,
}

pub const LANGDETECT_WEB: WebLanguageRecipe = WebLanguageRecipe {
    model_id: MODEL_ID,
    tatoeba: LanguageRecipe { model_id: MODEL_ID, ..LANGDETECT },
    web: WEB,
    compared: COMPARED,
};

/// One language's web windows, and once gated, how many held-out ones
/// each model got right.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct WebLanguageCount {
    pub tatoeba: String,
    pub training: usize,
    pub held_out: usize,
    /// Held-out windows the trained model got right.
    pub candidate_correct: usize,
    /// Held-out windows the reference got right.
    pub reference_correct: usize,
    /// Held-out windows [`COMPARED`] got right: none of a language it
    /// does not have.
    pub compared_correct: usize,
}

/// A model the recipe trained.
#[derive(Debug, Clone)]
pub struct TrainedWeb {
    /// The `.bin` file.
    pub bytes: Vec<u8>,
    /// The training sentences.
    pub sentences: usize,
    /// The training windows.
    pub windows: usize,
    pub events: usize,
    /// `Training-Eventhash`.
    pub event_hash: u32,
    /// Events whose predicates are not in the JVM's order
    /// (`turbo_train_components::langdetect`).
    pub inexact_events: usize,
    /// [`SENTENCES_FILE`] of the sentences trained on.
    pub sentences_file: Vec<u8>,
    /// [`DOCUMENTS_FILE`] of the windows trained on.
    pub documents_file: Vec<u8>,
}

/// What the gates and the comparisons scored and found.
#[derive(Debug)]
pub struct Scored {
    pub tatoeba_gate: ModelGateRecord,
    pub web_gate: ModelGateRecord,
    pub compare_tatoeba: ModelGateRecord,
    /// On the held-out windows of the languages [`COMPARED`] has.
    pub compare_web: ModelGateRecord,
    /// For each language of the plan, the held-out sentences
    /// [`COMPARED`] got right.
    pub compared_sentences: Vec<usize>,
    pub web_counts: Vec<WebLanguageCount>,
}

fn open(what: &str, bytes: &[u8]) -> Result<LanguageDetector> {
    ModelFile::open(bytes.to_vec())
        .map_err(|e| e.to_string())
        .and_then(|f| LanguageModel::open(f).map_err(|e| e.to_string()))
        .map(LanguageDetector::new)
        .map_err(|e| Error::Model(format!("{what}: {e}")))
}

impl WebLanguageRecipe {
    /// Trains the model on the plan's training sentences and the web
    /// plan's training windows, one event each.
    ///
    /// # Errors
    ///
    /// What the trainer fails on.
    pub fn train(&self, plan: &Plan<'_>, web: &WebPlan, files: &[WebFile]) -> Result<TrainedWeb> {
        let factory = LanguageDetectorFactory::new();
        let mut stream = LanguageEventStream::new(&factory);
        let languages = self.tatoeba.languages;
        let mut events = Vec::with_capacity(plan.training.len() + web.training.len());
        for s in &plan.training {
            events.push(stream.event(languages[s.language].tatoeba, &s.text));
        }
        for s in &web.training {
            events.push(stream.event(self.web.languages[s.language].tatoeba, &s.window));
        }
        let trained = train_language_model_from_events(
            &events,
            &factory,
            &self.tatoeba.parameters(),
            &mut TrainingConfiguration::default(),
            Stamp::at(TIMESTAMP_MS),
        )?;
        Ok(TrainedWeb {
            bytes: trained.package.to_bytes(),
            sentences: plan.training.len(),
            windows: web.training.len(),
            events: events.len(),
            event_hash: trained.report.event_hash.expect("trained from events"),
            inexact_events: stream.inexact_events(),
            sentences_file: langdetect::sentences_file(plan),
            documents_file: fineweb::documents_file(web, files),
        })
    }

    /// The reference's codes counted right for another language, over all
    /// of the recipe's languages: the web windows of every language are
    /// held out, whichever Tatoeba languages the plan keeps.
    pub fn web_mappings(&self) -> Vec<LabelMapping> {
        let mut out = Vec::new();
        for p in self.tatoeba.languages {
            for r in p.reference.iter().filter(|r| **r != p.tatoeba) {
                out.push(LabelMapping { held_out: p.tatoeba.to_owned(), reference: (*r).to_owned() });
            }
        }
        out
    }

    /// The SHA-256 of held-out windows as scored: for each, its
    /// language, its length in bytes and its text, then each of
    /// `mappings`.
    pub fn web_samples_digest<'s>(
        &self,
        samples: impl IntoIterator<Item = &'s WebSample>,
        mappings: &[LabelMapping],
    ) -> String {
        let mut text = String::new();
        for s in samples {
            text.push_str(&format!("{}\t{}\t{}\n", self.web.languages[s.language].tatoeba, s.window.len(), s.window));
        }
        for m in mappings {
            text.push_str(&format!("mapping\t{}\t{}\n", m.held_out, m.reference));
        }
        Sha256::of(text.as_bytes()).to_string()
    }

    /// The name of the held-out web set in the gate record.
    pub fn web_held_out_name(&self, lists_digest: &str) -> String {
        format!(
            "FineWeb-2 and FineWeb held out (at most {} windows of at most {} characters per language, lists {})",
            self.web.split.held_out,
            self.web.split.window,
            &lists_digest[..16.min(lists_digest.len())]
        )
    }

    /// Scores the trained model on both held-out sets against the
    /// reference and against [`COMPARED`], and fills in the plan's
    /// correct counts. `export_digest` and `lists_digest` name the sets.
    ///
    /// # Errors
    ///
    /// A model file that is not a language detector, or what the gate
    /// fails on.
    #[allow(clippy::too_many_arguments)]
    pub fn score(
        &self,
        trained: &[u8],
        reference: &[u8],
        compared: &[u8],
        plan: &mut Plan<'_>,
        web: &WebPlan,
        export_digest: &str,
        lists_digest: &str,
    ) -> Result<Scored> {
        let tatoeba_gate =
            self.tatoeba.gate(trained, reference, plan, &self.tatoeba.held_out_name(export_digest), export_digest)?;
        let candidate = open(self.model_id, trained)?;
        let reference_model = open(self.tatoeba.reference.id(), reference)?;
        let known: std::collections::HashSet<&str> = reference_model.model().supported_languages().collect();
        for s in &web.held_out {
            let p = self.tatoeba.languages[s.language];
            if let Some(code) = p.reference.iter().find(|c| !known.contains(*c)) {
                return Err(Error::Model(format!(
                    "{}: no outcome {code}, which the recipe counts right for {}",
                    self.tatoeba.reference.id(),
                    p.tatoeba
                )));
            }
        }
        let compared_model = open(self.compared.id(), compared)?;
        let compared_known: std::collections::HashSet<&str> = compared_model.model().supported_languages().collect();
        let mut scratch = Scratch::new();
        let mut right = |model: &LanguageDetector, text: &str, codes: &[&str]| {
            codes.contains(&model.predict_language(text, &mut scratch).code)
        };
        let candidate_id = ModelId { id: self.model_id.to_owned(), sha256: Sha256::of(trained).to_string() };
        let reference_id =
            ModelId { id: self.tatoeba.reference.id().to_owned(), sha256: self.tatoeba.reference.sha256()? };
        let compared_id = ModelId { id: self.compared.id().to_owned(), sha256: self.compared.sha256()? };
        let pair = |c: bool, r: bool| PairedUnit {
            candidate: Counts::accuracy(u64::from(c), 1),
            reference: Counts::accuracy(u64::from(r), 1),
        };

        // The held-out sentences against the compared model.
        let index: HashMap<&str, usize> =
            plan.languages.iter().enumerate().map(|(k, l)| (l.tatoeba.as_str(), k)).collect();
        let mut compared_sentences = vec![0; plan.languages.len()];
        let mut units = Vec::with_capacity(plan.held_out.len());
        for s in &plan.held_out {
            let code = self.tatoeba.languages[s.language].tatoeba;
            let c = right(&candidate, &s.text, &[code]);
            let o = right(&compared_model, &s.text, &[code]);
            compared_sentences[index[code]] += usize::from(o);
            units.push(pair(c, o));
        }
        // The compared model has Tatoeba's codes, so its set has no
        // mappings, and its samples' digest none either.
        let set = HeldOutSet {
            name: self.tatoeba.held_out_name(export_digest),
            corpus_sha256: export_digest.to_owned(),
            samples_sha256: self.tatoeba.samples_digest_with(plan, &[]),
            skipped_sentences: 0,
            scoring: Scoring::Labels(LabelScoring {
                samples: plan.held_out.len() as u64,
                reference_mappings: Vec::new(),
            }),
        };
        let compare_tatoeba =
            ModelGateRecord::new(Task::LanguageDetector, set, candidate_id.clone(), compared_id.clone(), units)?;

        // The held-out web windows against the reference and the
        // compared model.
        let mut web_counts: Vec<WebLanguageCount> = self
            .web
            .languages
            .iter()
            .map(|l| WebLanguageCount {
                tatoeba: l.tatoeba.to_owned(),
                training: 0,
                held_out: 0,
                candidate_correct: 0,
                reference_correct: 0,
                compared_correct: 0,
            })
            .collect();
        for s in &web.training {
            web_counts[s.language].training += 1;
        }
        let mut gate_units = Vec::with_capacity(web.held_out.len());
        let mut compare_units = Vec::with_capacity(web.held_out.len());
        let mut compared_windows = Vec::with_capacity(web.held_out.len());
        for s in &web.held_out {
            let p = self.tatoeba.languages[s.language];
            let c = right(&candidate, &s.window, &[p.tatoeba]);
            let r = right(&reference_model, &s.window, p.reference);
            let o = right(&compared_model, &s.window, &[p.tatoeba]);
            let count = &mut web_counts[s.language];
            count.held_out += 1;
            count.candidate_correct += usize::from(c);
            count.reference_correct += usize::from(r);
            count.compared_correct += usize::from(o);
            gate_units.push(pair(c, r));
            // The compared model cannot be right on a language it does
            // not have, so those windows say nothing of what the web text
            // changed.
            if compared_known.contains(p.tatoeba) {
                compare_units.push(pair(c, o));
                compared_windows.push(s);
            }
        }
        let mappings = self.web_mappings();
        let web_set = |name: String, samples: &[&WebSample], mappings: Vec<LabelMapping>| HeldOutSet {
            name,
            corpus_sha256: lists_digest.to_owned(),
            samples_sha256: self.web_samples_digest(samples.iter().copied(), &mappings),
            skipped_sentences: 0,
            scoring: Scoring::Labels(LabelScoring { samples: samples.len() as u64, reference_mappings: mappings }),
        };
        let all: Vec<&WebSample> = web.held_out.iter().collect();
        let web_gate = ModelGateRecord::new(
            Task::LanguageDetector,
            web_set(self.web_held_out_name(lists_digest), &all, mappings),
            candidate_id.clone(),
            reference_id,
            gate_units,
        )?;
        let compare_web = ModelGateRecord::new(
            Task::LanguageDetector,
            web_set(
                format!("{}, the languages of {}", self.web_held_out_name(lists_digest), self.compared.id()),
                &compared_windows,
                Vec::new(),
            ),
            candidate_id,
            compared_id,
            compare_units,
        )?;
        Ok(Scored { tatoeba_gate, web_gate, compare_tatoeba, compare_web, compared_sentences, web_counts })
    }
}

/// What the recipe made.
#[derive(Debug)]
pub struct WebOutcome {
    pub trained: TrainedWeb,
    pub scored: Scored,
    pub card: WebLanguageCard,
}

/// The web data a run reads: the files and ranges lists, whether they
/// are the recorded ones ([`fineweb::FILES`] and [`fineweb::RANGES`]),
/// and the directory the ranges were fetched into, made by
/// [`WebInput::checked`].
#[non_exhaustive]
pub struct WebInput<'a> {
    pub web: WebRecipe,
    pub files: &'a [WebFile],
    pub ranges: &'a [WebRange],
    pub recorded: bool,
    pub store: &'a Store<'a>,
}

impl<'a> WebInput<'a> {
    /// The web data of `files` and `ranges` in `store`, once the ranges
    /// listed are checked to be the ones `web` reads of the files, from
    /// the bytes the directory holds ([`fineweb::record`]).
    ///
    /// # Errors
    ///
    /// A range missing or not the listed bytes, a file not read as the
    /// recipe needs, or ranges listed that are not the ones the files give.
    pub fn checked(
        web: WebRecipe,
        files: &'a [WebFile],
        ranges: &'a [WebRange],
        recorded: bool,
        store: &'a Store<'a>,
    ) -> Result<WebInput<'a>> {
        let made = fineweb::record(files, &web, store)?;
        fineweb::check_ranges(ranges, &made)?;
        Ok(WebInput { web, files, ranges, recorded, store })
    }
}

/// Reads the recipe's sentences from `export` and its windows from
/// `web`, plans, trains, scores the model against `reference` (the bytes
/// of the reference's file) and `compared` (those of [`COMPARED`]'s) and
/// makes the card. `recorded_export` is whether the export is the
/// recorded one.
///
/// # Errors
///
/// A reference or compared model that is not the recorded one, web data
/// checked for another web recipe, or what reading, training or scoring
/// fail on.
pub fn run(
    recipe: &WebLanguageRecipe,
    export: &Export,
    recorded_export: bool,
    web: &WebInput<'_>,
    reference: &[u8],
    compared: &[u8],
) -> Result<WebOutcome> {
    recipe.tatoeba.reference.check(reference)?;
    recipe.compared.check(compared)?;
    let tatoeba: Vec<&str> = recipe.tatoeba.languages.iter().map(|p| p.tatoeba).collect();
    let web_codes: Vec<&str> = recipe.web.languages.iter().map(|l| l.tatoeba).collect();
    if tatoeba != web_codes {
        return Err(Error::Model(format!(
            "{}: the web languages are not the Tatoeba recipe's, in its order",
            recipe.model_id
        )));
    }
    if web.web != recipe.web {
        return Err(Error::Model(format!("{}: the web data was checked for another web recipe", recipe.model_id)));
    }
    let mut plan = recipe.tatoeba.plan(recipe.tatoeba.sentences(export)?)?;
    let web_plan = {
        let (n, min) = (recipe.web.languages.len(), recipe.web.split.overlap_min_chars);
        let avoid = Avoid {
            training: Overlap::new(n, min, plan.held_out.iter().map(|s| (s.language, s.text.as_ref()))),
            held_out: Overlap::new(n, min, plan.training.iter().map(|s| (s.language, s.text.as_ref()))),
        };
        fineweb::plan_web(web.files, &recipe.web, web.store, &avoid)?
    };
    let trained = recipe.train(&plan, &web_plan, web.files)?;
    let digest = export.digest();
    let lists = fineweb::lists_digest(web.files, web.ranges);
    let scored = recipe.score(&trained.bytes, reference, compared, &mut plan, &web_plan, &digest, &lists)?;
    let card = WebLanguageCard::new(recipe, export, recorded_export, web, &lists, &plan, &web_plan, &trained, &scored);
    Ok(WebOutcome { trained, scored, card })
}

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

/// A comparison with [`COMPARED`] as the card records it: the record's
/// units, both scores and the interval of their difference. It is not a
/// gate, so the record's pass is left out.
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
    fn of(record: &ModelGateRecord, file: &str) -> ComparisonCard {
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
    /// [`COMPARED`] has.
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
}

impl WebLanguageCard {
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
    /// for a `class` other than its licences give, or a field the index
    /// refuses.
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
        let bad = |field: &'static str, message: String| Error::InvalidCard { field, message };
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

/// Writes the model's directory `dir/<model id>/`: the model file,
/// [`SENTENCES_FILE`], [`DOCUMENTS_FILE`], [`CARD_FILE`], [`GATE_FILE`],
/// [`WEB_GATE_FILE`], [`COMPARE_TATOEBA_FILE`] and [`COMPARE_WEB_FILE`].
/// The directory must not exist.
///
/// # Errors
///
/// An existing directory, or a write that fails.
pub fn write_web_language_dir(dir: &Path, outcome: &WebOutcome) -> Result<()> {
    let card = &outcome.card;
    let out = dir.join(&card.id);
    std::fs::create_dir_all(dir).map_err(io(dir))?;
    std::fs::create_dir(&out).map_err(io(&out))?;
    let files: [(&str, &[u8]); 3] = [
        (&card.file.name, &outcome.trained.bytes),
        (&card.training_sentences.name, &outcome.trained.sentences_file),
        (&card.training_documents.name, &outcome.trained.documents_file),
    ];
    for (name, bytes) in files {
        let path = out.join(name);
        std::fs::write(&path, bytes).map_err(io(&path))?;
    }
    let card_path = out.join(CARD_FILE);
    std::fs::write(&card_path, card.to_json()).map_err(io(&card_path))?;
    let s = &outcome.scored;
    s.tatoeba_gate.write(&out.join(GATE_FILE))?;
    s.web_gate.write(&out.join(WEB_GATE_FILE))?;
    s.compare_tatoeba.write(&out.join(COMPARE_TATOEBA_FILE))?;
    Ok(s.compare_web.write(&out.join(COMPARE_WEB_FILE))?)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_recipe_is_the_tatoeba_recipe_under_its_own_id() {
        let r = LANGDETECT_WEB;
        assert_eq!(r.tatoeba.model_id, MODEL_ID);
        assert_eq!(r.tatoeba.reference.id(), LANGDETECT.reference.id());
        assert_eq!(r.tatoeba.split, LANGDETECT.split);
        assert_eq!(r.tatoeba.parameters(), LANGDETECT.parameters());
        assert_eq!(r.tatoeba.languages, LANGDETECT.languages);
        assert_eq!(r.web, WEB);
        assert_eq!(r.compared.id(), COMPARED.id());
        let id: turbo_zoo::ArtifactId = MODEL_ID.parse().unwrap();
        assert_eq!(id.as_str(), MODEL_ID);
        assert_eq!(COMPARED.id(), langdetect::MODEL_ID);
        // The mappings over every language are the Tatoeba recipe's pairs
        // whose codes differ.
        let m: Vec<(String, String)> = r.web_mappings().into_iter().map(|m| (m.held_out, m.reference)).collect();
        assert_eq!(m.len(), 7);
        assert!(m.contains(&("est".to_owned(), "ekk".to_owned())));
        assert!(m.contains(&("zsm".to_owned(), "msa".to_owned())));
    }

    /// fineweb-references.txt lists the reference and the compared model,
    /// each with its SHA-256 and its address, and the workflow reads it.
    #[test]
    fn the_references_list_names_the_reference_and_the_compared_model() {
        let crate_dir = Path::new(env!("CARGO_MANIFEST_DIR"));
        let listed = std::fs::read_to_string(crate_dir.join("fineweb-references.txt")).unwrap();
        let listed: Vec<(&str, &str)> = listed
            .lines()
            .filter(|l| !l.is_empty() && !l.starts_with('#'))
            .map(|l| l.split_once(' ').unwrap())
            .collect();
        let zoo = turbo_zoo::Index::builtin();
        let compared = &zoo.get(langdetect::MODEL_ID).unwrap().files[0];
        let reference = langdetect::LANGDETECT_183;
        assert_eq!(
            listed,
            [(reference.sha256, reference.url), (compared.sha256.to_string().as_str(), compared.url.as_str()),]
        );
        // Each is saved under its address's last segment, the name the run
        // reads it by.
        assert_eq!(reference.url.rsplit('/').next(), Some(reference.file));
        assert_eq!(compared.url.as_str().rsplit('/').next(), Some(COMPARED.file().unwrap().as_str()));
        let workflow = std::fs::read_to_string(crate_dir.join("../../.github/workflows/models-fineweb.yml")).unwrap();
        assert!(workflow.contains("analysis/train-recipes/fineweb-references.txt"));
        assert!(workflow.contains(&format!("id={MODEL_ID}")));
        for f in [DOCUMENTS_FILE, SENTENCES_FILE, CARD_FILE, GATE_FILE, WEB_GATE_FILE, COMPARE_TATOEBA_FILE] {
            assert!(workflow.contains(f), "{f}");
        }
        assert!(workflow.contains(COMPARE_WEB_FILE));
    }

    #[test]
    fn the_training_data_is_permissive_with_attribution() {
        let data = training_data(&"0".repeat(64));
        assert_eq!(data.len(), 3);
        assert_eq!(data[1].license, Licence::OdcBy10);
        assert_eq!(data[1].name, "FineWeb-2 af9c13333eb9");
        assert_eq!(data[2].name, "FineWeb 9bb295ddab0e");
        let class = class_of(std::iter::once(MODEL_LICENCE).chain(data.iter().map(|t| t.license)));
        assert_eq!(class, LicenceClass::Permissive);
    }
}
