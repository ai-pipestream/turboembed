//! The language detector recipe (issue #429): `LanguageDetectorME`
//! trained on Tatoeba's CC BY 2.0 FR sentences, gated against OpenNLP's
//! `langdetect-183.bin` on held-out Tatoeba sentences of the languages
//! both cover.
//!
//! - Data: the sentences of a Tatoeba export ([`crate::tatoeba`]) whose
//!   language is one of [`LANGUAGES`]. The licence check keeps a sentence
//!   only when its licence is one of [`ACCEPTED_LICENCES`], CC BY 2.0 FR,
//!   as the issue asks; the CC0 sentences, which the policy would also
//!   allow, are left out and counted per language. A text that occurs
//!   twice in a language is kept once, with its smallest id, so no text is
//!   both trained on and held out.
//! - Languages: [`LANGUAGES`] maps each Tatoeba code to the codes of
//!   `langdetect-183.bin` counted right for it: the 96 codes both use, and
//!   seven pairs of a macrolanguage and one of its individual languages
//!   where the two use different codes of ISO 639-3 (Tatoeba's `est`,
//!   `pes` and `lvs`, which the reference also has, beside its `ekk`,
//!   `fas` and `lav`; Tatoeba's `zsm`, `npi`, `mlg` and `swh` for its
//!   `msa`, `nep`, `plt` and `swa`). A language with fewer than
//!   [`Split::min_sentences`] sentences left after the licence check is
//!   left out of training and of the gate, and listed in the card.
//! - Split: per language, the sentences in id order shuffled as
//!   `Collections.shuffle(list, new Random(seed))` does; the first
//!   [`Split::held_out`] are held out and the next ones, at most
//!   [`Split::max_training`], trained on. Each language has its own
//!   `Random` with the same seed, so a language's split depends on its
//!   own sentences alone.
//! - Training: `LanguageDetectorME.train` with the default factory and the
//!   parameters `langdetect-183.bin` is reproduced with, those of the
//!   command line trainer given none (`MAXENT`, 100 iterations, cutoff 5,
//!   one thread, the `OnePass` indexer `train` puts in), and a fixed
//!   manifest `Timestamp`, so a run gives the same bytes.
//! - Gate: every held-out sentence is a unit of `turbo_eval::model_gate`
//!   ([`Task::LanguageDetector`], [`Scoring::Labels`]), right when the
//!   model's best language (`predictLanguage`) is its language: as written
//!   for the trained model, through [`LANGUAGES`] for the reference. Both
//!   are run by turbo-langdetect, whose probabilities are the JVM's bit for
//!   bit. The reference is only scored; it is never shipped or bundled.

use std::collections::{HashMap, HashSet};
use std::path::Path;

use serde::{Deserialize, Serialize};
use turbo_eval::model_gate::{
    Counts, HeldOutSet, LabelMapping, LabelScoring, ModelGateRecord, ModelId, PairedUnit, Scoring, Task,
};
use turbo_jvm::random::JavaRandom;
use turbo_langdetect::{LanguageDetector, LanguageModel, Scratch};
use turbo_maxent::ModelFile;
use turbo_train::{Algorithm, Key, TrainingConfiguration, TrainingParameters};
use turbo_train_components::Stamp;
use turbo_train_components::langdetect::{
    LanguageDetectorFactory, LanguageEventStream, ONE_PASS, train_language_model_from_events,
};
use turbo_zoo::{Artifact, File, Kind, Licence, LicenceClass, Sha256, Sha512, TrainingData, class_of, index::Format};

use crate::card::{
    GATE_FILE, GateCard, MODEL_LICENCE, MODEL_LICENCE_TERMS, MODEL_VERSION, ModelFileCard, publishable_class,
};
use crate::error::{Error, Result, io};
use crate::segmentation::TIMESTAMP_MS;
use crate::tatoeba::{self, Export, ExportFile, SentenceLicence, Sentences, TatoebaError};
use crate::{RecordedReference, Reference};

/// The trained model's zoo id, also its file name without `.bin`.
pub const MODEL_ID: &str = "turbo-mul-tatoeba-langdetect-1";
/// The language tag the zoo lists the model under: several languages.
pub const ZOO_LANGUAGE: &str = "mul";
/// The card's schema version.
pub const CARD_VERSION: u32 = 1;
/// The card's file name in the model's directory.
pub const CARD_FILE: &str = "card.json";

/// OpenNLP's language detector, trained on the Leipzig corpora: the
/// reference the trained model is gated against. The zoo does not list
/// it (its data's terms are not confirmed, docs/licences.md), so the
/// recipe records it.
pub const LANGDETECT_183: RecordedReference = RecordedReference {
    id: "opennlp-langdetect-183",
    file: "langdetect-183.bin",
    url: "https://downloads.apache.org/opennlp/models/langdetect/1.8.3/langdetect-183.bin",
    size: 10_568_188,
    sha256: "2ddf585fac2e02a9dcfb9a4a9cc9417562eaac351be2efb506a2eaa87f19e9d4",
};

/// The sentence licences the licence check keeps.
pub const ACCEPTED_LICENCES: [SentenceLicence; 1] = [SentenceLicence::CcBy20Fr];

/// A Tatoeba language and the codes of the reference counted right for
/// it.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct LanguagePair {
    /// Tatoeba's code, the trained model's outcome.
    pub tatoeba: &'static str,
    /// The reference's codes counted right for it.
    pub reference: &'static [&'static str],
}

macro_rules! pairs {
    ($($tatoeba:literal => [$($reference:literal),+]),* $(,)?) => {
        [$(LanguagePair { tatoeba: $tatoeba, reference: &[$($reference),+] }),*]
    };
}

/// Every Tatoeba language whose sentences the recipe reads, in Tatoeba
/// code order, with the codes of `langdetect-183.bin` (ISO 639-3) counted
/// right for it. The reference's 103 codes are each in exactly one pair.
pub const LANGUAGES: [LanguagePair; 100] = pairs![
    "afr" => ["afr"], "ara" => ["ara"], "ast" => ["ast"], "aze" => ["aze"], "bak" => ["bak"],
    "bel" => ["bel"], "ben" => ["ben"], "bos" => ["bos"], "bre" => ["bre"], "bul" => ["bul"],
    "cat" => ["cat"], "ceb" => ["ceb"], "ces" => ["ces"], "che" => ["che"], "cmn" => ["cmn"],
    "cym" => ["cym"], "dan" => ["dan"], "deu" => ["deu"], "ell" => ["ell"], "eng" => ["eng"],
    "epo" => ["epo"], "est" => ["est", "ekk"], "eus" => ["eus"], "fao" => ["fao"], "fin" => ["fin"],
    "fra" => ["fra"], "fry" => ["fry"], "gle" => ["gle"], "glg" => ["glg"], "gsw" => ["gsw"],
    "guj" => ["guj"], "heb" => ["heb"], "hin" => ["hin"], "hrv" => ["hrv"], "hun" => ["hun"],
    "hye" => ["hye"], "ind" => ["ind"], "isl" => ["isl"], "ita" => ["ita"], "jav" => ["jav"],
    "jpn" => ["jpn"], "kan" => ["kan"], "kat" => ["kat"], "kaz" => ["kaz"], "kir" => ["kir"],
    "kor" => ["kor"], "lat" => ["lat"], "lim" => ["lim"], "lit" => ["lit"], "ltz" => ["ltz"],
    "lvs" => ["lvs", "lav"], "mal" => ["mal"], "mar" => ["mar"], "min" => ["min"], "mkd" => ["mkd"],
    "mlg" => ["plt"], "mlt" => ["mlt"], "mon" => ["mon"], "mri" => ["mri"], "nan" => ["nan"],
    "nds" => ["nds"], "nld" => ["nld"], "nno" => ["nno"], "nob" => ["nob"], "npi" => ["nep"],
    "oci" => ["oci"], "pan" => ["pan"], "pes" => ["pes", "fas"], "pnb" => ["pnb"], "pol" => ["pol"],
    "por" => ["por"], "pus" => ["pus"], "ron" => ["ron"], "rus" => ["rus"], "san" => ["san"],
    "sin" => ["sin"], "slk" => ["slk"], "slv" => ["slv"], "som" => ["som"], "spa" => ["spa"],
    "sqi" => ["sqi"], "srp" => ["srp"], "sun" => ["sun"], "swe" => ["swe"], "swh" => ["swa"],
    "tam" => ["tam"], "tat" => ["tat"], "tel" => ["tel"], "tgk" => ["tgk"], "tgl" => ["tgl"],
    "tha" => ["tha"], "tur" => ["tur"], "ukr" => ["ukr"], "urd" => ["urd"], "uzb" => ["uzb"],
    "vie" => ["vie"], "vol" => ["vol"], "war" => ["war"], "zsm" => ["msa"], "zul" => ["zul"],
];

/// How each language's sentences are split: see the module comment.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Split {
    /// The seed of each language's `java.util.Random`.
    pub seed: i64,
    /// The sentences held out of each language.
    pub held_out: usize,
    /// The most sentences of a language trained on.
    pub max_training: usize,
    /// The fewest sentences a language needs after the licence check and
    /// de-duplication, or it is left out: the held-out sentences and at
    /// least `min_sentences - held_out` to train on.
    pub min_sentences: usize,
}

/// The recipe's split: 500 held-out sentences per language, enough for the
/// gate's interval over about 50,000 units to be narrow beside its
/// half-point margin; at most 5,000 training sentences per language,
/// which keeps the events of 100 languages within a few gigabytes; and
/// at least 1,000 to train on.
pub const SPLIT: Split = Split { seed: 429, held_out: 500, max_training: 5_000, min_sentences: 1_500 };

/// What the recipe recorded of an export it may ship a model from.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum ExportRecord {
    /// No export is recorded: a model trained on any export is not a
    /// release.
    NotRecorded,
    /// The export of this digest ([`Export::digest`]).
    Recorded { digest: &'static str },
}

impl ExportRecord {
    /// Whether `digest` is the recorded export's.
    pub fn is(&self, digest: &str) -> bool {
        matches!(self, ExportRecord::Recorded { digest: d } if *d == digest)
    }
}

/// The export the recipe ships from. No full run has been made yet (see
/// the crate's README), so none is recorded, and every card is marked
/// `recorded: false`.
pub const EXPORT: ExportRecord = ExportRecord::NotRecorded;

/// Checks the export's digest against `record`: `Ok(true)` for the
/// recorded export, `Ok(false)` for another when `allow_unrecorded`.
///
/// # Errors
///
/// [`TatoebaError::ExportDigest`] for another export, or
/// [`TatoebaError::NotRecorded`] when none is recorded, unless
/// `allow_unrecorded`.
pub fn check_export(export: &Export, record: ExportRecord, allow_unrecorded: bool) -> Result<bool> {
    let digest = export.digest();
    match record {
        _ if record.is(&digest) => Ok(true),
        _ if allow_unrecorded => Ok(false),
        ExportRecord::NotRecorded => Err(TatoebaError::NotRecorded.into()),
        ExportRecord::Recorded { digest: expected } => {
            Err(TatoebaError::ExportDigest { expected: expected.to_owned(), found: digest }.into())
        }
    }
}

/// The recipe: the model, its reference, the languages, the split and the
/// training parameters.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct LanguageRecipe {
    pub model_id: &'static str,
    pub reference: Reference,
    pub languages: &'static [LanguagePair],
    pub split: Split,
    pub iterations: i32,
    pub cutoff: i32,
}

/// The language detector recipe.
pub const LANGDETECT: LanguageRecipe = LanguageRecipe {
    model_id: MODEL_ID,
    reference: Reference::Recorded(LANGDETECT_183),
    languages: &LANGUAGES,
    split: SPLIT,
    iterations: turbo_train::params::DEFAULT_ITERATIONS,
    cutoff: turbo_train::params::DEFAULT_CUTOFF,
};

/// One held-out or training sentence: the index of its language in the
/// recipe's [`LanguageRecipe::languages`], its id and its text.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Sample<'a> {
    pub language: usize,
    pub id: u32,
    pub text: std::borrow::Cow<'a, str>,
}

/// One language of a plan: its sentences and how they were split, and
/// once gated, how many held-out sentences each model got right.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct LanguageCount {
    pub tatoeba: String,
    pub reference: Vec<String>,
    /// The language's sentences in the export, of any licence.
    pub sentences: usize,
    /// Sentences the licence check left out, their licence not one of
    /// [`ACCEPTED_LICENCES`].
    pub licence_left_out: usize,
    /// Sentences left out as a text the language already has.
    pub duplicates: usize,
    pub training: usize,
    pub held_out: usize,
    /// Held-out sentences the trained model got right.
    pub candidate_correct: usize,
    /// Held-out sentences the reference got right.
    pub reference_correct: usize,
}

/// A language left out for having fewer than [`Split::min_sentences`]
/// sentences after the licence check and de-duplication.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ExcludedLanguage {
    pub tatoeba: String,
    /// Its sentences in the export, of any licence.
    pub sentences: usize,
    /// Those left after the licence check and de-duplication.
    pub usable: usize,
}

/// The split of an export's sentences.
#[derive(Clone, Debug)]
pub struct Plan<'a> {
    /// Language by language in the recipe's order, each in its shuffled
    /// order.
    pub training: Vec<Sample<'a>>,
    /// The same way.
    pub held_out: Vec<Sample<'a>>,
    /// The languages trained on and held out, in the recipe's order.
    pub languages: Vec<LanguageCount>,
    pub excluded: Vec<ExcludedLanguage>,
    /// The records of the detailed export.
    pub records: usize,
    /// Its records with no language.
    pub no_language: usize,
    /// Its records of a language the recipe does not read.
    pub other_languages: usize,
}

/// `Collections.shuffle(list, rnd)` for a list that allows random access:
/// from the last position down to the second, each element swapped with
/// one at `rnd.nextInt(i)`.
pub fn java_shuffle<T>(list: &mut [T], random: &mut JavaRandom) {
    for i in (2..=list.len()).rev() {
        let n = i32::try_from(i).expect("a list java.util.Random can index");
        list.swap(i - 1, random.next_int_bounded(n) as usize);
    }
}

/// A model the recipe trained.
#[derive(Debug, Clone)]
pub struct TrainedLanguages {
    /// The `.bin` file.
    pub bytes: Vec<u8>,
    pub samples: usize,
    pub events: usize,
    /// `Training-Eventhash`.
    pub event_hash: u32,
    /// Events whose predicates are not in the JVM's order
    /// (`turbo_train_components::langdetect`).
    pub inexact_events: usize,
}

impl LanguageRecipe {
    /// The training parameters: `MAXENT`, the recipe's iterations and
    /// cutoff; `train` adds the `OnePass` indexer.
    pub fn parameters(&self) -> TrainingParameters {
        let mut p = TrainingParameters::new();
        p.put(Key::ALGORITHM, Algorithm::Maxent.name());
        p.put(Key::ITERATIONS, self.iterations);
        p.put(Key::CUTOFF, self.cutoff);
        p
    }

    /// The pair of a Tatoeba code, by its index in the recipe's languages.
    fn index(&self) -> HashMap<&'static str, usize> {
        self.languages.iter().enumerate().map(|(i, p)| (p.tatoeba, i)).collect()
    }

    /// Reads the sentences of the recipe's languages from `export`.
    ///
    /// # Errors
    ///
    /// What [`Export::sentences`] fails on.
    pub fn sentences<'a>(&self, export: &'a Export) -> Result<Sentences<'a>> {
        let index = self.index();
        Ok(export.sentences(|lang| index.contains_key(lang))?)
    }

    /// The licence check, de-duplication and split of `sentences`.
    ///
    /// # Errors
    ///
    /// [`Error::EmptySplit`] when no language has
    /// [`Split::min_sentences`] sentences.
    pub fn plan<'a>(&self, sentences: Sentences<'a>) -> Result<Plan<'a>> {
        let index = self.index();
        let mut by_language: Vec<Vec<tatoeba::Sentence<'a>>> = vec![Vec::new(); self.languages.len()];
        let kept = sentences.kept.len();
        for s in sentences.kept {
            let i = *index.get(&*s.language).expect("the export kept the recipe's languages only");
            by_language[i].push(s);
        }
        let mut plan = Plan {
            training: Vec::new(),
            held_out: Vec::new(),
            languages: Vec::new(),
            excluded: Vec::new(),
            records: sentences.records,
            no_language: sentences.no_language,
            other_languages: sentences.records - sentences.no_language - kept,
        };
        for (language, mut all) in by_language.into_iter().enumerate() {
            let pair = self.languages[language];
            let total = all.len();
            all.retain(|s| ACCEPTED_LICENCES.contains(&s.licence));
            let licence_left_out = total - all.len();
            all.sort_by_key(|s| s.id);
            let mut texts = HashSet::new();
            let mut usable: Vec<tatoeba::Sentence<'a>> = Vec::with_capacity(all.len());
            for s in all {
                if texts.insert(s.text.clone()) {
                    usable.push(s);
                }
            }
            let duplicates = total - licence_left_out - usable.len();
            if usable.len() < self.split.min_sentences {
                plan.excluded.push(ExcludedLanguage {
                    tatoeba: pair.tatoeba.to_owned(),
                    sentences: total,
                    usable: usable.len(),
                });
                continue;
            }
            java_shuffle(&mut usable, &mut JavaRandom::new(self.split.seed));
            let held_out = self.split.held_out.min(usable.len());
            let training = self.split.max_training.min(usable.len() - held_out);
            let mut it = usable.into_iter();
            for s in it.by_ref().take(held_out) {
                plan.held_out.push(Sample { language, id: s.id, text: s.text });
            }
            for s in it.take(training) {
                plan.training.push(Sample { language, id: s.id, text: s.text });
            }
            plan.languages.push(LanguageCount {
                tatoeba: pair.tatoeba.to_owned(),
                reference: pair.reference.iter().map(|c| (*c).to_owned()).collect(),
                sentences: total,
                licence_left_out,
                duplicates,
                training,
                held_out,
                candidate_correct: 0,
                reference_correct: 0,
            });
        }
        if plan.languages.is_empty() {
            return Err(Error::EmptySplit { what: "language with enough sentences" });
        }
        Ok(plan)
    }

    /// Trains the model on the plan's training sentences, one event each.
    ///
    /// # Errors
    ///
    /// What the trainer fails on.
    pub fn train(&self, plan: &Plan<'_>) -> Result<TrainedLanguages> {
        let factory = LanguageDetectorFactory::new();
        let mut stream = LanguageEventStream::new(&factory);
        let events: Vec<_> =
            plan.training.iter().map(|s| stream.event(self.languages[s.language].tatoeba, &s.text)).collect();
        let trained = train_language_model_from_events(
            &events,
            &factory,
            &self.parameters(),
            &mut TrainingConfiguration::default(),
            Stamp::at(TIMESTAMP_MS),
        )?;
        Ok(TrainedLanguages {
            bytes: trained.package.to_bytes(),
            samples: plan.training.len(),
            events: events.len(),
            event_hash: trained.report.event_hash.expect("trained from events"),
            inexact_events: stream.inexact_events(),
        })
    }

    /// The reference's codes counted right for another held-out language,
    /// over the plan's languages.
    pub fn reference_mappings(&self, plan: &Plan<'_>) -> Vec<LabelMapping> {
        let mut out = Vec::new();
        for l in &plan.languages {
            for r in l.reference.iter().filter(|r| **r != l.tatoeba) {
                out.push(LabelMapping { held_out: l.tatoeba.clone(), reference: r.clone() });
            }
        }
        out
    }

    /// The SHA-256 of the held-out samples as scored: for each, its
    /// language, its length in bytes and its text, then each reference
    /// mapping, so two records scoring the same samples the same way have
    /// the same digest.
    pub fn samples_digest(&self, plan: &Plan<'_>) -> String {
        let mut text = String::new();
        for s in &plan.held_out {
            text.push_str(&format!("{}\t{}\t{}\n", self.languages[s.language].tatoeba, s.text.len(), s.text));
        }
        for m in self.reference_mappings(plan) {
            text.push_str(&format!("mapping\t{}\t{}\n", m.held_out, m.reference));
        }
        Sha256::of(text.as_bytes()).to_string()
    }

    /// Scores the trained model and the reference on the plan's held-out
    /// sentences, one unit each, and runs the gate; `set_name` and
    /// `corpus_sha256` name the held-out set. Fills in each language's
    /// correct counts.
    ///
    /// # Errors
    ///
    /// A model file that is not a language detector, a reference without a
    /// code the plan's languages map to, or what the gate fails on.
    pub fn gate(
        &self,
        trained: &[u8],
        reference: &[u8],
        plan: &mut Plan<'_>,
        set_name: &str,
        corpus_sha256: &str,
    ) -> Result<ModelGateRecord> {
        let open = |what: &str, bytes: &[u8]| {
            ModelFile::open(bytes.to_vec())
                .map_err(|e| e.to_string())
                .and_then(|f| LanguageModel::open(f).map_err(|e| e.to_string()))
                .map(LanguageDetector::new)
                .map_err(|e| Error::Model(format!("{what}: {e}")))
        };
        let candidate = open(self.model_id, trained)?;
        let reference_model = open(self.reference.id(), reference)?;
        let known: HashSet<&str> = reference_model.model().supported_languages().collect();
        for l in &plan.languages {
            if let Some(code) = l.reference.iter().find(|c| !known.contains(c.as_str())) {
                return Err(Error::Model(format!(
                    "{}: no outcome {code}, which the recipe counts right for {}",
                    self.reference.id(),
                    l.tatoeba
                )));
            }
        }
        let index = self.index();
        let position: HashMap<usize, usize> =
            plan.languages.iter().enumerate().map(|(k, l)| (index[l.tatoeba.as_str()], k)).collect();
        let mut scratch = Scratch::new();
        let mut units = Vec::with_capacity(plan.held_out.len());
        for s in &plan.held_out {
            let pair = self.languages[s.language];
            let c = candidate.predict_language(&s.text, &mut scratch).code == pair.tatoeba;
            let r = pair.reference.contains(&reference_model.predict_language(&s.text, &mut scratch).code);
            let count = &mut plan.languages[position[&s.language]];
            count.candidate_correct += usize::from(c);
            count.reference_correct += usize::from(r);
            units.push(PairedUnit {
                candidate: Counts::accuracy(u64::from(c), 1),
                reference: Counts::accuracy(u64::from(r), 1),
            });
        }
        let set = HeldOutSet {
            name: set_name.to_owned(),
            corpus_sha256: corpus_sha256.to_owned(),
            samples_sha256: self.samples_digest(plan),
            skipped_sentences: 0,
            scoring: Scoring::Labels(LabelScoring {
                samples: plan.held_out.len() as u64,
                reference_mappings: self.reference_mappings(plan),
            }),
        };
        let candidate_id = ModelId { id: self.model_id.to_owned(), sha256: Sha256::of(trained).to_string() };
        let reference_id = ModelId { id: self.reference.id().to_owned(), sha256: self.reference.sha256()? };
        Ok(ModelGateRecord::new(Task::LanguageDetector, set, candidate_id, reference_id, units)?)
    }

    /// The name of the held-out set in the gate record.
    pub fn held_out_name(&self, export_digest: &str) -> String {
        format!(
            "Tatoeba export {} held out ({} sentences per language, seed {})",
            &export_digest[..16.min(export_digest.len())],
            self.split.held_out,
            self.split.seed
        )
    }
}

/// What the recipe made.
#[derive(Debug)]
pub struct LanguageOutcome {
    pub trained: TrainedLanguages,
    pub gate: ModelGateRecord,
    pub card: LanguageCard,
}

/// Reads the recipe's sentences from `export`, plans, trains, gates the
/// model against `reference` (the bytes of the reference's file) and
/// makes the card. `recorded` is whether the export is the recorded one.
///
/// # Errors
///
/// What reading, training, scoring or the gate fail on.
pub fn run(recipe: &LanguageRecipe, export: &Export, recorded: bool, reference: &[u8]) -> Result<LanguageOutcome> {
    recipe.reference.check(reference)?;
    let mut plan = recipe.plan(recipe.sentences(export)?)?;
    let trained = recipe.train(&plan)?;
    let digest = export.digest();
    let gate = recipe.gate(&trained.bytes, reference, &mut plan, &recipe.held_out_name(&digest), &digest)?;
    let card = LanguageCard::new(recipe, export, recorded, &plan, &trained, &gate);
    Ok(LanguageOutcome { trained, gate, card })
}

/// The export the model was trained and gated on.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ExportCard {
    pub detailed: ExportFile,
    pub cc0: ExportFile,
    /// [`Export::digest`].
    pub digest: String,
    /// Whether it is [`EXPORT`]. A model trained on another export is not
    /// a release.
    pub recorded: bool,
    /// The records of the detailed export.
    pub records: usize,
    pub no_language: usize,
    pub other_languages: usize,
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
    /// The credit CC BY 2.0 FR asks for, to be shipped with the model.
    pub attribution: String,
    pub export: ExportCard,
    pub split: Split,
    pub per_language: Vec<LanguageCount>,
    pub excluded_languages: Vec<ExcludedLanguage>,
    pub training: LanguageTrainingCard,
    pub held_out: LanguageHeldOutCard,
    pub gate: GateCard,
}

/// The data the model is trained on, as the zoo lists it.
pub fn training_data(export: &Export) -> TrainingData {
    TrainingData {
        name: format!("Tatoeba export {}", &export.digest()[..16]),
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
        let trained_on = vec![training_data(export)];
        let class = class_of(std::iter::once(MODEL_LICENCE).chain(trained_on.iter().map(|t| t.license)));
        let r = &gate.result;
        let digest = export.digest();
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
                "Trained on sentences from Tatoeba (https://tatoeba.org), the export of digest {digest}, licensed \
                 under CC BY 2.0 FR (https://creativecommons.org/licenses/by/2.0/fr/); the export's username column \
                 names each sentence's contributor."
            ),
            export: ExportCard {
                detailed: export.detailed.clone(),
                cc0: export.cc0.clone(),
                digest,
                recorded,
                records: plan.records,
                no_language: plan.no_language,
                other_languages: plan.other_languages,
                accepted_licences: ACCEPTED_LICENCES.to_vec(),
            },
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

    /// The model's zoo entry once the file is published at `url`.
    ///
    /// # Errors
    ///
    /// [`Error::NotShippable`] when the model may not leave the machine it
    /// was trained on, it is not [`LANGDETECT`]'s model, it was not gated
    /// against the recipe's reference, the gate failed, or the export is
    /// not the recorded one; [`Error::InvalidCard`] for a `class` other
    /// than its licences give, or a field the index refuses.
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
/// [`CARD_FILE`] and [`GATE_FILE`]. The directory must not exist yet.
///
/// # Errors
///
/// An existing directory, or a write that fails.
pub fn write_language_dir(dir: &Path, card: &LanguageCard, model: &[u8], gate: &ModelGateRecord) -> Result<()> {
    let out = dir.join(&card.id);
    std::fs::create_dir_all(dir).map_err(io(dir))?;
    std::fs::create_dir(&out).map_err(io(&out))?;
    let model_path = out.join(&card.file.name);
    std::fs::write(&model_path, model).map_err(io(&model_path))?;
    let card_path = out.join(CARD_FILE);
    std::fs::write(&card_path, card.to_json()).map_err(io(&card_path))?;
    Ok(gate.write(&out.join(GATE_FILE))?)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_reference_code_is_in_one_pair_and_the_tatoeba_codes_are_unique() {
        let mut tatoeba: Vec<&str> = LANGUAGES.iter().map(|p| p.tatoeba).collect();
        let sorted = {
            let mut s = tatoeba.clone();
            s.sort_unstable();
            s
        };
        assert_eq!(tatoeba, sorted, "in Tatoeba code order");
        tatoeba.dedup();
        assert_eq!(tatoeba.len(), 100);
        let mut reference: Vec<&str> = LANGUAGES.iter().flat_map(|p| p.reference.iter().copied()).collect();
        reference.sort_unstable();
        let all = reference.len();
        reference.dedup();
        assert_eq!((all, reference.len()), (103, 103), "each of the reference's 103 codes once");
        for p in LANGUAGES {
            for code in std::iter::once(p.tatoeba).chain(p.reference.iter().copied()) {
                assert!(code.len() == 3 && code.bytes().all(|b| b.is_ascii_lowercase()), "{code}");
            }
        }
        // The pairs whose codes differ: a macrolanguage and one of its
        // individual languages.
        let differing: Vec<(&str, &[&str])> =
            LANGUAGES.iter().filter(|p| p.reference != [p.tatoeba]).map(|p| (p.tatoeba, p.reference)).collect();
        assert_eq!(
            differing,
            [
                ("est", &["est", "ekk"][..]),
                ("lvs", &["lvs", "lav"][..]),
                ("mlg", &["plt"][..]),
                ("npi", &["nep"][..]),
                ("pes", &["pes", "fas"][..]),
                ("swh", &["swa"][..]),
                ("zsm", &["msa"][..]),
            ]
        );
    }

    #[test]
    fn the_recipe_trains_with_the_parameters_langdetect_183_is_reproduced_with() {
        let ours = LANGDETECT.parameters();
        let theirs = turbo_train_components::langdetect::default_training_parameters();
        assert_eq!(ours, theirs);
        assert_eq!(LANGDETECT.reference.id(), "opennlp-langdetect-183");
        assert!(turbo_zoo::Index::builtin().get(LANGDETECT_183.id).is_err(), "the zoo does not list the reference");
        let id: turbo_zoo::ArtifactId = MODEL_ID.parse().unwrap();
        assert_eq!(id.as_str(), MODEL_ID);
        assert!(ZOO_LANGUAGE.parse::<turbo_zoo::Language>().is_ok());
    }

    #[test]
    fn the_shuffle_is_collections_shuffle() {
        // Collections.shuffle swaps positions 4, 3, 2 and 1 of a list of
        // five with nextInt(5), nextInt(4), nextInt(3) and nextInt(2) of
        // its Random, in that order (turbo-jvm's tests check the draws
        // against the JDK's).
        let mut r = JavaRandom::new(42);
        let draws: Vec<i32> = (2..=5).rev().map(|n| r.next_int_bounded(n)).collect();
        let mut list = [0, 1, 2, 3, 4];
        java_shuffle(&mut list, &mut JavaRandom::new(42));
        let mut expected = [0, 1, 2, 3, 4];
        for (i, d) in (2..=5usize).rev().zip(draws) {
            expected.swap(i - 1, d as usize);
        }
        assert_eq!(list, expected);
        let mut one = [7];
        java_shuffle(&mut one, &mut JavaRandom::new(42));
        assert_eq!(one, [7]);
    }
}
