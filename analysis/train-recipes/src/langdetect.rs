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
//!   parameters `langdetect-183.bin` was most likely trained with, those
//!   of the command line trainer given none (`MAXENT`, 100 iterations,
//!   cutoff 5, one thread, the `OnePass` indexer `train` puts in), and a
//!   fixed manifest `Timestamp`, so a run gives the same bytes.
//! - Gate: every held-out sentence is a unit of `turbo_eval::model_gate`
//!   ([`Task::LanguageDetector`](turbo_eval::model_gate::Task::LanguageDetector),
//!   [`Scoring::Labels`](turbo_eval::model_gate::Scoring::Labels)), right when the
//!   model's best language (`predictLanguage`) is its language: as written
//!   for the trained model, through [`LANGUAGES`] for the reference. Both
//!   are run by turbo-langdetect, whose probabilities are the JVM's bit for
//!   bit. The reference is only scored; it is never shipped or bundled.

mod card;
mod plan;

use serde::{Deserialize, Serialize};
use turbo_eval::model_gate::ModelGateRecord;

pub use card::{
    ExportCard, LanguageCard, LanguageHeldOutCard, LanguageTrainingCard, SentencesFileCard, export_card, training_data,
    write_language_dir,
};

use crate::error::Result;
use crate::tatoeba::{Export, SentenceLicence, TatoebaError};
use crate::{RecordedReference, Reference};

/// The trained model's zoo id, also its file name without `.bin`.
pub const MODEL_ID: &str = "turbo-mul-tatoeba-langdetect-1";
/// The language tag the zoo lists the model under: several languages.
pub const ZOO_LANGUAGE: &str = "mul";
/// The card's schema version.
pub const CARD_VERSION: u32 = 1;
/// The card's file name in the model's directory.
pub const CARD_FILE: &str = "card.json";
/// The file of the model's directory that lists the ids of the sentences
/// it was trained on ([`sentences_file`]): the credit the card's
/// attribution points at.
pub const SENTENCES_FILE: &str = "training-sentences.txt";
/// Where Tatoeba shows a sentence, with its author: this address followed
/// by the sentence's id.
pub const SENTENCE_URL: &str = "https://tatoeba.org/sentences/show/";

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

    /// The recorded export's digest; `None` when none is recorded.
    pub fn digest(&self) -> Option<&'static str> {
        match self {
            ExportRecord::NotRecorded => None,
            ExportRecord::Recorded { digest } => Some(digest),
        }
    }
}

/// The export the recipe ships from: Tatoeba's weekly export of
/// 2026-10-10, which the full run read (the crate's README, "Full run").
/// Tatoeba replaces its export every week at the same addresses, so a
/// release of this repository ([`crate::release::TATOEBA_EXPORT_RELEASE`])
/// keeps this one's two archives.
pub const EXPORT: ExportRecord =
    ExportRecord::Recorded { digest: "4421126b36dc0ce206b222296d696aff04a399620f1ca5dbefbfa50a1890aa92" };

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
    /// The records of the CC0 export, those whose id the detailed export
    /// holds with another language or text, and those whose id it does
    /// not hold ([`Sentences`](crate::tatoeba::Sentences)).
    pub cc0_records: usize,
    pub cc0_differing: usize,
    pub cc0_not_in_detailed: usize,
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
    /// [`SENTENCES_FILE`] of the samples trained on.
    pub sentences: Vec<u8>,
}

/// The contents of [`SENTENCES_FILE`] for `plan`: the id of each training
/// sentence in ascending order, a line each, every line ending in a line
/// feed. No username is stored: each sentence's page names its author.
pub fn sentences_file(plan: &Plan<'_>) -> Vec<u8> {
    let mut ids: Vec<u32> = plan.training.iter().map(|s| s.id).collect();
    ids.sort_unstable();
    let mut out = String::with_capacity(ids.len() * 9);
    for id in ids {
        out.push_str(&id.to_string());
        out.push('\n');
    }
    out.into_bytes()
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

#[cfg(test)]
mod tests;
