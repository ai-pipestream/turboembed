//! Reproducing the published OpenNLP UD 1.3 models (sentence detector,
//! tokenizer, part-of-speech tagger and lemmatizer): what Apache OpenNLP
//! trained them from and with, as typed recipes, and the comparison of a
//! model trained here with the published file.
//!
//! The models were made by the scripts in `apache/opennlp-models`
//! (`opennlp-models-training/opennlp-models-training-ud/src/main/resources/
//! ud-train.sh` and `ud-train.conf`) with the OpenNLP 2.5.4 command line:
//! `TokenizerTrainer.conllu -params ud-train.conf -lang <code> -data
//! <treebank>-ud-train.conllu -encoding UTF-8`,
//! `SentenceDetectorTrainer.conllu ... -sentencesPerSample 10`,
//! `POSTaggerTrainer.conllu ...` and `LemmatizerTrainerME.conllu ...`,
//! over the `train` file of each treebank at Universal Dependencies
//! release 2.16. `ud-train.conf` is `Algorithm=MAXENT`, `Iterations=150`,
//! `Cutoff=5`, `Threads=4`. The tools' defaults supply the rest: the
//! tokenizer with no abbreviation dictionary, the alphanumeric
//! optimization off and the language's pattern; the sentence detector
//! with `useTokenEnd` on, no dictionary and the language's end
//! characters; the tagger with the default feature generator descriptor,
//! no tag dictionary and the universal tags (`-tagset u`, column 4); the
//! lemmatizer with its one context generator. OpenNLP 2.5.4 tests
//! whitespace with `Character.isWhitespace` or the `Zs` category
//! everywhere (`WhitespaceMode::Legacy` here), and the JVM that wrote the
//! files had its clock in Central European Summer Time.
//!
//! The iteration count is the one setting the repository does not give
//! correctly: the published English sentence model's parameters are those
//! of a run of 300 iterations (the parameters after every iteration of a
//! longer run were compared with the file; 300 agrees to within a few
//! thousand units in the last place, 299 and 301 do not), so the models
//! were trained with `Iterations=300`, not the committed 150. The recipes
//! use 300 ([`ITERATIONS`]); [`CONF_ITERATIONS`] is what the file says.
//!
//! With those settings the event checksum (`Training-Eventhash`) and the
//! manifest of a published model come out exactly, and the model entry's
//! labels and outcome patterns too. The parameters differ from the
//! published ones in their last bits: for the eighteen models, between
//! 2.2e-15 (German sentences) and 9.8e-15 (French tokens) for the sentence
//! and token models, 1.6e-14 (English) and 1.4e-13 (Spanish) for the
//! taggers, and 8.0e-15 (English) and 1.8e-14 (Portuguese) for the
//! lemmatizers, in absolute value at most, with 60 to 97 percent of the
//! parameters differing; the widest, 1.37e-13, is about seven times under
//! the 1e-12 the reproduction test allows. For the eight sentence and
//! token models, the JVM of the goldens container (Temurin 21 on x86-64),
//! running the same training with the fork's trainer, writes exactly the
//! parameters this crate writes and shows the same residual against the
//! published file, so for them the residual is not between this port and
//! the fork's trainer; the taggers and lemmatizers have not been run
//! through the fork's trainer in the container. Its cause is not
//! known: `GISTrainer` uses `StrictMath` throughout and merges its
//! threads' sums in a fixed order, so neither platform `Math` intrinsics
//! nor thread timing explain it. The next experiment is to run the OpenNLP
//! 2.5.4 command line itself in the container on the same treebank and
//! compare its parameters with both.
//!
//! The treebanks are CC BY-SA 4.0, so they are fetched through the zoo,
//! by digest, after the user accepts that licence, and are never checked
//! in. The reproduction itself is the test `tests/reproduce.rs`, ignored
//! by default.

use std::path::{Path, PathBuf};

use turbo_formats::conllu::{ConlluLemmaReader, ConlluPosReader, ConlluSentenceReader, ConlluTokenReader, Tagset};
use turbo_maxent::{MaxentModel, ModelFile, sha256};
use turbo_train::package::MANIFEST_ENTRY;
use turbo_train::{GisModel, Key, TimeZone, TrainingConfiguration, TrainingParameters};
use turbo_unicode::WhitespaceMode;
use turbo_zoo::{
    Acceptances, Artifact, Http, Index, Kind, Licence, LicenceClass, Policy, UserId, Zoo, index::Format as ZooFormat,
};

use crate::error::{Error, Result};
use crate::lemmatize::{self, LemmatizerFactory};
use crate::package::Stamp;
use crate::postag::{self, PosTaggerFactory};
use crate::sentdetect::{self, SentenceDetectorFactory};
use crate::tokenize::{self, TokenizerFactory};

/// The OpenNLP that trained the published models.
pub const OPENNLP_VERSION: &str = "2.5.4";
/// The time zone of the JVM that wrote them.
pub const ZONE: TimeZone = TimeZone::CEST;
/// OpenNLP 2.5.4's whitespace definition.
pub const WHITESPACE: WhitespaceMode = WhitespaceMode::Legacy;
/// `-sentencesPerSample 10` of `SentenceDetectorTrainer.conllu`.
pub const SENTENCES_PER_SAMPLE: usize = 10;
/// The default `-tagset` of `POSTaggerTrainer.conllu` and
/// `LemmatizerTrainerME.conllu`: the universal tags.
pub const TAGSET: Tagset = Tagset::U;
/// The Universal Dependencies release.
pub const UD_RELEASE: &str = "r2.16";
/// The iterations the published models' parameters agree with (see the
/// module documentation).
pub const ITERATIONS: i32 = 300;
/// `Iterations` as `ud-train.conf` in `apache/opennlp-models` has it,
/// which the published models do not agree with.
pub const CONF_ITERATIONS: i32 = 150;
/// `ud-train.conf`.
pub const CUTOFF: i32 = 5;
/// `ud-train.conf`.
pub const THREADS: i32 = 4;

/// The training parameters as `CmdLineUtil.loadTrainingParameters` reads
/// a parameters file: every value a string. `ud-train.conf` with
/// `Iterations` at [`ITERATIONS`].
pub fn training_parameters() -> TrainingParameters {
    let mut p = TrainingParameters::new();
    p.put_str(Key::ALGORITHM, "MAXENT");
    p.put_str(Key::ITERATIONS, &ITERATIONS.to_string());
    p.put_str(Key::CUTOFF, &CUTOFF.to_string());
    p.put_str(Key::THREADS, &THREADS.to_string());
    p
}

/// One treebank's training file at the release.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Treebank {
    /// The repository under `github.com/UniversalDependencies`.
    pub repository: &'static str,
    /// The `train` file in it.
    pub train_file: &'static str,
    pub size: u64,
    pub sha256: &'static str,
    pub sha512: &'static str,
    pub licence: Licence,
}

impl Treebank {
    /// The `README.md` of the repository at the release, where the
    /// licence is stated.
    pub fn licence_terms(&self) -> String {
        format!("https://github.com/UniversalDependencies/{}/blob/{UD_RELEASE}/README.md", self.repository)
    }

    /// The file's address at the release.
    pub fn url(&self) -> String {
        format!(
            "https://raw.githubusercontent.com/UniversalDependencies/{}/{UD_RELEASE}/{}",
            self.repository, self.train_file
        )
    }

    /// The zoo id of the dataset artifact: the repository name lowercased
    /// with the release and `train`.
    pub fn dataset_id(&self) -> String {
        format!("{}-{UD_RELEASE}-train", self.repository.to_ascii_lowercase().replace('_', "-"))
    }

    /// The treebank's `train` file as a zoo artifact.
    pub fn dataset(&self) -> Artifact {
        fn parse<T: std::str::FromStr<Err = String>>(what: &str, s: &str) -> T {
            s.parse().unwrap_or_else(|e| panic!("{what}: {e}"))
        }
        Artifact {
            id: parse("id", &self.dataset_id()),
            kind: Kind::Dataset,
            component: None,
            language: None,
            format: ZooFormat::Conllu,
            version: UD_RELEASE.to_owned(),
            license: self.licence,
            license_terms: parse("terms", &self.licence_terms()),
            trained_on: Vec::new(),
            class: LicenceClass::ShareAlike,
            origin: None,
            files: vec![turbo_zoo::File {
                path: parse("path", self.train_file),
                url: parse("url", &self.url()),
                size: self.size,
                sha256: parse("sha256", self.sha256),
                sha512: parse("sha512", self.sha512),
            }],
        }
    }
}

pub const ENGLISH_EWT: Treebank = Treebank {
    repository: "UD_English-EWT",
    train_file: "en_ewt-ud-train.conllu",
    size: 14203025,
    sha256: "3fb78e2b55482c8ee00caa653af436c88c9e5bc9a73ce26e4ae3ddc79d2b7e7b",
    sha512: "79379a97c1d61fce860762f966940e6da5ec68fdcd8aa985aa06f889aa24f704ed7793db507f84de53d40bae2c3a20a1a1970df80ae8210a181d8967837eb7ca",
    licence: Licence::CcBySa40,
};

pub const GERMAN_GSD: Treebank = Treebank {
    repository: "UD_German-GSD",
    train_file: "de_gsd-ud-train.conllu",
    size: 19627297,
    sha256: "b01561c37fafa1b505630f69d173ca0c015a91e444a2cfe7aeca67288b1c77f2",
    sha512: "5b8876ecab65c557dd01be86205b576384e9bc836f562e7cd5750460ecbde4fb4daa67843075e79d583e60783cc6a1f5e4f7ff84077e0840db34ac27115a0666",
    licence: Licence::CcBySa40,
};

pub const SPANISH_GSD: Treebank = Treebank {
    repository: "UD_Spanish-GSD",
    train_file: "es_gsd-ud-train.conllu",
    size: 23913739,
    sha256: "6118dd7ae44a67a8a79acecf90c77a9f580961ff55417457e17ca3799adc2810",
    sha512: "f974c94cc67c05a28764eb3f54305438cac80a37ad4f4a91e55b0a40ce3b5b2fd80a98bf437c5a257b037a2415b26e5998c13d90d4a8f0c6f3c14f67e265b379",
    licence: Licence::CcBySa40,
};

pub const FRENCH_GSD: Treebank = Treebank {
    repository: "UD_French-GSD",
    train_file: "fr_gsd-ud-train.conllu",
    size: 23020440,
    sha256: "3682d4276e12f422816f86ee10fa0a0525eca750a8e502fd914e18511dd7695d",
    sha512: "4bbc04b5c6d1df2b9a3f07a851899e4329a7e23ac911497b5258109fba248a692f55ee327753e0baa6c5f8b6ab0505fc21f461652c60cc906c756e8883e21922",
    licence: Licence::CcBySa40,
};

pub const PORTUGUESE_GSD: Treebank = Treebank {
    repository: "UD_Portuguese-GSD",
    train_file: "pt_gsd-ud-train.conllu",
    size: 13521544,
    sha256: "c28bdc7ff8ddf03cba586062be0b041308da36bd82c321b437564d2b6c8664bb",
    sha512: "4b2496b8ea55e9672872c8cf8c53f644837fc616f3d21f7b72e5f011127267b491b1c40523e2a4f2f79c191739a5975208eb2034e577b6884e66306ce7ce7bd7",
    licence: Licence::CcBySa40,
};

/// The treebanks the published models come from.
pub const TREEBANKS: [Treebank; 5] = [ENGLISH_EWT, GERMAN_GSD, SPANISH_GSD, FRENCH_GSD, PORTUGUESE_GSD];

/// Which model of a treebank.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Component {
    Sentence,
    Tokens,
    Pos,
    Lemmas,
}

impl Component {
    /// The word in the published file names.
    pub fn name(self) -> &'static str {
        match self {
            Component::Sentence => "sentence",
            Component::Tokens => "tokens",
            Component::Pos => "pos",
            Component::Lemmas => "lemmas",
        }
    }

    /// The model entry of the `.bin`.
    pub fn model_entry(self) -> &'static str {
        match self {
            Component::Sentence => sentdetect::MODEL_ENTRY,
            Component::Tokens => tokenize::MODEL_ENTRY,
            Component::Pos => postag::MODEL_ENTRY,
            Component::Lemmas => lemmatize::MODEL_ENTRY,
        }
    }
}

/// One published model and what it was trained from.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Recipe {
    /// The language code the trainer was given (`-lang`).
    pub language: &'static str,
    /// The treebank's short name in the model id (`ewt`, `gsd`).
    pub treebank_name: &'static str,
    pub component: Component,
    pub treebank: Treebank,
}

impl Recipe {
    /// The zoo id of the published model, which is also its file name
    /// without `.bin`.
    pub fn model_id(&self) -> String {
        format!("opennlp-{}-ud-{}-{}-1.3-{OPENNLP_VERSION}", self.language, self.treebank_name, self.component.name())
    }

    /// The published file's name.
    pub fn model_file(&self) -> String {
        format!("{}.bin", self.model_id())
    }

    /// The factory the trainer tool built: the language, and every other
    /// setting at its default.
    pub fn sentence_factory(&self) -> Result<SentenceDetectorFactory> {
        SentenceDetectorFactory::new(self.language, true, None, None)
    }

    /// The factory the trainer tool built: the language, and every other
    /// setting at its default.
    pub fn tokenizer_factory(&self) -> Result<TokenizerFactory> {
        TokenizerFactory::new(self.language, None, false, None)
    }

    /// The factory the trainer tool built: the language, the default
    /// feature generator descriptor and no tag dictionary.
    pub fn pos_factory(&self) -> Result<PosTaggerFactory> {
        PosTaggerFactory::new(self.language, None, None)
    }

    /// The factory the trainer tool built: the language.
    pub fn lemmatizer_factory(&self) -> LemmatizerFactory {
        LemmatizerFactory::new(self.language)
    }

    /// Trains the model from the treebank's CoNLL-U text as the published
    /// one was trained, stamped with `timestamp_ms` (the published
    /// manifest's `Timestamp`, to compare the manifests).
    pub fn train(&self, conllu: &str, timestamp_ms: i64, config: &mut TrainingConfiguration) -> Result<Trained> {
        let stamp = Stamp { timestamp_ms, version: OPENNLP_VERSION, zone: ZONE };
        let params = training_parameters();
        match self.component {
            // The samples are read one at a time and dropped after their
            // events, so only the events of a treebank are held at once.
            Component::Sentence => {
                let factory = self.sentence_factory()?;
                let mut stream = sentdetect::SdEventStream::new(&factory, WHITESPACE);
                let mut events = Vec::new();
                for sample in ConlluSentenceReader::new(conllu, SENTENCES_PER_SAMPLE)? {
                    stream.events(&sample?, &mut events);
                }
                let t = sentdetect::train_sentence_model_from_events(&events, &factory, &params, config, stamp)?;
                Ok(Trained { events: events.len(), model: t.model, report: t.report, package: t.package })
            }
            Component::Tokens => {
                let factory = self.tokenizer_factory()?;
                let mut stream = tokenize::TokSpanEventStream::new(&factory, WHITESPACE);
                let mut events = Vec::new();
                for sample in ConlluTokenReader::new(conllu, WHITESPACE)? {
                    stream.events(&sample?, &mut events)?;
                }
                let t = tokenize::train_tokenizer_model_from_events(&events, &factory, &params, config, stamp)?;
                Ok(Trained { events: events.len(), model: t.model, report: t.report, package: t.package })
            }
            Component::Pos => {
                let factory = self.pos_factory()?;
                let mut stream = postag::PosEventStream::new(&factory)?;
                let mut events = Vec::new();
                for sample in ConlluPosReader::new(conllu, TAGSET)? {
                    stream.events(&sample?, &mut events);
                }
                let t = postag::train_pos_model_from_events(&events, &factory, &params, config, stamp)?;
                Ok(Trained { events: events.len(), model: t.model, report: t.report, package: t.package })
            }
            Component::Lemmas => {
                let factory = self.lemmatizer_factory();
                let mut stream = lemmatize::LemmaEventStream::new(&factory);
                let mut events = Vec::new();
                for sample in ConlluLemmaReader::new(conllu, TAGSET)? {
                    stream.events(&sample?, &mut events)?;
                }
                let t = lemmatize::train_lemmatizer_model_from_events(&events, &factory, &params, config, stamp)?;
                Ok(Trained { events: events.len(), model: t.model, report: t.report, package: t.package })
            }
        }
    }

    /// Trains from the treebank file and compares with the published
    /// file, taking the timestamp from the latter.
    pub fn reproduce(
        &self,
        treebank: &Path,
        published: &Path,
        config: &mut TrainingConfiguration,
    ) -> Result<(Trained, Comparison)> {
        let read = |p: &Path| std::fs::read(p).map_err(|e| Error::Io { path: p.to_path_buf(), message: e.to_string() });
        let conllu = String::from_utf8(read(treebank)?)
            .map_err(|e| Error::Io { path: treebank.to_path_buf(), message: e.to_string() })?;
        let published = ModelFile::open(read(published)?).map_err(Error::Model)?;
        let timestamp: i64 = published
            .manifest()
            .get("Timestamp")
            .ok_or(Error::MissingManifestEntry("Timestamp"))?
            .parse()
            .map_err(|_| Error::MissingManifestEntry("Timestamp"))?;
        let trained = self.train(&conllu, timestamp, config)?;
        drop(conllu);
        let comparison = Comparison::of(self, &trained, &published)?;
        Ok((trained, comparison))
    }
}

/// The eighteen published models: sentence detectors for three languages,
/// and tokenizers, taggers and lemmatizers for five.
pub const RECIPES: [Recipe; 18] = [
    Recipe { language: "en", treebank_name: "ewt", component: Component::Sentence, treebank: ENGLISH_EWT },
    Recipe { language: "en", treebank_name: "ewt", component: Component::Tokens, treebank: ENGLISH_EWT },
    Recipe { language: "en", treebank_name: "ewt", component: Component::Pos, treebank: ENGLISH_EWT },
    Recipe { language: "en", treebank_name: "ewt", component: Component::Lemmas, treebank: ENGLISH_EWT },
    Recipe { language: "de", treebank_name: "gsd", component: Component::Sentence, treebank: GERMAN_GSD },
    Recipe { language: "de", treebank_name: "gsd", component: Component::Tokens, treebank: GERMAN_GSD },
    Recipe { language: "de", treebank_name: "gsd", component: Component::Pos, treebank: GERMAN_GSD },
    Recipe { language: "de", treebank_name: "gsd", component: Component::Lemmas, treebank: GERMAN_GSD },
    Recipe { language: "pt", treebank_name: "gsd", component: Component::Sentence, treebank: PORTUGUESE_GSD },
    Recipe { language: "pt", treebank_name: "gsd", component: Component::Tokens, treebank: PORTUGUESE_GSD },
    Recipe { language: "pt", treebank_name: "gsd", component: Component::Pos, treebank: PORTUGUESE_GSD },
    Recipe { language: "pt", treebank_name: "gsd", component: Component::Lemmas, treebank: PORTUGUESE_GSD },
    Recipe { language: "es", treebank_name: "gsd", component: Component::Tokens, treebank: SPANISH_GSD },
    Recipe { language: "es", treebank_name: "gsd", component: Component::Pos, treebank: SPANISH_GSD },
    Recipe { language: "es", treebank_name: "gsd", component: Component::Lemmas, treebank: SPANISH_GSD },
    Recipe { language: "fr", treebank_name: "gsd", component: Component::Tokens, treebank: FRENCH_GSD },
    Recipe { language: "fr", treebank_name: "gsd", component: Component::Pos, treebank: FRENCH_GSD },
    Recipe { language: "fr", treebank_name: "gsd", component: Component::Lemmas, treebank: FRENCH_GSD },
];

/// A model trained from a recipe.
#[derive(Debug, Clone)]
pub struct Trained {
    /// How many events the stream made.
    pub events: usize,
    pub model: GisModel,
    pub report: turbo_train::TrainingReport,
    pub package: turbo_train::ModelPackage,
}

/// Whether a trained value agrees with the published one.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Agreement<T> {
    Same(T),
    Differ { published: T, trained: T },
}

impl<T> Agreement<T> {
    fn of(published: T, trained: T) -> Agreement<T>
    where
        T: PartialEq,
    {
        if published == trained { Agreement::Same(trained) } else { Agreement::Differ { published, trained } }
    }

    pub fn is_same(&self) -> bool {
        matches!(self, Agreement::Same(_))
    }
}

/// A model entry's size and SHA-256.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EntryDigest {
    pub len: usize,
    pub sha256: String,
}

impl EntryDigest {
    fn of(bytes: &[u8]) -> EntryDigest {
        EntryDigest { len: bytes.len(), sha256: sha256(bytes).to_hex() }
    }
}

/// How far the parameters of two models of the same predicates and
/// patterns are apart.
#[derive(Debug, Clone, PartialEq)]
pub struct ParameterDistance {
    pub total: usize,
    /// How many parameters are not bit for bit the published ones.
    pub differing: usize,
    /// The widest gap in units in the last place.
    pub max_ulps: u64,
    /// The widest gap as a number.
    pub max_abs: f64,
}

/// A trained model against the published one: the event checksum, the
/// model entry, the manifest text and the zip entries. The zip container
/// itself is not compared: the JVM deflates its entries and this port
/// stores them.
#[derive(Debug, Clone, PartialEq)]
pub struct Comparison {
    pub model: String,
    pub events: usize,
    pub event_hash: Agreement<u32>,
    pub model_entry: Agreement<EntryDigest>,
    /// The first byte at which the model entries differ, when they do.
    pub first_difference: Option<usize>,
    /// When the entries differ: whether the published model has the same
    /// predicates with the same outcome patterns, and if so how far the
    /// parameters are apart. `None` when the entries agree.
    pub parameters: Option<std::result::Result<ParameterDistance, String>>,
    pub manifest: Agreement<String>,
    pub entry_names: Agreement<Vec<String>>,
}

impl Comparison {
    fn of(recipe: &Recipe, trained: &Trained, published: &ModelFile) -> Result<Comparison> {
        let hash: u32 = published
            .manifest()
            .get("Training-Eventhash")
            .ok_or(Error::MissingManifestEntry("Training-Eventhash"))?
            .parse()
            .map_err(|_| Error::MissingManifestEntry("Training-Eventhash"))?;
        let entry = recipe.component.model_entry();
        let theirs = published.entry(entry).map_err(Error::Model)?;
        let ours = trained
            .package
            .entries()
            .find(|(n, _)| *n == entry)
            .map(|(_, b)| b)
            .expect("the package holds its model entry");
        let first_difference = theirs
            .iter()
            .zip(ours)
            .position(|(a, b)| a != b)
            .or_else(|| (theirs.len() != ours.len()).then_some(theirs.len().min(ours.len())));
        let parameters = first_difference.map(|_| {
            let theirs = published.maxent(entry).map_err(|e| e.to_string())?;
            parameter_distance(&trained.model, &theirs)
        });
        let their_manifest =
            String::from_utf8_lossy(&published.entry(MANIFEST_ENTRY).map_err(Error::Model)?).into_owned();
        let their_names: Vec<String> =
            published.archive().names().map(|n| String::from_utf8_lossy(n).into_owned()).collect();
        let our_names: Vec<String> = trained.package.entry_names().into_iter().map(str::to_owned).collect();
        Ok(Comparison {
            model: recipe.model_id(),
            events: trained.events,
            event_hash: Agreement::of(hash, trained.report.event_hash.expect("trained from events")),
            model_entry: Agreement::of(EntryDigest::of(&theirs), EntryDigest::of(ours)),
            first_difference,
            parameters,
            manifest: Agreement::of(their_manifest, trained.package.manifest().to_properties()),
            entry_names: Agreement::of(their_names, our_names),
        })
    }

    /// Whether every compared part agrees, the model entry byte for byte.
    pub fn reproduced(&self) -> bool {
        self.event_hash.is_same() && self.model_entry.is_same() && self.manifest.is_same() && self.entry_names.is_same()
    }

    /// Whether everything but the parameters' last bits agrees: the same
    /// events, manifest and zip entries, the same predicates and patterns,
    /// and no parameter further than `max_abs` from the published one. The
    /// published models differ from this port in their parameters' last
    /// bits for a reason not yet known (the module comment says how far),
    /// so this is the agreement they can show.
    pub fn agrees_within(&self, max_abs: f64) -> bool {
        let parameters = match &self.parameters {
            None => self.model_entry.is_same(),
            // A NaN distance (a NaN parameter on either side) is no
            // agreement; the comparison is false for it.
            Some(Ok(d)) => d.max_abs <= max_abs,
            Some(Err(_)) => false,
        };
        self.event_hash.is_same() && self.manifest.is_same() && self.entry_names.is_same() && parameters
    }

    /// A few lines for a report.
    pub fn report(&self) -> String {
        let mut s = format!("{}: {} events\n", self.model, self.events);
        match &self.event_hash {
            Agreement::Same(h) => s.push_str(&format!("  event hash {h}: same\n")),
            Agreement::Differ { published, trained } => {
                s.push_str(&format!("  event hash: published {published}, trained {trained}\n"));
            }
        }
        match &self.model_entry {
            Agreement::Same(d) => s.push_str(&format!("  model entry {} bytes {}: same\n", d.len, d.sha256)),
            Agreement::Differ { published, trained } => s.push_str(&format!(
                "  model entry: published {} bytes {}, trained {} bytes {}, first difference at byte {}\n",
                published.len,
                published.sha256,
                trained.len,
                trained.sha256,
                self.first_difference.map_or("none".to_owned(), |i| i.to_string())
            )),
        }
        match &self.parameters {
            None => {}
            Some(Ok(d)) => s.push_str(&format!(
                "  same predicates and patterns; {} of {} parameters differ, at most {} ulps, {:e} in value\n",
                d.differing, d.total, d.max_ulps, d.max_abs
            )),
            Some(Err(why)) => s.push_str(&format!("  parameters not comparable: {why}\n")),
        }
        match &self.manifest {
            Agreement::Same(_) => s.push_str("  manifest: same\n"),
            Agreement::Differ { published, trained } => {
                s.push_str("  manifest differs:\n");
                // Every line of either, so extra lines on one side show.
                let (mut p, mut t) = (published.lines(), trained.lines());
                loop {
                    match (p.next(), t.next()) {
                        (None, None) => break,
                        (a, b) if a == b => {}
                        (a, b) => s.push_str(&format!(
                            "    published: {}\n    trained:   {}\n",
                            a.unwrap_or("(no line)"),
                            b.unwrap_or("(no line)")
                        )),
                    }
                }
            }
        }
        match &self.entry_names {
            Agreement::Same(_) => s.push_str("  zip entries: same names in the same order\n"),
            Agreement::Differ { published, trained } => {
                s.push_str(&format!("  zip entries: published {published:?}, trained {trained:?}\n"));
            }
        }
        s
    }
}

/// The distance between the trained parameters and the published model's,
/// matched by predicate name and outcome label, or why they cannot be
/// matched.
fn parameter_distance(ours: &GisModel, theirs: &MaxentModel) -> std::result::Result<ParameterDistance, String> {
    if ours.num_predicates() != theirs.num_predicates() {
        return Err(format!("{} predicates trained, {} published", ours.num_predicates(), theirs.num_predicates()));
    }
    let mut d = ParameterDistance { total: 0, differing: 0, max_ulps: 0, max_abs: 0.0 };
    for (p, label) in ours.pred_labels().iter().enumerate() {
        let Some(q) = theirs.predicate_index(label.as_bytes()) else {
            return Err(format!("the published model lacks the predicate {}", label.to_string_lossy()));
        };
        let our_outcomes: Vec<&str> =
            ours.pattern(p).iter().map(|&o| ours.outcome_labels()[o as usize].as_str()).collect();
        let their_outcomes: Vec<&str> = theirs.pattern(q).iter().map(|&o| theirs.outcome(o as usize)).collect();
        if our_outcomes != their_outcomes {
            return Err(format!(
                "the predicate {} has the pattern {our_outcomes:?} trained and {their_outcomes:?} published",
                label.to_string_lossy()
            ));
        }
        for (&a, &b) in ours.parameters(p).iter().zip(theirs.parameters(q)) {
            d.total += 1;
            if a.to_bits() != b.to_bits() {
                d.differing += 1;
                d.max_ulps = d.max_ulps.max(ulps_apart(a, b));
                let gap = (a - b).abs();
                // Not `f64::max`, which would drop a NaN gap: a NaN wins
                // and stays, so `agrees_within` sees it.
                if gap.is_nan() || d.max_abs.is_nan() || gap > d.max_abs {
                    d.max_abs = if d.max_abs.is_nan() { d.max_abs } else { gap };
                }
            }
        }
    }
    Ok(d)
}

/// How many representable numbers lie between `a` and `b`: the distance
/// of their bit patterns once each is mapped from sign and magnitude to a
/// number line, so that two values on either side of zero are as far
/// apart as the numbers between them and not half the bit space.
fn ulps_apart(a: f64, b: f64) -> u64 {
    fn ordered(x: f64) -> i64 {
        let bits = x.to_bits() as i64;
        if bits < 0 { i64::MIN - bits } else { bits }
    }
    ordered(a).abs_diff(ordered(b))
}

/// An index of the treebank datasets alone, for a zoo that fetches them.
pub fn datasets() -> Index {
    let artifacts: Vec<Artifact> = TREEBANKS.iter().map(Treebank::dataset).collect();
    Index::parse(Index::to_canonical_json(&artifacts).as_bytes()).expect("the recipes make a valid index")
}

/// The zoo's home as the `turbo-zoo` command finds it: `$TURBO_ZOO_HOME`,
/// else `$XDG_DATA_HOME/turbo-zoo`, else `~/.local/share/turbo-zoo`.
pub fn zoo_home() -> Option<PathBuf> {
    let var = |k: &str| std::env::var_os(k).filter(|v| !v.is_empty()).map(PathBuf::from);
    var("TURBO_ZOO_HOME")
        .or_else(|| var("XDG_DATA_HOME").map(|x| x.join("turbo-zoo")))
        .or_else(|| var("HOME").map(|h| h.join(".local/share/turbo-zoo")))
}

/// Fetches a treebank's `train` file into the zoo cache under `home`,
/// if `user` has accepted its licence there (`turbo-zoo accept
/// CC-BY-SA-4.0 --yes`), and returns its path.
pub fn fetch_treebank(treebank: &Treebank, home: &Path, user: &UserId) -> Result<PathBuf> {
    let zoo = Zoo::new(datasets(), home.join("artifacts"), Policy::default(), &Http);
    let acceptances = home.join("acceptances.json");
    let fetched = zoo.fetch_with(&treebank.dataset_id(), user, || Acceptances::load(&acceptances))?;
    Ok(fetched.single().expect("one file per treebank").to_path_buf())
}

/// Fetches a published model into the zoo cache under `home` through the
/// zoo's built-in index, where every UD 1.3 model is listed under
/// [`Recipe::model_id`] (Apache License 2.0, nothing to accept), and
/// returns its path.
pub fn fetch_published_model(recipe: &Recipe, home: &Path, user: &UserId) -> Result<PathBuf> {
    let zoo = Zoo::new(Index::builtin(), home.join("artifacts"), Policy::default(), &Http);
    let acceptances = home.join("acceptances.json");
    let fetched = zoo.fetch_with(&recipe.model_id(), user, || Acceptances::load(&acceptances))?;
    Ok(fetched.single().expect("one file per model").to_path_buf())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ulps_count_the_numbers_between_two_values_across_zero() {
        assert_eq!(ulps_apart(1.0, 1.0), 0);
        assert_eq!(ulps_apart(1.0, f64::from_bits(1.0f64.to_bits() + 3)), 3);
        assert_eq!(ulps_apart(0.0, -0.0), 0);
        let tiny = f64::from_bits(2);
        assert_eq!(ulps_apart(tiny, -tiny), 4);
        assert_eq!(ulps_apart(-1.0, -f64::from_bits(1.0f64.to_bits() + 1)), 1);
    }

    fn comparison_with(parameters: Option<std::result::Result<ParameterDistance, String>>) -> Comparison {
        let digest = EntryDigest { len: 1, sha256: String::new() };
        Comparison {
            model: String::new(),
            events: 0,
            event_hash: Agreement::Same(1),
            model_entry: Agreement::Differ { published: digest.clone(), trained: digest },
            first_difference: Some(0),
            parameters,
            manifest: Agreement::Same(String::new()),
            entry_names: Agreement::Same(Vec::new()),
        }
    }

    #[test]
    fn a_nan_distance_is_no_agreement_and_the_report_shows_every_manifest_line() {
        let close = ParameterDistance { total: 2, differing: 1, max_ulps: 1, max_abs: 1e-15 };
        assert!(comparison_with(Some(Ok(close.clone()))).agrees_within(1e-12));
        let nan = ParameterDistance { max_abs: f64::NAN, ..close.clone() };
        assert!(!comparison_with(Some(Ok(nan))).agrees_within(1e-12));
        assert!(!comparison_with(Some(Err("shapes".to_owned()))).agrees_within(1e-12));
        assert!(!comparison_with(None).agrees_within(1e-12));
        let mut c = comparison_with(Some(Ok(close)));
        c.manifest = Agreement::Differ { published: "a=1\nb=2\nc=3".to_owned(), trained: "a=1\nb=x".to_owned() };
        let report = c.report();
        assert!(report.contains("    published: b=2\n    trained:   b=x\n"), "{report}");
        assert!(report.contains("    published: c=3\n    trained:   (no line)\n"), "{report}");
    }

    /// Trains the English sentence recipe on a few CoNLL-U sentences with
    /// `iterations` iterations.
    fn small_training(iterations: i32) -> Trained {
        let conllu = "# text = Dr. Who left.\n1\tDr.\t_\t_\t_\t_\t_\t_\t_\t_\n2\tWho\t_\t_\t_\t_\t_\t_\t_\t_\n\
                      3\tleft.\t_\t_\t_\t_\t_\t_\t_\t_\n\n\
                      # text = Two? Yes!\n1\tTwo?\t_\t_\t_\t_\t_\t_\t_\t_\n2\tYes!\t_\t_\t_\t_\t_\t_\t_\t_\n\n\
                      # text = Mr. Smith came.\n1\tMr.\t_\t_\t_\t_\t_\t_\t_\t_\n2\tSmith\t_\t_\t_\t_\t_\t_\t_\t_\n\
                      3\tcame.\t_\t_\t_\t_\t_\t_\t_\t_\n\n";
        let recipe = RECIPES[0];
        let factory = recipe.sentence_factory().unwrap();
        let mut events = Vec::new();
        let mut stream = sentdetect::SdEventStream::new(&factory, WHITESPACE);
        for sample in ConlluSentenceReader::new(conllu, SENTENCES_PER_SAMPLE).unwrap() {
            stream.events(&sample.unwrap(), &mut events);
        }
        let mut params = training_parameters();
        params.put(Key::ITERATIONS, iterations);
        params.put(Key::CUTOFF, 0);
        params.put(Key::THREADS, 1);
        let stamp = Stamp { timestamp_ms: 1_749_698_375_454, version: OPENNLP_VERSION, zone: ZONE };
        let t = sentdetect::train_sentence_model_from_events(
            &events,
            &factory,
            &params,
            &mut TrainingConfiguration::default(),
            stamp,
        )
        .unwrap();
        Trained { events: events.len(), model: t.model, report: t.report, package: t.package }
    }

    #[test]
    fn a_model_compared_with_itself_is_reproduced_and_with_another_run_is_not() {
        let recipe = RECIPES[0];
        let ten = small_training(10);
        assert_eq!(ten.events, 6);
        let published = ModelFile::open(ten.package.to_bytes()).unwrap();
        let same = Comparison::of(&recipe, &ten, &published).unwrap();
        assert!(same.reproduced() && same.agrees_within(0.0), "{}", same.report());
        assert!(same.parameters.is_none() && same.first_difference.is_none());
        assert_eq!(same.model, "opennlp-en-ud-ewt-sentence-1.3-2.5.4");

        let twenty = small_training(20);
        let other = Comparison::of(&recipe, &twenty, &published).unwrap();
        assert!(!other.reproduced() && !other.agrees_within(1e-12), "{}", other.report());
        assert!(other.event_hash.is_same() && other.manifest.is_same() && other.entry_names.is_same());
        let distance = other.parameters.as_ref().unwrap().as_ref().unwrap();
        assert_eq!(distance.total, ten.model.all_parameters().len());
        assert!(distance.differing > 0 && distance.max_ulps > 0 && distance.max_abs > 1e-12);
        assert!(other.report().contains("same predicates and patterns;"), "{}", other.report());

        // The published model's predicates, patterns and parameters are read
        // back as the trained model holds them.
        let theirs = published.maxent(sentdetect::MODEL_ENTRY).unwrap();
        for (p, label) in ten.model.pred_labels().iter().enumerate() {
            let q = theirs.predicate_index(label.as_bytes()).unwrap();
            let ours: Vec<&str> =
                ten.model.pattern(p).iter().map(|&o| ten.model.outcome_labels()[o as usize].as_str()).collect();
            let published: Vec<&str> = theirs.pattern(q).iter().map(|&o| theirs.outcome(o as usize)).collect();
            assert_eq!(ours, published);
            assert_eq!(ten.model.parameters(p), theirs.parameters(q));
        }
    }

    #[test]
    fn the_datasets_make_a_valid_share_alike_index() {
        let index = datasets();
        assert_eq!(index.list().len(), 5);
        let ewt = index.get("ud-english-ewt-r2.16-train").unwrap();
        assert_eq!(ewt.class, LicenceClass::ShareAlike);
        assert_eq!(ewt.files[0].size, 14203025);
        assert_eq!(
            ewt.files[0].url.as_str(),
            "https://raw.githubusercontent.com/UniversalDependencies/UD_English-EWT/r2.16/en_ewt-ud-train.conllu"
        );
        assert_eq!(
            ewt.license_terms.as_str(),
            "https://github.com/UniversalDependencies/UD_English-EWT/blob/r2.16/README.md"
        );
        // Fetching one without an acceptance is refused before any transfer.
        let zoo = Zoo::new(datasets(), PathBuf::from("/nonexistent"), Policy::default(), &turbo_zoo::Offline);
        let user: UserId = "someone".parse().unwrap();
        assert!(zoo.permit("ud-english-ewt-r2.16-train", &user, &Acceptances::default()).is_err());
    }

    #[test]
    fn the_recipes_name_the_published_files_and_the_training_settings() {
        let files: Vec<String> = RECIPES.iter().map(Recipe::model_file).collect();
        assert_eq!(
            files,
            [
                "opennlp-en-ud-ewt-sentence-1.3-2.5.4.bin",
                "opennlp-en-ud-ewt-tokens-1.3-2.5.4.bin",
                "opennlp-en-ud-ewt-pos-1.3-2.5.4.bin",
                "opennlp-en-ud-ewt-lemmas-1.3-2.5.4.bin",
                "opennlp-de-ud-gsd-sentence-1.3-2.5.4.bin",
                "opennlp-de-ud-gsd-tokens-1.3-2.5.4.bin",
                "opennlp-de-ud-gsd-pos-1.3-2.5.4.bin",
                "opennlp-de-ud-gsd-lemmas-1.3-2.5.4.bin",
                "opennlp-pt-ud-gsd-sentence-1.3-2.5.4.bin",
                "opennlp-pt-ud-gsd-tokens-1.3-2.5.4.bin",
                "opennlp-pt-ud-gsd-pos-1.3-2.5.4.bin",
                "opennlp-pt-ud-gsd-lemmas-1.3-2.5.4.bin",
                "opennlp-es-ud-gsd-tokens-1.3-2.5.4.bin",
                "opennlp-es-ud-gsd-pos-1.3-2.5.4.bin",
                "opennlp-es-ud-gsd-lemmas-1.3-2.5.4.bin",
                "opennlp-fr-ud-gsd-tokens-1.3-2.5.4.bin",
                "opennlp-fr-ud-gsd-pos-1.3-2.5.4.bin",
                "opennlp-fr-ud-gsd-lemmas-1.3-2.5.4.bin",
            ]
        );
        // Every model is in the zoo's index under its id.
        let index = Index::builtin();
        for r in &RECIPES {
            assert!(index.get(&r.model_id()).is_ok(), "{}", r.model_id());
        }
        let p = training_parameters();
        assert_eq!(p.get_int(Key::ITERATIONS).unwrap(), Some(300));
        assert_eq!(CONF_ITERATIONS, 150);
        assert_eq!(p.get_int(Key::CUTOFF).unwrap(), Some(5));
        assert_eq!(p.get_int(Key::THREADS).unwrap(), Some(4));
        assert_eq!(p.algorithm().unwrap(), Some("MAXENT"));
        let pt = RECIPES[8].sentence_factory().unwrap();
        assert_eq!(pt.eos_string(), ".?!;:()\u{AB}\u{BB}'\"");
        let de = RECIPES[5].tokenizer_factory().unwrap();
        assert!(!de.use_alpha_numeric_optimization());
        assert_eq!(de.alpha_numeric_pattern(), turbo_tokenize::me::alnum::language_pattern("de"));
        let pos = RECIPES[2].pos_factory().unwrap();
        assert_eq!(pos.descriptor(), turbo_sequence::postag::DEFAULT_DESCRIPTOR.as_bytes());
        assert!(pos.tag_dictionary().is_none());
        assert_eq!(RECIPES[3].lemmatizer_factory().language(), "en");
        assert_eq!(RECIPES[6].component.model_entry(), "pos.model");
        assert_eq!(RECIPES[7].component.model_entry(), "lemmatizer.model");
    }

    #[test]
    fn the_tagger_and_lemmatizer_recipes_train_from_conllu_and_compare_with_themselves() {
        let two = "# text = Dogs ran.\n\
            1\tDogs\tdog\tNOUN\tNNS\t_\t2\tnsubj\t_\t_\n\
            2\tran\trun\tVERB\tVBD\t_\t0\troot\t_\tSpaceAfter=No\n\
            3\t.\t.\tPUNCT\t.\t_\t2\tpunct\t_\t_\n\n\
            # text = A dog runs.\n\
            1\tA\ta\tDET\tDT\t_\t2\tdet\t_\t_\n\
            2\tdog\tdog\tNOUN\tNN\t_\t3\tnsubj\t_\t_\n\
            3\truns\trun\tVERB\tVBZ\t_\t0\troot\t_\tSpaceAfter=No\n\
            4\t.\t.\tPUNCT\t.\t_\t3\tpunct\t_\t_\n\n";
        // Five times, so every feature passes the recipe's cutoff of 5.
        let conllu = two.repeat(5);
        let mut config = TrainingConfiguration::default();
        let pos = RECIPES[2].train(&conllu, 1_749_698_398_335, &mut config).unwrap();
        assert_eq!(pos.events, 35);
        let outcomes: Vec<&str> = pos.model.outcome_labels().iter().map(String::as_str).collect();
        assert_eq!(outcomes, ["NOUN", "VERB", "PUNCT", "DET"]);
        assert_eq!(pos.package.entry_names(), ["manifest.properties", "generator.featuregen", "pos.model"]);
        let manifest = pos.package.manifest().to_properties();
        assert!(manifest.contains("Component-Name=POSTaggerME\n"), "{manifest}");
        assert!(manifest.contains("BeamSize=3\n") && manifest.contains("Language=en\n"), "{manifest}");
        let published = ModelFile::open(pos.package.to_bytes()).unwrap();
        let same = Comparison::of(&RECIPES[2], &pos, &published).unwrap();
        assert!(same.reproduced(), "{}", same.report());

        let lemmas = RECIPES[3].train(&conllu, 1_749_698_430_433, &mut config).unwrap();
        assert_eq!(lemmas.events, 35);
        let outcomes: Vec<&str> = lemmas.model.outcome_labels().iter().map(String::as_str).collect();
        assert_eq!(outcomes, ["D0s", "R1au", "O"]);
        assert_eq!(lemmas.package.entry_names(), ["manifest.properties", "lemmatizer.model"]);
        assert_eq!(lemmas.package.manifest().component_name(), "StatisticalLemmatizer");
        let published = ModelFile::open(lemmas.package.to_bytes()).unwrap();
        let same = Comparison::of(&RECIPES[3], &lemmas, &published).unwrap();
        assert!(same.reproduced(), "{}", same.report());
    }
}
