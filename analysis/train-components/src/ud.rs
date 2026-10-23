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
//! A treebank's dataset takes its licence class from its licence
//! (`turbo_zoo::class_of`): the five treebanks here are CC BY-SA 4.0, so
//! their class is share-alike and they are fetched through the zoo, by
//! digest, after the user accepts that licence; a CC BY 4.0 treebank comes
//! out permissive, with nothing to accept. They are never checked in. The
//! reproduction itself is the test `tests/reproduce.rs`, ignored by
//! default.

mod compare;

use std::path::{Path, PathBuf};

use turbo_formats::conllu::{ConlluLemmaReader, ConlluPosReader, ConlluSentenceReader, ConlluTokenReader, Tagset};
use turbo_maxent::ModelFile;
use turbo_train::{GisModel, Key, TimeZone, TrainingConfiguration, TrainingParameters};
use turbo_unicode::WhitespaceMode;
use turbo_zoo::{
    Acceptances, Artifact, Http, Index, Kind, Licence, Policy, UserId, Zoo, class_of, index::Format as ZooFormat,
};

use crate::error::{Error, Result};
use crate::lemmatize::{self, LemmatizerFactory};
use crate::package::Stamp;
use crate::postag::{self, PosTaggerFactory};
use crate::sentdetect::{self, SentenceDetectorFactory};
use crate::tokenize::{self, TokenizerFactory};

pub use compare::{Agreement, Comparison, EntryDigest, ParameterDistance};

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

    /// The treebank's `train` file as a zoo artifact, its class computed
    /// from its licence.
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
            class: class_of([self.licence]),
            attribution: Some(format!(
                "{} {UD_RELEASE} ({}, {}).",
                self.repository,
                self.licence,
                self.licence_terms()
            )),
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
                    stream.events(&sample?, &mut events);
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

/// Fetches a treebank's `train` file into the zoo cache under `home` and
/// returns its path. A share-alike treebank needs `user`'s acceptance of
/// its licence there first (`turbo-zoo accept CC-BY-SA-4.0 --yes` for the
/// five in [`TREEBANKS`]); a permissive one needs none.
pub fn fetch_treebank(treebank: &Treebank, home: &Path, user: &UserId) -> Result<PathBuf> {
    let zoo = Zoo::new(datasets(), home.join("artifacts"), Policy::default(), &Http);
    let acceptances = home.join("acceptances.json");
    let fetched = zoo.fetch_with(&treebank.dataset_id(), user, || Acceptances::load(&acceptances))?;
    Ok(fetched.single().expect("one file per treebank").to_path_buf())
}

/// Fetches a published model into the zoo cache under `home` through the
/// zoo's built-in index, where every UD 1.3 model is listed under
/// [`Recipe::model_id`], and returns its path. The models are Apache
/// License 2.0, but the index classes each by its treebank too: the ones
/// trained on a CC BY-SA treebank, every model of [`RECIPES`] among them,
/// are share-alike, so the fetch needs `user`'s acceptance of
/// CC-BY-SA-4.0 under `home` (`turbo-zoo accept CC-BY-SA-4.0 --yes`);
/// only the Catalan AnCora models (CC BY 4.0) are permissive.
pub fn fetch_published_model(recipe: &Recipe, home: &Path, user: &UserId) -> Result<PathBuf> {
    let zoo = Zoo::new(Index::builtin(), home.join("artifacts"), Policy::default(), &Http);
    let acceptances = home.join("acceptances.json");
    let fetched = zoo.fetch_with(&recipe.model_id(), user, || Acceptances::load(&acceptances))?;
    Ok(fetched.single().expect("one file per model").to_path_buf())
}

#[cfg(test)]
mod tests;
