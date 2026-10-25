//! The model card of a model a UD recipe trained, and its zoo entry.
//!
//! A trained model's directory holds the model file, `card.json` (this
//! card) and `gate.json`, the gate record ([`ModelGateRecord`]) of the
//! version the MASC recipes write. The card says what the model was
//! trained from and with, the treebank's licence, the attribution and
//! citation it asks for, what the gate read from the `test` file and how
//! the gate went, the last two in the MASC card's terms ([`HeldOutCard`],
//! [`GateCard`]). It is strict JSON: every field is required and an
//! unknown field is an error.

use std::path::Path;

use serde::{Deserialize, Serialize};
use turbo_eval::model_gate::ModelGateRecord;
use turbo_train_components::ud as published;
use turbo_zoo::{Artifact, File, Kind, Licence, LicenceClass, Sha256, Sha512, TrainingData, class_of, index::Format};

use crate::Trained;
use crate::card::{
    GateCard, HeldOutCard, LeftOutFor, MODEL_LICENCE, MODEL_LICENCE_TERMS, MODEL_VERSION, ModelFileCard,
    ParametersCard, Summary, TagFormat, TrainingCard, publishable_class, read_card, write_dir,
};
use crate::error::{Error, Result};
use crate::postag::CANDIDATE_FORMAT;
use crate::segmentation::TIMESTAMP_MS;
use crate::treebank::{Grant, RELEASE, TextRights, Treebank, TreebankFile};
use crate::ud::{SENTENCES_PER_SAMPLE, TAG_FORMAT, UdRecipe};

/// The schema version of [`UdModelCard`]: 4, with the training counts,
/// held-out counts and gate of the MASC card's version 5
/// ([`TrainingCard`], [`HeldOutCard`], [`GateCard`]), whose chunk and name
/// counts a UD card leaves `null`, and a lemmatizer's held-out tokens
/// counted as a tagger's are ([`crate::ud::ReferenceClasses`]).
pub const UD_CARD_VERSION: u32 = 4;

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
    pub other_licence_offered: Option<Licence>,
    pub licence_file: TreebankFileCard,
    pub readme: TreebankFileCard,
    /// UD's own split: the model is trained on `train` and gated on
    /// `test`; `dev` is checked against its digest and not read.
    pub train: TreebankFileCard,
    pub dev: TreebankFileCard,
    pub test: TreebankFileCard,
}

impl TreebankCard {
    /// The record of `t` at the release the recipes read.
    pub fn of(t: &Treebank) -> TreebankCard {
        TreebankCard {
            repository: t.repository.to_owned(),
            release: RELEASE.to_owned(),
            license: crate::treebank::LICENCE,
            license_terms: t.licence_terms(),
            other_licence_offered: match t.grant {
                Grant::CcBy40 => None,
                Grant::CcBy40OrOther { other } => Some(other),
            },
            licence_file: TreebankFileCard::of(&t.licence_file),
            readme: TreebankFileCard::of(&t.readme),
            train: TreebankFileCard::of(&t.train),
            dev: TreebankFileCard::of(&t.dev),
            test: TreebankFileCard::of(&t.test),
        }
    }
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
    /// What the gate read from `test`: the sentences of a tagger's or a
    /// lemmatizer's samples with no word once empty nodes are left out are
    /// the ones skipped (none for the sentence detector and the
    /// tokenizer), and for a tagger or a lemmatizer, how its tokens were
    /// scored.
    pub held_out: HeldOutCard,
    pub gate: GateCard,
}

impl UdModelCard {
    /// The card of a model `recipe` trained, gated by `gate` on what
    /// `held_out` describes.
    pub fn new(recipe: &UdRecipe, trained: &Trained, held_out: HeldOutCard, gate: &ModelGateRecord) -> UdModelCard {
        let t = &recipe.treebank;
        let trained_on = vec![t.training_data()];
        let class = class_of(std::iter::once(MODEL_LICENCE).chain(trained_on.iter().map(|d| d.license)));
        let segmentation = matches!(recipe.component, published::Component::Sentence | published::Component::Tokens);
        let (candidate_tag_format, reference_tag_format) = match recipe.component {
            published::Component::Pos => (TagFormat::of(CANDIDATE_FORMAT), TagFormat::of(TAG_FORMAT)),
            _ => (None, None),
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
            treebank: TreebankCard::of(t),
            training: TrainingCard {
                parameters: ParametersCard {
                    algorithm: "MAXENT".to_owned(),
                    iterations: published::ITERATIONS,
                    cutoff: published::CUTOFF,
                    threads: crate::ud::THREADS,
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
                names: None,
                chunks: None,
            },
            held_out,
            gate: GateCard::of(gate, candidate_tag_format, reference_tag_format),
        }
    }

    /// The model's zoo entry once the file is published at `url`.
    ///
    /// # Errors
    ///
    /// [`Error::NotShippable`] when the model may not leave the machine it
    /// was trained on ([`publishable_class`]), no recipe of
    /// [`crate::ud::UD_RECIPES`] trains the model, the rights to its
    /// treebank's text are not settled ([`TextRights::NotSettled`], whatever
    /// its gate says), the gate was not run against the recipe's reference
    /// (its id and SHA-256), the gate failed, or the card's treebank and
    /// training data are not the recipe's treebank at the release, as
    /// recorded with its files' sizes and digests; [`Error::InvalidCard`] for
    /// a `class` other than the one the licences give, a component or
    /// language other than its recipe's, or an address or field the index
    /// refuses.
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
        let recipe = UdRecipe::trains(&self.id).ok_or(not("no recipe trains it"))?;
        if let TextRights::NotSettled { .. } = recipe.treebank.text_rights {
            return Err(not("the rights to its treebank's text are not settled"));
        }
        if self.gate.reference != recipe.reference.id() || self.gate.reference_sha256 != recipe.reference.sha256()? {
            return Err(not("it was not gated against its recipe's reference"));
        }
        if !self.gate.pass {
            return Err(not("it failed its gate"));
        }
        let t = &recipe.treebank;
        if self.trained_on != [t.training_data()] || self.treebank != TreebankCard::of(t) {
            return Err(not("it was not trained and gated on its recipe's treebank"));
        }
        let bad = |field: &'static str, message: String| Error::InvalidCard { field, message };
        if self.component != recipe.zoo_component() {
            return Err(bad(
                "component",
                format!("{} is not {}, its recipe's", self.component, recipe.zoo_component()),
            ));
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
            }],
        })
    }

    /// The card as written: pretty JSON with a final newline.
    pub fn to_json(&self) -> String {
        let mut s = serde_json::to_string_pretty(self).expect("a card serializes");
        s.push('\n');
        s
    }

    /// Reads a card written by [`write_ud_model_dir`], its `card_version`
    /// first, as [`crate::card::ModelCard::read`] reads a MASC card.
    ///
    /// # Errors
    ///
    /// A file that cannot be read, is not JSON with a `card_version`, is of
    /// another version ([`Error::CardVersion`]), or is not a card of this
    /// version.
    pub fn read(path: &Path) -> Result<UdModelCard> {
        read_card(path, UD_CARD_VERSION)
    }

    /// What a run prints of the card before its gate record, as for a MASC
    /// card ([`Summary`]).
    pub fn summary(&self) -> Summary<'_> {
        let left_out_for = match self.component {
            turbo_zoo::Component::Lemmatizer => LeftOutFor::Lemma,
            _ => LeftOutFor::Tag,
        };
        Summary::new(&self.id, &self.training, &self.held_out, left_out_for)
    }
}

/// Writes a trained model's directory `dir/<model id>/`: the model file,
/// [`crate::card::CARD_FILE`] and [`crate::card::GATE_FILE`], as
/// [`crate::card::write_model_dir`] writes a MASC model's. The directory
/// must not exist yet, so nothing is replaced.
///
/// # Errors
///
/// An existing directory, or a write that fails.
pub fn write_ud_model_dir(dir: &Path, card: &UdModelCard, model: &[u8], gate: &ModelGateRecord) -> Result<()> {
    write_dir(dir, &card.id, (&card.file.name, model), &card.to_json(), gate)
}
