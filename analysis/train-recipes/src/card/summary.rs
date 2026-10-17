//! Reading and writing model card directories, licence publishability
//! checks, and run summary formatting.

use std::fmt;
use std::path::Path;

use serde::Deserialize;
use turbo_eval::model_gate::ModelGateRecord;
use turbo_zoo::{Licence, LicenceClass, TrainingData, class_of};

use super::{CARD_FILE, GATE_FILE, HeldOutCard, ModelCard, TrainingCard};
use crate::error::{Error, Result, io};

/// Reads a card of the schema version `expected` from `path`: its
/// `card_version` first, alone, so a card of another version is
/// [`Error::CardVersion`] whatever fields it has, then the card.
///
/// # Errors
///
/// A file that cannot be read, is not JSON with a `card_version`, is of
/// another version, or is not a card of this version.
pub(crate) fn read_card<T: serde::de::DeserializeOwned>(path: &Path, expected: u32) -> Result<T> {
    /// The one field every version of a card has.
    #[derive(Deserialize)]
    struct Version {
        card_version: u32,
    }
    let bytes = std::fs::read(path).map_err(io(path))?;
    let json = |e: serde_json::Error| Error::Json { path: path.to_owned(), message: e.to_string() };
    let Version { card_version } = serde_json::from_slice(&bytes).map_err(json)?;
    if card_version != expected {
        return Err(Error::CardVersion { path: path.to_owned(), found: card_version, expected });
    }
    serde_json::from_slice(&bytes).map_err(json)
}

/// Why the held-out tokens a card lists were left out of the scores.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LeftOutFor {
    /// The reference tagger cannot give their tag.
    Tag,
    /// The reference lemmatizer cannot give the lemma their gold lemma
    /// class gives ([`crate::ud::ReferenceClasses`]); they are listed by
    /// tag.
    Lemma,
}

/// What a run prints of a card before its gate record
/// ([`ModelCard::summary`], and [`crate::ud_card::UdModelCard::summary`]):
/// a line of the training and held-out samples and the sentences each
/// side skipped; for a tagger or a UD lemmatizer a second line of the
/// held-out tokens left out of the scores, by their tags as written; and
/// for a chunker a line for each side of its chunks in no sentence its
/// reader reads ([`super::ChunksCard`]).
#[derive(Clone, Copy, Debug)]
pub struct Summary<'c> {
    id: &'c str,
    training: &'c TrainingCard,
    held_out: &'c HeldOutCard,
    left_out_for: LeftOutFor,
}

impl<'c> Summary<'c> {
    /// The summary of the card of the model `id`, whose held-out tokens
    /// are left out for `left_out_for`.
    pub(crate) fn new(
        id: &'c str,
        training: &'c TrainingCard,
        held_out: &'c HeldOutCard,
        left_out_for: LeftOutFor,
    ) -> Summary<'c> {
        Summary { id, training, held_out, left_out_for }
    }
}

impl fmt::Display for Summary<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let Summary { id, training, held_out, left_out_for } = *self;
        write!(
            f,
            "{id}: trained on {} samples, {} sentences skipped, {} events; held out {} samples, {} sentences skipped",
            training.samples, training.skipped_sentences, training.events, held_out.samples, held_out.skipped_sentences,
        )?;
        if let Some(tokens) = &held_out.tokens {
            let count: u64 = tokens.left_out.iter().map(|t| t.tokens).sum();
            let why = match left_out_for {
                LeftOutFor::Tag => "their tags not the reference's",
                LeftOutFor::Lemma => "their lemmas not the reference's, by tag",
            };
            write!(f, "\n{id}: {count} of {} held-out tokens left out, {why}:", tokens.tokens)?;
            if tokens.left_out.is_empty() {
                f.write_str(" none")?;
            }
            for (i, t) in tokens.left_out.iter().enumerate() {
                write!(f, "{} {} ({})", if i == 0 { "" } else { "," }, t.tag, t.tokens)?;
            }
        }
        for (side, names) in [("training", &training.names), ("held-out", &held_out.names)] {
            if let Some(n) = names {
                write!(
                    f,
                    "\n{id}: {} names in the {side} samples of {} documents, {} with a gap; {} overlapping and {} of \
                     other types left out; {} outside every sentence region and {} in sentence regions the reader \
                     drops, in no sample",
                    n.names,
                    n.documents,
                    n.gapped,
                    n.overlapping,
                    n.other_types,
                    n.outside_sentences,
                    n.in_dropped_sentences
                )?;
            }
        }
        for (side, chunks) in [("training", &training.chunks), ("held-out", &held_out.chunks)] {
            if let Some(c) = chunks {
                write!(
                    f,
                    "\n{id}: of {} {side} chunks, {} outside every kept sentence, {} across sentences, {} without tokens",
                    c.chunks, c.outside_sentences, c.across_sentences, c.without_tokens
                )?;
            }
        }
        Ok(())
    }
}

/// The class of a model under `license` trained on `trained_on`, computed
/// from the licences ([`class_of`]), when that class lets the model leave
/// the machine it was trained on ([`LicenceClass::may_redistribute`]). A
/// model trained on non-commercial or restricted data is
/// `CLASS_LOCAL_ONLY`: it may be trained, written and evaluated locally
/// ([`write_model_dir`]), but it is never published, so it gets no zoo
/// entry.
///
/// # Errors
///
/// [`Error::NotShippable`] for a local-only or refused class.
pub fn publishable_class(id: &str, license: Licence, trained_on: &[TrainingData]) -> Result<LicenceClass> {
    let class = class_of(std::iter::once(license).chain(trained_on.iter().map(|t| t.license)));
    if class.may_redistribute() {
        return Ok(class);
    }
    let reason = if class == LicenceClass::LocalOnly {
        "it was trained on non-commercial or restricted data, so it stays on the machine it was trained on"
    } else {
        "it was trained on data under terms the zoo refuses"
    };
    Err(Error::NotShippable { id: id.to_owned(), reason })
}

/// Writes a trained model's directory `dir/<model id>/`: the model file,
/// [`CARD_FILE`] and [`GATE_FILE`]. The directory must not exist yet, so
/// nothing is replaced.
///
/// # Errors
///
/// An existing directory, or a write that fails.
pub fn write_model_dir(dir: &Path, card: &ModelCard, model: &[u8], gate: &ModelGateRecord) -> Result<()> {
    write_dir(dir, &card.id, (&card.file.name, model), &card.to_json(), gate)
}

/// Writes `dir/<id>/` with the model file (its name and bytes),
/// [`CARD_FILE`] (`card_json`) and [`GATE_FILE`]; the directory must not
/// exist yet.
pub(crate) fn write_dir(
    dir: &Path,
    id: &str,
    (model_name, model): (&str, &[u8]),
    card_json: &str,
    gate: &ModelGateRecord,
) -> Result<()> {
    let out = dir.join(id);
    std::fs::create_dir_all(dir).map_err(io(dir))?;
    std::fs::create_dir(&out).map_err(io(&out))?;
    let model_path = out.join(model_name);
    std::fs::write(&model_path, model).map_err(io(&model_path))?;
    let card_path = out.join(CARD_FILE);
    std::fs::write(&card_path, card_json).map_err(io(&card_path))?;
    Ok(gate.write(&out.join(GATE_FILE))?)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::card::{
        CARD_VERSION, CorpusCard, GATE_FILE, GateCard, MODEL_LICENCE, MODEL_LICENCE_TERMS, ModelFileCard,
        ParametersCard,
    };
    use crate::masc::{self, SPLIT};
    use crate::segmentation::{LANGUAGE, TIMESTAMP_MS};

    fn data(name: &str, license: Licence) -> TrainingData {
        TrainingData { name: name.to_owned(), license, license_terms: "https://example.org/terms".parse().unwrap() }
    }

    #[test]
    fn a_model_trained_on_non_commercial_data_is_local_only_and_gets_no_zoo_entry() {
        let gum = data("UD_English-GUM r2.16", Licence::CcByNcSa40);
        let trained_on = [masc::training_data(), gum];
        // The card's class, as `ModelCard::new` computes it.
        let class = class_of(std::iter::once(MODEL_LICENCE).chain(trained_on.iter().map(|t| t.license)));
        assert_eq!(class, LicenceClass::LocalOnly);
        assert!(!class.may_bundle() && !class.may_redistribute());
        match publishable_class("m", MODEL_LICENCE, &trained_on) {
            Err(Error::NotShippable { id, reason }) => {
                assert_eq!(id, "m");
                assert!(reason.contains("non-commercial"), "{reason}");
            }
            other => panic!("{other:?}"),
        }
        let nc = [data("non-commercial", Licence::CcByNc40)];
        assert!(matches!(publishable_class("m", MODEL_LICENCE, &nc), Err(Error::NotShippable { .. })));
    }

    #[test]
    fn models_that_may_leave_the_machine_keep_their_class() {
        assert_eq!(publishable_class("m", MODEL_LICENCE, &[masc::training_data()]).unwrap(), LicenceClass::Permissive);
        let ewt = data("UD_English-EWT r2.16", Licence::CcBySa40);
        assert_eq!(publishable_class("m", MODEL_LICENCE, &[ewt]).unwrap(), LicenceClass::ShareAlike);
        // No-derivatives data is refused, not local-only.
        let nd = [data("no-derivatives", Licence::CcByNcNd40)];
        match publishable_class("m", MODEL_LICENCE, &nd) {
            Err(Error::NotShippable { reason, .. }) => assert!(reason.contains("refuses"), "{reason}"),
            other => panic!("{other:?}"),
        }
    }

    #[test]
    fn model_card_artifact_validates_trained_on_component_and_language() {
        let recipe = &crate::RECIPES[0];
        let ref_sha = recipe.reference().sha256().unwrap();
        let sha64 = "a".repeat(64);
        let sha128 = "b".repeat(128);
        let card = ModelCard {
            card_version: CARD_VERSION,
            id: recipe.model_id().into(),
            component: recipe.component(),
            language: LANGUAGE.into(),
            file: ModelFileCard {
                name: format!("{}.bin", recipe.model_id()),
                size: 3,
                sha256: sha64.clone(),
                sha512: sha128,
            },
            license: MODEL_LICENCE,
            license_terms: MODEL_LICENCE_TERMS.into(),
            trained_on: vec![masc::training_data()],
            class: LicenceClass::Permissive,
            attribution: masc::ATTRIBUTION.into(),
            corpus: CorpusCard {
                name: masc::NAME.into(),
                corpus_sha256: sha64,
                release_recorded: true,
                split: SPLIT,
                training_documents: 1,
                held_out_documents: Vec::new(),
                excluded_documents: Vec::new(),
            },
            training: TrainingCard {
                parameters: ParametersCard { algorithm: "MAXENT".into(), iterations: 300, cutoff: 5, threads: 4 },
                sentences_per_sample: Some(20),
                whitespace: Some("legacy".into()),
                timestamp_ms: TIMESTAMP_MS,
                seed: None,
                samples: 1,
                skipped_sentences: 0,
                events: 1,
                event_hash: 0,
                names: None,
                chunks: None,
            },
            held_out: HeldOutCard { samples: 1, skipped_sentences: 0, tokens: None, names: None, chunks: None },
            gate: GateCard {
                record: GATE_FILE.into(),
                reference: recipe.reference_id().into(),
                reference_sha256: ref_sha,
                candidate_tag_format: None,
                reference_tag_format: None,
                candidate_score: 100.0,
                reference_score: 100.0,
                difference_low: 0.0,
                difference_high: 0.0,
                pass: true,
            },
        };
        let url = "https://example.org/m.bin";
        assert!(card.artifact(url).is_ok());

        let mut bad_trained = card.clone();
        bad_trained.trained_on = vec![data("other", Licence::CcBy40)];
        assert!(matches!(bad_trained.artifact(url), Err(Error::NotShippable { .. })));

        let mut bad_component = card.clone();
        bad_component.component = turbo_zoo::Component::Pos;
        assert!(matches!(bad_component.artifact(url), Err(Error::InvalidCard { field: "component", .. })));

        let mut bad_language = card;
        bad_language.language = "de".into();
        assert!(matches!(bad_language.artifact(url), Err(Error::InvalidCard { field: "language", .. })));
    }
}
