//! Reference models a recipe's model is gated against, either listed in the
//! zoo's built-in index or recorded by size and SHA-256.

use turbo_zoo::{Index, Sha256};

use crate::error::{Error, Result};

/// The OpenNLP model a recipe's model is gated against.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Reference {
    /// A model of the zoo's built-in index, by its id.
    Zoo(&'static str),
    /// A model the zoo does not list, recorded by the recipe.
    Recorded(RecordedReference),
}

/// A reference model the zoo does not list: its file as published, which
/// a run reads from its references directory and checks against the
/// recorded digest. Such a model is only read to score it; it is never
/// shipped or bundled.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RecordedReference {
    /// The id the gate record names it by.
    pub id: &'static str,
    /// The file name as published.
    pub file: &'static str,
    /// Where it is published.
    pub url: &'static str,
    pub size: u64,
    pub sha256: &'static str,
}

impl Reference {
    /// The id the gate record names the model by: its zoo id, or the
    /// recorded one.
    pub fn id(&self) -> &'static str {
        match self {
            Reference::Zoo(id) => id,
            Reference::Recorded(r) => r.id,
        }
    }

    /// The model's file name: the one the zoo's built-in index gives it,
    /// or the recorded one.
    ///
    /// # Errors
    ///
    /// A zoo id the index does not list.
    pub fn file(&self) -> Result<String> {
        match self {
            Reference::Zoo(id) => Ok(Index::builtin().get(id)?.files[0].path.to_string()),
            Reference::Recorded(r) => Ok(r.file.to_owned()),
        }
    }

    /// The model file's SHA-256: the one the zoo's built-in index lists, or
    /// the recorded one.
    ///
    /// # Errors
    ///
    /// A zoo id the index does not list.
    pub fn sha256(&self) -> Result<String> {
        match self {
            Reference::Zoo(id) => Ok(Index::builtin().get(id)?.files[0].sha256.to_string()),
            Reference::Recorded(r) => Ok(r.sha256.to_owned()),
        }
    }

    /// Checks that `bytes` are the model: the SHA-256 the zoo's built-in
    /// index lists, or the recorded size and SHA-256.
    ///
    /// # Errors
    ///
    /// [`Error::ReferenceDigest`] for other bytes, or a zoo id the index
    /// does not list.
    pub fn check(&self, bytes: &[u8]) -> Result<()> {
        let expected = self.sha256()?;
        let size = match self {
            Reference::Zoo(_) => None,
            Reference::Recorded(r) => Some(r.size),
        };
        let found = Sha256::of(bytes).to_string();
        if found == expected && size.is_none_or(|n| n == bytes.len() as u64) {
            Ok(())
        } else {
            Err(Error::ReferenceDigest { id: self.id().to_owned(), expected, found })
        }
    }
}

#[cfg(test)]
mod tests {
    use turbo_eval::model_gate::Task;

    use super::*;
    use crate::card::ParametersCard;
    use crate::{LeftOut, Limit, RECIPES, Recipe, check_skipped, chunk, postag};

    #[test]
    fn every_recipe_names_a_reference_and_a_valid_model_id() {
        let ids: Vec<&str> = RECIPES.iter().map(Recipe::model_id).collect();
        assert_eq!(
            ids,
            [
                "turbo-en-masc-3.0.0-sentence-1",
                "turbo-en-masc-3.0.0-tokens-1",
                "turbo-en-masc-3.0.0-pos-1",
                "turbo-en-masc-3.0.0-chunks-1",
                "turbo-en-masc-3.0.0-ner-person-1",
                "turbo-en-masc-3.0.0-ner-location-1",
                "turbo-en-masc-3.0.0-ner-organization-1",
                "turbo-en-masc-3.0.0-ner-date-1",
            ]
        );
        for r in RECIPES {
            assert!(matches!(r.check_reference(b"not the model"), Err(Error::ReferenceDigest { .. })));
            let id: turbo_zoo::ArtifactId = r.model_id().parse().unwrap();
            assert_eq!(id.as_str(), r.model_id());
            match r.reference() {
                // The reference is a model of the component the trained
                // one is, under its zoo file name.
                Reference::Zoo(id) => {
                    assert_eq!(r.reference_file().unwrap(), format!("{id}.bin"));
                    assert_eq!(Index::builtin().get(id).unwrap().component, Some(r.component()));
                }
                // A recorded reference is one the zoo does not list, so it
                // can never be fetched or shipped through it.
                Reference::Recorded(rec) => {
                    assert_eq!(r.reference_file().unwrap(), rec.file);
                    assert!(Index::builtin().get(rec.id).is_err());
                }
            }
        }
        assert_eq!(RECIPES[2].task(), Task::PartOfSpeechTagger);
        assert_eq!(RECIPES[2].component(), turbo_zoo::Component::Pos);
        assert_eq!(RECIPES[2].reference_id(), "opennlp-en-pos-maxent-1.5");
        assert_eq!(RECIPES[2].reference().sha256().unwrap(), postag::EN_POS_MAXENT.sha256);
        for r in &RECIPES[4..] {
            assert_eq!(r.task(), Task::NameFinder);
            assert_eq!(r.component(), turbo_zoo::Component::NameFinder);
            assert_eq!(
                r.parameters(),
                ParametersCard { algorithm: "MAXENT".into(), iterations: 300, cutoff: 0, threads: 1 }
            );
        }
        assert_eq!(
            RECIPES[0].parameters(),
            ParametersCard { algorithm: "MAXENT".into(), iterations: 300, cutoff: 5, threads: 4 }
        );
        for r in RECIPES {
            assert_eq!(Recipe::trains(r.model_id()), Some(r));
        }
        assert_eq!(RECIPES[3].task(), Task::Chunker);
        assert_eq!(RECIPES[3].component(), turbo_zoo::Component::Chunker);
        assert_eq!(RECIPES[3].reference_id(), "opennlp-en-chunker-1.5");
        assert_eq!(RECIPES[3].reference().sha256().unwrap(), chunk::EN_CHUNKER.sha256);
        let p = RECIPES[3].parameters();
        assert_eq!((p.algorithm.as_str(), p.iterations, p.cutoff, p.threads), ("MAXENT", 100, 5, 1));
        let p = RECIPES[2].parameters();
        assert_eq!((p.algorithm.as_str(), p.iterations, p.cutoff, p.threads), ("MAXENT", 300, 5, 4));
        assert!(RECIPES[3].reads_chunks() && !RECIPES[2].reads_chunks());
        assert_eq!(Recipe::trains("turbo-en-masc-3.0.0-unknown-1"), None);
        assert!(Reference::Zoo("not-in-the-zoo").file().is_err());
    }

    #[test]
    fn a_recorded_reference_is_checked_by_size_and_digest() {
        let bytes = b"abc";
        let rec = RecordedReference {
            id: "x",
            file: "x.bin",
            url: "https://example.org/x.bin",
            size: 3,
            sha256: "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad",
        };
        Reference::Recorded(rec).check(bytes).unwrap();
        assert_eq!(Reference::Recorded(rec).sha256().unwrap(), rec.sha256);
        assert!(Reference::Recorded(RecordedReference { size: 4, ..rec }).check(bytes).is_err());
        let other = Reference::Recorded(rec).check(b"abd").unwrap_err();
        // The message says what was recorded, not where.
        assert_eq!(
            other.to_string(),
            format!("reference model x: SHA-256 {}, expected {}", Sha256::of(b"abd"), rec.sha256)
        );
    }

    #[test]
    fn limit_allows_guards_zero_denominator_and_excess_left_out() {
        let limit = Limit { count: 1, of: 1000 };
        assert!(limit.allows(0, 0));
        assert!(limit.allows(1, 1000));
        assert!(!limit.allows(2, 1000));
        assert!(!limit.allows(1, 0));
        assert!(!Limit { count: 1, of: 0 }.allows(0, 10));
        assert!(!Limit { count: 1, of: 0 }.allows(5, 10));
        assert!(!Limit { count: 2, of: 1 }.allows(5, 3));
        assert!(check_skipped(LeftOut::TrainingSentences, 1, 999).is_ok());
        assert!(check_skipped(LeftOut::TrainingSentences, 2, 998).is_err());
    }
}
