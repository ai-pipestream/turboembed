//! Evaluation references: native gold-label baselines or remaining legacy
//! model references scheduled for retirement.

use turbo_zoo::{Index, Sha256};

use crate::error::{Error, Result};

/// The reference policy a recipe uses.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Reference {
    /// Native predicted spans against gold labels and a newline baseline. No model file.
    GoldSentences,
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
            Reference::GoldSentences => crate::sentence_gold::BASELINE_ID,
            Reference::Zoo(id) => id,
            Reference::Recorded(r) => r.id,
        }
    }

    /// The external model file name. Native gold-label evaluation has no file.
    ///
    /// # Errors
    ///
    /// A native gold-label baseline, or a zoo id the index does not list.
    pub fn file(&self) -> Result<String> {
        match self {
            Reference::GoldSentences => {
                Err(Error::Model("gold sentence evaluation has no reference model file".into()))
            }
            Reference::Zoo(id) => Ok(Index::builtin().get(id)?.files[0].path.to_string()),
            Reference::Recorded(r) => Ok(r.file.to_owned()),
        }
    }

    /// A model-file SHA-256, or the native baseline algorithm identity hash.
    ///
    /// # Errors
    ///
    /// A zoo id the index does not list.
    pub fn sha256(&self) -> Result<String> {
        match self {
            Reference::GoldSentences => Ok(crate::sentence_gold::baseline().sha256),
            Reference::Zoo(id) => Ok(Index::builtin().get(id)?.files[0].sha256.to_string()),
            Reference::Recorded(r) => Ok(r.sha256.to_owned()),
        }
    }

    /// Checks that `bytes` are the model: the SHA-256 the zoo's built-in
    /// index lists, or the recorded size and SHA-256.
    ///
    /// # Errors
    ///
    /// Native gold-label baselines reject all byte inputs. Model references
    /// reject a wrong digest or an id the index does not list.
    pub fn check(&self, bytes: &[u8]) -> Result<()> {
        if matches!(self, Self::GoldSentences) {
            return Err(Error::Model("gold sentence evaluation accepts spans, not reference model bytes".into()));
        }
        let expected = self.sha256()?;
        let size = match self {
            Reference::GoldSentences | Reference::Zoo(_) => None,
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

    use super::*;

    use crate::{LeftOut, Limit, check_skipped};

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
