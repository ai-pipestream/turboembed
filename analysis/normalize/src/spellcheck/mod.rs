//! Spelling correction with symmetric delete (SymSpell), ported from
//! OpenNLP's spellcheck add-on.
//!
//! - [`SymSpell`] is the engine: a deletes index over a frequency
//!   dictionary, with `lookup` for one term and `lookup_compound` for a
//!   phrase, which also splits and merges words. It is configured by
//!   [`SymSpellConfig`] and verifies candidates with a [`Metric`].
//! - [`FrequencyDictionary`] reads the plain-text dictionaries and
//!   [`SymSpellModel`] the binary models of OpenNLP's
//!   `SymSpellModelSerializer`; no dictionary ships with this crate.
//! - [`SpellCorrect`] is the normalizer: per token or per phrase, with an
//!   exact alignment, and as an annotator a `Corrections` layer.
//!
//! The SymSpell algorithm is Wolf Garbe's (MIT licensed reference
//! implementation); the behaviour ported is OpenNLP's.

mod casing;
mod compound;
mod config;
mod correct;
mod deletes;
mod dictionary;
mod distance;
mod guards;
mod model;
mod suggestion;
mod symspell;
mod table;

pub use compound::{Compound, CompoundPart};
pub use config::{ConfigError, DERIVE_CORPUS_WORD_COUNT, MAX_COUNT, SymSpellConfig, SymSpellConfigBuilder};
pub use correct::{DEFAULT_MIN_TOKEN_LENGTH, SpellCorrect, SpellCorrectMode};
pub use dictionary::{FrequencyDictionary, MalformedLine, MalformedReason};
pub use distance::{DamerauOsa, EditDistance, Levenshtein, Metric};
pub use model::{
    DEFAULT_MODEL_NAME, DEFAULT_MODEL_VERSION, FORMAT_VERSION, MAGIC, MAX_ENTRIES, ModelError, SymSpellModel,
};
pub use suggestion::{Suggestion, Verbosity};
pub use symspell::{LookupError, SymSpell};

/// The noise scorer's word list: a word is known when it is indexed, its
/// count at or above the threshold.
impl crate::noise::Dictionary for SymSpell {
    fn contains(&self, word: &str) -> bool {
        SymSpell::contains(self, word)
    }
}
