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
//! implementation); the behaviour ported is OpenNLP's, except that
//! [`SymSpell::add_bigram`] ignores zero counts and keeps the true minimum
//! positive bigram count when existing bigrams are incremented (rather than
//! OpenNLP's `SymSpell.addBigram`, which inserts zero counts and leaves a
//! stale pre-increment minimum).

mod casing;
mod compound;
mod config;
mod confusables;
mod context;
mod correct;
mod deletes;
mod dictionary;
mod distance;
mod guards;
mod lookup;
mod model;
mod suggestion;
mod symspell;
mod table;
mod trellis;

pub use compound::{Compound, CompoundPart};
pub use config::{ConfigError, DERIVE_CORPUS_WORD_COUNT, MAX_COUNT, SymSpellConfig, SymSpellConfigBuilder};
pub use confusables::ConfusableSet;
pub use context::{
    ContextSpellAnnotator, ContextSpellCorrector, DEFAULT_CONTEXT_MIN_TOKEN_LENGTH, DEFAULT_EDIT_WEIGHT,
    DEFAULT_MAX_BEAM_WIDTH, DEFAULT_MAX_CANDIDATES,
};
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

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;
    use turbo_analysis::{Annotator, Document, Span, Spans, View};
    use turbo_normalize_layers::{PlaceholderKind, Placeholders};

    #[test]
    fn annotate_normalizes_nested_and_unsorted_placeholders_and_rejects_oob() {
        let mut engine = SymSpell::default();
        engine.add("hello", 100);
        engine.add("world", 100);
        let corrector = SpellCorrect::new(Arc::new(engine)).use_placeholders(true);

        let doc = Document::new("helo wrld");
        let view = doc.get::<View>().unwrap();

        // Unsorted non-overlapping placeholders: [5, 9) before [0, 1) must still protect "helo".
        let mut unsorted = Spans::new(view.id());
        unsorted.push(Span::new(5, 9));
        unsorted.push(Span::new(0, 1));
        let placeholders = Placeholders { spans: unsorted, kind: vec![PlaceholderKind::Url, PlaceholderKind::Url] };
        let (corrected_view, corrections) = corrector.annotate((view, Some(&placeholders))).unwrap();
        assert_eq!(corrected_view.text(), "helo wrld");
        assert!(corrections.spans.is_empty());

        // Nested placeholders: [0, 9) containing [1, 2).
        let mut nested = Spans::new(view.id());
        nested.push(Span::new(0, 9));
        nested.push(Span::new(1, 2));
        let placeholders = Placeholders { spans: nested, kind: vec![PlaceholderKind::Url, PlaceholderKind::Url] };
        let (corrected_view, corrections) = corrector.annotate((view, Some(&placeholders))).unwrap();
        assert_eq!(corrected_view.text(), "helo wrld");
        assert!(corrections.spans.is_empty());

        let mut oob = Spans::new(view.id());
        oob.push(Span::new(0, 99));
        let oob_placeholders = Placeholders { spans: oob, kind: vec![PlaceholderKind::Url] };
        let err = corrector
            .annotate((view, Some(&oob_placeholders)))
            .err()
            .expect("out-of-bounds placeholder span must fail");
        assert!(err.to_string().contains("out of bounds"));
    }
}
