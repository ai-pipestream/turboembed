//! The lexicon lookups as [`LemmaDictionary`] implementations, standing
//! over the lemmatizer head where they list a lemma (lemma.rs): a
//! Morfologik dictionary (turbo-morfologik, a `.dict` and `.info` pair)
//! and WordNet's Morphy (turbo-wordnet) over any lexicon it reads,
//! Princeton's database files or an Open English WordNet LMF file.
//!
//! Each asks under the tag the head decoded for the word, through an
//! optional tag map (a UPOS head over a dictionary with its own tagset),
//! and gives no override to a head without a `pos_input`: a lexicon lists
//! lemmas by part of speech, and one under the wrong part of speech is
//! worse than the head's. Each answer is fixed by the lexicon alone:
//! Morfologik's is the smallest listed lemma in byte order, Morphy's its
//! first candidate (an exception, the word itself when the lexicon has
//! it, else the first detachment rule whose result it has).

use std::borrow::Cow;

use turbo_morfologik::MorfologikLemmatizer;
use turbo_wordnet::{LexicalKnowledgeBase, MorphyLemmatizer, UNKNOWN_LEMMA};

use crate::lemma::LemmaDictionary;

/// The head's tag as the lexicon tags words; None asks nothing.
type TagMap = Box<dyn Fn(&str) -> Option<String> + Send + Sync>;

/// The tag a lookup asks under: the head's own, or the map's for it.
fn mapped<'a>(map: Option<&TagMap>, tag: Option<&'a str>) -> Option<Cow<'a, str>> {
    let tag = tag?;
    match map {
        Some(m) => m(tag).map(Cow::Owned),
        None => Some(Cow::Borrowed(tag)),
    }
}

/// A Morfologik dictionary over the head: of the lemmas it lists for the
/// word under the tag (the word lowercased, as the dictionary is keyed),
/// the smallest in byte order.
pub struct MorfologikLemmas {
    lemmatizer: MorfologikLemmatizer,
    tags: Option<TagMap>,
}

impl MorfologikLemmas {
    pub fn new(lemmatizer: MorfologikLemmatizer) -> MorfologikLemmas {
        MorfologikLemmas { lemmatizer, tags: None }
    }

    /// This lookup asking under `map(tag)` in place of the head's tag; a
    /// tag the map answers with None is not looked up.
    pub fn with_tag_map(mut self, map: impl Fn(&str) -> Option<String> + Send + Sync + 'static) -> MorfologikLemmas {
        self.tags = Some(Box::new(map));
        self
    }

    pub fn lemmatizer(&self) -> &MorfologikLemmatizer {
        &self.lemmatizer
    }
}

impl LemmaDictionary for MorfologikLemmas {
    fn lemma(&self, form: &str, tag: Option<&str>) -> Option<String> {
        let tag = mapped(self.tags.as_ref(), tag)?;
        self.lemmatizer.lemmas_for(form, &tag).into_iter().min()
    }
}

/// WordNet's Morphy over the head: its first candidate for the word under
/// the tag's part of speech (`MorphyLemmatizer::pos_from_tag` reads UPOS,
/// Penn and WordNet's own codes), None for a tag outside noun, verb,
/// adjective and adverb or a word the lexicon does not know.
pub struct MorphyLemmas<L: LexicalKnowledgeBase> {
    morphy: MorphyLemmatizer<L>,
    tags: Option<TagMap>,
}

impl<L: LexicalKnowledgeBase> MorphyLemmas<L> {
    pub fn new(morphy: MorphyLemmatizer<L>) -> MorphyLemmas<L> {
        MorphyLemmas { morphy, tags: None }
    }

    /// As [`MorfologikLemmas::with_tag_map`].
    pub fn with_tag_map(mut self, map: impl Fn(&str) -> Option<String> + Send + Sync + 'static) -> MorphyLemmas<L> {
        self.tags = Some(Box::new(map));
        self
    }

    pub fn morphy(&self) -> &MorphyLemmatizer<L> {
        &self.morphy
    }
}

impl<L: LexicalKnowledgeBase + Send + Sync> LemmaDictionary for MorphyLemmas<L> {
    fn lemma(&self, form: &str, tag: Option<&str>) -> Option<String> {
        let tag = mapped(self.tags.as_ref(), tag)?;
        let lemma = self.morphy.lemmatize_one(form, &tag);
        (lemma != UNKNOWN_LEMMA).then_some(lemma)
    }
}
