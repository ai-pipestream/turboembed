//! Lemmatization directly from a compiled Morfologik dictionary.
use crate::dictionary::MorfologikDictionary;
use crate::error::MorfologikError;
use crate::reader::NOT_FOUND;
use std::path::Path;
use std::sync::Arc;
use turbo_analysis::{Annotator, Error, ModelRef, Refs, View};
use turbo_analysis_layers::{Lemmas, PosTags, Strs, Tokens};
use turbo_unicode::java::lowercase_cow;

/// Lemmatizer backed by a [`MorfologikDictionary`], ported from
/// `opennlp.morfologik.lemmatizer.MorfologikLemmatizer`.
#[derive(Debug, Clone)]
pub struct MorfologikLemmatizer {
    dictionary: Arc<MorfologikDictionary>,
}

impl MorfologikLemmatizer {
    /// Initializes a `MorfologikLemmatizer` from a `.dict` path and its sibling `.info` file.
    pub fn read(dict_path: impl AsRef<Path>) -> Result<Self, MorfologikError> {
        Ok(Self::new(Arc::new(MorfologikDictionary::read(dict_path)?)))
    }

    /// Initializes a `MorfologikLemmatizer` from `.dict` and `.info` bytes.
    pub fn from_bytes(dict_bytes: &[u8], info_bytes: &[u8]) -> Result<Self, MorfologikError> {
        Ok(Self::new(Arc::new(MorfologikDictionary::from_bytes(dict_bytes, info_bytes)?)))
    }

    /// Initializes a `MorfologikLemmatizer` from an existing [`MorfologikDictionary`].
    pub fn new(dictionary: Arc<MorfologikDictionary>) -> Self {
        Self { dictionary }
    }

    /// Returns all distinct lemmas for `(word, postag)`, lowercasing `word` with
    /// `Locale.ROOT` (`turbo_unicode::java::to_lowercase`).
    pub fn lemmas_for(&self, word: &str, postag: &str) -> Vec<String> {
        let lower = lowercase_cow(word);
        let mut lemmas = Vec::new();
        for data in self.dictionary.lookup(&lower) {
            if data.tag.as_deref() == Some(postag) && !lemmas.contains(&data.stem) {
                lemmas.push(data.stem);
            }
        }
        lemmas
    }

    /// `MorfologikLemmatizer.lemmatize(String[] toks, String[] tags)`: returns `Some(lemma)`
    /// for each token and tag, or `None` (`null` in Java) when no lemma matches.
    pub fn lemmatize<S1: AsRef<str>, S2: AsRef<str>>(&self, tokens: &[S1], tags: &[S2]) -> Vec<Option<String>> {
        assert_eq!(tokens.len(), tags.len(), "tokens and tags must have the same length");
        tokens
            .iter()
            .zip(tags.iter())
            .map(|(tok, tag)| self.dictionary.first_lemma(&lowercase_cow(tok.as_ref()), tag.as_ref()))
            .collect()
    }

    /// `MorfologikLemmatizer.lemmatize(List<String> toks, List<String> tags)`: returns
    /// the list of distinct lemmas for each token and tag (empty `Vec` on a miss).
    pub fn lemmatize_all<S1: AsRef<str>, S2: AsRef<str>>(&self, tokens: &[S1], tags: &[S2]) -> Vec<Vec<String>> {
        assert_eq!(tokens.len(), tags.len(), "tokens and tags must have the same length");
        tokens.iter().zip(tags.iter()).map(|(tok, tag)| self.lemmas_for(tok.as_ref(), tag.as_ref())).collect()
    }
}

impl Annotator for MorfologikLemmatizer {
    const ID: &'static str = "turbo-morfologik:lemmatizer";
    const VERSION: &'static str = "2";
    type Reads = (View, Tokens, PosTags);
    type Writes = (Lemmas,);

    fn annotate(&self, (view, tokens, tags): Refs<'_, Self>) -> Result<(Lemmas,), Error> {
        if tokens.spans.view() != view.id() {
            return Err(Error::Annotator {
                annotator: Self::ID,
                detail: format!(
                    "the tokens are in view {}, not in view {} that is lemmatized",
                    tokens.spans.view().0,
                    view.id().0
                ),
            });
        }
        if tokens.spans.len() != tags.tags.len() {
            return Err(Error::Annotator {
                annotator: Self::ID,
                detail: format!("{} tokens but {} POS tags", tokens.spans.len(), tags.tags.len()),
            });
        }
        let text = view.text();
        let mut out = Strs::default();
        for ((_, span), &tag) in tokens.spans.iter().zip(&tags.tags) {
            let word = text.get(span.start as usize..span.end as usize).ok_or_else(|| Error::Annotator {
                annotator: Self::ID,
                detail: format!("token span {}..{} out of bounds", span.start, span.end),
            })?;
            if usize::from(tag.0) >= tags.tagset.len() {
                return Err(Error::Annotator {
                    annotator: Self::ID,
                    detail: format!("POS tag {} outside tag set", tag.0),
                });
            }
            let pos = tags.tagset.label(tag);
            match self.dictionary.first_lemma(&lowercase_cow(word), pos) {
                Some(lemma) => out.push(&lemma),
                None => out.push(NOT_FOUND),
            }
        }
        Ok((Lemmas { lemma: out },))
    }

    fn model(&self) -> Option<ModelRef> {
        Some(self.dictionary.model_ref().clone())
    }
}
