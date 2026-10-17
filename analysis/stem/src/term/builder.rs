//! Builder for [`TermAnalyzer`] (`opennlp.tools.util.normalizer.TermAnalyzer.Builder`).

use std::num::{NonZeroU32, NonZeroUsize};
use std::sync::Arc;

use turbo_normalize::{AccentFold, DynNormalizer, Normalizer, StrokeLetters};
use turbo_tokenize::WordTokenizer;
use turbo_unicode::script::Script;

use super::{
    Dimension, LocaleCaseFold, NUM_DIMENSIONS, TermAnalyzer, TermAnalyzerInner, TermError, TermLemmatizer,
    TurkicCaseFold,
};
use crate::cache::CachingStemmer;
use crate::stemmer::Stemmer;

/// A builder for [`TermAnalyzer`] (`TermAnalyzer.Builder`).
pub struct TermAnalyzerBuilder {
    enabled: [bool; NUM_DIMENSIONS],
    transforms: [Option<Box<dyn DynNormalizer>>; NUM_DIMENSIONS],
    stemmer: Option<Box<dyn Stemmer>>,
    lemmatizer: Option<Box<dyn TermLemmatizer>>,
    tokenizer: WordTokenizer,
}

impl Default for TermAnalyzerBuilder {
    fn default() -> Self {
        Self::new()
    }
}

impl TermAnalyzerBuilder {
    /// Creates a new builder with no dimensions enabled and the default
    /// [`WordTokenizer`].
    pub fn new() -> TermAnalyzerBuilder {
        TermAnalyzerBuilder {
            enabled: [false; NUM_DIMENSIONS],
            transforms: std::array::from_fn(|_| None),
            stemmer: None,
            lemmatizer: None,
            tokenizer: WordTokenizer::new(),
        }
    }

    /// Enables [`Dimension::Nfc`].
    pub fn nfc(mut self) -> Self {
        self.enabled[Dimension::Nfc.index()] = true;
        self
    }

    /// Enables [`Dimension::Nfkc`].
    pub fn nfkc(mut self) -> Self {
        self.enabled[Dimension::Nfkc.index()] = true;
        self
    }

    /// Enables [`Dimension::Whitespace`].
    pub fn whitespace(mut self) -> Self {
        self.enabled[Dimension::Whitespace.index()] = true;
        self
    }

    /// Enables [`Dimension::Whitespace`] with a custom [`Normalizer`].
    pub fn whitespace_with<N: Normalizer>(self, normalizer: N) -> Result<Self, TermError> {
        self.transform(Dimension::Whitespace, normalizer)
    }

    /// Enables [`Dimension::Dash`].
    pub fn dash(mut self) -> Self {
        self.enabled[Dimension::Dash.index()] = true;
        self
    }

    /// Enables [`Dimension::Dash`] with a custom [`Normalizer`].
    pub fn dash_with<N: Normalizer>(self, normalizer: N) -> Result<Self, TermError> {
        self.transform(Dimension::Dash, normalizer)
    }

    /// Enables [`Dimension::CaseFold`] (`Locale.ROOT` lowercasing).
    ///
    /// # Errors
    ///
    /// Returns [`TermError::BothCaseFolds`] if [`Dimension::FullCaseFold`] is
    /// already configured.
    pub fn case_fold(mut self) -> Result<Self, TermError> {
        self.require_not_both_case_folds(Dimension::CaseFold)?;
        self.enabled[Dimension::CaseFold.index()] = true;
        Ok(self)
    }

    /// Enables [`Dimension::CaseFold`] using the given locale tag's case rules
    /// (for example `"tr"` for Turkish dotted/dotless `i`).
    ///
    /// # Errors
    ///
    /// Returns [`TermError::BothCaseFolds`] if [`Dimension::FullCaseFold`] is
    /// already configured.
    pub fn case_fold_locale(self, locale_tag: &str) -> Result<Self, TermError> {
        let primary = locale_tag.split(['-', '_']).next().unwrap_or("").trim().to_ascii_lowercase();
        if matches!(primary.as_str(), "tr" | "tur" | "az" | "aze") {
            self.transform(Dimension::CaseFold, TurkicCaseFold)
        } else {
            self.transform(Dimension::CaseFold, LocaleCaseFold)
        }
    }

    /// Enables [`Dimension::FullCaseFold`], Unicode full case folding.
    ///
    /// # Errors
    ///
    /// Returns [`TermError::BothCaseFolds`] if [`Dimension::CaseFold`] is
    /// already configured.
    pub fn full_case_fold(mut self) -> Result<Self, TermError> {
        self.require_not_both_case_folds(Dimension::FullCaseFold)?;
        self.enabled[Dimension::FullCaseFold.index()] = true;
        Ok(self)
    }

    fn require_not_both_case_folds(&self, adding: Dimension) -> Result<(), TermError> {
        let other = if adding == Dimension::CaseFold { Dimension::FullCaseFold } else { Dimension::CaseFold };
        if self.enabled[other.index()] {
            return Err(TermError::BothCaseFolds);
        }
        Ok(())
    }

    /// Enables [`Dimension::AccentFold`].
    pub fn accent_fold(mut self) -> Self {
        self.enabled[Dimension::AccentFold.index()] = true;
        self
    }

    /// Enables [`Dimension::AccentFold`] restricted to `scripts` and whether
    /// stroke letters (`ø`, `ł`, `ß`, ...) are folded.
    pub fn accent_fold_scripts(self, scripts: &[Script], fold_stroke_letters: bool) -> Self {
        let strokes = if fold_stroke_letters { StrokeLetters::OpenNlp } else { StrokeLetters::Off };
        let norm = AccentFold::for_scripts(scripts).with_stroke_letters(strokes);
        self.transform(Dimension::AccentFold, norm).expect("AccentFold is a character-level dimension")
    }

    /// Enables [`Dimension::EmojiFold`].
    pub fn emoji_fold(mut self) -> Self {
        self.enabled[Dimension::EmojiFold.index()] = true;
        self
    }

    /// Enables [`Dimension::ConfusableFold`].
    pub fn confusable_fold(mut self) -> Self {
        self.enabled[Dimension::ConfusableFold.index()] = true;
        self
    }

    /// Enables a character-level `dimension` with a specific `normalizer`,
    /// overriding its default.
    ///
    /// # Errors
    ///
    /// Returns [`TermError::NonCharacterTransform`] if `dimension` is
    /// [`Dimension::Original`], [`Dimension::Stem`], or [`Dimension::Lemma`],
    /// or [`TermError::BothCaseFolds`] if `dimension` is a case fold and the
    /// other case fold is already configured.
    pub fn transform<N: Normalizer>(mut self, dimension: Dimension, normalizer: N) -> Result<Self, TermError> {
        if !dimension.is_character_level() {
            return Err(TermError::NonCharacterTransform(dimension));
        }
        if matches!(dimension, Dimension::CaseFold | Dimension::FullCaseFold) {
            self.require_not_both_case_folds(dimension)?;
        }
        self.transforms[dimension.index()] = Some(Box::new(normalizer));
        self.enabled[dimension.index()] = true;
        Ok(self)
    }

    /// Enables [`Dimension::Stem`] through `stemmer`.
    pub fn stem<S: Stemmer + 'static>(mut self, stemmer: S) -> Self {
        self.stemmer = Some(Box::new(stemmer));
        self.enabled[Dimension::Stem.index()] = true;
        self
    }

    /// Enables [`Dimension::Stem`] wrapped in a [`CachingStemmer`] with
    /// [`CachingStemmer::DEFAULT_CAPACITY`] (`Builder.stem(StemmerFactory)`).
    pub fn stem_cached<S: Stemmer + 'static>(self, stemmer: S) -> Self {
        self.stem(CachingStemmer::new(stemmer))
    }

    /// Enables [`Dimension::Stem`] wrapped in a [`CachingStemmer`] with an
    /// explicit `cache_size` (`Builder.stem(StemmerFactory, int)`).
    ///
    /// # Errors
    ///
    /// Returns [`TermError::InvalidCacheCapacity`] if `cache_size == 0`.
    pub fn stem_cached_with_capacity<S: Stemmer + 'static>(
        self,
        stemmer: S,
        cache_size: usize,
    ) -> Result<Self, TermError> {
        let cap = NonZeroUsize::new(cache_size).ok_or(TermError::InvalidCacheCapacity(cache_size))?;
        Ok(self.stem(CachingStemmer::with_capacity(stemmer, cap)))
    }

    /// Enables [`Dimension::Lemma`] through `lemmatizer`.
    pub fn lemmatize<L: TermLemmatizer + 'static>(mut self, lemmatizer: L) -> Self {
        self.lemmatizer = Some(Box::new(lemmatizer));
        self.enabled[Dimension::Lemma.index()] = true;
        self
    }

    /// Sets the [`WordTokenizer`] used by [`TermAnalyzer::analyze`].
    pub fn tokenizer(mut self, tokenizer: WordTokenizer) -> Self {
        self.tokenizer = tokenizer;
        self
    }

    /// Sets the maximum token length (in UTF-16 code units) of the tokenizer
    /// used by [`TermAnalyzer::analyze`].
    ///
    /// # Errors
    ///
    /// Returns [`TermError::InvalidMaxTokenLength`] if `max_token_length == 0`.
    pub fn max_token_length(mut self, max_token_length: u32) -> Result<Self, TermError> {
        let len = NonZeroU32::new(max_token_length).ok_or(TermError::InvalidMaxTokenLength(max_token_length))?;
        self.tokenizer = WordTokenizer::with_max_token_length(len);
        Ok(self)
    }

    /// Builds the immutable [`TermAnalyzer`].
    pub fn build(self) -> TermAnalyzer {
        let mut chain = Vec::new();
        for dim in Dimension::ALL {
            if dim != Dimension::Original && self.enabled[dim.index()] {
                chain.push(dim);
            }
        }
        let final_dimension = chain.last().copied().unwrap_or(Dimension::Original);
        TermAnalyzer {
            inner: Arc::new(TermAnalyzerInner {
                chain,
                final_dimension,
                transforms: self.transforms,
                stemmer: self.stemmer,
                lemmatizer: self.lemmatizer,
                tokenizer: self.tokenizer,
            }),
        }
    }
}
