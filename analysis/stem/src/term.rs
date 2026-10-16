//! OpenNLP's `Dimension`, `Term`, and `TermAnalyzer`: layered per-token
//! normalization with eager evaluation of configured dimensions and thread-safe
//! lazy memoization of unconfigured dimensions.

use std::fmt;
use std::num::{NonZeroU32, NonZeroUsize};
use std::sync::{Arc, LazyLock, OnceLock};

use turbo_analysis::{Span, ViewBuilder};
use turbo_analysis_layers::Strs;
use turbo_lemmatizer::{DictionaryLemmatizer, MaxentLemmatizer, TokenTag};
use turbo_normalize::{
    AccentFold, CaseFold, ConfusableSkeleton, Dashes, DynNormalizer, EmojiToEmoticon, Normalizer, StrokeLetters,
    UnicodeForm, Whitespace,
};
use turbo_tokenize::{WordTokenizer, word_tokens};
use turbo_unicode::script::Script;

use crate::cache::CachingStemmer;
use crate::stemmer::Stemmer;

/// Total number of [`Dimension`] variants.
const NUM_DIMENSIONS: usize = 12;

/// A layer of the [`Term`] normalization stack, in increasing order of
/// aggressiveness (`opennlp.tools.util.normalizer.Dimension`).
///
/// The declaration order is the canonical application order, because the
/// transforms do not commute. [`Dimension::Original`] is the source token text
/// and is always present. [`Dimension::Stem`] and [`Dimension::Lemma`] are
/// token-level and have no default normalizer; they require a [`Stemmer`] or
/// [`TermLemmatizer`] on the analyzer ([`Dimension::Lemma`] also requires a
/// part-of-speech tag).
#[derive(Copy, Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[repr(u8)]
pub enum Dimension {
    /// The original token text, the source of truth (`ORIGINAL`).
    Original = 0,
    /// Unicode canonical composition (`NFC`); lossless under canonical equivalence.
    Nfc = 1,
    /// Unicode compatibility composition (`NFKC`); lossy (for example superscripts to digits).
    Nfkc = 2,
    /// Unicode whitespace folded to ASCII spaces (`WHITESPACE`).
    Whitespace = 3,
    /// Unicode dashes folded to the ASCII hyphen-minus (`DASH`).
    Dash = 4,
    /// Case folding (`CASE_FOLD`); `Locale.ROOT` lowercasing by default.
    CaseFold = 5,
    /// Unicode full case folding (`FULL_CASE_FOLD`, UTS #21); expanding (`ß` to `ss`, ligatures).
    FullCaseFold = 6,
    /// Diacritic and accent folding (`ACCENT_FOLD`).
    AccentFold = 7,
    /// Emoji folded to ASCII emoticons where a mapping exists (`EMOJI_FOLD`).
    EmojiFold = 8,
    /// Confusable (homoglyph) skeleton folding per UTS #39 (`CONFUSABLE_FOLD`).
    ConfusableFold = 9,
    /// Stemming through a configured [`Stemmer`] (`STEM`).
    Stem = 10,
    /// Lemmatization through a configured [`TermLemmatizer`] (`LEMMA`).
    Lemma = 11,
}

static DEFAULT_NFC: LazyLock<UnicodeForm> = LazyLock::new(UnicodeForm::nfc);
static DEFAULT_NFKC: LazyLock<UnicodeForm> = LazyLock::new(UnicodeForm::nfkc);
static DEFAULT_WHITESPACE: LazyLock<Whitespace> = LazyLock::new(Whitespace::collapse);
static DEFAULT_DASH: LazyLock<Dashes> = LazyLock::new(Dashes::new);
static DEFAULT_CASE_FOLD: LazyLock<CaseFold> = LazyLock::new(CaseFold::lowercase);
static DEFAULT_FULL_CASE_FOLD: LazyLock<CaseFold> = LazyLock::new(CaseFold::full);
static DEFAULT_ACCENT_FOLD: LazyLock<AccentFold> = LazyLock::new(AccentFold::new);
static DEFAULT_EMOJI_FOLD: EmojiToEmoticon = EmojiToEmoticon;
static DEFAULT_CONFUSABLE_FOLD: ConfusableSkeleton = ConfusableSkeleton;

impl Dimension {
    /// All twelve dimensions in canonical pipeline order.
    pub const ALL: [Dimension; NUM_DIMENSIONS] = [
        Dimension::Original,
        Dimension::Nfc,
        Dimension::Nfkc,
        Dimension::Whitespace,
        Dimension::Dash,
        Dimension::CaseFold,
        Dimension::FullCaseFold,
        Dimension::AccentFold,
        Dimension::EmojiFold,
        Dimension::ConfusableFold,
        Dimension::Stem,
        Dimension::Lemma,
    ];

    /// Returns whether this dimension is a character-level normalizer
    /// ([`Dimension::Nfc`] through [`Dimension::ConfusableFold`]).
    pub const fn is_character_level(self) -> bool {
        matches!(
            self,
            Dimension::Nfc
                | Dimension::Nfkc
                | Dimension::Whitespace
                | Dimension::Dash
                | Dimension::CaseFold
                | Dimension::FullCaseFold
                | Dimension::AccentFold
                | Dimension::EmojiFold
                | Dimension::ConfusableFold
        )
    }

    /// Returns the default character-level normalizer for this dimension, or
    /// `None` for [`Dimension::Original`], [`Dimension::Stem`], and
    /// [`Dimension::Lemma`] (`Dimension.defaultNormalizer()`).
    pub fn default_normalizer(self) -> Option<&'static dyn DynNormalizer> {
        match self {
            Dimension::Original | Dimension::Stem | Dimension::Lemma => None,
            Dimension::Nfc => Some(&*DEFAULT_NFC),
            Dimension::Nfkc => Some(&*DEFAULT_NFKC),
            Dimension::Whitespace => Some(&*DEFAULT_WHITESPACE),
            Dimension::Dash => Some(&*DEFAULT_DASH),
            Dimension::CaseFold => Some(&*DEFAULT_CASE_FOLD),
            Dimension::FullCaseFold => Some(&*DEFAULT_FULL_CASE_FOLD),
            Dimension::AccentFold => Some(&*DEFAULT_ACCENT_FOLD),
            Dimension::EmojiFold => Some(&DEFAULT_EMOJI_FOLD),
            Dimension::ConfusableFold => Some(&DEFAULT_CONFUSABLE_FOLD),
        }
    }

    const fn index(self) -> usize {
        self as usize
    }
}

impl fmt::Display for Dimension {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let name = match self {
            Dimension::Original => "ORIGINAL",
            Dimension::Nfc => "NFC",
            Dimension::Nfkc => "NFKC",
            Dimension::Whitespace => "WHITESPACE",
            Dimension::Dash => "DASH",
            Dimension::CaseFold => "CASE_FOLD",
            Dimension::FullCaseFold => "FULL_CASE_FOLD",
            Dimension::AccentFold => "ACCENT_FOLD",
            Dimension::EmojiFold => "EMOJI_FOLD",
            Dimension::ConfusableFold => "CONFUSABLE_FOLD",
            Dimension::Stem => "STEM",
            Dimension::Lemma => "LEMMA",
        };
        f.write_str(name)
    }
}

/// Errors returned when configuring or running a [`TermAnalyzer`], [`Term`], or
/// [`crate::profile::NormalizationProfile`].
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum TermError {
    /// Both `caseFold()` and `fullCaseFold()` were configured on the same builder.
    BothCaseFolds,
    /// `transform(...)` was called with a non-character dimension (`ORIGINAL`, `STEM`, or `LEMMA`).
    NonCharacterTransform(Dimension),
    /// `maxTokenLength` was zero.
    InvalidMaxTokenLength(u32),
    /// `cacheSize` was zero.
    InvalidCacheCapacity(usize),
    /// `analyze_tokens` received slices of different lengths.
    TokensTagsLengthMismatch { tokens: usize, tags: usize },
    /// `analyze(text)` was called when `Dimension::Lemma` is configured in the chain.
    LemmaRequiresPosTagsInAnalyzeText,
    /// `Dimension::Stem` was requested without a configured [`Stemmer`].
    MissingStemmer,
    /// `Dimension::Lemma` was requested without a configured [`TermLemmatizer`].
    MissingLemmatizer,
    /// `Dimension::Lemma` was requested for a token whose POS tag was `None`.
    MissingPosTag { token: String },
    /// The configured [`TermLemmatizer`] returned no lemma for `token`.
    EmptyLemma { token: String },
    /// A [`crate::profile::NormalizationProfile`] was given a blank language code.
    BlankLanguage,
}

impl fmt::Display for TermError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            TermError::BothCaseFolds => f.write_str(
                "caseFold() and fullCaseFold() must not both be configured; fullCaseFold() already includes case folding plus the expanding folds, use one or the other",
            ),
            TermError::NonCharacterTransform(dim) => {
                write!(f, "transform(...) only applies to character-level dimensions, not {dim}")
            }
            TermError::InvalidMaxTokenLength(len) => {
                write!(f, "maxTokenLength must be at least 1, got {len}")
            }
            TermError::InvalidCacheCapacity(cap) => {
                write!(f, "cacheSize must be positive, got {cap}")
            }
            TermError::TokensTagsLengthMismatch { tokens, tags } => {
                write!(f, "tokens and tags must be the same length, got {tokens} and {tags}")
            }
            TermError::LemmaRequiresPosTagsInAnalyzeText => f.write_str(
                "Dimension LEMMA requires part-of-speech tags, which analyze(CharSequence) cannot supply; use analyze_tokens(tokens, tags)",
            ),
            TermError::MissingStemmer => {
                f.write_str("Dimension STEM requires a Stemmer; configure it with builder().stem(...)")
            }
            TermError::MissingLemmatizer => {
                f.write_str("Dimension LEMMA requires a Lemmatizer; configure it with builder().lemmatize(...)")
            }
            TermError::MissingPosTag { token } => write!(
                f,
                "Dimension LEMMA requires a part-of-speech tag, but the tag for token '{token}' was null; use analyze_tokens(tokens, tags) with a non-null tag per token"
            ),
            TermError::EmptyLemma { token } => {
                write!(f, "The Lemmatizer returned no lemma for token '{token}'")
            }
            TermError::BlankLanguage => f.write_str("language must not be blank"),
        }
    }
}

impl std::error::Error for TermError {}

/// Single-token lemmatization adapter used by [`TermAnalyzer`] for [`Dimension::Lemma`].
pub trait TermLemmatizer: Send + Sync {
    /// Lemmatizes a single `(token, pos_tag)` pair, or returns `None` if no
    /// lemma could be produced.
    fn lemmatize_token(&self, token: &str, pos_tag: &str) -> Option<String>;
}

impl TermLemmatizer for MaxentLemmatizer {
    fn lemmatize_token(&self, token: &str, pos_tag: &str) -> Option<String> {
        let mut lemmas = Strs::default();
        let mut probs = Vec::with_capacity(1);
        let ok = self.lemmatize(&[TokenTag { token, tag: pos_tag }], &mut lemmas, &mut probs).ok()?;
        if ok && !lemmas.is_empty() { Some(lemmas.get(0).to_owned()) } else { None }
    }
}

impl TermLemmatizer for DictionaryLemmatizer {
    fn lemmatize_token(&self, token: &str, pos_tag: &str) -> Option<String> {
        Some(self.lemma(token, pos_tag).to_owned())
    }
}

impl<F> TermLemmatizer for F
where
    F: Fn(&str, &str) -> Option<String> + Send + Sync,
{
    fn lemmatize_token(&self, token: &str, pos_tag: &str) -> Option<String> {
        (self)(token, pos_tag)
    }
}

fn apply_dyn_normalizer(normalizer: &dyn DynNormalizer, input: &str) -> String {
    let mut out = ViewBuilder::new(input);
    normalizer.normalize_dyn(input, &mut out);
    let (text, _) = out.finish();
    text
}

struct TermAnalyzerInner {
    chain: Vec<Dimension>,
    final_dimension: Dimension,
    transforms: [Option<Box<dyn DynNormalizer>>; NUM_DIMENSIONS],
    stemmer: Option<Box<dyn Stemmer>>,
    lemmatizer: Option<Box<dyn TermLemmatizer>>,
    tokenizer: WordTokenizer,
}

impl TermAnalyzerInner {
    fn apply(&self, dimension: Dimension, input: &str, pos_tag: Option<&str>) -> Result<String, TermError> {
        match dimension {
            Dimension::Original => Ok(input.to_owned()),
            Dimension::Stem => {
                let stemmer = self.stemmer.as_deref().ok_or(TermError::MissingStemmer)?;
                Ok(stemmer.stem(input))
            }
            Dimension::Lemma => {
                let lemmatizer = self.lemmatizer.as_deref().ok_or(TermError::MissingLemmatizer)?;
                let tag = pos_tag.ok_or_else(|| TermError::MissingPosTag { token: input.to_owned() })?;
                lemmatizer.lemmatize_token(input, tag).ok_or_else(|| TermError::EmptyLemma { token: input.to_owned() })
            }
            _ => {
                let normalizer = self.transforms[dimension.index()]
                    .as_deref()
                    .or_else(|| dimension.default_normalizer())
                    .expect("every character-level dimension has a default normalizer");
                Ok(apply_dyn_normalizer(normalizer, input))
            }
        }
    }
}

/// One token as a stack of normalization layers (`opennlp.tools.util.normalizer.Term`).
///
/// The [`Term::original`] form is the source of truth; the other layers are
/// derived, increasingly aggressive [`Dimension`]s tuned for matching. The
/// dimensions configured on the producing [`TermAnalyzer`] are computed
/// eagerly and cached; any other dimension is computed on first request,
/// applied on top of [`Term::normalized`], and then cached in a lock-free
/// [`OnceLock`] slot.
#[derive(Clone)]
pub struct Term {
    analyzer: Arc<TermAnalyzerInner>,
    span: Option<Span>,
    pos_tag: Option<String>,
    layers: [OnceLock<String>; NUM_DIMENSIONS],
}

impl PartialEq for Term {
    fn eq(&self, other: &Term) -> bool {
        self.original() == other.original()
            && self.normalized() == other.normalized()
            && self.span == other.span
            && self.pos_tag == other.pos_tag
    }
}

impl Eq for Term {}

impl fmt::Debug for Term {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Term")
            .field("original", &self.original())
            .field("normalized", &self.normalized())
            .field("span", &self.span)
            .field("pos_tag", &self.pos_tag)
            .finish()
    }
}

impl Term {
    fn new(
        analyzer: Arc<TermAnalyzerInner>,
        original: &str,
        span: Option<Span>,
        pos_tag: Option<&str>,
    ) -> Result<Term, TermError> {
        let layers: [OnceLock<String>; NUM_DIMENSIONS] = std::array::from_fn(|_| OnceLock::new());
        let mut value = original.to_owned();
        let _ = layers[Dimension::Original.index()].set(value.clone());
        for &dimension in &analyzer.chain {
            value = analyzer.apply(dimension, &value, pos_tag)?;
            let _ = layers[dimension.index()].set(value.clone());
        }
        Ok(Term { analyzer, span, pos_tag: pos_tag.map(str::to_owned), layers })
    }

    /// Returns the source byte span of this token, or `None` if it was
    /// supplied as a pre-tokenized string (`Term.span()`).
    pub fn span(&self) -> Option<Span> {
        self.span
    }

    /// Returns the original token text (`Term.original()`).
    pub fn original(&self) -> &str {
        self.layers[Dimension::Original.index()].get().expect("Dimension::Original is always populated on construction")
    }

    /// Returns the token at the analyzer's final configured dimension; equal
    /// to [`Term::original`] when no dimensions were configured (`Term.normalized()`).
    pub fn normalized(&self) -> &str {
        self.layers[self.analyzer.final_dimension.index()]
            .get()
            .expect("final configured dimension is always populated on construction")
    }

    /// Returns the token at `dimension` (`Term.at(Dimension)`).
    ///
    /// Configured dimensions are cached at construction; an unconfigured
    /// dimension is computed by applying its transform to [`Term::normalized`]
    /// and then cached so subsequent calls return the identical `&str` slice.
    pub fn at(&self, dimension: Dimension) -> Result<&str, TermError> {
        let slot = &self.layers[dimension.index()];
        if let Some(cached) = slot.get() {
            return Ok(cached.as_str());
        }
        let value = self.analyzer.apply(dimension, self.normalized(), self.pos_tag.as_deref())?;
        let _ = slot.set(value);
        Ok(slot.get().expect("slot was just initialized").as_str())
    }

    /// Returns the token at the dimension just below the final configured one
    /// (`Term.peel()`): the last-applied layer removed (for example the form
    /// before stemming when [`Dimension::Stem`] is the final dimension), or
    /// [`Term::original`] when at most one dimension is configured.
    pub fn peel(&self) -> &str {
        let chain = &self.analyzer.chain;
        if chain.len() < 2 {
            self.original()
        } else {
            let prev = chain[chain.len() - 2];
            self.layers[prev.index()].get().expect("all configured chain dimensions are populated on construction")
        }
    }
}

/// Builds [`Term`]s by segmenting text and applying a configured stack of
/// normalization [`Dimension`]s to each token (`opennlp.tools.util.normalizer.TermAnalyzer`).
#[derive(Clone)]
pub struct TermAnalyzer {
    inner: Arc<TermAnalyzerInner>,
}

impl fmt::Debug for TermAnalyzer {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("TermAnalyzer")
            .field("dimensions", &self.inner.chain)
            .field("final_dimension", &self.inner.final_dimension)
            .field("has_stemmer", &self.inner.stemmer.is_some())
            .field("has_lemmatizer", &self.inner.lemmatizer.is_some())
            .field("tokenizer", &self.inner.tokenizer)
            .finish()
    }
}

impl TermAnalyzer {
    /// Returns a new [`TermAnalyzerBuilder`] starting with no dimensions
    /// enabled and the default UAX #29 [`WordTokenizer`].
    pub fn builder() -> TermAnalyzerBuilder {
        TermAnalyzerBuilder::new()
    }

    /// Returns the configured dimensions that are computed eagerly, in
    /// canonical pipeline order (never including [`Dimension::Original`]).
    pub fn dimensions(&self) -> &[Dimension] {
        &self.inner.chain
    }

    /// Returns the last configured dimension in pipeline order, or
    /// [`Dimension::Original`] when none are configured.
    pub fn final_dimension(&self) -> Dimension {
        self.inner.final_dimension
    }

    /// Whether `self` and `other` are clones of one built analyzer, sharing its
    /// normalizers and stem cache.
    #[cfg(test)]
    pub(crate) fn shares_state_with(&self, other: &TermAnalyzer) -> bool {
        Arc::ptr_eq(&self.inner, &other.inner)
    }

    /// Returns whether this analyzer has a configured [`Stemmer`].
    pub fn has_stemmer(&self) -> bool {
        self.inner.stemmer.is_some()
    }

    /// Returns whether this analyzer has a configured [`TermLemmatizer`].
    pub fn has_lemmatizer(&self) -> bool {
        self.inner.lemmatizer.is_some()
    }

    /// Segments `text` with the configured UAX #29 [`WordTokenizer`] and
    /// returns one [`Term`] per word token, in order (`TermAnalyzer.analyze(CharSequence)`).
    ///
    /// # Errors
    ///
    /// Returns [`TermError::LemmaRequiresPosTagsInAnalyzeText`] if
    /// [`Dimension::Lemma`] is configured, because no part-of-speech tags are
    /// available from raw text.
    pub fn analyze(&self, text: &str) -> Result<Vec<Term>, TermError> {
        if self.inner.chain.contains(&Dimension::Lemma) {
            return Err(TermError::LemmaRequiresPosTagsInAnalyzeText);
        }
        let mut spans = Vec::new();
        let mut types = Vec::new();
        word_tokens(text, self.inner.tokenizer.max_token_length(), &mut spans, &mut types);
        let mut terms = Vec::with_capacity(spans.len());
        for span in spans {
            let token_text = &text[span.range()];
            terms.push(Term::new(Arc::clone(&self.inner), token_text, Some(span), None)?);
        }
        Ok(terms)
    }

    /// Returns one [`Term`] per supplied token, attaching the matching
    /// part-of-speech tag (`Some(tag)` or `None`) so [`Dimension::Lemma`] can
    /// be computed (`TermAnalyzer.analyze(String[], String[])`). The returned
    /// terms have `span() == None`.
    pub fn analyze_tokens(&self, tokens: &[&str], tags: &[Option<&str>]) -> Result<Vec<Term>, TermError> {
        if tokens.len() != tags.len() {
            return Err(TermError::TokensTagsLengthMismatch { tokens: tokens.len(), tags: tags.len() });
        }
        let mut terms = Vec::with_capacity(tokens.len());
        for (&token, &tag) in tokens.iter().zip(tags) {
            terms.push(Term::new(Arc::clone(&self.inner), token, None, tag)?);
        }
        Ok(terms)
    }

    /// Convenience wrapper around [`TermAnalyzer::analyze_tokens`] when every
    /// token has a part-of-speech tag.
    pub fn analyze_tagged(&self, tokens: &[&str], tags: &[&str]) -> Result<Vec<Term>, TermError> {
        if tokens.len() != tags.len() {
            return Err(TermError::TokensTagsLengthMismatch { tokens: tokens.len(), tags: tags.len() });
        }
        let mut terms = Vec::with_capacity(tokens.len());
        for (&token, &tag) in tokens.iter().zip(tags) {
            terms.push(Term::new(Arc::clone(&self.inner), token, None, Some(tag))?);
        }
        Ok(terms)
    }
}

/// Locale-aware lowercaser for `TermAnalyzerBuilder::case_fold_locale`:
/// applies Turkic simple case folding for Turkish (`tr`/`tur`) and Azeri
/// (`az`/`aze`), and `Locale.ROOT` lowercasing otherwise.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
struct LocaleCaseFold {
    turkic: bool,
}

impl Normalizer for LocaleCaseFold {
    const ID: &'static str = "turbo:normalize:locale-case-fold";

    fn normalize(&self, text: &str, out: &mut ViewBuilder<'_>) {
        if self.turkic {
            CaseFold::turkic_simple().normalize(text, out);
        } else {
            CaseFold::lowercase().normalize(text, out);
        }
    }
}

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
        let turkic = matches!(primary.as_str(), "tr" | "tur" | "az" | "aze");
        self.transform(Dimension::CaseFold, LocaleCaseFold { turkic })
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

#[cfg(test)]
mod tests {
    use std::sync::Barrier;

    use turbo_normalize::{ConfusableSkeleton, GermanUmlauts, Whitespace, WhitespaceMode, apply};

    use super::*;
    use crate::porter::PorterStemmer;
    use crate::snowball::{SnowballAlgorithm, SnowballStemmer};

    #[test]
    fn dimension_exposes_character_defaults_and_none_for_token_dimensions() {
        for dim in [
            Dimension::Nfc,
            Dimension::Nfkc,
            Dimension::Whitespace,
            Dimension::Dash,
            Dimension::CaseFold,
            Dimension::FullCaseFold,
            Dimension::AccentFold,
            Dimension::EmojiFold,
            Dimension::ConfusableFold,
        ] {
            assert!(dim.is_character_level(), "{dim}");
            assert!(dim.default_normalizer().is_some(), "{dim}");
        }
        for dim in [Dimension::Original, Dimension::Stem, Dimension::Lemma] {
            assert!(!dim.is_character_level(), "{dim}");
            assert!(dim.default_normalizer().is_none(), "{dim}");
        }
    }

    #[test]
    fn no_dimensions_leaves_token_unchanged() {
        let analyzer = TermAnalyzer::builder().build();
        let terms = analyzer.analyze("Hello").unwrap();
        let term = &terms[0];
        assert_eq!(term.original(), "Hello");
        assert_eq!(term.normalized(), "Hello");
        assert_eq!(term.peel(), "Hello");
        assert!(analyzer.dimensions().is_empty());
    }

    #[test]
    fn chain_applies_in_pipeline_order_regardless_of_builder_order() {
        let analyzer = TermAnalyzer::builder().accent_fold().case_fold().unwrap().build();
        assert_eq!(analyzer.dimensions(), &[Dimension::CaseFold, Dimension::AccentFold]);
        let terms = analyzer.analyze("CAF\u{00C9}").unwrap();
        let term = &terms[0];
        assert_eq!(term.original(), "CAF\u{00C9}");
        assert_eq!(term.normalized(), "cafe");
        assert_eq!(term.peel(), "caf\u{00E9}");
    }

    #[test]
    fn full_case_fold_orders_before_accent_fold_regardless_of_builder_order() {
        let analyzer = TermAnalyzer::builder().accent_fold().full_case_fold().unwrap().build();
        assert_eq!(analyzer.dimensions(), &[Dimension::FullCaseFold, Dimension::AccentFold]);
        let terms = analyzer.analyze("CAF\u{00C9}").unwrap();
        let term = &terms[0];
        assert_eq!(term.normalized(), "cafe");
        assert_eq!(term.peel(), "caf\u{00E9}");
    }

    #[test]
    fn stem_is_the_top_layer_and_peel_returns_layer_below() {
        let analyzer = TermAnalyzer::builder().case_fold().unwrap().stem(PorterStemmer).build();
        let terms = analyzer.analyze("Running").unwrap();
        let term = &terms[0];
        assert_eq!(term.peel(), "running");
        assert_eq!(term.normalized(), "run");
        assert_eq!(term.at(Dimension::Stem).unwrap(), "run");
    }

    #[test]
    fn unconfigured_char_dimension_is_computed_lazily_and_memoized() {
        let analyzer = TermAnalyzer::builder().build();
        let terms = analyzer.analyze("HELLO").unwrap();
        let term = &terms[0];
        assert_eq!(term.normalized(), "HELLO");
        let first = term.at(Dimension::CaseFold).unwrap();
        let second = term.at(Dimension::CaseFold).unwrap();
        assert_eq!(first, "hello");
        assert!(std::ptr::eq(first, second));
    }

    #[test]
    fn stem_and_lemma_without_engines_fail_loudly() {
        let analyzer = TermAnalyzer::builder().case_fold().unwrap().build();
        let terms = analyzer.analyze("running").unwrap();
        let term = &terms[0];
        assert_eq!(term.at(Dimension::Stem), Err(TermError::MissingStemmer));
        assert_eq!(term.at(Dimension::Lemma), Err(TermError::MissingLemmatizer));
    }

    #[test]
    fn analyze_text_produces_spans_and_analyze_tokens_has_none() {
        let analyzer = TermAnalyzer::builder().case_fold().unwrap().build();
        let terms = analyzer.analyze("The Cats").unwrap();
        assert_eq!(terms.len(), 2);
        assert_eq!(terms[0].original(), "The");
        assert_eq!(terms[0].normalized(), "the");
        assert_eq!(terms[0].span(), Some(Span::new(0, 3)));
        assert_eq!(terms[1].original(), "Cats");
        assert_eq!(terms[1].span(), Some(Span::new(4, 8)));

        let pre = analyzer.analyze_tagged(&["Cats"], &["NNS"]).unwrap();
        assert_eq!(pre[0].span(), None);
        assert_eq!(pre[0].normalized(), "cats");
    }

    #[test]
    fn lemma_dimension_with_lemmatizer_and_error_cases() {
        let analyzer = TermAnalyzer::builder()
            .case_fold()
            .unwrap()
            .lemmatize(|tok: &str, tag: &str| if tok == "was" && tag == "VBD" { Some("be".to_owned()) } else { None })
            .build();
        // Fails fast on raw text even if empty.
        assert_eq!(analyzer.analyze(""), Err(TermError::LemmaRequiresPosTagsInAnalyzeText));
        assert_eq!(analyzer.analyze("was"), Err(TermError::LemmaRequiresPosTagsInAnalyzeText));

        let terms = analyzer.analyze_tagged(&["was"], &["VBD"]).unwrap();
        assert_eq!(terms[0].normalized(), "be");

        // Missing POS tag fails loudly.
        assert_eq!(
            analyzer.analyze_tokens(&["was"], &[None]).unwrap_err(),
            TermError::MissingPosTag { token: "was".to_owned() }
        );
        // Lemmatizer returning None fails loudly instead of looping.
        assert_eq!(
            analyzer.analyze_tagged(&["unknown"], &["NN"]).unwrap_err(),
            TermError::EmptyLemma { token: "unknown".to_owned() }
        );
    }

    #[test]
    fn builder_guards_and_options_match_opennlp() {
        assert_eq!(TermAnalyzer::builder().case_fold().unwrap().full_case_fold().err(), Some(TermError::BothCaseFolds));
        assert_eq!(TermAnalyzer::builder().full_case_fold().unwrap().case_fold().err(), Some(TermError::BothCaseFolds));
        assert_eq!(
            TermAnalyzer::builder().full_case_fold().unwrap().case_fold_locale("tr").err(),
            Some(TermError::BothCaseFolds)
        );
        assert_eq!(
            TermAnalyzer::builder().transform(Dimension::Stem, CaseFold::lowercase()).err(),
            Some(TermError::NonCharacterTransform(Dimension::Stem))
        );
        assert_eq!(TermAnalyzer::builder().max_token_length(0).err(), Some(TermError::InvalidMaxTokenLength(0)));
        assert_eq!(
            TermAnalyzer::builder()
                .stem_cached_with_capacity(SnowballStemmer::new(SnowballAlgorithm::English), 0)
                .err(),
            Some(TermError::InvalidCacheCapacity(0))
        );
    }

    #[test]
    fn multilingual_and_character_dimension_behaviors_match_opennlp() {
        // German umlauts fold to digraphs.
        let german = TermAnalyzer::builder()
            .case_fold()
            .unwrap()
            .transform(Dimension::AccentFold, GermanUmlauts::new())
            .unwrap()
            .build();
        assert_eq!(german.analyze("GR\u{00DC}\u{00DF}E").unwrap()[0].normalized(), "gruesse");
        assert_eq!(german.analyze("\u{00DC}ber").unwrap()[0].normalized(), "ueber");

        // Generic accent fold strips Romance diacritics.
        let romance = TermAnalyzer::builder().case_fold().unwrap().accent_fold().build();
        assert_eq!(romance.analyze("\u{00C9}l\u{00E8}ve").unwrap()[0].normalized(), "eleve");
        assert_eq!(romance.analyze("Ni\u{00F1}o").unwrap()[0].normalized(), "nino");
        assert_eq!(romance.analyze("Canci\u{00F3}n").unwrap()[0].normalized(), "cancion");

        // Cyrillic case fold.
        let cyrillic = TermAnalyzer::builder().case_fold().unwrap().build();
        assert_eq!(
            cyrillic.analyze("\u{041C}\u{041E}\u{0421}\u{041A}\u{0412}\u{0410}").unwrap()[0].normalized(),
            "\u{043C}\u{043E}\u{0441}\u{043A}\u{0432}\u{0430}"
        );

        // Fullwidth A under NFKC + caseFold.
        let nfkc = TermAnalyzer::builder().nfkc().case_fold().unwrap().build();
        assert_eq!(nfkc.analyze("\u{FF21}").unwrap()[0].normalized(), "a");

        // Kanji unchanged under NFC.
        let nfc = TermAnalyzer::builder().nfc().build();
        assert_eq!(nfc.analyze("\u{6771}").unwrap()[0].normalized(), "\u{6771}");

        // Confusable fold composes with case fold.
        let confusable = TermAnalyzer::builder().case_fold().unwrap().confusable_fold().build();
        let spoof = "P\u{0430}yp\u{0430}l";
        assert_eq!(confusable.analyze(spoof).unwrap()[0].normalized(), apply(&ConfusableSkeleton, "paypal"));

        // Turkish locale case fold maps I to dotless i (U+0131).
        let tr = TermAnalyzer::builder().case_fold_locale("tr").unwrap().build();
        assert_eq!(tr.analyze("I").unwrap()[0].normalized(), "\u{0131}");

        // Accent fold restricted to Latin without stroke letters.
        let lat_only = TermAnalyzer::builder().accent_fold_scripts(&[Script::Latin], false).build();
        assert_eq!(lat_only.analyze("caf\u{00E9}").unwrap()[0].normalized(), "cafe");

        // Custom whitespace normalizer.
        let ws =
            TermAnalyzer::builder().whitespace_with(Whitespace::new(WhitespaceMode::KeepLineBreaks)).unwrap().build();
        assert_eq!(ws.analyze_tagged(&["a\n\n\tb"], &["X"]).unwrap()[0].normalized(), "a\nb");

        // Full case fold expands sharp s.
        let full = TermAnalyzer::builder().full_case_fold().unwrap().build();
        let term = &full.analyze("Ma\u{00DF}e").unwrap()[0];
        assert_eq!(term.normalized(), "masse");
        assert_eq!(term.at(Dimension::FullCaseFold).unwrap(), "masse");

        // Emoji fold layers emoticon on token.
        let emoji = TermAnalyzer::builder().emoji_fold().build();
        let grin = &emoji.analyze("\u{1F600}").unwrap()[0];
        assert_eq!(grin.at(Dimension::EmojiFold).unwrap(), ":D");
        assert_eq!(grin.at(Dimension::Original).unwrap(), "\u{1F600}");
        assert_eq!(emoji.analyze("hello").unwrap()[0].at(Dimension::EmojiFold).unwrap(), "hello");

        // Max token length chops tokens in UTF-16 units.
        let chopped = TermAnalyzer::builder().max_token_length(3).unwrap().build();
        let parts = chopped.analyze("abcdefg").unwrap();
        let originals: Vec<&str> = parts.iter().map(Term::original).collect();
        assert_eq!(originals, ["abc", "def", "g"]);
    }

    #[test]
    fn concurrent_lazy_access_returns_same_cached_slice() {
        let analyzer = TermAnalyzer::builder().build();
        let term = Arc::new(analyzer.analyze("HELLO").unwrap().into_iter().next().unwrap());
        let threads = 8;
        let barrier = Arc::new(Barrier::new(threads));
        let mut handles = Vec::with_capacity(threads);
        for _ in 0..threads {
            let t = Arc::clone(&term);
            let b = Arc::clone(&barrier);
            handles.push(std::thread::spawn(move || {
                b.wait();
                let s = t.at(Dimension::CaseFold).unwrap();
                (s.to_owned(), s.as_ptr() as usize)
            }));
        }
        let first = handles.remove(0).join().unwrap();
        assert_eq!(first.0, "hello");
        for h in handles {
            let res = h.join().unwrap();
            assert_eq!(res, first);
        }
    }
}
