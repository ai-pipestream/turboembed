//! OpenNLP's `Dimension`, `Term`, and `TermAnalyzer`: layered per-token
//! normalization with eager evaluation of configured dimensions and thread-safe
//! lazy memoization of unconfigured dimensions.

mod builder;
#[cfg(test)]
mod tests;

use std::fmt;
use std::sync::{Arc, LazyLock, OnceLock};

pub use builder::TermAnalyzerBuilder;
use turbo_analysis::{Span, ViewBuilder};
use turbo_analysis_layers::Strs;
use turbo_lemmatizer::{DictionaryLemmatizer, MaxentLemmatizer, TokenTag};
use turbo_normalize::{
    AccentFold, CaseFold, ConfusableSkeleton, Dashes, DynNormalizer, EmojiToEmoticon, Normalizer, UnicodeForm,
    Whitespace,
};
use turbo_tokenize::{WordTokenizer, word_tokens};

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

    pub(crate) const fn index(self) -> usize {
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

    /// Returns `(is_custom, normalizer_id)` for a character-level `dimension`,
    /// or `None` for non-character dimensions.
    pub(crate) fn normalizer_info(&self, dimension: Dimension) -> Option<(bool, &'static str)> {
        if let Some(custom) = &self.inner.transforms[dimension.index()] {
            Some((true, custom.id()))
        } else {
            dimension.default_normalizer().map(|n| (false, n.id()))
        }
    }

    /// Evaluates the configured normalization chain for a single token in
    /// [`crate::profile::TermAnnotator`], returning `(folded, stem, lemma)`
    /// without allocating a [`Term`] or its [`OnceLock`] layer table.
    ///
    /// When `skip_lemma` is `true` (because the document already supplies an
    /// `opennlp:lemmas` layer), [`Dimension::Lemma`] is not evaluated.
    pub(crate) fn analyze_for_annotator(
        &self,
        token: &str,
        pos_tag: Option<&str>,
        skip_lemma: bool,
    ) -> Result<(String, Option<String>, Option<String>), TermError> {
        let mut folded = token.to_owned();
        for &dim in &self.inner.chain {
            if dim.is_character_level() {
                folded = self.inner.apply(dim, &folded, pos_tag)?;
            }
        }
        let stem = if self.inner.stemmer.is_some() {
            Some(self.inner.apply(Dimension::Stem, &folded, pos_tag)?)
        } else {
            None
        };
        let lemma = if !skip_lemma && self.inner.lemmatizer.is_some() {
            let input = stem.as_deref().unwrap_or(&folded);
            Some(self.inner.apply(Dimension::Lemma, input, pos_tag)?)
        } else {
            None
        };
        Ok((folded, stem, lemma))
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

/// Turkic simple case folding for `TermAnalyzerBuilder::case_fold_locale`
/// when the locale is Turkish (`tr`/`tur`) or Azeri (`az`/`aze`).
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
struct TurkicCaseFold;

impl Normalizer for TurkicCaseFold {
    const ID: &'static str = "turbo:normalize:turkic-case-fold";

    fn normalize(&self, text: &str, out: &mut ViewBuilder<'_>) {
        CaseFold::turkic_simple().normalize(text, out);
    }
}

/// Non-Turkic `Locale.ROOT` lowercaser for `TermAnalyzerBuilder::case_fold_locale`.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
struct LocaleCaseFold;

impl Normalizer for LocaleCaseFold {
    const ID: &'static str = "turbo:normalize:locale-case-fold";

    fn normalize(&self, text: &str, out: &mut ViewBuilder<'_>) {
        CaseFold::lowercase().normalize(text, out);
    }
}
