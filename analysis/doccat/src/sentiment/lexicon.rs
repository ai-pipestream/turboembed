//! Deterministic lexicon and negation-scope sentiment scorer (#320).
//!
//! Ports the rule-based sentiment model of Hutto and Gilbert (2014),
//! *VADER: A Parsimonious Rule-based Model for Sentiment Analysis of Social
//! Media Text* (ICWSM-14), onto the layered analysis API:
//!
//! - [`VaderLexicon`] loads the MIT-licensed VADER lexicon (`vader_lexicon.txt`)
//!   or a caller-supplied TSV table (`token\tvalence`).
//! - [`LexiconSentiment::score_text`] runs in VADER-compatible raw-text mode
//!   and reproduces Python `vaderSentiment`'s `SentimentIntensityAnalyzer`
//!   scores (`neg`, `neu`, `pos`, `compound`) across its reference suite.
//! - [`LexiconSentiment::score_sentence`] and the [`Annotator`] implementation
//!   (`turbo-doccat:lexicon-sentiment`) read `opennlp:sentences` and
//!   `opennlp:tokens`, plus optional `opennlp:dependencies` and `turbo:emoji`,
//!   and write `turbo:opinions` ([`Opinions`], with exact token evidence spans)
//!   and sentence-level polarity in `turbo:categories<opennlp:sentences>`
//!   ([`Categories<Sentences>`]).
//! - Emoji scoring uses OpenNLP's own Apache-2.0 `emoji-annotations.txt`
//!   ([`turbo_normalize::emoji::opennlp_annotation`]) rather than any external
//!   CC BY-SA emoji ranking table.

use std::collections::HashMap;
use std::hash::{Hash, Hasher};
use std::sync::{Arc, OnceLock};

use turbo_analysis::{Annotator, Bound, Error, Ix, IxRange, Key, Layer, ModelRef, Refs, Span, Tag, TagSet, View};
use turbo_analysis_layers::{Categories, Dependencies, Opinion, Opinions, Polarity, Sentences, Tokens};
use turbo_maxent::sha256;
use turbo_normalize::emoji::opennlp_annotation;
use turbo_normalize_layers::Emoji;

/// Empirically derived mean sentiment intensity rating increase for booster words.
pub const B_INCR: f64 = 0.293;

/// Empirically derived mean sentiment intensity rating decrease for dampener words.
pub const B_DECR: f64 = -0.293;

/// Empirically derived mean sentiment intensity rating increase for ALLCAPS emphasis.
pub const C_INCR: f64 = 0.733;

/// Negation multiplier applied to negated valence scores.
pub const N_SCALAR: f64 = -0.74;

/// Normalization constant (`alpha`) for computing the compound score in `[-1.0, 1.0]`.
pub const DEFAULT_ALPHA: f64 = 15.0;

/// Standard VADER compound threshold (`0.05`) for classifying positive (`>= 0.05`),
/// negative (`<= -0.05`), and neutral (`(-0.05, 0.05)`) sentences.
pub const COMPOUND_THRESHOLD: f64 = 0.05;

/// Named instance of `Categories<Sentences>` written by [`LexiconSentiment::bound`].
pub const SENTENCE_SENTIMENT: Key<Categories<Sentences>> = Key::named("sentiment");

const BUNDLED_LEXICON_TSV: &str = include_str!("vader_lexicon.txt");

/// The 59 negation words in `vaderSentiment`.
const NEGATE_WORDS: [&str; 59] = [
    "aint",
    "arent",
    "cannot",
    "cant",
    "couldnt",
    "darent",
    "didnt",
    "doesnt",
    "ain't",
    "aren't",
    "can't",
    "couldn't",
    "daren't",
    "didn't",
    "doesn't",
    "dont",
    "hadnt",
    "hasnt",
    "havent",
    "isnt",
    "mightnt",
    "mustnt",
    "neither",
    "don't",
    "hadn't",
    "hasn't",
    "haven't",
    "isn't",
    "mightn't",
    "mustn't",
    "neednt",
    "needn't",
    "never",
    "none",
    "nope",
    "nor",
    "not",
    "nothing",
    "nowhere",
    "oughtnt",
    "shant",
    "shouldnt",
    "uhuh",
    "wasnt",
    "werent",
    "oughtn't",
    "shan't",
    "shouldn't",
    "uh-uh",
    "wasn't",
    "weren't",
    "without",
    "wont",
    "wouldnt",
    "won't",
    "wouldn't",
    "rarely",
    "seldom",
    "despite",
];

/// Booster and dampener degree modifiers (`BOOSTER_DICT` in `vaderSentiment`).
const BOOSTERS: &[(&str, f64)] = &[
    ("absolutely", B_INCR),
    ("amazingly", B_INCR),
    ("awfully", B_INCR),
    ("completely", B_INCR),
    ("considerable", B_INCR),
    ("considerably", B_INCR),
    ("decidedly", B_INCR),
    ("deeply", B_INCR),
    ("effing", B_INCR),
    ("enormous", B_INCR),
    ("enormously", B_INCR),
    ("entirely", B_INCR),
    ("especially", B_INCR),
    ("exceptional", B_INCR),
    ("exceptionally", B_INCR),
    ("extreme", B_INCR),
    ("extremely", B_INCR),
    ("fabulously", B_INCR),
    ("flipping", B_INCR),
    ("flippin", B_INCR),
    ("frackin", B_INCR),
    ("fracking", B_INCR),
    ("fricking", B_INCR),
    ("frickin", B_INCR),
    ("frigging", B_INCR),
    ("friggin", B_INCR),
    ("fully", B_INCR),
    ("fuckin", B_INCR),
    ("fucking", B_INCR),
    ("fuggin", B_INCR),
    ("fugging", B_INCR),
    ("greatly", B_INCR),
    ("hella", B_INCR),
    ("highly", B_INCR),
    ("hugely", B_INCR),
    ("incredible", B_INCR),
    ("incredibly", B_INCR),
    ("intensely", B_INCR),
    ("major", B_INCR),
    ("majorly", B_INCR),
    ("more", B_INCR),
    ("most", B_INCR),
    ("particularly", B_INCR),
    ("purely", B_INCR),
    ("quite", B_INCR),
    ("really", B_INCR),
    ("remarkably", B_INCR),
    ("so", B_INCR),
    ("substantially", B_INCR),
    ("thoroughly", B_INCR),
    ("total", B_INCR),
    ("totally", B_INCR),
    ("tremendous", B_INCR),
    ("tremendously", B_INCR),
    ("uber", B_INCR),
    ("unbelievably", B_INCR),
    ("unusually", B_INCR),
    ("utter", B_INCR),
    ("utterly", B_INCR),
    ("very", B_INCR),
    ("almost", B_DECR),
    ("barely", B_DECR),
    ("hardly", B_DECR),
    ("just enough", B_DECR),
    ("kind of", B_DECR),
    ("kinda", B_DECR),
    ("kindof", B_DECR),
    ("kind-of", B_DECR),
    ("less", B_DECR),
    ("little", B_DECR),
    ("marginal", B_DECR),
    ("marginally", B_DECR),
    ("occasional", B_DECR),
    ("occasionally", B_DECR),
    ("partly", B_DECR),
    ("scarce", B_DECR),
    ("scarcely", B_DECR),
    ("slight", B_DECR),
    ("slightly", B_DECR),
    ("somewhat", B_DECR),
    ("sort of", B_DECR),
    ("sorta", B_DECR),
    ("sortof", B_DECR),
    ("sort-of", B_DECR),
];

/// Special-case idioms containing lexicon words (`SPECIAL_CASES` in `vaderSentiment`).
const SPECIAL_CASES: &[(&str, f64)] = &[
    ("the shit", 3.0),
    ("the bomb", 3.0),
    ("bad ass", 1.5),
    ("badass", 1.5),
    ("bus stop", 0.0),
    ("yeah right", -2.0),
    ("kiss of death", -1.5),
    ("to die for", 3.0),
    ("beating heart", 3.1),
    ("broken heart", -2.9),
];

#[inline]
fn booster_value(lower: &str) -> Option<f64> {
    BOOSTERS.iter().find(|&&(k, _)| k == lower).map(|&(_, v)| v)
}

#[inline]
fn special_case_value(lower: &str) -> Option<f64> {
    SPECIAL_CASES.iter().find(|&&(k, _)| k == lower).map(|&(_, v)| v)
}

/// Returns `true` if `word_lower` is a VADER negation word or contains `"n't"`.
#[inline]
pub fn is_negation_word(word_lower: &str) -> bool {
    NEGATE_WORDS.contains(&word_lower) || word_lower.contains("n't")
}

/// Normalizes a raw valence sum into `[-1.0, 1.0]` using `score / sqrt(score^2 + alpha)`.
#[inline]
pub fn normalize_score(score: f64, alpha: f64) -> f64 {
    let norm = score / (score * score + alpha).sqrt();
    norm.clamp(-1.0, 1.0)
}

/// Checks whether a token is ALL CAPS in Python's `str.isupper()` sense:
/// at least one cased character, and no lowercase character.
#[inline]
pub fn is_all_caps(word: &str) -> bool {
    let mut has_cased = false;
    for ch in word.chars() {
        if ch.is_lowercase() {
            return false;
        }
        if ch.is_uppercase() {
            has_cased = true;
        }
    }
    has_cased
}

/// Returns `true` if some (`> 0`) but not all (`< words.len()`) tokens are ALL CAPS.
pub fn allcap_differential<S: AsRef<str>>(words: &[S]) -> bool {
    let allcap_words = words.iter().filter(|w| is_all_caps(w.as_ref())).count();
    allcap_words > 0 && allcap_words < words.len()
}

/// Strips leading and trailing ASCII punctuation from `token` if the stripped
/// string has more than 2 characters (`chars().count() > 2`), matching
/// `SentiText._strip_punc_if_word` in `vaderSentiment`.
pub fn strip_punc_if_word(token: &str) -> &str {
    let stripped = token.trim_matches(|c: char| c.is_ascii_punctuation());
    if stripped.chars().count() <= 2 { token } else { stripped }
}

/// Computes the booster/dampener scalar for a preceding modifier token,
/// negated when `valence < 0.0` and amplified by `C_INCR` when `word` is
/// ALL CAPS in a mixed-case sentence.
pub fn scalar_inc_dec(word: &str, valence: f64, is_cap_diff: bool) -> f64 {
    let lower = word.to_lowercase();
    let Some(mut scalar) = booster_value(&lower) else {
        return 0.0;
    };
    if valence < 0.0 {
        scalar *= -1.0;
    }
    if is_all_caps(word) && is_cap_diff {
        if valence > 0.0 {
            scalar += C_INCR;
        } else {
            scalar -= C_INCR;
        }
    }
    scalar
}

/// Rounds `x` to `decimals` decimal places using Python's round-half-to-even rule.
fn round_half_to_even(x: f64, decimals: u32) -> f64 {
    let factor = 10_f64.powi(decimals as i32);
    let scaled = x * factor;
    let floor = scaled.floor();
    let diff = scaled - floor;
    let eps = 1e-12;
    let rounded = if (diff - 0.5).abs() <= eps {
        let f_int = floor as i64;
        if f_int % 2 == 0 { floor } else { floor + 1.0 }
    } else {
        scaled.round()
    };
    rounded / factor
}

/// Sentiment lexicon mapping lowercase token strings to prior valence scores.
#[derive(Clone, Debug)]
pub struct VaderLexicon {
    entries: HashMap<String, f64>,
    sha256_hex: String,
}

static BUNDLED_LEXICON: OnceLock<Arc<VaderLexicon>> = OnceLock::new();

impl VaderLexicon {
    /// Returns the bundled MIT-licensed VADER lexicon (`vader_lexicon.txt`).
    pub fn bundled() -> Arc<VaderLexicon> {
        BUNDLED_LEXICON
            .get_or_init(|| {
                Arc::new(VaderLexicon::from_tsv(BUNDLED_LEXICON_TSV).expect("bundled vader_lexicon.txt is valid TSV"))
            })
            .clone()
    }

    /// Parses a tab-separated lexicon (`word\tvalence`, ignoring empty lines
    /// and `#` comment lines). Extra columns, such as the standard deviation
    /// and raw ratings in the upstream file, are ignored.
    pub fn from_tsv(tsv: &str) -> Result<VaderLexicon, Error> {
        let invalid = |line_no: usize, detail: String| Error::Annotator {
            annotator: LexiconSentiment::ID,
            detail: format!("lexicon line {}: {detail}", line_no + 1),
        };
        let mut entries = HashMap::new();
        for (line_no, raw_line) in tsv.lines().enumerate() {
            let line = raw_line.trim();
            if line.is_empty() || line.starts_with('#') {
                continue;
            }
            let mut fields = line.split('\t');
            let Some(word) = fields.next() else {
                return Err(invalid(line_no, "missing word column".into()));
            };
            let Some(measure_str) = fields.next() else {
                return Err(invalid(line_no, format!("missing valence column for {word:?}")));
            };
            let measure: f64 = measure_str
                .trim()
                .parse()
                .map_err(|e| invalid(line_no, format!("invalid valence {measure_str:?}: {e}")))?;
            if !measure.is_finite() {
                return Err(invalid(line_no, format!("non-finite valence for {word:?}")));
            }
            entries.insert(word.to_string(), measure);
        }
        let mut lexicon = VaderLexicon { entries, sha256_hex: String::new() };
        lexicon.recompute_digest();
        Ok(lexicon)
    }

    /// Recomputes the digest over the entries in canonical form (sorted
    /// `word\tvalence` lines), so two tables with the same entries share a
    /// digest whatever their comments or column layout.
    fn recompute_digest(&mut self) {
        let mut words: Vec<&String> = self.entries.keys().collect();
        words.sort_unstable();
        let mut canonical = String::new();
        for word in words {
            canonical.push_str(word);
            canonical.push('\t');
            canonical.push_str(&self.entries[word].to_string());
            canonical.push('\n');
        }
        self.sha256_hex = sha256(canonical.as_bytes()).to_hex();
    }

    /// Looks up the prior valence of `word_lower`.
    #[inline]
    pub fn get(&self, word_lower: &str) -> Option<f64> {
        self.entries.get(word_lower).copied()
    }

    /// Returns `true` if `word_lower` is present in the lexicon.
    #[inline]
    pub fn contains(&self, word_lower: &str) -> bool {
        self.entries.contains_key(word_lower)
    }

    /// Inserts or updates a lowercase lexicon entry and recomputes the digest.
    pub fn insert(&mut self, word_lower: impl Into<String>, valence: f64) {
        self.entries.insert(word_lower.into(), valence);
        self.recompute_digest();
    }

    /// Number of entries in the lexicon.
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Whether the lexicon has no entries.
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// SHA-256 hex digest of the entries in canonical form (sorted
    /// `word\tvalence` lines).
    pub fn sha256_hex(&self) -> &str {
        &self.sha256_hex
    }
}

/// How negation scope is determined when scoring tokenized sentences.
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq, Hash)]
pub enum NegationMode {
    /// Use the `opennlp:dependencies` layer when present on the document,
    /// falling back to VADER's three-token preceding window when absent.
    #[default]
    Auto,
    /// Always use VADER's three-token preceding window, even if a dependency
    /// layer is present.
    Window,
    /// Use dependency edges when provided, falling back to VADER's three-token
    /// window only when no dependency layer is supplied.
    Dependency,
}

impl NegationMode {
    pub const fn model_id(self) -> &'static str {
        match self {
            NegationMode::Auto => "vader:lexicon:auto",
            NegationMode::Window => "vader:lexicon:window",
            NegationMode::Dependency => "vader:lexicon:dependency",
        }
    }
}

/// VADER polarity and compound scores for a text or sentence.
#[derive(Copy, Clone, Debug, Default, PartialEq)]
pub struct VaderScore {
    /// Negative proportion (`0.0..=1.0`).
    pub neg: f64,
    /// Neutral proportion (`0.0..=1.0`).
    pub neu: f64,
    /// Positive proportion (`0.0..=1.0`).
    pub pos: f64,
    /// Normalized, weighted composite score (`-1.0..=1.0`).
    pub compound: f64,
}

impl VaderScore {
    /// `neg` rounded to 3 decimal places as `vaderSentiment` reports it.
    pub fn neg_rounded(self) -> f64 {
        round_half_to_even(self.neg, 3)
    }

    /// `neu` rounded to 3 decimal places as `vaderSentiment` reports it.
    pub fn neu_rounded(self) -> f64 {
        round_half_to_even(self.neu, 3)
    }

    /// `pos` rounded to 3 decimal places as `vaderSentiment` reports it.
    pub fn pos_rounded(self) -> f64 {
        round_half_to_even(self.pos, 3)
    }

    /// `compound` rounded to 4 decimal places as `vaderSentiment` reports it.
    pub fn compound_rounded(self) -> f64 {
        round_half_to_even(self.compound, 4)
    }

    /// Classifies `compound` using VADER's standard `+/- 0.05` threshold.
    pub fn polarity(self) -> Polarity {
        if self.compound >= COMPOUND_THRESHOLD {
            Polarity::Positive
        } else if self.compound <= -COMPOUND_THRESHOLD {
            Polarity::Negative
        } else {
            Polarity::Neutral
        }
    }
}

/// Detailed sentiment result for one tokenized sentence.
#[derive(Clone, Debug, PartialEq)]
pub struct SentenceSentiment {
    /// Sentence-level `neg`, `neu`, `pos`, and `compound` scores.
    pub score: VaderScore,
    /// Opinion evidence spans (`IxRange<Tokens>`) that contributed non-zero
    /// sentiment in this sentence, in token order.
    pub opinions: Vec<Opinion>,
    /// Whether dependency-based negation scope was used for this sentence.
    pub used_dependencies: bool,
}

/// Deterministic VADER-style lexicon and negation sentiment scorer (#320).
#[derive(Clone, Debug)]
pub struct LexiconSentiment {
    lexicon: Arc<VaderLexicon>,
    negation_mode: NegationMode,
    use_opennlp_emoji: bool,
    alpha: f64,
    categories: Arc<TagSet>,
    model_ref: ModelRef,
}

impl Default for LexiconSentiment {
    fn default() -> Self {
        Self::new()
    }
}

impl LexiconSentiment {
    /// Creates a `LexiconSentiment` scorer with the bundled VADER lexicon,
    /// [`NegationMode::Auto`], and OpenNLP emoji sentiment annotations enabled.
    pub fn new() -> LexiconSentiment {
        Self::with_lexicon(VaderLexicon::bundled())
    }

    /// Creates a `LexiconSentiment` scorer with a custom [`VaderLexicon`].
    pub fn with_lexicon(lexicon: Arc<VaderLexicon>) -> LexiconSentiment {
        let negation_mode = NegationMode::Auto;
        let model_ref =
            ModelRef { id: negation_mode.model_id().into(), revision: "1".into(), sha256: lexicon.sha256_hex().into() };
        let categories = TagSet::new("sentiment:polarity", &["positive", "negative", "neutral"]);
        LexiconSentiment {
            lexicon,
            negation_mode,
            use_opennlp_emoji: true,
            alpha: DEFAULT_ALPHA,
            categories,
            model_ref,
        }
    }

    /// Configures how negation scope is resolved ([`NegationMode::Auto`],
    /// [`NegationMode::Window`], or [`NegationMode::Dependency`]).
    pub fn with_negation_mode(mut self, mode: NegationMode) -> LexiconSentiment {
        self.negation_mode = mode;
        self.model_ref =
            ModelRef { id: mode.model_id().into(), revision: "1".into(), sha256: self.lexicon.sha256_hex().into() };
        self
    }

    /// Enables or disables fallback to OpenNLP's `emoji-annotations.txt`
    /// coarse sentiment scores (`-2..=2`) for Unicode emoji tokens.
    pub fn with_opennlp_emoji(mut self, enabled: bool) -> LexiconSentiment {
        self.use_opennlp_emoji = enabled;
        self
    }

    /// Returns the negation mode.
    pub fn negation_mode(&self) -> NegationMode {
        self.negation_mode
    }

    /// Returns the underlying lexicon.
    pub fn lexicon(&self) -> &VaderLexicon {
        &self.lexicon
    }

    /// Returns the `TagSet` (`["positive", "negative", "neutral"]`) used for
    /// `Categories<Sentences>`.
    pub fn categories_tagset(&self) -> &Arc<TagSet> {
        &self.categories
    }

    /// Binds the second output (`Categories<Sentences>`) to the `"sentiment"`
    /// instance ([`SENTENCE_SENTIMENT`]).
    pub fn bound(self) -> Bound<LexiconSentiment> {
        Bound::new(self).write(1, SENTENCE_SENTIMENT.name)
    }

    /// Scores raw text in VADER-compatible mode, matching Python
    /// `vaderSentiment`'s `SentimentIntensityAnalyzer.polarity_scores(text)`.
    pub fn score_text(&self, text: &str) -> VaderScore {
        let raw_tokens: Vec<&str> = text.split_whitespace().collect();
        let words_and_emoticons: Vec<&str> = raw_tokens.iter().map(|&t| strip_punc_if_word(t)).collect();
        let lower_words: Vec<String> = words_and_emoticons.iter().map(|w| w.to_lowercase()).collect();
        let is_cap_diff = allcap_differential(&words_and_emoticons);

        let n = words_and_emoticons.len();
        let mut sentiments = Vec::with_capacity(n);

        for i in 0..n {
            let item = words_and_emoticons[i];
            let item_lower = &lower_words[i];
            if booster_value(item_lower).is_some() {
                sentiments.push(0.0);
                continue;
            }
            if i + 1 < n && item_lower == "kind" && lower_words[i + 1] == "of" {
                sentiments.push(0.0);
                continue;
            }
            let (valence, _, _) =
                self.window_token_valence(item, item_lower, i, &words_and_emoticons, &lower_words, is_cap_diff, None);
            sentiments.push(valence);
        }

        apply_but_check_vader(&lower_words, &mut sentiments);
        compute_vader_score(&sentiments, text, self.alpha)
    }

    /// Scores a single sentence over `tokens` in `tok_range` (`start..end`),
    /// producing both the [`VaderScore`] and the exact [`Opinion`] evidence
    /// spans (`IxRange<Tokens>`).
    ///
    /// When `deps` (`opennlp:dependencies`) is supplied and `self.negation_mode`
    /// is [`NegationMode::Auto`] or [`NegationMode::Dependency`], negation scope
    /// is read from dependency relations within the sentence rather than the
    /// fixed 3-token window.
    pub fn score_sentence(
        &self,
        view_text: &str,
        sentence_span: Span,
        tokens: &Tokens,
        tok_range: IxRange<Tokens>,
        deps: Option<&Dependencies>,
        emoji: Option<&Emoji>,
    ) -> SentenceSentiment {
        let start = tok_range.start.get();
        let end = tok_range.end.get();
        let use_deps = match self.negation_mode {
            NegationMode::Auto | NegationMode::Dependency => deps.is_some(),
            NegationMode::Window => false,
        };
        if start >= end {
            return SentenceSentiment {
                score: VaderScore::default(),
                opinions: Vec::new(),
                used_dependencies: use_deps,
            };
        }

        let n = end - start;
        let mut raw_surfaces = Vec::with_capacity(n);
        let mut words = Vec::with_capacity(n);
        let mut lower_words = Vec::with_capacity(n);
        for tok_idx in start..end {
            let sp = tokens.spans.get(Ix::new(tok_idx as u32));
            let surf = &view_text[sp.range()];
            let stripped = strip_punc_if_word(surf);
            raw_surfaces.push(surf);
            words.push(stripped);
            lower_words.push(stripped.to_lowercase());
        }

        let is_cap_diff = allcap_differential(&words);
        let mut sentiments = Vec::with_capacity(n);
        let mut priors = Vec::with_capacity(n);

        for i in 0..n {
            let tok_global = start + i;
            let item = words[i];
            let item_lower = &lower_words[i];
            if booster_value(item_lower).is_some() || (i + 1 < n && item_lower == "kind" && lower_words[i + 1] == "of")
            {
                sentiments.push(0.0);
                priors.push(0.0);
                continue;
            }

            let emoji_prior = if self.use_opennlp_emoji {
                lookup_emoji_prior(raw_surfaces[i], tokens.spans.get(Ix::new(tok_global as u32)), view_text, emoji)
            } else {
                None
            };

            let (valence, prior, _negated) = match deps {
                Some(deps) if use_deps => self.dependency_token_valence(
                    item,
                    item_lower,
                    i,
                    start,
                    end,
                    &words,
                    &lower_words,
                    is_cap_diff,
                    emoji_prior,
                    deps,
                ),
                _ => self.window_token_valence(item, item_lower, i, &words, &lower_words, is_cap_diff, emoji_prior),
            };
            sentiments.push(valence);
            priors.push(prior);
        }

        // Contrastive "but": by token position, not by vaderSentiment's value
        // lookup (`sentiments.index(sentiment)`), which rescales the first token
        // carrying a repeated value instead of the token itself. `score_text`
        // keeps that quirk for parity; evidence spans must point at the token
        // that was actually rescaled.
        if let Some(but_idx) = lower_words.iter().position(|w| w == "but") {
            for (sidx, s) in sentiments.iter_mut().enumerate() {
                if sidx < but_idx {
                    *s *= 0.5;
                } else if sidx > but_idx {
                    *s *= 1.5;
                }
            }
        }

        let mut opinions = Vec::new();
        for i in 0..n {
            let final_val = sentiments[i];
            let prior = priors[i];
            if final_val == 0.0 {
                continue;
            }
            // `negated` must agree with the sign of the adjusted valence so that
            // `Opinion::effective_polarity` reproduces it: a doubly negated token
            // ("no", then "not") ends up positive and is reported as not negated.
            // A token whose prior was zeroed ("NO" before a lexicon word) but
            // that the ALLCAPS or booster rules still scored keeps its evidence
            // row with the adjusted valence as its polarity.
            let (polarity, intensity, negated) = if prior == 0.0 {
                (final_val, 1.0, false)
            } else {
                (prior, (final_val / prior).abs(), final_val.signum() != prior.signum())
            };
            let tok_ix = (start + i) as u32;
            opinions.push(Opinion { tokens: IxRange::new(tok_ix, tok_ix + 1), polarity, intensity, negated });
        }

        let sent_text = &view_text[sentence_span.range()];
        let score = compute_vader_score(&sentiments, sent_text, self.alpha);
        SentenceSentiment { score, opinions, used_dependencies: use_deps }
    }

    #[allow(clippy::too_many_arguments)]
    fn window_token_valence(
        &self,
        item: &str,
        item_lower: &str,
        i: usize,
        words: &[&str],
        lower_words: &[String],
        is_cap_diff: bool,
        emoji_prior: Option<f64>,
    ) -> (f64, f64, bool) {
        let base_opt = self.lexicon.get(item_lower).or(emoji_prior);
        let Some(base_valence) = base_opt else {
            return (0.0, 0.0, false);
        };
        let n = words.len();
        let mut prior = base_valence;
        let mut valence = base_valence;
        let mut negated = false;

        // "no" immediately before a lexicon word acts as a negator rather than its
        // own lexicon item. Its valence becomes 0.0 but, as in vaderSentiment, the
        // ALLCAPS and booster rules below still apply to it ("NO good" scores
        // "NO" at -0.733).
        if item_lower == "no" && i + 1 < n && self.lexicon.contains(&lower_words[i + 1]) {
            valence = 0.0;
            prior = 0.0;
        }
        if (i > 0 && lower_words[i - 1] == "no")
            || (i > 1 && lower_words[i - 2] == "no")
            || (i > 2 && lower_words[i - 3] == "no" && matches!(lower_words[i - 1].as_str(), "or" | "nor"))
        {
            valence = base_valence * N_SCALAR;
            negated = !negated;
        }

        if is_all_caps(item) && is_cap_diff {
            if valence > 0.0 {
                valence += C_INCR;
            } else {
                valence -= C_INCR;
            }
        }

        for start_i in 0..3 {
            if i > start_i {
                let prev_idx = i - (start_i + 1);
                let prev_lower = &lower_words[prev_idx];
                if !self.lexicon.contains(prev_lower) {
                    let mut s = scalar_inc_dec(words[prev_idx], valence, is_cap_diff);
                    if start_i == 1 && s != 0.0 {
                        s *= 0.95;
                    }
                    if start_i == 2 && s != 0.0 {
                        s *= 0.9;
                    }
                    valence += s;
                    let (v_after_neg, neg_flipped) = negation_check(valence, lower_words, start_i, i);
                    valence = v_after_neg;
                    if neg_flipped {
                        negated = !negated;
                    }
                    if start_i == 2 {
                        let (v_after_idiom, idiom_prior) = special_idioms_check(valence, lower_words, i);
                        valence = v_after_idiom;
                        if let Some(ip) = idiom_prior {
                            prior = ip;
                            negated = false;
                        }
                    }
                }
            }
        }

        let (v_after_least, least_flipped) = least_check(&self.lexicon, valence, lower_words, i);
        valence = v_after_least;
        if least_flipped {
            negated = !negated;
        }

        (valence, prior, negated)
    }

    #[allow(clippy::too_many_arguments)]
    fn dependency_token_valence(
        &self,
        item: &str,
        item_lower: &str,
        i: usize,
        sent_start: usize,
        sent_end: usize,
        words: &[&str],
        lower_words: &[String],
        is_cap_diff: bool,
        emoji_prior: Option<f64>,
        deps: &Dependencies,
    ) -> (f64, f64, bool) {
        let base_opt = self.lexicon.get(item_lower).or(emoji_prior);
        let Some(base_valence) = base_opt else {
            return (0.0, 0.0, false);
        };
        let n = words.len();
        let mut prior = base_valence;
        let mut valence = base_valence;

        // "no" acts as a negator rather than its own lexicon item when it precedes a
        // lexicon word or is a dependent of a lexicon head in this sentence. Its
        // valence becomes 0.0 but the ALLCAPS and booster rules below still apply,
        // as in the window path.
        let tok_global = sent_start + i;
        if item_lower == "no" {
            let precedes_lexicon_word = i + 1 < n && self.lexicon.contains(&lower_words[i + 1]);
            let modifies_lexicon_head = match deps.head.get(tok_global) {
                Some(Some(head_ix)) => {
                    let h = head_ix.get();
                    h >= sent_start
                        && h < sent_end
                        && h != tok_global
                        && self.lexicon.contains(&lower_words[h - sent_start])
                }
                _ => false,
            };
            if precedes_lexicon_word || modifies_lexicon_head {
                valence = 0.0;
                prior = 0.0;
            }
        }

        if is_all_caps(item) && is_cap_diff {
            if valence > 0.0 {
                valence += C_INCR;
            } else {
                valence -= C_INCR;
            }
        }

        // Apply booster/dampener scalars and special-case idioms (plus "never so/this"
        // and "without doubt" amplifier patterns), while leaving negation scope to the
        // dependency tree.
        for start_i in 0..3 {
            if i > start_i {
                let prev_idx = i - (start_i + 1);
                let prev_lower = &lower_words[prev_idx];
                if !self.lexicon.contains(prev_lower) {
                    let mut s = scalar_inc_dec(words[prev_idx], valence, is_cap_diff);
                    if start_i == 1 && s != 0.0 {
                        s *= 0.95;
                    }
                    if start_i == 2 && s != 0.0 {
                        s *= 0.9;
                    }
                    valence += s;
                    if (start_i == 1
                        && lower_words[i - 2] == "never"
                        && (lower_words[i - 1] == "so" || lower_words[i - 1] == "this"))
                        || (start_i == 2
                            && ((lower_words[i - 3] == "never"
                                && (lower_words[i - 2] == "so" || lower_words[i - 2] == "this"))
                                || (lower_words[i - 1] == "so" || lower_words[i - 1] == "this")))
                    {
                        valence *= 1.25;
                    }
                    if start_i == 2 {
                        let (v_after_idiom, idiom_prior) = special_idioms_check(valence, lower_words, i);
                        valence = v_after_idiom;
                        if let Some(ip) = idiom_prior {
                            prior = ip;
                        }
                    }
                }
            }
        }

        // Check whether `never so/this` or `without doubt` suppressed negation locally.
        let never_so_this =
            (i >= 2 && lower_words[i - 2] == "never" && (lower_words[i - 1] == "so" || lower_words[i - 1] == "this"))
                || (i >= 3
                    && lower_words[i - 3] == "never"
                    && (lower_words[i - 2] == "so"
                        || lower_words[i - 2] == "this"
                        || lower_words[i - 1] == "so"
                        || lower_words[i - 1] == "this"));
        let without_doubt = (i >= 2 && lower_words[i - 2] == "without" && lower_words[i - 1] == "doubt")
            || (i >= 3
                && lower_words[i - 3] == "without"
                && (lower_words[i - 2] == "doubt" || lower_words[i - 1] == "doubt"));

        let mut negated = false;
        if !never_so_this
            && !without_doubt
            && is_negated_in_dependencies(i, sent_start, sent_end, lower_words, &self.lexicon, deps)
        {
            valence *= N_SCALAR;
            negated = true;
        }

        (valence, prior, negated)
    }
}

/// Checks whether token `i` (at document token index `sent_start + i`) is under
/// negation scope in `deps`:
///
/// 1. Any dependent `d` in `[sent_start, sent_end)` with `head[d] == tok` that
///    either has relation `"neg"` or is a negation token (`is_negation_word`,
///    `"no"`, or `"least"` not preceded by `"at"`/`"very"`).
/// 2. If `tok` is an adjectival, clausal, or open complement or conjunction
///    (`amod`, `acomp`, `xcomp`, `ccomp`, `conj`, `advmod`, `dep`, `attr`) of
///    a head `h` in `[sent_start, sent_end)` (and `tok` is not separated from
///    `h` by a contrastive `"but"`), checks whether `h` itself has a direct
///    negation dependent (for example, `"The movie was not good"` where `"good"`
///    is head with `"not"` attached, or copular/verbal constructions where
///    `"was"` governs `"not"` and `"good"`).
fn is_negated_in_dependencies(
    i: usize,
    sent_start: usize,
    sent_end: usize,
    lower_words: &[String],
    lexicon: &VaderLexicon,
    deps: &Dependencies,
) -> bool {
    let tok_global = sent_start + i;
    if has_direct_negation_dependent(tok_global, sent_start, sent_end, lower_words, lexicon, deps) {
        return true;
    }

    let Some(Some(head_ix)) = deps.head.get(tok_global) else {
        return false;
    };
    let head_global = head_ix.get();
    if head_global < sent_start || head_global >= sent_end || head_global == tok_global {
        return false;
    }

    // Do not propagate negation from a head across a contrastive "but" between head and token.
    let lo = head_global.min(tok_global) - sent_start;
    let hi = head_global.max(tok_global) - sent_start;
    if lower_words[lo..=hi].iter().any(|w| w == "but") {
        return false;
    }

    let rel = deps.relation.get(tok_global).map(|&t| deps.relations.label(t)).unwrap_or("");
    let base_rel = rel.split(':').next().unwrap_or(rel);
    if matches!(base_rel, "acomp" | "xcomp" | "ccomp" | "amod" | "conj" | "advmod" | "attr" | "oprd" | "dep")
        && has_direct_negation_dependent(head_global, sent_start, sent_end, lower_words, lexicon, deps)
    {
        return true;
    }

    false
}

fn has_direct_negation_dependent(
    target_global: usize,
    sent_start: usize,
    sent_end: usize,
    lower_words: &[String],
    lexicon: &VaderLexicon,
    deps: &Dependencies,
) -> bool {
    for dep_global in sent_start..sent_end {
        if dep_global == target_global {
            continue;
        }
        let Some(Some(h)) = deps.head.get(dep_global) else {
            continue;
        };
        if h.get() != target_global {
            continue;
        }
        let local_idx = dep_global - sent_start;
        let w = lower_words[local_idx].as_str();
        let rel = deps.relation.get(dep_global).map(|&t| deps.relations.label(t)).unwrap_or("");
        let base_rel = rel.split(':').next().unwrap_or(rel);
        if base_rel == "neg" || is_negation_word(w) || w == "no" {
            return true;
        }
        if w == "least" && !lexicon.contains(w) {
            let preceded_by_at_or_very = local_idx > 0 && matches!(lower_words[local_idx - 1].as_str(), "at" | "very");
            if !preceded_by_at_or_very {
                return true;
            }
        }
    }
    false
}

fn lookup_emoji_prior(surface: &str, tok_span: Span, view_text: &str, emoji_layer: Option<&Emoji>) -> Option<f64> {
    if let Some(a) = opennlp_annotation(surface)
        && a.sentiment != 0
    {
        return Some(a.sentiment as f64);
    }
    if let Some(emoji) = emoji_layer {
        for (_, esp) in emoji.spans.iter() {
            if esp.start >= tok_span.start
                && esp.end <= tok_span.end
                && let Some(a) = opennlp_annotation(&view_text[esp.range()])
                && a.sentiment != 0
            {
                return Some(a.sentiment as f64);
            }
        }
    }
    None
}

fn negation_check(valence: f64, lower_words: &[String], start_i: usize, i: usize) -> (f64, bool) {
    if start_i == 0 {
        if is_negation_word(&lower_words[i - 1]) {
            return (valence * N_SCALAR, true);
        }
    } else if start_i == 1 {
        if lower_words[i - 2] == "never" && (lower_words[i - 1] == "so" || lower_words[i - 1] == "this") {
            return (valence * 1.25, false);
        } else if lower_words[i - 2] == "without" && lower_words[i - 1] == "doubt" {
            return (valence, false);
        } else if is_negation_word(&lower_words[i - 2]) {
            return (valence * N_SCALAR, true);
        }
    } else if start_i == 2 {
        if (lower_words[i - 3] == "never" && (lower_words[i - 2] == "so" || lower_words[i - 2] == "this"))
            || (lower_words[i - 1] == "so" || lower_words[i - 1] == "this")
        {
            return (valence * 1.25, false);
        } else if lower_words[i - 3] == "without" && (lower_words[i - 2] == "doubt" || lower_words[i - 1] == "doubt") {
            return (valence, false);
        } else if is_negation_word(&lower_words[i - 3]) {
            return (valence * N_SCALAR, true);
        }
    }
    (valence, false)
}

fn special_idioms_check(valence: f64, lower_words: &[String], i: usize) -> (f64, Option<f64>) {
    let mut out = valence;
    let mut matched_idiom = None;

    let onezero = format!("{} {}", lower_words[i - 1], lower_words[i]);
    let twoonezero = format!("{} {} {}", lower_words[i - 2], lower_words[i - 1], lower_words[i]);
    let twoone = format!("{} {}", lower_words[i - 2], lower_words[i - 1]);
    let threetwoone = format!("{} {} {}", lower_words[i - 3], lower_words[i - 2], lower_words[i - 1]);
    let threetwo = format!("{} {}", lower_words[i - 3], lower_words[i - 2]);

    for seq in [&onezero, &twoonezero, &twoone, &threetwoone, &threetwo] {
        if let Some(v) = special_case_value(seq) {
            out = v;
            matched_idiom = Some(v);
            break;
        }
    }

    if lower_words.len() - 1 > i {
        let zeroone = format!("{} {}", lower_words[i], lower_words[i + 1]);
        if let Some(v) = special_case_value(&zeroone) {
            out = v;
            matched_idiom = Some(v);
        }
    }
    if lower_words.len() - 1 > i + 1 {
        let zeroonetwo = format!("{} {} {}", lower_words[i], lower_words[i + 1], lower_words[i + 2]);
        if let Some(v) = special_case_value(&zeroonetwo) {
            out = v;
            matched_idiom = Some(v);
        }
    }

    for ngram in [&threetwoone, &threetwo, &twoone] {
        if let Some(b) = booster_value(ngram) {
            out += b;
        }
    }

    (out, matched_idiom)
}

fn least_check(lexicon: &VaderLexicon, valence: f64, lower_words: &[String], i: usize) -> (f64, bool) {
    if i > 1 && !lexicon.contains(&lower_words[i - 1]) && lower_words[i - 1] == "least" {
        if lower_words[i - 2] != "at" && lower_words[i - 2] != "very" {
            return (valence * N_SCALAR, true);
        }
    } else if i > 0 && !lexicon.contains(&lower_words[i - 1]) && lower_words[i - 1] == "least" {
        return (valence * N_SCALAR, true);
    }
    (valence, false)
}

/// Applies `_but_check` exactly as `vaderSentiment.py` does.
fn apply_but_check_vader(lower_words: &[String], sentiments: &mut [f64]) {
    let Some(bi) = lower_words.iter().position(|w| w == "but") else {
        return;
    };
    let snapshot = sentiments.to_vec();
    for sentiment in snapshot {
        if let Some(si) = sentiments.iter().position(|&s| s == sentiment) {
            if si < bi {
                sentiments[si] = sentiment * 0.5;
            } else if si > bi {
                sentiments[si] = sentiment * 1.5;
            }
        }
    }
}

fn punctuation_emphasis(text: &str) -> f64 {
    let ep_count = text.bytes().filter(|&b| b == b'!').count().min(4);
    let ep_amp = (ep_count as f64) * 0.292;

    let qm_count = text.bytes().filter(|&b| b == b'?').count();
    let qm_amp = if qm_count > 1 { if qm_count <= 3 { (qm_count as f64) * 0.18 } else { 0.96 } } else { 0.0 };

    ep_amp + qm_amp
}

fn sift_sentiment_scores(sentiments: &[f64]) -> (f64, f64, usize) {
    let mut pos_sum = 0.0;
    let mut neg_sum = 0.0;
    let mut neu_count = 0usize;
    for &s in sentiments {
        if s > 0.0 {
            pos_sum += s + 1.0;
        }
        if s < 0.0 {
            neg_sum += s - 1.0;
        }
        if s == 0.0 {
            neu_count += 1;
        }
    }
    (pos_sum, neg_sum, neu_count)
}

fn compute_vader_score(sentiments: &[f64], text: &str, alpha: f64) -> VaderScore {
    if sentiments.is_empty() {
        return VaderScore::default();
    }
    let mut sum_s: f64 = sentiments.iter().sum();
    let punct_amp = punctuation_emphasis(text);
    if sum_s > 0.0 {
        sum_s += punct_amp;
    } else if sum_s < 0.0 {
        sum_s -= punct_amp;
    }
    let compound = normalize_score(sum_s, alpha);

    let (mut pos_sum, mut neg_sum, neu_count) = sift_sentiment_scores(sentiments);
    if pos_sum > neg_sum.abs() {
        pos_sum += punct_amp;
    } else if pos_sum < neg_sum.abs() {
        neg_sum -= punct_amp;
    }

    let total = pos_sum + neg_sum.abs() + (neu_count as f64);
    if total == 0.0 {
        return VaderScore { neg: 0.0, neu: 0.0, pos: 0.0, compound };
    }
    let pos = (pos_sum / total).abs();
    let neg = (neg_sum / total).abs();
    let neu = ((neu_count as f64) / total).abs();
    VaderScore { neg, neu, pos, compound }
}

impl Annotator for LexiconSentiment {
    const ID: &'static str = "turbo-doccat:lexicon-sentiment";
    type Reads = (View, Sentences, Tokens, Option<Dependencies>, Option<Emoji>);
    type Writes = (Opinions, Categories<Sentences>);

    fn annotate(
        &self,
        (view, sentences, tokens, deps, emoji): Refs<'_, Self>,
    ) -> Result<(Opinions, Categories<Sentences>), Error> {
        if sentences.spans.view() != view.id() {
            return Err(Error::Annotator { annotator: Self::ID, detail: "sentences are not in the given view".into() });
        }
        if tokens.spans.view() != view.id() {
            return Err(Error::Annotator { annotator: Self::ID, detail: "tokens are not in the given view".into() });
        }
        if let Some(e) = emoji
            && e.spans.view() != view.id()
        {
            return Err(Error::Annotator {
                annotator: Self::ID,
                detail: "emoji layer is not in the given view".into(),
            });
        }

        let text = view.text();
        let mut opinions = Opinions::new();
        let mut categories = Categories::<Sentences>::with_capacity(self.categories.clone(), sentences.len());

        let tok_count = tokens.len();
        let mut tok_cursor = 0usize;

        for (_, sent_span) in sentences.spans.iter() {
            while tok_cursor < tok_count && tokens.spans.get(Ix::new(tok_cursor as u32)).end <= sent_span.start {
                tok_cursor += 1;
            }
            let sent_tok_start = tok_cursor;
            while tok_cursor < tok_count && tokens.spans.get(Ix::new(tok_cursor as u32)).start < sent_span.end {
                tok_cursor += 1;
            }
            let sent_tok_end = tok_cursor;
            let range = IxRange::new(sent_tok_start as u32, sent_tok_end as u32);

            let result = self.score_sentence(text, sent_span, tokens, range, deps, emoji);
            for op in result.opinions {
                opinions.push(op);
            }

            let (pos, neg, mut neu) = (result.score.pos, result.score.neg, result.score.neu);
            if pos == 0.0 && neg == 0.0 && neu == 0.0 {
                neu = 1.0;
            }
            // The best category follows the compound threshold, not the largest
            // proportion: "not bad" is mostly neutral tokens but classifies positive.
            let best_tag = match result.score.polarity() {
                Polarity::Positive => Tag(0),
                Polarity::Negative => Tag(1),
                Polarity::Neutral => Tag(2),
            };
            categories.push(best_tag, &[pos, neg, neu]);
        }

        Ok((opinions, categories))
    }

    fn model(&self) -> Option<ModelRef> {
        Some(self.model_ref.clone())
    }

    fn config_hash(&self) -> u64 {
        let mut h = std::collections::hash_map::DefaultHasher::new();
        self.negation_mode.hash(&mut h);
        self.use_opennlp_emoji.hash(&mut h);
        self.alpha.to_bits().hash(&mut h);
        self.lexicon.sha256_hex().hash(&mut h);
        h.finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use turbo_analysis::{Analyzer, Document, Provenance, Spans, ViewId};

    /// All 28 test sentences from Python `vaderSentiment.py`'s `__main__`
    /// block (`sentences` and `tricky_sentences`), with their exact
    /// `(neg, neu, pos, compound)` scores from `SentimentIntensityAnalyzer`.
    const VADER_REFERENCE_SUITE: &[(&str, f64, f64, f64, f64)] = &[
        ("VADER is smart, handsome, and funny.", 0.000, 0.254, 0.746, 0.8316),
        ("VADER is smart, handsome, and funny!", 0.000, 0.248, 0.752, 0.8439),
        ("VADER is very smart, handsome, and funny.", 0.000, 0.299, 0.701, 0.8545),
        ("VADER is VERY SMART, handsome, and FUNNY.", 0.000, 0.246, 0.754, 0.9227),
        ("VADER is VERY SMART, handsome, and FUNNY!!!", 0.000, 0.233, 0.767, 0.9342),
        ("VADER is VERY SMART, uber handsome, and FRIGGIN FUNNY!!!", 0.000, 0.294, 0.706, 0.9469),
        ("VADER is not smart, handsome, nor funny.", 0.646, 0.354, 0.000, -0.7424),
        ("The book was good.", 0.000, 0.508, 0.492, 0.4404),
        ("At least it isn't a horrible book.", 0.000, 0.678, 0.322, 0.4310),
        ("The book was only kind of good.", 0.000, 0.697, 0.303, 0.3832),
        (
            "The plot was good, but the characters are uncompelling and the dialog is not great.",
            0.327,
            0.579,
            0.094,
            -0.7042,
        ),
        ("Today SUX!", 0.779, 0.221, 0.000, -0.5461),
        ("Today only kinda sux! But I'll get by, lol", 0.127, 0.556, 0.317, 0.5249),
        ("Make sure you :) or :D today!", 0.000, 0.294, 0.706, 0.8633),
        ("Not bad at all", 0.000, 0.513, 0.487, 0.4310),
        ("Sentiment analysis has never been good.", 0.325, 0.675, 0.000, -0.3412),
        ("Sentiment analysis has never been this good!", 0.000, 0.621, 0.379, 0.5672),
        ("Most automated sentiment analysis tools are shit.", 0.375, 0.625, 0.000, -0.5574),
        ("With VADER, sentiment analysis is the shit!", 0.000, 0.583, 0.417, 0.6476),
        ("Other sentiment analysis tools can be quite bad.", 0.351, 0.649, 0.000, -0.5849),
        ("On the other hand, VADER is quite bad ass", 0.000, 0.423, 0.577, 0.8020),
        ("VADER is such a badass!", 0.000, 0.598, 0.402, 0.4003),
        ("Without a doubt, excellent idea.", 0.000, 0.341, 0.659, 0.7013),
        ("Roger Dodger is one of the most compelling variations on this theme.", 0.000, 0.834, 0.166, 0.2944),
        ("Roger Dodger is at least compelling as a variation on the theme.", 0.000, 0.853, 0.147, 0.2263),
        ("Roger Dodger is one of the least compelling variations on this theme.", 0.132, 0.868, 0.000, -0.1695),
        ("Not such a badass after all.", 0.289, 0.711, 0.000, -0.2584),
        ("Without a doubt, an excellent idea.", 0.000, 0.408, 0.592, 0.7013),
    ];

    /// Sentences outside the upstream suite where an earlier port diverged,
    /// scored with vaderSentiment 3.3.2. "no" before a lexicon word keeps its
    /// 0.0 valence through the ALLCAPS and booster rules; a dampener before
    /// "no" and a leading idiom are both scored by the preceding-word rules.
    const VADER_EXTRA_SUITE: &[(&str, f64, f64, f64, f64)] = &[
        ("NO good", 1.000, 0.000, 0.000, -0.4835),
        ("very no good", 0.539, 0.201, 0.260, -0.3381),
        ("yeah right bad ass", 0.686, 0.098, 0.216, -0.7003),
        ("screwer NO bravely favoritisms", 1.000, 0.000, 0.000, -0.7313),
        ("no no good", 0.454, 0.189, 0.357, -0.1326),
        ("no not good", 0.420, 0.191, 0.389, -0.0412),
    ];

    fn make_sentences_and_tokens(text: &str, words: &[&str]) -> (Sentences, Tokens) {
        let mut s_spans = Spans::new(ViewId::SOURCE);
        s_spans.push(Span::new(0, text.len() as u32));
        let sentences = Sentences { spans: s_spans };

        let mut t_spans = Spans::new(ViewId::SOURCE);
        let mut cursor = 0usize;
        for &w in words {
            let pos = text[cursor..].find(w).unwrap() + cursor;
            t_spans.push(Span::new(pos as u32, (pos + w.len()) as u32));
            cursor = pos + w.len();
        }
        let tokens = Tokens { spans: t_spans, sentence: None };
        (sentences, tokens)
    }

    #[test]
    fn matches_python_vader_sentiment_on_all_reference_sentences() {
        let scorer = LexiconSentiment::new();
        assert_eq!(scorer.lexicon().len(), 7506);

        for &(sentence, exp_neg, exp_neu, exp_pos, exp_compound) in
            VADER_REFERENCE_SUITE.iter().chain(VADER_EXTRA_SUITE)
        {
            let s = scorer.score_text(sentence);
            assert_eq!(
                (s.neg_rounded(), s.neu_rounded(), s.pos_rounded(), s.compound_rounded()),
                (exp_neg, exp_neu, exp_pos, exp_compound),
                "mismatch on sentence: {sentence:?}"
            );
        }
    }

    #[test]
    fn digest_follows_entries_not_source_bytes() {
        let a = VaderLexicon::from_tsv("# comment\ngood\t1.9\nbad\t-2.5\n").unwrap();
        let b = VaderLexicon::from_tsv("bad\t-2.5\t0.5\t[-3, -2]\ngood\t1.9\n").unwrap();
        assert_eq!(a.sha256_hex(), b.sha256_hex());
        let mut c = a.clone();
        c.insert("meh", -0.5);
        assert_ne!(a.sha256_hex(), c.sha256_hex());
        assert!(VaderLexicon::from_tsv("good\tnot-a-number\n").is_err());
    }

    #[test]
    fn a_zeroed_no_that_still_scores_keeps_its_evidence_row() {
        let text = "NO good";
        let words = ["NO", "good"];
        let (_s, tokens) = make_sentences_and_tokens(text, &words);
        let scorer = LexiconSentiment::new().with_negation_mode(NegationMode::Window);
        let res = scorer.score_sentence(text, Span::new(0, 7), &tokens, IxRange::new(0, 2), None, None);
        assert_eq!(res.opinions.len(), 2);
        let no = res.opinions[0];
        assert_eq!(no.tokens, IxRange::new(0, 1));
        assert!((no.effective_polarity() + C_INCR).abs() < 1e-12);
        let good = res.opinions[1];
        assert_eq!(good.tokens, IxRange::new(1, 2));
        assert!((good.effective_polarity() - 1.9 * N_SCALAR).abs() < 1e-12);
        let sum: f64 = res.opinions.iter().map(|o| o.effective_polarity()).sum();
        assert!((normalize_score(sum, DEFAULT_ALPHA) - res.score.compound).abs() < 1e-12);
    }

    #[test]
    fn double_negation_is_reported_as_not_negated() {
        // "no" two tokens back applies the "no" rule, then "not" applies the
        // negation window: the valence flips twice and ends positive.
        let text = "no not good";
        let words = ["no", "not", "good"];
        let (_s, tokens) = make_sentences_and_tokens(text, &words);
        let scorer = LexiconSentiment::new().with_negation_mode(NegationMode::Window);
        let res = scorer.score_sentence(text, Span::new(0, 11), &tokens, IxRange::new(0, 3), None, None);
        // "no" keeps its own lexicon valence here because "not" is not a lexicon word.
        assert_eq!(res.opinions.len(), 2);
        let op = res.opinions[1];
        assert_eq!(op.tokens, IxRange::new(2, 3));
        assert!(!op.negated);
        assert!((op.effective_polarity() - 1.9 * N_SCALAR * N_SCALAR).abs() < 1e-12);
    }

    #[test]
    fn evidence_spans_point_at_exact_scored_tokens_and_preserve_effective_polarity() {
        let text = "The plot was good, but the dialog is not great.";
        let word_slices = ["The", "plot", "was", "good", ",", "but", "the", "dialog", "is", "not", "great", "."];
        let (sentences, tokens) = make_sentences_and_tokens(text, &word_slices);

        let mut doc = Document::new(text);
        doc.add(&Key::<Sentences>::DEFAULT, sentences, Provenance::by("test")).unwrap();
        doc.add(&Key::<Tokens>::DEFAULT, tokens, Provenance::by("test")).unwrap();

        let analyzer = Analyzer::builder()
            .given(Key::<Sentences>::DEFAULT)
            .given(Key::<Tokens>::DEFAULT)
            .add(LexiconSentiment::new())
            .build()
            .unwrap();
        let mut docs = [doc];
        for r in analyzer.run(&mut docs) {
            r.unwrap();
        }
        let doc = &docs[0];

        let opinions = doc.get::<Opinions>().unwrap();
        let toks = doc.get::<Tokens>().unwrap();
        assert_eq!(opinions.len(), 2);

        // First opinion: "good" at token index 3 (before "but", scaled by 0.5).
        let op0 = opinions.get(Ix::new(0));
        assert_eq!(op0.tokens, IxRange::new(3, 4));
        assert_eq!(&text[opinions.span(Ix::new(0), toks).range()], "good");
        assert!(!op0.negated);
        assert!((op0.polarity - 1.9).abs() < 1e-12);
        assert!((op0.effective_polarity() - 1.9 * 0.5).abs() < 1e-12);

        // Second opinion: "great" at token index 10 (negated by "not" and after "but", scaled by -0.74 * 1.5).
        let op1 = opinions.get(Ix::new(1));
        assert_eq!(op1.tokens, IxRange::new(10, 11));
        assert_eq!(&text[opinions.span(Ix::new(1), toks).range()], "great");
        assert!(op1.negated);
        assert!((op1.polarity - 3.1).abs() < 1e-12);
        assert!((op1.effective_polarity() - 3.1 * N_SCALAR * 1.5).abs() < 1e-12);

        let cats = doc.get::<Categories<Sentences>>().unwrap();
        assert_eq!(cats.len(), 1);
        assert_eq!(cats.best_category(0), "negative");
    }

    #[test]
    fn dependency_negation_scope_handles_long_distance_and_clause_boundaries() {
        // Sentence where negation "not" is 4 tokens before "good" ("is not in any way good"):
        // in a 3-token window "not" is too far to negate "good", whereas in the dependency parse
        // "not" is a direct `neg` dependent of "good".
        let text = "The service is not in any way good.";
        let words = ["The", "service", "is", "not", "in", "any", "way", "good", "."];
        let (sentences, tokens) = make_sentences_and_tokens(text, &words);

        // Window mode: "not" is at index 3, "good" is at index 7 (distance 4 > 3), so window mode misses negation.
        let scorer_win = LexiconSentiment::new().with_negation_mode(NegationMode::Window);
        let win_res = scorer_win.score_sentence(
            text,
            Span::new(0, text.len() as u32),
            &tokens,
            IxRange::new(0, words.len() as u32),
            None,
            None,
        );
        assert!(!win_res.used_dependencies);
        assert!(win_res.score.compound > 0.0);
        assert!(!win_res.opinions[0].negated);

        // Dependency mode: "not" (token 3) has head "good" (token 7) with relation "neg".
        let rel_tags = TagSet::new("ud", &["root", "nsubj", "cop", "neg", "case", "det", "obl", "punct"]);
        let deps = Dependencies {
            head: vec![
                Some(Ix::new(1)), // The -> service
                Some(Ix::new(7)), // service -> good
                Some(Ix::new(7)), // is -> good
                Some(Ix::new(7)), // not -> good
                Some(Ix::new(6)), // in -> way
                Some(Ix::new(6)), // any -> way
                Some(Ix::new(7)), // way -> good
                None,             // good (root)
                Some(Ix::new(7)), // . -> good
            ],
            relation: vec![Tag(5), Tag(1), Tag(2), Tag(3), Tag(4), Tag(5), Tag(6), Tag(0), Tag(7)],
            relations: rel_tags,
        };

        let mut doc = Document::new(text);
        doc.add(&Key::<Sentences>::DEFAULT, sentences, Provenance::by("test")).unwrap();
        doc.add(&Key::<Tokens>::DEFAULT, tokens, Provenance::by("test")).unwrap();
        doc.add(&Key::<Dependencies>::DEFAULT, deps, Provenance::by("test")).unwrap();

        let analyzer = Analyzer::builder()
            .given(Key::<Sentences>::DEFAULT)
            .given(Key::<Tokens>::DEFAULT)
            .given(Key::<Dependencies>::DEFAULT)
            .add(LexiconSentiment::new())
            .build()
            .unwrap();
        let mut docs = [doc];
        for r in analyzer.run(&mut docs) {
            r.unwrap();
        }
        let doc = &docs[0];

        let opinions = doc.get::<Opinions>().unwrap();
        assert_eq!(opinions.len(), 1);
        let op = opinions.get(Ix::new(0));
        assert_eq!(op.tokens, IxRange::new(7, 8));
        assert!(op.negated);
        assert!(op.effective_polarity() < 0.0);

        let cats = doc.get::<Categories<Sentences>>().unwrap();
        assert_eq!(cats.best_category(0), "negative");

        // Provenance records that `opennlp:dependencies` was read.
        let prov = doc.provenance(Key::<Opinions>::DEFAULT.id()).unwrap();
        assert!(prov.reads.contains(&Key::<Dependencies>::DEFAULT.id()));
    }

    #[test]
    fn dependency_negation_avoids_false_negation_across_clauses() {
        // "Not" negates "sure", not "good" inside the complement clause ("Not sure the food is good"):
        // in 3-token window mode ("Not sure the good"), "not" is 3 tokens before "good" and negates it,
        // whereas in dependency mode "not" attaches to "sure" and "good" is the head of "ccomp" without
        // negation on "good" (when "sure" is not in the lexicon, or when "not" attaches to an unrelated verb).
        let text = "Not knowing the food is good.";
        let words = ["Not", "knowing", "the", "food", "is", "good", "."];
        // And in a 3-token distance case: "Not saying food is good" -> "Not"(0) "saying"(1) "it"(2) "good"(3):
        let text2 = "Not saying it is good.";
        let words2 = ["Not", "saying", "it", "is", "good", "."];
        let (_s2, tokens2) = make_sentences_and_tokens(text2, &words2);

        // In window mode, "Not" is 4 tokens before "good" in text2, or 3 tokens before "good" in "Not saying anything good.":
        let text3 = "Not saying things are good.";
        let words3 = ["Not", "saying", "things", "are", "good", "."];
        let _ = (text, words, tokens2, text3, words3);

        // 3-token window case: "Not"(0) "saying"(1) "everything"(2) "good"(3):
        // Window mode sees "Not" at i - 3 and negates "good", whereas in dependency mode "Not" modifies
        // "saying" (root) and "good" modifies "everything" (obj -> amod on "everything", whose head is "saying",
        // so "good"'s head is "everything", which has NO negation dependent!).
        let sent_text = "Not saying everything good.";
        let sent_words = ["Not", "saying", "everything", "good", "."];
        let (_sents, toks) = make_sentences_and_tokens(sent_text, &sent_words);

        let win_scorer = LexiconSentiment::new().with_negation_mode(NegationMode::Window);
        let win_res = win_scorer.score_sentence(
            sent_text,
            Span::new(0, sent_text.len() as u32),
            &toks,
            IxRange::new(0, sent_words.len() as u32),
            None,
            None,
        );
        assert!(win_res.opinions[0].negated);
        assert!(win_res.score.compound < 0.0);

        let rel_tags = TagSet::new("ud", &["root", "neg", "obj", "amod", "punct"]);
        let deps = Dependencies {
            head: vec![
                Some(Ix::new(1)), // Not -> saying
                None,             // saying (root)
                Some(Ix::new(1)), // everything -> saying
                Some(Ix::new(2)), // good -> everything
                Some(Ix::new(1)), // . -> saying
            ],
            relation: vec![Tag(1), Tag(0), Tag(2), Tag(3), Tag(4)],
            relations: rel_tags,
        };
        let dep_scorer = LexiconSentiment::new().with_negation_mode(NegationMode::Dependency);
        let dep_res = dep_scorer.score_sentence(
            sent_text,
            Span::new(0, sent_text.len() as u32),
            &toks,
            IxRange::new(0, sent_words.len() as u32),
            Some(&deps),
            None,
        );
        assert!(dep_res.used_dependencies);
        assert!(!dep_res.opinions[0].negated);
        assert!(dep_res.score.compound > 0.0);
    }

    #[test]
    fn opennlp_emoji_annotations_score_unicode_emoji_tokens() {
        let text = "Great update \u{1F600}";
        let words = ["Great", "update", "\u{1F600}"];
        let (sentences, tokens) = make_sentences_and_tokens(text, &words);

        let mut doc = Document::new(text);
        doc.add(&Key::<Sentences>::DEFAULT, sentences, Provenance::by("test")).unwrap();
        doc.add(&Key::<Tokens>::DEFAULT, tokens, Provenance::by("test")).unwrap();

        let analyzer = Analyzer::builder()
            .given(Key::<Sentences>::DEFAULT)
            .given(Key::<Tokens>::DEFAULT)
            .add(LexiconSentiment::new())
            .build()
            .unwrap();
        let mut docs = [doc];
        for r in analyzer.run(&mut docs) {
            r.unwrap();
        }
        let doc = &docs[0];

        let opinions = doc.get::<Opinions>().unwrap();
        // Both "Great" (token 0) and grinning face U+1F600 (token 2) produce positive opinions.
        assert_eq!(opinions.len(), 2);
        assert_eq!(opinions.get(Ix::new(0)).tokens, IxRange::new(0, 1));
        assert_eq!(opinions.get(Ix::new(1)).tokens, IxRange::new(2, 3));
        assert!(opinions.get(Ix::new(1)).polarity > 0.0);
    }
}
