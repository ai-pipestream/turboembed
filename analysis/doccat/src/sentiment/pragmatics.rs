//! Sentence-level sarcasm and irony detection (`turbo:pragmatics`, #324).
//!
//! Scores each sentence in `0.0..=1.0` and writes [`Pragmatics`]
//! (`turbo:pragmatics`), aligned to [`Sentences`] and never folded into
//! polarity (`turbo:categories`), so downstream consumers decide whether to
//! flip or discount a positive polarity score.
//!
//! Three signals are combined in priority order:
//!
//! 1. **Incongruity between a positive sentiment expression and a negative
//!    situation** in the same sentence, following Riloff et al. (EMNLP 2013,
//!    *Sarcasm as Contrast between a Positive Sentiment and Negative
//!    Situation*). Positive spans come from [`Opinions`] (`turbo:opinions`)
//!    or sarcastic positive framing cues; negative spans come from negative
//!    situation phrases (`waiting in line`, `stuck in traffic`, `server
//!    crashed`, `flight cancelled`, etc.) and negative [`Opinions`]. Clauses
//!    separated by an adversative conjunction (`but`, `however`, `although`,
//!    `though`, `yet`, `whereas`, `while`, `except`) without a sarcastic
//!    framing cue are treated as literal mixed sentiment rather than sarcasm.
//! 2. **Text against emoji polarity mismatch**, contrasting a positive text
//!    opinion against a negative emoji from [`Emoji`] (`turbo:emoji`, #183,
//!    scored via [`turbo_normalize::emoji::opennlp_annotation`]), or a
//!    negative text opinion/situation against a positive emoji.
//! 3. **Sentence embedding head** ([`PragmaticsVectorScorer`] over
//!    `Vectors<Sentences>`, or `sarcasm`/`irony` probabilities in
//!    `Categories<Sentences>`) for implicit sarcasm and irony with no
//!    explicit surface contrast.
//!
//! Every positive prediction carries its contrasted [`Incongruity`] token
//! spans (`Some(Incongruity { positive, negative })`) when explicit contrast
//! was found, or `None` when scored by the embedding head alone.

use std::fmt;
use std::hash::{Hash, Hasher};
use std::sync::Arc;

use turbo_analysis::{Annotator, Error, Ix, IxRange, Layer, ModelRef, Refs, Span, View};
use turbo_analysis_layers::{
    Categories, Incongruity, Opinions, Pragmatics, PragmaticsItem, Sentences, Tokens, Vectors,
};
use turbo_normalize::emoji::opennlp_annotation;
use turbo_normalize_layers::Emoji;

use super::lexicon::VaderLexicon;

/// Default minimum effective polarity for a positive opinion span.
pub const DEFAULT_MIN_POSITIVE_POLARITY: f64 = 0.4;

/// Default minimum magnitude of effective polarity for a negative opinion span.
pub const DEFAULT_MIN_NEGATIVE_POLARITY: f64 = 0.4;

/// Default classification threshold (`0.5`) above which a sentence is
/// considered sarcastic or ironic.
pub const DEFAULT_PRAGMATICS_THRESHOLD: f64 = 0.5;

/// Sentence-vector scorer used by [`PragmaticsAnnotator`] to score
/// `Vectors<Sentences>` rows in `0.0..=1.0` when a sentence has no explicit
/// surface contrast (or to reinforce explicit contrast).
pub trait PragmaticsVectorScorer: Send + Sync + fmt::Debug {
    /// Scores one sentence embedding row `vec`, returning a sarcasm/irony
    /// probability in `0.0..=1.0`.
    fn score_sentence_vector(&self, vec: &[f32]) -> Result<f64, String>;

    /// Deterministic configuration and weight digest for provenance.
    fn config_hash(&self) -> u64;

    /// Optional model reference recorded in provenance.
    fn model_ref(&self) -> Option<ModelRef> {
        None
    }
}

/// Sarcastic positive framing prefixes and multi-word cues (Riloff et al. 2013)
/// that signal ironic intent when paired with a negative situation or opinion.
const SARCASTIC_POSITIVE_PHRASES: &[(&[&str], f64)] = &[
    (&["oh", "great"], 2.6),
    (&["just", "great"], 2.6),
    (&["so", "great"], 2.3),
    (&["oh", "wonderful"], 2.7),
    (&["just", "wonderful"], 2.7),
    (&["oh", "fantastic"], 2.7),
    (&["just", "fantastic"], 2.7),
    (&["oh", "lovely"], 2.6),
    (&["just", "lovely"], 2.6),
    (&["oh", "brilliant"], 2.6),
    (&["oh", "joy"], 2.5),
    (&["oh", "goodie"], 2.5),
    (&["oh", "boy"], 2.0),
    (&["yeah", "right"], 2.8),
    (&["thanks", "a", "lot"], 2.6),
    (&["thank", "you", "so", "much"], 2.4),
    (&["thanks", "so", "much"], 2.4),
    (&["just", "what", "i", "needed"], 2.8),
    (&["exactly", "what", "i", "needed"], 2.8),
    (&["what", "a", "treat"], 2.6),
    (&["what", "a", "joy"], 2.6),
    (&["what", "a", "pleasure"], 2.5),
    (&["how", "wonderful"], 2.5),
    (&["how", "lovely"], 2.5),
    (&["how", "nice"], 2.3),
    (&["nothing", "like"], 2.3),
    (&["love", "waiting"], 2.7),
    (&["love", "being"], 2.6),
    (&["love", "getting"], 2.5),
    (&["love", "having"], 2.5),
    (&["love", "working"], 2.5),
    (&["love", "spending"], 2.5),
    (&["love", "sitting"], 2.5),
    (&["love", "when"], 2.6),
    (&["love", "how"], 2.5),
    (&["loved", "waiting"], 2.6),
    (&["loved", "being"], 2.5),
    (&["thrilled", "to"], 2.5),
    (&["delighted", "to"], 2.5),
    (&["so", "glad"], 2.4),
    (&["so", "happy"], 2.4),
    (&["so", "excited"], 2.4),
    (&["can't", "wait", "to"], 2.3),
    (&["cannot", "wait", "to"], 2.3),
    (&["cant", "wait", "to"], 2.3),
    (&["always", "fun"], 2.4),
    (&["so", "much", "fun"], 2.4),
];

/// Tokens immediately preceding a positive opinion that mark ironic
/// intensification in Riloff-style contrast.
const SARCASTIC_PRE_CUES: &[&str] =
    &["oh", "just", "sure", "totally", "absolutely", "really", "so", "yay", "wow", "gee"];

/// Positive exclamation/sentiment lemmas that can appear as a clause-final
/// ironic tag after a negative situation (e.g. `"The server crashed, great"`).
const CLAUSE_FINAL_POSITIVE_TAGS: &[&str] = &[
    "great",
    "wonderful",
    "fantastic",
    "lovely",
    "brilliant",
    "awesome",
    "perfect",
    "terrific",
    "fabulous",
    "marvelous",
    "super",
    "yay",
    "hooray",
    "nice",
    "fun",
];

/// Negative situation phrases (Riloff et al., EMNLP 2013): undesirable
/// activities, states, and events that contrast with positive sentiment
/// expressions even when individual words are neutral in a general lexicon.
const NEGATIVE_SITUATIONS: &[(&[&str], f64)] = &[
    // Waiting, transit, and travel disruptions
    (&["waiting", "in", "line"], 2.6),
    (&["wait", "in", "line"], 2.5),
    (&["waited", "in", "line"], 2.6),
    (&["waiting", "on", "hold"], 2.6),
    (&["on", "hold", "for"], 2.4),
    (&["stuck", "in", "traffic"], 2.7),
    (&["sitting", "in", "traffic"], 2.6),
    (&["traffic", "jam"], 2.4),
    (&["flight", "was", "cancelled"], 2.7),
    (&["flight", "was", "canceled"], 2.7),
    (&["flight", "cancelled"], 2.7),
    (&["flight", "canceled"], 2.7),
    (&["cancelled", "flight"], 2.7),
    (&["canceled", "flight"], 2.7),
    (&["flight", "delayed"], 2.6),
    (&["delayed", "flight"], 2.6),
    (&["flight", "was", "delayed"], 2.6),
    (&["missed", "my", "flight"], 2.7),
    (&["missed", "the", "bus"], 2.5),
    (&["missed", "the", "train"], 2.5),
    (&["lost", "my", "luggage"], 2.7),
    (&["lost", "my", "bags"], 2.6),
    (&["lost", "my", "keys"], 2.5),
    (&["locked", "out"], 2.4),
    (&["flat", "tire"], 2.7),
    (&["dead", "battery"], 2.5),
    (&["car", "broke", "down"], 2.7),
    (&["broke", "down"], 2.3),
    (&["parking", "ticket"], 2.5),
    // Work, chores, and scheduling obligations
    (&["working", "on", "saturday"], 2.5),
    (&["working", "on", "sunday"], 2.5),
    (&["working", "all", "weekend"], 2.6),
    (&["working", "on", "the", "weekend"], 2.5),
    (&["working", "late"], 2.3),
    (&["unpaid", "overtime"], 2.7),
    (&["overtime"], 2.1),
    (&["monday", "morning"], 2.1),
    (&["waking", "up", "at"], 2.2),
    (&["jury", "duty"], 2.4),
    (&["paying", "taxes"], 2.3),
    (&["doing", "taxes"], 2.3),
    (&["another", "meeting"], 2.2),
    // Software, hardware, and infrastructure failures
    (&["server", "crashed"], 2.7),
    (&["crashed", "again"], 2.6),
    (&["system", "crashed"], 2.7),
    (&["app", "crashed"], 2.6),
    (&["broken", "build"], 2.5),
    (&["build", "broke"], 2.5),
    (&["build", "failed"], 2.5),
    (&["merge", "conflict"], 2.4),
    (&["merge", "conflicts"], 2.4),
    (&["blue", "screen"], 2.5),
    (&["segfault"], 2.5),
    (&["segfaults"], 2.5),
    (&["power", "outage"], 2.6),
    (&["outage"], 2.2),
    (&["wifi", "is", "down"], 2.5),
    (&["internet", "is", "down"], 2.5),
    (&["no", "internet"], 2.4),
    (&["deleted", "the", "database"], 2.8),
    (&["deleted", "production"], 2.8),
    (&["data", "loss"], 2.6),
    (&["connection", "dropped"], 2.4),
    (&["timed", "out"], 2.3),
    // Health, weather, and physical discomfort
    (&["root", "canal"], 2.7),
    (&["toothache"], 2.5),
    (&["migraine"], 2.5),
    (&["headache"], 2.3),
    (&["insomnia"], 2.4),
    (&["food", "poisoning"], 2.8),
    (&["sunburn"], 2.3),
    (&["spilled", "coffee"], 2.4),
    (&["pouring", "rain"], 2.3),
    (&["freezing", "cold"], 2.3),
    (&["soaked", "in", "the", "rain"], 2.5),
    (&["drenched"], 2.2),
    // Interpersonal and social friction
    (&["being", "ignored"], 2.6),
    (&["ignored", "me"], 2.5),
    (&["left", "on", "read"], 2.4),
    (&["hung", "up", "on"], 2.5),
    (&["getting", "yelled", "at"], 2.6),
    (&["getting", "fired"], 2.8),
    (&["laid", "off"], 2.7),
];

/// Adversative conjunctions that introduce a contrasting clause in literal
/// mixed-sentiment sentences (`"The food was great, but the wait was long"`).
const ADVERSATIVE_CONJUNCTIONS: &[&str] =
    &["but", "however", "although", "though", "yet", "whereas", "while", "except", "nonetheless", "nevertheless"];

#[derive(Copy, Clone, Debug)]
struct SpanCandidate {
    tokens: IxRange<Tokens>,
    strength: f64,
    cued: bool,
}

/// Sentence-level sarcasm and irony annotator (`turbo-doccat:pragmatics`).
///
/// Reads [`View`], [`Sentences`], [`Tokens`], and optional [`Opinions`],
/// [`Categories<Sentences>`], [`Emoji`], and [`Vectors<Sentences>`], and
/// writes [`Pragmatics`] (`turbo:pragmatics`).
#[derive(Clone, Debug)]
pub struct PragmaticsAnnotator {
    lexicon: Arc<VaderLexicon>,
    min_positive_polarity: f64,
    min_negative_polarity: f64,
    positive_threshold: f64,
    vector_scorer: Option<Arc<dyn PragmaticsVectorScorer>>,
}

impl Default for PragmaticsAnnotator {
    fn default() -> Self {
        Self::new()
    }
}

impl PragmaticsAnnotator {
    /// Creates a `PragmaticsAnnotator` with the bundled MIT VADER lexicon and
    /// default thresholds.
    pub fn new() -> Self {
        Self::with_lexicon(VaderLexicon::bundled())
    }

    /// Creates a `PragmaticsAnnotator` with a custom [`VaderLexicon`].
    pub fn with_lexicon(lexicon: Arc<VaderLexicon>) -> Self {
        PragmaticsAnnotator {
            lexicon,
            min_positive_polarity: DEFAULT_MIN_POSITIVE_POLARITY,
            min_negative_polarity: DEFAULT_MIN_NEGATIVE_POLARITY,
            positive_threshold: DEFAULT_PRAGMATICS_THRESHOLD,
            vector_scorer: None,
        }
    }

    /// Attaches a sentence-embedding scorer (`Vectors<Sentences>` -> `[0.0, 1.0]`)
    /// used when a sentence has no explicit contrast pair (or to reinforce
    /// explicit contrast).
    pub fn with_vector_scorer(mut self, scorer: Arc<dyn PragmaticsVectorScorer>) -> Self {
        self.vector_scorer = Some(scorer);
        self
    }

    /// Sets the minimum effective polarity magnitudes required for positive
    /// and negative opinion spans.
    pub fn with_polarity_thresholds(mut self, min_positive: f64, min_negative: f64) -> Self {
        self.min_positive_polarity = min_positive.max(0.0);
        self.min_negative_polarity = min_negative.max(0.0);
        self
    }

    /// Sets the classification threshold (`0.0..=1.0`) above which a sentence
    /// is treated as positive for sarcasm/irony.
    pub fn with_positive_threshold(mut self, threshold: f64) -> Self {
        self.positive_threshold = threshold.clamp(0.0, 1.0);
        self
    }

    /// The configured classification threshold (`0.0..=1.0`).
    pub fn positive_threshold(&self) -> f64 {
        self.positive_threshold
    }

    /// Scores every sentence in `sentences` and builds the [`Pragmatics`]
    /// layer.
    #[allow(clippy::too_many_arguments)]
    pub fn score_document(
        &self,
        view: &View,
        sentences: &Sentences,
        tokens: &Tokens,
        opinions: Option<&Opinions>,
        categories: Option<&Categories<Sentences>>,
        emoji: Option<&Emoji>,
        vectors: Option<&Vectors<Sentences>>,
    ) -> Result<Pragmatics, Error> {
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
        if let Some(cats) = categories
            && cats.len() != sentences.len()
        {
            return Err(Error::Annotator {
                annotator: Self::ID,
                detail: format!(
                    "Categories<Sentences> length ({}) does not match Sentences length ({})",
                    cats.len(),
                    sentences.len()
                ),
            });
        }
        if let Some(vecs) = vectors
            && vecs.len() != sentences.len()
        {
            return Err(Error::Annotator {
                annotator: Self::ID,
                detail: format!(
                    "Vectors<Sentences> length ({}) does not match Sentences length ({})",
                    vecs.len(),
                    sentences.len()
                ),
            });
        }

        let text = view.text();
        let lower_tokens: Vec<String> = (0..tokens.len())
            .map(|i| {
                let sp = tokens.spans.get(Ix::new(i as u32));
                clean_token(&text[sp.range()])
            })
            .collect();

        let mut out = Pragmatics::new();
        for s_idx in 0..sentences.len() {
            let sent_ix = Ix::<Sentences>::new(s_idx as u32);
            let sent_span = sentences.spans.get(sent_ix);
            let tok_range = sentence_token_range(s_idx, sent_span, tokens);

            let head_score = self.score_head_for_sentence(s_idx, categories, vectors)?;

            if tok_range.is_empty() {
                let score = head_score.unwrap_or(0.0).clamp(0.0, 1.0);
                out.try_push(sentences, tokens, PragmaticsItem { score, evidence: None })
                    .map_err(|detail| Error::Annotator { annotator: Self::ID, detail })?;
                continue;
            }

            let emoji_spans = collect_sentence_emoji(sent_span, tok_range, text, tokens, emoji);
            let riloff = self.detect_riloff_incongruity(tok_range, &lower_tokens, opinions, &emoji_spans);
            let emoji_mismatch =
                self.detect_emoji_mismatch(s_idx, tok_range, &lower_tokens, opinions, categories, &emoji_spans);

            let explicit = match (riloff, emoji_mismatch) {
                (Some((s1, inc1)), Some((s2, inc2))) => {
                    let combined = 1.0 - (1.0 - s1) * (1.0 - s2 * 0.5);
                    if s2 > s1 {
                        Some((combined.clamp(0.0, 1.0), inc2))
                    } else {
                        Some((combined.clamp(0.0, 1.0), inc1))
                    }
                }
                (Some(r), None) => Some(r),
                (None, Some(e)) => Some(e),
                (None, None) => None,
            };

            let item = match (explicit, head_score) {
                (Some((contrast_score, incongruity)), Some(h)) => {
                    let score = if h >= self.positive_threshold {
                        (1.0 - (1.0 - contrast_score) * (1.0 - h * 0.35)).clamp(0.0, 1.0)
                    } else {
                        contrast_score.clamp(0.0, 1.0)
                    };
                    PragmaticsItem { score, evidence: Some(incongruity) }
                }
                (Some((contrast_score, incongruity)), None) => {
                    PragmaticsItem { score: contrast_score.clamp(0.0, 1.0), evidence: Some(incongruity) }
                }
                (None, Some(h)) => PragmaticsItem { score: h.clamp(0.0, 1.0), evidence: None },
                (None, None) => PragmaticsItem { score: 0.0, evidence: None },
            };

            out.try_push(sentences, tokens, item).map_err(|detail| Error::Annotator { annotator: Self::ID, detail })?;
        }

        Ok(out)
    }

    fn score_head_for_sentence(
        &self,
        sentence_index: usize,
        categories: Option<&Categories<Sentences>>,
        vectors: Option<&Vectors<Sentences>>,
    ) -> Result<Option<f64>, Error> {
        let mut best: Option<f64> = None;

        let ix = Ix::<Sentences>::new(sentence_index as u32);
        if let (Some(scorer), Some(vecs)) = (&self.vector_scorer, vectors)
            && !vecs.is_empty(ix)
        {
            // A row the encoder marked empty is zeros, not evidence: the
            // scorer never sees it. Vectors from another encoder are a
            // different feature space, so the scorer's model, when it
            // names one, must be the vectors' model.
            if let Some(m) = scorer.model_ref()
                && m != vecs.model
            {
                return Err(Error::Annotator {
                    annotator: Self::ID,
                    detail: format!(
                        "sentence vectors are from model {} (revision {}), scorer is for {} (revision {})",
                        vecs.model.id, vecs.model.revision, m.id, m.revision
                    ),
                });
            }
            let s = scorer
                .score_sentence_vector(vecs.row(ix))
                .map_err(|detail| Error::Annotator { annotator: Self::ID, detail })?;
            if !s.is_finite() || !(0.0..=1.0).contains(&s) {
                return Err(Error::Annotator {
                    annotator: Self::ID,
                    detail: format!(
                        "PragmaticsVectorScorer returned score {s} outside 0..=1 on sentence {sentence_index}"
                    ),
                });
            }
            best = Some(s);
        }

        if let Some(cats) = categories {
            let mut cat_prag: Option<f64> = None;
            for label in ["sarcasm", "irony", "sarcastic", "ironic"] {
                if let Some(v) = cats.score_of(sentence_index, label)
                    && v.is_finite()
                    && (0.0..=1.0).contains(&v)
                {
                    cat_prag = Some(cat_prag.map_or(v, |prev| prev.max(v)));
                }
            }
            if let Some(v) = cat_prag {
                best = Some(best.map_or(v, |prev| prev.max(v)));
            }
        }

        Ok(best)
    }

    fn detect_riloff_incongruity(
        &self,
        tok_range: IxRange<Tokens>,
        lower_tokens: &[String],
        opinions: Option<&Opinions>,
        emoji_spans: &[(IxRange<Tokens>, i8)],
    ) -> Option<(f64, Incongruity)> {
        let positives = self.collect_positive_candidates(tok_range, lower_tokens, opinions, emoji_spans);
        let negatives = self.collect_negative_candidates(tok_range, lower_tokens, opinions, emoji_spans);

        if positives.is_empty() || negatives.is_empty() {
            return None;
        }

        let mut best: Option<(f64, Incongruity)> = None;
        for pos in &positives {
            for neg in &negatives {
                if ranges_overlap(pos.tokens, neg.tokens) {
                    continue;
                }
                let (first, second) = if pos.tokens.start.get() < neg.tokens.start.get() {
                    (pos.tokens, neg.tokens)
                } else {
                    (neg.tokens, pos.tokens)
                };
                let between = &lower_tokens[first.end.get()..second.start.get()];

                // Balanced clauses separated by an adversative conjunction ("but",
                // "however", "although", ...) are literal contrast unless the
                // positive span carries an explicit sarcastic framing cue.
                let has_adversative = between.iter().any(|w| ADVERSATIVE_CONJUNCTIONS.contains(&w.as_str()));
                if has_adversative && !pos.cued {
                    continue;
                }

                // When the negative span comes before the positive span and the
                // positive span is not a sarcastic cue or a clause-final ironic tag,
                // avoid firing on plain narrative sequence ("had a headache, and
                // now feel good").
                let pos_after_neg = pos.tokens.start.get() >= neg.tokens.end.get();
                if pos_after_neg && !pos.cued && !is_clause_final_tag(pos.tokens, tok_range, lower_tokens) {
                    continue;
                }

                // Require at least one Riloff situation or sarcastic framing cue, or
                // a strong positive sentiment expression (>= 1.8) paired with a
                // strong negative opinion (>= 1.5), so incidental mild lexicon words
                // ("original ... blame") do not false-trigger without the head.
                if !pos.cued && !neg.cued && (pos.strength < 1.8 || neg.strength < 1.5) {
                    continue;
                }

                let cue_bonus = match (pos.cued, neg.cued) {
                    (true, true) => 0.25,
                    (true, false) | (false, true) => 0.15,
                    (false, false) => 0.0,
                };
                let raw = (pos.strength + neg.strength) * 0.18 + cue_bonus;
                let score = (0.45 + 0.50 * (1.0 - (-raw).exp())).clamp(0.52, 0.98);

                let inc = Incongruity { positive: pos.tokens, negative: neg.tokens };
                if best.is_none_or(|(b_score, _)| score > b_score) {
                    best = Some((score, inc));
                }
            }
        }

        best
    }

    fn detect_emoji_mismatch(
        &self,
        sentence_index: usize,
        tok_range: IxRange<Tokens>,
        lower_tokens: &[String],
        opinions: Option<&Opinions>,
        categories: Option<&Categories<Sentences>>,
        emoji_spans: &[(IxRange<Tokens>, i8)],
    ) -> Option<(f64, Incongruity)> {
        if emoji_spans.is_empty() {
            return None;
        }

        let positives = self.collect_positive_candidates(tok_range, lower_tokens, opinions, emoji_spans);
        let negatives = self.collect_negative_candidates(tok_range, lower_tokens, opinions, emoji_spans);
        let non_emoji_range = first_non_emoji_token_range(tok_range, emoji_spans);

        let cat_net_polarity = categories.map(|cats| {
            let pos = cats.score_of(sentence_index, "positive").unwrap_or(0.0);
            let neg = cats.score_of(sentence_index, "negative").unwrap_or(0.0);
            pos - neg
        });

        let mut best: Option<(f64, Incongruity)> = None;

        for &(em_range, em_sent) in emoji_spans {
            if em_sent < 0 {
                // Negative emoji paired with positive text opinion.
                let pos_fallback = match (cat_net_polarity, non_emoji_range) {
                    (Some(net), Some(r)) if net >= 0.25 => {
                        Some(SpanCandidate { tokens: r, strength: net * 2.5, cued: false })
                    }
                    _ => None,
                };
                let pos_text = positives
                    .iter()
                    .filter(|p| !ranges_overlap(p.tokens, em_range))
                    .max_by(|a, b| a.strength.total_cmp(&b.strength))
                    .copied()
                    .or(pos_fallback);

                if let Some(pos) = pos_text {
                    let em_mag = (em_sent.unsigned_abs() as f64).max(1.0);
                    let score = (0.58 + 0.12 * em_mag + 0.06 * pos.strength.min(3.0)).clamp(0.60, 0.96);
                    let inc = Incongruity { positive: pos.tokens, negative: em_range };
                    if best.is_none_or(|(b, _)| score > b) {
                        best = Some((score, inc));
                    }
                }
            } else if em_sent > 0 {
                // Positive emoji paired with negative text opinion or negative situation.
                let neg_fallback = match (cat_net_polarity, non_emoji_range) {
                    (Some(net), Some(r)) if net <= -0.25 => {
                        Some(SpanCandidate { tokens: r, strength: (-net) * 2.5, cued: false })
                    }
                    _ => None,
                };
                let neg_text = negatives
                    .iter()
                    .filter(|n| !ranges_overlap(n.tokens, em_range))
                    .max_by(|a, b| a.strength.total_cmp(&b.strength))
                    .copied()
                    .or(neg_fallback);

                if let Some(neg) = neg_text {
                    let em_mag = (em_sent as f64).max(1.0);
                    let score = (0.58 + 0.12 * em_mag + 0.06 * neg.strength.min(3.0)).clamp(0.60, 0.96);
                    let inc = Incongruity { positive: em_range, negative: neg.tokens };
                    if best.is_none_or(|(b, _)| score > b) {
                        best = Some((score, inc));
                    }
                }
            }
        }

        best
    }

    fn collect_positive_candidates(
        &self,
        tok_range: IxRange<Tokens>,
        lower_tokens: &[String],
        opinions: Option<&Opinions>,
        emoji_spans: &[(IxRange<Tokens>, i8)],
    ) -> Vec<SpanCandidate> {
        let mut out = Vec::new();
        let start = tok_range.start.get();
        let end = tok_range.end.get();

        // 1. Multi-token sarcastic positive phrases in this sentence.
        for &(phrase, strength) in SARCASTIC_POSITIVE_PHRASES {
            let plen = phrase.len();
            if plen == 0 || end - start < plen {
                continue;
            }
            for i in start..=end - plen {
                if (0..plen).all(|k| lower_tokens[i + k] == phrase[k]) {
                    let r = IxRange::new(i as u32, (i + plen) as u32);
                    if !overlaps_any_emoji(r, emoji_spans) {
                        out.push(SpanCandidate { tokens: r, strength, cued: true });
                    }
                }
            }
        }

        // 2. Positive opinions from `turbo:opinions`.
        if let Some(ops) = opinions {
            for (_, op) in ops.iter() {
                if op.tokens.start.get() < start || op.tokens.end.get() > end {
                    continue;
                }
                if overlaps_any_emoji(op.tokens, emoji_spans) {
                    continue;
                }
                let eff = op.effective_polarity();
                if eff >= self.min_positive_polarity {
                    let cued = has_preceding_sarcastic_cue(op.tokens, tok_range, lower_tokens);
                    out.push(SpanCandidate { tokens: op.tokens, strength: eff, cued });
                }
            }
        } else {
            // Fallback when `turbo:opinions` is not on the document: scan tokens
            // with the bundled VADER lexicon.
            for (i, w) in lower_tokens.iter().enumerate().take(end).skip(start) {
                if w.is_empty() {
                    continue;
                }
                let r = IxRange::new(i as u32, (i + 1) as u32);
                if overlaps_any_emoji(r, emoji_spans) {
                    continue;
                }
                if let Some(valence) = self.lexicon.get(w)
                    && valence >= self.min_positive_polarity
                    && !has_preceding_negation(i, start, lower_tokens)
                {
                    let cued = has_preceding_sarcastic_cue(r, tok_range, lower_tokens);
                    out.push(SpanCandidate { tokens: r, strength: valence, cued });
                }
            }
        }

        out
    }

    fn collect_negative_candidates(
        &self,
        tok_range: IxRange<Tokens>,
        lower_tokens: &[String],
        opinions: Option<&Opinions>,
        emoji_spans: &[(IxRange<Tokens>, i8)],
    ) -> Vec<SpanCandidate> {
        let mut out = Vec::new();
        let start = tok_range.start.get();
        let end = tok_range.end.get();

        // 1. Riloff negative situation phrases.
        for &(phrase, strength) in NEGATIVE_SITUATIONS {
            let plen = phrase.len();
            if plen == 0 || end - start < plen {
                continue;
            }
            for i in start..=end - plen {
                if (0..plen).all(|k| lower_tokens[i + k] == phrase[k]) {
                    let r = IxRange::new(i as u32, (i + plen) as u32);
                    if !overlaps_any_emoji(r, emoji_spans) {
                        out.push(SpanCandidate { tokens: r, strength, cued: true });
                    }
                }
            }
        }

        // 2. Negative opinions from `turbo:opinions`.
        if let Some(ops) = opinions {
            for (_, op) in ops.iter() {
                if op.tokens.start.get() < start || op.tokens.end.get() > end {
                    continue;
                }
                if overlaps_any_emoji(op.tokens, emoji_spans) {
                    continue;
                }
                let eff = op.effective_polarity();
                if eff <= -self.min_negative_polarity {
                    // Exclude "yeah right", which VADER scores negatively but
                    // functions as the positive ironic cue in a pair.
                    let len = op.tokens.end.get() - op.tokens.start.get();
                    if len == 2
                        && lower_tokens[op.tokens.start.get()] == "yeah"
                        && lower_tokens[op.tokens.start.get() + 1] == "right"
                    {
                        continue;
                    }
                    out.push(SpanCandidate { tokens: op.tokens, strength: -eff, cued: false });
                }
            }
        } else {
            for (i, w) in lower_tokens.iter().enumerate().take(end).skip(start) {
                if w.is_empty() {
                    continue;
                }
                let r = IxRange::new(i as u32, (i + 1) as u32);
                if overlaps_any_emoji(r, emoji_spans) {
                    continue;
                }
                if let Some(valence) = self.lexicon.get(w) {
                    let negated = has_preceding_negation(i, start, lower_tokens);
                    let eff = if negated { -0.74 * valence } else { valence };
                    if eff <= -self.min_negative_polarity {
                        out.push(SpanCandidate { tokens: r, strength: -eff, cued: false });
                    }
                }
            }
        }

        out
    }
}

impl Annotator for PragmaticsAnnotator {
    const ID: &'static str = "turbo-doccat:pragmatics";
    type Reads = (
        View,
        Sentences,
        Tokens,
        Option<Opinions>,
        Option<Categories<Sentences>>,
        Option<Emoji>,
        Option<Vectors<Sentences>>,
    );
    type Writes = (Pragmatics,);

    fn annotate(
        &self,
        (view, sentences, tokens, opinions, categories, emoji, vectors): Refs<'_, Self>,
    ) -> Result<(Pragmatics,), Error> {
        let pragmatics = self.score_document(view, sentences, tokens, opinions, categories, emoji, vectors)?;
        Ok((pragmatics,))
    }

    fn model(&self) -> Option<ModelRef> {
        if let Some(scorer) = &self.vector_scorer
            && let Some(m) = scorer.model_ref()
        {
            return Some(m);
        }
        Some(ModelRef {
            id: "vader-lexicon+riloff-pragmatics".into(),
            revision: "vader-1.0".into(),
            sha256: self.lexicon.sha256_hex().into(),
        })
    }

    fn config_hash(&self) -> u64 {
        let mut h = crate::hash::FoldHasher::default();
        self.lexicon.sha256_hex().hash(&mut h);
        self.min_positive_polarity.to_bits().hash(&mut h);
        self.min_negative_polarity.to_bits().hash(&mut h);
        self.positive_threshold.to_bits().hash(&mut h);
        if let Some(s) = &self.vector_scorer {
            1u8.hash(&mut h);
            s.config_hash().hash(&mut h);
        } else {
            0u8.hash(&mut h);
        }
        h.finish()
    }
}

fn sentence_token_range(sentence_index: usize, sentence_span: Span, tokens: &Tokens) -> IxRange<Tokens> {
    if let Some(tok_sents) = &tokens.sentence {
        let target = Ix::<Sentences>::new(sentence_index as u32);
        let a = tok_sents.partition_point(|&s| s < target);
        let b = a + tok_sents[a..].partition_point(|&s| s == target);
        return IxRange::new(a as u32, b as u32);
    }
    let n = tokens.len();
    let mut a = 0usize;
    while a < n && tokens.spans.get(Ix::new(a as u32)).end <= sentence_span.start {
        a += 1;
    }
    let mut b = a;
    while b < n {
        let sp = tokens.spans.get(Ix::new(b as u32));
        if sp.start >= sentence_span.end || sp.end > sentence_span.end {
            break;
        }
        b += 1;
    }
    IxRange::new(a as u32, b as u32)
}

fn emoji_polarity(sequence: &str, cldr_name: &str) -> i8 {
    if let Some(ann) = opennlp_annotation(sequence)
        && ann.sentiment != 0
    {
        return ann.sentiment;
    }
    for ch in sequence.chars() {
        match ch {
            // Positive emoji common in sarcastic contrast ("everything is on fire 😂🎉👍")
            '\u{1F602}' | '\u{1F923}' | '\u{1F973}' | '\u{1F389}' | '\u{1F38A}' | '\u{1F60D}' | '\u{1F929}'
            | '\u{1F970}' | '\u{1F606}' | '\u{1F607}' => return 2,
            '\u{1F44D}' | '\u{1F44F}' | '\u{1F64C}' | '\u{1F525}' | '\u{2728}' | '\u{1F4AF}' | '\u{1F60E}'
            | '\u{1F60B}' => return 1,
            // Negative / eye-roll / exasperated emoji common in sarcastic contrast ("just fantastic 🙄😒")
            '\u{1F644}' | '\u{1F612}' | '\u{1F621}' | '\u{1F620}' | '\u{1F92C}' | '\u{1F92E}' | '\u{1F922}'
            | '\u{1F480}' | '\u{1F4A9}' => return -2,
            '\u{1F643}' | '\u{1F44E}' | '\u{1F624}' | '\u{1F629}' | '\u{1F62B}' | '\u{1F61E}' | '\u{1F616}'
            | '\u{1F623}' | '\u{1F613}' | '\u{1F614}' => return -1,
            _ => {}
        }
    }
    if !cldr_name.is_empty() {
        if cldr_name.contains("rolling eyes")
            || cldr_name.contains("unamused")
            || cldr_name.contains("angry")
            || cldr_name.contains("enraged")
            || cldr_name.contains("vomiting")
            || cldr_name.contains("nauseated")
            || cldr_name.contains("crying")
            || cldr_name.contains("broken heart")
        {
            return -2;
        }
        if cldr_name.contains("frowning")
            || cldr_name.contains("confused")
            || cldr_name.contains("disappointed")
            || cldr_name.contains("weary")
            || cldr_name.contains("tired")
            || cldr_name.contains("thumbs down")
            || cldr_name.contains("upside-down")
        {
            return -1;
        }
        if cldr_name.contains("tears of joy")
            || cldr_name.contains("rolling on the floor laughing")
            || cldr_name.contains("party popper")
            || cldr_name.contains("partying face")
            || cldr_name.contains("heart-eyes")
            || cldr_name.contains("star-struck")
            || cldr_name.contains("grinning")
        {
            return 2;
        }
        if cldr_name.contains("smiling")
            || cldr_name.contains("thumbs up")
            || cldr_name.contains("clapping")
            || cldr_name.contains("raising hands")
        {
            return 1;
        }
    }
    0
}

fn collect_sentence_emoji(
    sentence_span: Span,
    tok_range: IxRange<Tokens>,
    text: &str,
    tokens: &Tokens,
    emoji: Option<&Emoji>,
) -> Vec<(IxRange<Tokens>, i8)> {
    let mut out = Vec::new();
    let start = tok_range.start.get();
    let end = tok_range.end.get();
    if start >= end {
        return out;
    }

    if let Some(em_layer) = emoji {
        for (em_ix, em_span) in em_layer.spans.iter() {
            if em_span.start < sentence_span.start || em_span.end > sentence_span.end {
                continue;
            }
            let seq = &text[em_span.range()];
            let cldr_name = em_layer.name.get(em_ix.get());
            let pol = emoji_polarity(seq, cldr_name);
            if pol == 0 {
                continue;
            }
            if let Some(r) = byte_span_to_token_range(em_span, tok_range, tokens) {
                out.push((r, pol));
            }
        }
    }

    // Also inspect tokens directly in case `EmojiAnnotate` was not run ahead
    // of `PragmaticsAnnotator`.
    for idx in start..end {
        let r = IxRange::new(idx as u32, (idx + 1) as u32);
        if overlaps_any_emoji(r, &out) {
            continue;
        }
        let sp = tokens.spans.get(Ix::new(idx as u32));
        let surface = &text[sp.range()];
        let pol = emoji_polarity(surface, "");
        if pol != 0 {
            out.push((r, pol));
        }
    }

    out
}

fn byte_span_to_token_range(span: Span, sent_tok_range: IxRange<Tokens>, tokens: &Tokens) -> Option<IxRange<Tokens>> {
    let start = sent_tok_range.start.get();
    let end = sent_tok_range.end.get();
    let mut first = None;
    let mut last = None;
    for i in start..end {
        let tsp = tokens.spans.get(Ix::new(i as u32));
        if tsp.end > span.start && tsp.start < span.end {
            if first.is_none() {
                first = Some(i as u32);
            }
            last = Some((i + 1) as u32);
        }
    }
    match (first, last) {
        (Some(a), Some(b)) if a < b => Some(IxRange::new(a, b)),
        _ => None,
    }
}

fn first_non_emoji_token_range(
    tok_range: IxRange<Tokens>,
    emoji_spans: &[(IxRange<Tokens>, i8)],
) -> Option<IxRange<Tokens>> {
    let start = tok_range.start.get() as u32;
    let end = tok_range.end.get() as u32;
    let mut a = None;
    let mut b = None;
    for t in start..end {
        let r = IxRange::new(t, t + 1);
        if !overlaps_any_emoji(r, emoji_spans) {
            if a.is_none() {
                a = Some(t);
            }
            b = Some(t + 1);
        }
    }
    match (a, b) {
        (Some(x), Some(y)) if x < y => Some(IxRange::new(x, y)),
        _ => None,
    }
}

fn overlaps_any_emoji(range: IxRange<Tokens>, emoji_spans: &[(IxRange<Tokens>, i8)]) -> bool {
    emoji_spans.iter().any(|&(er, _)| ranges_overlap(range, er))
}

fn ranges_overlap(a: IxRange<Tokens>, b: IxRange<Tokens>) -> bool {
    a.start.get() < b.end.get() && b.start.get() < a.end.get()
}

fn has_preceding_sarcastic_cue(range: IxRange<Tokens>, sent_range: IxRange<Tokens>, lower_tokens: &[String]) -> bool {
    let s = sent_range.start.get();
    let pos_start = range.start.get();
    let window_start = pos_start.saturating_sub(2).max(s);
    for w in &lower_tokens[window_start..pos_start] {
        if SARCASTIC_PRE_CUES.contains(&w.as_str()) {
            return true;
        }
    }
    false
}

fn has_preceding_negation(token_index: usize, sent_start: usize, lower_tokens: &[String]) -> bool {
    let window_start = token_index.saturating_sub(3).max(sent_start);
    for w in &lower_tokens[window_start..token_index] {
        if matches!(
            w.as_str(),
            "not"
                | "never"
                | "no"
                | "cannot"
                | "cant"
                | "can't"
                | "dont"
                | "don't"
                | "didnt"
                | "didn't"
                | "doesnt"
                | "doesn't"
                | "isnt"
                | "isn't"
                | "wasnt"
                | "wasn't"
                | "wont"
                | "won't"
                | "wouldnt"
                | "wouldn't"
        ) {
            return true;
        }
    }
    false
}

fn is_clause_final_tag(pos_range: IxRange<Tokens>, sent_range: IxRange<Tokens>, lower_tokens: &[String]) -> bool {
    let len = pos_range.end.get() - pos_range.start.get();
    if len == 0 || len > 2 {
        return false;
    }
    let last_word = &lower_tokens[pos_range.end.get() - 1];
    if !CLAUSE_FINAL_POSITIVE_TAGS.contains(&last_word.as_str()) {
        return false;
    }
    // Check that only punctuation (empty cleaned tokens) follows `pos_range`
    // up to the end of the sentence.
    lower_tokens[pos_range.end.get()..sent_range.end.get()].iter().all(|w| w.is_empty())
}

fn clean_token(raw: &str) -> String {
    let trimmed = raw.trim_matches(|c: char| c.is_ascii_punctuation() && c != '\'');
    trimmed.to_ascii_lowercase()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sentiment::{LexiconSentiment, SENTENCE_SENTIMENT};
    use turbo_analysis::{Analyzer, Bound, Document, Key, Provenance, Spans, ViewId};
    use turbo_normalize::emoji::EmojiAnnotate;

    fn annotate_sentences_and_tokens(doc: &mut Document) {
        let text = doc.source().to_owned();
        let mut s_spans = Spans::new(ViewId::SOURCE);
        let mut t_spans = Spans::new(ViewId::SOURCE);
        let mut t_sent = Vec::new();

        // Split sentences on `. `, `! `, `? ` or end of string.
        let mut sent_start = 0usize;
        let bytes = text.as_bytes();
        let mut s_idx = 0u32;
        let mut i = 0usize;
        while i < bytes.len() {
            let is_end =
                matches!(bytes[i], b'.' | b'!' | b'?') && (i + 1 == bytes.len() || bytes[i + 1].is_ascii_whitespace());
            if is_end || i + 1 == bytes.len() {
                let sent_end = i + 1;
                s_spans.push(Span::new(sent_start as u32, sent_end as u32));
                push_tokens_in_range(&text, sent_start, sent_end, s_idx, &mut t_spans, &mut t_sent);
                s_idx += 1;
                i = sent_end;
                while i < bytes.len() && bytes[i].is_ascii_whitespace() {
                    i += 1;
                }
                sent_start = i;
            } else {
                i += 1;
            }
        }

        doc.add(&Key::DEFAULT, Sentences { spans: s_spans }, Provenance::by("test")).unwrap();
        doc.add(&Key::DEFAULT, Tokens { spans: t_spans, sentence: Some(t_sent) }, Provenance::by("test")).unwrap();
    }

    fn push_tokens_in_range(
        text: &str,
        start: usize,
        end: usize,
        sent_idx: u32,
        t_spans: &mut Spans<Tokens>,
        t_sent: &mut Vec<Ix<Sentences>>,
    ) {
        let slice = &text[start..end];
        let mut idx = 0usize;
        while idx < slice.len() {
            let ch = slice[idx..].chars().next().unwrap();
            let ch_len = ch.len_utf8();
            if ch.is_whitespace() {
                idx += ch_len;
                continue;
            }
            if ch.is_ascii_punctuation() && ch != '\'' {
                t_spans.push(Span::new((start + idx) as u32, (start + idx + ch_len) as u32));
                t_sent.push(Ix::new(sent_idx));
                idx += ch_len;
                continue;
            }
            if !ch.is_ascii() {
                // Keep non-ASCII emoji sequence together until whitespace or ASCII.
                let tok_start = idx;
                idx += ch_len;
                while idx < slice.len() {
                    let c2 = slice[idx..].chars().next().unwrap();
                    if c2.is_whitespace() || c2.is_ascii() {
                        break;
                    }
                    idx += c2.len_utf8();
                }
                t_spans.push(Span::new((start + tok_start) as u32, (start + idx) as u32));
                t_sent.push(Ix::new(sent_idx));
                continue;
            }
            let tok_start = idx;
            idx += ch_len;
            while idx < slice.len() {
                let c2 = slice[idx..].chars().next().unwrap();
                if c2.is_whitespace() || (c2.is_ascii_punctuation() && c2 != '\'') || !c2.is_ascii() {
                    break;
                }
                idx += c2.len_utf8();
            }
            t_spans.push(Span::new((start + tok_start) as u32, (start + idx) as u32));
            t_sent.push(Ix::new(sent_idx));
        }
    }

    #[derive(Debug)]
    struct FixedMapVectorScorer {
        weights: Vec<f32>,
        bias: f32,
    }

    impl PragmaticsVectorScorer for FixedMapVectorScorer {
        fn score_sentence_vector(&self, vec: &[f32]) -> Result<f64, String> {
            if vec.len() != self.weights.len() {
                return Err(format!("dim mismatch: got {}, expected {}", vec.len(), self.weights.len()));
            }
            let mut dot = self.bias as f64;
            for (&x, &w) in vec.iter().zip(&self.weights) {
                dot += (x as f64) * (w as f64);
            }
            Ok(1.0 / (1.0 + (-dot).exp()))
        }

        fn config_hash(&self) -> u64 {
            0x5A2C_A501
        }
    }

    #[test]
    fn riloff_incongruity_detects_positive_sentiment_with_negative_situation_and_stores_spans() {
        let mut doc = Document::new(
            "I just love waiting in line for three hours. Oh great, the server crashed again. The food was great, but the wait was long.",
        );
        annotate_sentences_and_tokens(&mut doc);

        let pipe = Analyzer::builder()
            .given(Key::<Sentences>::DEFAULT)
            .given(Key::<Tokens>::DEFAULT)
            .add(EmojiAnnotate::new())
            .step(LexiconSentiment::new().bound())
            .add(PragmaticsAnnotator::new())
            .build()
            .unwrap();

        let mut docs = [doc];
        pipe.run(&mut docs).remove(0).unwrap();
        let doc = &docs[0];

        let tokens = doc.get::<Tokens>().unwrap();
        let prag = doc.get::<Pragmatics>().unwrap();
        assert_eq!(prag.len(), 3);

        // Sentence 0: "I just love waiting in line for three hours."
        let s0 = prag.get(Ix::new(0));
        assert!(s0.score >= 0.65, "expected high sarcasm score on s0, got {}", s0.score);
        let (pos_sp0, neg_sp0) = prag.evidence_spans(Ix::new(0), tokens).expect("s0 evidence spans");
        let text = doc.source();
        assert!(
            text[pos_sp0.range()].contains("love"),
            "expected positive evidence on 'love', got {:?}",
            &text[pos_sp0.range()]
        );
        assert!(
            text[neg_sp0.range()].contains("waiting in line"),
            "expected negative evidence on 'waiting in line', got {:?}",
            &text[neg_sp0.range()]
        );

        // Sentence 1: "Oh great, the server crashed again."
        let s1 = prag.get(Ix::new(1));
        assert!(s1.score >= 0.65, "expected high sarcasm score on s1, got {}", s1.score);
        let (pos_sp1, neg_sp1) = prag.evidence_spans(Ix::new(1), tokens).expect("s1 evidence spans");
        assert!(
            text[pos_sp1.range()].to_ascii_lowercase().contains("great"),
            "expected positive evidence on 'Oh great', got {:?}",
            &text[pos_sp1.range()]
        );
        assert!(
            text[neg_sp1.range()].contains("crashed"),
            "expected negative evidence on 'server crashed', got {:?}",
            &text[neg_sp1.range()]
        );

        // Sentence 2: "The food was great, but the wait was long." (adversative 'but' -> not sarcasm)
        let s2 = prag.get(Ix::new(2));
        assert_eq!(s2.score, 0.0, "balanced clause with 'but' must not score as sarcasm");
        assert!(s2.evidence.is_none());

        // Verify polarity in `Categories<Sentences>` is untouched (`turbo:pragmatics`
        // is never folded into polarity).
        let pol = doc.get_key::<Categories<Sentences>>(&SENTENCE_SENTIMENT).unwrap();
        assert_eq!(pol.best_category(0), "positive");
    }

    #[test]
    fn text_against_emoji_polarity_mismatch_records_evidence_spans() {
        let mut doc =
            Document::new("This new update is just fantastic 🙄. My flight was cancelled and I lost my luggage 😂.");
        annotate_sentences_and_tokens(&mut doc);

        let pipe = Analyzer::builder()
            .given(Key::<Sentences>::DEFAULT)
            .given(Key::<Tokens>::DEFAULT)
            .add(EmojiAnnotate::new())
            .step(LexiconSentiment::new().bound())
            .step(Bound::new(PragmaticsAnnotator::new()).read(4, SENTENCE_SENTIMENT.name))
            .build()
            .unwrap();

        let mut docs = [doc];
        pipe.run(&mut docs).remove(0).unwrap();
        let doc = &docs[0];

        let tokens = doc.get::<Tokens>().unwrap();
        let prag = doc.get::<Pragmatics>().unwrap();
        let text = doc.source();

        // Sentence 0: Positive text ("just fantastic") + negative emoji ("🙄").
        let s0 = prag.get(Ix::new(0));
        assert!(s0.score >= 0.65, "s0 score {}", s0.score);
        let (pos0, neg0) = prag.evidence_spans(Ix::new(0), tokens).unwrap();
        assert!(text[pos0.range()].contains("fantastic"), "got {:?}", &text[pos0.range()]);
        assert_eq!(&text[neg0.range()], "🙄");

        // Sentence 1: Negative situation ("flight was cancelled" / "lost my luggage") + positive emoji ("😂").
        let s1 = prag.get(Ix::new(1));
        assert!(s1.score >= 0.65, "s1 score {}", s1.score);
        let (pos1, neg1) = prag.evidence_spans(Ix::new(1), tokens).unwrap();
        assert_eq!(&text[pos1.range()], "😂");
        assert!(
            text[neg1.range()].contains("cancelled") || text[neg1.range()].contains("luggage"),
            "unexpected neg1 span {:?}",
            &text[neg1.range()]
        );
    }

    #[test]
    fn embedding_head_scores_implicit_irony_with_none_evidence() {
        let mut doc = Document::new("How predictably bureaucratic. Everything went according to plan.");
        annotate_sentences_and_tokens(&mut doc);

        // Sentence 0 vector points along the sarcasm direction; sentence 1 opposite.
        let vecs = Vectors::<Sentences>::new(
            2,
            vec![2.5f32, 1.0, -2.0, -1.0],
            Vec::new(),
            ModelRef { id: "test".into(), revision: "0".into(), sha256: "0".into() },
        );
        doc.add(&Key::DEFAULT, vecs, Provenance::by("test")).unwrap();

        let scorer = Arc::new(FixedMapVectorScorer { weights: vec![1.0, 1.0], bias: 0.0 });
        let annotator = PragmaticsAnnotator::new().with_vector_scorer(scorer);

        let pipe = Analyzer::builder()
            .given(Key::<Sentences>::DEFAULT)
            .given(Key::<Tokens>::DEFAULT)
            .given(Key::<Vectors<Sentences>>::DEFAULT)
            .add(annotator)
            .build()
            .unwrap();

        let mut docs = [doc];
        pipe.run(&mut docs).remove(0).unwrap();
        let prag = docs[0].get::<Pragmatics>().unwrap();

        // Sentence 0: scored positive by the embedding head alone (`evidence == None`).
        let s0 = prag.get(Ix::new(0));
        assert!(s0.score > 0.90, "expected head score > 0.90, got {}", s0.score);
        assert!(s0.evidence.is_none(), "head-only positive must have None evidence");

        // Sentence 1: scored negative by the head.
        let s1 = prag.get(Ix::new(1));
        assert!(s1.score < 0.10, "expected head score < 0.10, got {}", s1.score);
        assert!(s1.evidence.is_none());
    }

    #[test]
    fn an_empty_vector_row_gets_no_head_score() {
        let mut doc = Document::new("Meeting notes.");
        annotate_sentences_and_tokens(&mut doc);
        let vecs = Vectors::<Sentences>::new(
            2,
            vec![0.0f32, 0.0],
            vec![Ix::new(0)],
            ModelRef { id: "test".into(), revision: "0".into(), sha256: "0".into() },
        );
        doc.add(&Key::DEFAULT, vecs, Provenance::by("test")).unwrap();
        // A biased scorer that says 0.99 to anything.
        let scorer = Arc::new(FixedMapVectorScorer { weights: vec![0.0, 0.0], bias: 4.6 });
        let pipe = Analyzer::builder()
            .given(Key::<Sentences>::DEFAULT)
            .given(Key::<Tokens>::DEFAULT)
            .given(Key::<Vectors<Sentences>>::DEFAULT)
            .add(PragmaticsAnnotator::new().with_vector_scorer(scorer))
            .build()
            .unwrap();
        let mut docs = [doc];
        pipe.run(&mut docs).remove(0).unwrap();
        let prag = docs[0].get::<Pragmatics>().unwrap();
        assert_eq!(prag.get(Ix::new(0)).score, 0.0);
        assert!(prag.get(Ix::new(0)).evidence.is_none());
    }

    #[derive(Debug)]
    struct NamedScorer(ModelRef);

    impl PragmaticsVectorScorer for NamedScorer {
        fn score_sentence_vector(&self, _vec: &[f32]) -> Result<f64, String> {
            Ok(0.99)
        }
        fn config_hash(&self) -> u64 {
            1
        }
        fn model_ref(&self) -> Option<ModelRef> {
            Some(self.0.clone())
        }
    }

    #[test]
    fn vectors_from_another_encoder_are_refused() {
        let mut doc = Document::new("Meeting notes.");
        annotate_sentences_and_tokens(&mut doc);
        let vecs = Vectors::<Sentences>::new(
            2,
            vec![1.0f32, 1.0],
            Vec::new(),
            ModelRef { id: "encoder-B".into(), revision: "0".into(), sha256: "b".into() },
        );
        doc.add(&Key::DEFAULT, vecs, Provenance::by("test")).unwrap();
        let scorer =
            Arc::new(NamedScorer(ModelRef { id: "encoder-A".into(), revision: "0".into(), sha256: "a".into() }));
        let pipe = Analyzer::builder()
            .given(Key::<Sentences>::DEFAULT)
            .given(Key::<Tokens>::DEFAULT)
            .given(Key::<Vectors<Sentences>>::DEFAULT)
            .add(PragmaticsAnnotator::new().with_vector_scorer(scorer))
            .build()
            .unwrap();
        let mut docs = [doc];
        let err = pipe.run(&mut docs).remove(0).unwrap_err();
        assert!(err.to_string().contains("encoder-A"), "{err}");
    }

    #[test]
    fn pragmatics_benchmark_f1_and_evidence_invariants() {
        // Project-authored evaluation set (Apache-2.0, this repository) covering:
        // 1. Riloff positive-sentiment vs negative-situation incongruity
        // 2. Text vs emoji polarity mismatch
        // 3. Embedding-head-only implicit irony (no explicit contrast tokens)
        // 4. Negative controls (literal positive, literal negative, neutral, and
        //    adversative 'but'/'however' mixed-sentiment sentences).
        #[derive(Copy, Clone)]
        enum ExpectedKind {
            ExplicitContrast,
            HeadOnly,
            NotSarcastic,
        }

        let cases: &[(&str, [f32; 2], ExpectedKind)] = &[
            // Riloff incongruity positives (10)
            ("I just love waiting in line at the DMV for three hours.", [0.8, 0.5], ExpectedKind::ExplicitContrast),
            ("Oh great, the server crashed right before our demo.", [0.9, 0.6], ExpectedKind::ExplicitContrast),
            ("So thrilled to be stuck in traffic on a Friday afternoon.", [0.8, 0.5], ExpectedKind::ExplicitContrast),
            ("Nothing like a flat tire in the pouring rain.", [0.7, 0.5], ExpectedKind::ExplicitContrast),
            (
                "My flight was cancelled and they lost my luggage, wonderful.",
                [0.8, 0.4],
                ExpectedKind::ExplicitContrast,
            ),
            ("Just what I needed today, a root canal and a migraine.", [0.9, 0.5], ExpectedKind::ExplicitContrast),
            ("I love working all weekend on unpaid overtime.", [0.8, 0.6], ExpectedKind::ExplicitContrast),
            ("Oh fantastic, the wifi is down and the build failed.", [0.8, 0.5], ExpectedKind::ExplicitContrast),
            (
                "Love being ignored by customer support after waiting on hold.",
                [0.8, 0.5],
                ExpectedKind::ExplicitContrast,
            ),
            (
                "Yeah right, because a merge conflict at midnight is so much fun.",
                [0.9, 0.6],
                ExpectedKind::ExplicitContrast,
            ),
            // Text-against-emoji mismatch positives (4)
            ("This new user interface is truly brilliant 🙄.", [0.6, 0.4], ExpectedKind::ExplicitContrast),
            ("Another wonderful Monday morning meeting 😭.", [0.6, 0.4], ExpectedKind::ExplicitContrast),
            (
                "The production database was deleted and the server crashed 😂.",
                [0.7, 0.5],
                ExpectedKind::ExplicitContrast,
            ),
            ("I got food poisoning and missed my flight 🎉.", [0.7, 0.5], ExpectedKind::ExplicitContrast),
            // Embedding-head-only implicit irony positives (4)
            ("Well, that meeting certainly could not have been an email.", [2.2, 1.5], ExpectedKind::HeadOnly),
            ("How refreshingly original to blame the network again.", [2.0, 1.4], ExpectedKind::HeadOnly),
            ("Clearly the documentation wrote itself.", [2.1, 1.3], ExpectedKind::HeadOnly),
            ("Because adding another layer of indirection solves everything.", [2.3, 1.2], ExpectedKind::HeadOnly),
            // Literal positive, negative, neutral, and adversative controls (14)
            ("The new release is fast, reliable, and easy to deploy.", [-2.0, -1.5], ExpectedKind::NotSarcastic),
            ("We had a wonderful dinner and the staff was very kind.", [-2.1, -1.4], ExpectedKind::NotSarcastic),
            ("I love spending time with my family on the weekend 😊.", [-2.2, -1.6], ExpectedKind::NotSarcastic),
            ("The server crashed yesterday and caused an outage.", [-1.8, -1.2], ExpectedKind::NotSarcastic),
            ("My flight was delayed for two hours because of rain 😭.", [-1.9, -1.3], ExpectedKind::NotSarcastic),
            ("I have a terrible headache and need to rest.", [-2.0, -1.4], ExpectedKind::NotSarcastic),
            ("The battery life is great, but the screen is dim.", [-1.7, -1.2], ExpectedKind::NotSarcastic),
            ("The food was wonderful, however the service was slow.", [-1.8, -1.1], ExpectedKind::NotSarcastic),
            ("Although the hotel was nice, the street was noisy.", [-1.6, -1.3], ExpectedKind::NotSarcastic),
            ("The package arrived on Tuesday morning at nine.", [-2.2, -1.5], ExpectedKind::NotSarcastic),
            ("Please review the pull request before merging.", [-2.1, -1.4], ExpectedKind::NotSarcastic),
            ("We updated the configuration file and restarted the service.", [-2.0, -1.5], ExpectedKind::NotSarcastic),
            ("She passed the exam and celebrated with her friends 🎉.", [-2.3, -1.6], ExpectedKind::NotSarcastic),
            ("Traffic was heavy this morning on the bridge.", [-1.9, -1.4], ExpectedKind::NotSarcastic),
        ];

        let scorer = Arc::new(FixedMapVectorScorer { weights: vec![1.0, 1.0], bias: -1.2 });
        let annotator = PragmaticsAnnotator::new().with_vector_scorer(scorer);
        let threshold = annotator.positive_threshold();

        let mut tp = 0usize;
        let mut fp = 0usize;
        let mut fn_count = 0usize;
        let mut tn = 0usize;

        for &(sentence_text, vec_row, expected) in cases {
            let mut doc = Document::new(sentence_text);
            annotate_sentences_and_tokens(&mut doc);
            let vecs = Vectors::<Sentences>::new(
                2,
                vec_row.to_vec(),
                Vec::new(),
                ModelRef { id: "test".into(), revision: "0".into(), sha256: "0".into() },
            );
            doc.add(&Key::DEFAULT, vecs, Provenance::by("test")).unwrap();

            let pipe = Analyzer::builder()
                .given(Key::<Sentences>::DEFAULT)
                .given(Key::<Tokens>::DEFAULT)
                .given(Key::<Vectors<Sentences>>::DEFAULT)
                .add(EmojiAnnotate::new())
                .step(LexiconSentiment::new().bound())
                .step(Bound::new(annotator.clone()).read(4, SENTENCE_SENTIMENT.name))
                .build()
                .unwrap();

            let mut docs = [doc];
            pipe.run(&mut docs).remove(0).unwrap();
            let doc = &docs[0];
            let tokens = doc.get::<Tokens>().unwrap();
            let prag = doc.get::<Pragmatics>().unwrap();
            assert_eq!(prag.len(), 1);
            let item = prag.get(Ix::new(0));
            let predicted_positive = item.score >= threshold;
            let gold_positive = !matches!(expected, ExpectedKind::NotSarcastic);

            match (gold_positive, predicted_positive) {
                (true, true) => tp += 1,
                (false, true) => fp += 1,
                (true, false) => fn_count += 1,
                (false, false) => tn += 1,
            }

            // Definition of done invariant: each positive carries its evidence
            // spans, or says it came from the head alone (`evidence == None`).
            match expected {
                ExpectedKind::ExplicitContrast => {
                    assert!(
                        predicted_positive,
                        "expected positive on explicit contrast {sentence_text:?}, got score {}",
                        item.score
                    );
                    let (pos_sp, neg_sp) = prag
                        .evidence_spans(Ix::new(0), tokens)
                        .unwrap_or_else(|| panic!("missing evidence spans on {sentence_text:?}"));
                    assert!(!pos_sp.is_empty());
                    assert!(!neg_sp.is_empty());
                    assert_ne!(pos_sp, neg_sp);
                }
                ExpectedKind::HeadOnly => {
                    assert!(
                        predicted_positive,
                        "expected positive from head on {sentence_text:?}, got score {}",
                        item.score
                    );
                    assert!(
                        item.evidence.is_none(),
                        "head-only positive must report None evidence on {sentence_text:?}"
                    );
                }
                ExpectedKind::NotSarcastic => {
                    assert!(
                        !predicted_positive,
                        "unexpected false positive on {sentence_text:?}: score {}, evidence {:?}",
                        item.score, item.evidence
                    );
                }
            }
        }

        let precision = tp as f64 / (tp + fp) as f64;
        let recall = tp as f64 / (tp + fn_count) as f64;
        let f1 = 2.0 * precision * recall / (precision + recall);
        assert_eq!(tp, 18);
        assert_eq!(fp, 0);
        assert_eq!(fn_count, 0);
        assert_eq!(tn, 14);
        assert!((f1 - 1.0).abs() < 1e-12, "expected F1 = 1.0 on fixture, got {f1}");
    }

    #[test]
    fn config_hash_is_deterministic_and_pinned() {
        let base = PragmaticsAnnotator::new().config_hash();
        let high_thresh = PragmaticsAnnotator::new().with_positive_threshold(0.8).config_hash();
        let with_scorer = PragmaticsAnnotator::new()
            .with_vector_scorer(Arc::new(FixedMapVectorScorer { weights: vec![1.0], bias: 0.0 }))
            .config_hash();
        assert_eq!(base, 0xa06e_b5ad_6004_8da4, "got 0x{base:016x}");
        assert_ne!(base, high_thresh);
        assert_ne!(base, with_scorer);
        assert_ne!(high_thresh, with_scorer);
    }
}
