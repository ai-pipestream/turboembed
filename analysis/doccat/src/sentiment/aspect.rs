//! Aspect and opinion extraction for aspect-based sentiment analysis
//! (`turbo:aspects` and `turbo:opinions`, #322).
//!
//! Extracts explicit aspect terms (`AspectTarget::Term`), implicit sentence
//! aspects (`AspectTarget::Implicit`), and opinion expressions (`Opinions`),
//! links each aspect to its backing opinion expressions (`Ix<Opinions>`),
//! assigns an optional aspect category from a [`TagSet`], and scores three-way
//! [`PolarityScores`] (`positive`, `negative`, `neutral`) per aspect.
//!
//! This module implements the rule-based path (`RuleAspectSentiment`):
//!
//! - **Double propagation over dependency relations** following Qiu, Liu, Bu,
//!   and Chen (2011, *Opinion Word Expansion and Target Extraction through
//!   Double Propagation*, Computational Linguistics 37(1)):
//!   - **R1 (`O -> T`)**: Known opinion words extract modified aspect targets
//!     via direct syntactic relations (`amod`, `nsubj`, `obj`, `dobj`, `obl`,
//!     `nmod`, `acl`, `xcomp`, `acomp`) or shared heads (`O <- H -> T` in
//!     copular and verb-complement constructions).
//!   - **R2 (`T -> O`)**: Extracted aspect targets extract predicate and
//!     modifier opinion words attached to them.
//!   - **R3 (`T -> T'`)**: Extracted aspect targets propagate to conjoined
//!     noun targets (`conj`, `appos`) within the same polarity clause.
//!   - **R4 (`O -> O'`)**: Opinion words propagate to conjoined predicates and
//!     adjectives (`conj`), inverting polarity across adversative conjunctions.
//! - **Clause-aware noun-phrase and lexical fallback**: When `opennlp:dependencies`
//!   is absent (or alongside sparse trees), `RuleAspectSentiment` uses
//!   `opennlp:chunks` (`PhraseChunks` `NP` spans), `opennlp:pos` (`PosTags`),
//!   and compound aspect phrase tables, segmented at clause boundaries (`but`,
//!   `however`, `yet`, `although`, `though`, `whereas`, `while`, `;`, `,`) so
//!   contrastive sentences such as *"battery life is great but the screen
//!   scratches"* yield two aspects with opposite polarity.
//! - **Implicit aspects**: When a sentence contains opinion expressions that do
//!   not attach to any explicit aspect term in that sentence (for example,
//!   *"Way too expensive and completely overpriced."*), an implicit aspect
//!   (`AspectTarget::Implicit { sentence }`) is emitted, anchored to the
//!   sentence, linked to those opinions, and categorized into the aspect
//!   `TagSet`.

use std::fmt;
use std::hash::{Hash, Hasher};
use std::sync::Arc;

use turbo_analysis::{Annotator, Error, Ix, IxRange, Layer, ModelRef, Refs, Tag, TagSet, View};
use turbo_analysis_layers::{
    Aspect, AspectTarget, Aspects, Dependencies, Opinion, Opinions, PhraseChunks, Polarity, PolarityScores, PosTags,
    Sentences, Tokens,
};

use super::lexicon::VaderLexicon;

/// Default aspect category labels used when no custom `TagSet` is supplied.
pub const DEFAULT_ASPECT_CATEGORIES: &[&str] = &[
    "battery",
    "display",
    "performance",
    "build",
    "software",
    "camera",
    "audio",
    "service",
    "food",
    "price",
    "ambience",
    "general",
];

/// Optional scorer that assigns an aspect category [`Tag`] from a [`TagSet`]
/// to an aspect (explicit or implicit) given its surface text and enclosing
/// sentence text.
///
/// Implemented in `turbo-analysis-embed` by [`LabelCentroids`]-backed category
/// scorers as well as keyword/centroid hybrid scorers.
pub trait AspectCategoryScorer: Send + Sync + fmt::Debug {
    /// The aspect category [`TagSet`] produced by this scorer.
    fn categories(&self) -> &Arc<TagSet>;

    /// Scores the aspect described by `aspect_text` (empty for implicit
    /// aspects), `opinion_text`, and `sentence_text`, returning the best
    /// category [`Tag`] if confidence exceeds the scorer's threshold.
    fn score_category(&self, aspect_text: &str, opinion_text: &str, sentence_text: &str) -> Option<Tag>;

    /// Deterministic configuration digest for provenance.
    fn config_hash(&self) -> u64;
}

/// Domain opinion words and predicate verbs that often carry aspect-specific
/// sentiment in product, software, and hospitality reviews even when absent or
/// weakly weighted in general-purpose social lexicons.
const DOMAIN_OPINION_LEXICON: &[(&str, f64)] = &[
    // Negative aspect predicates & adjectives
    ("scratches", -2.2),
    ("scratched", -2.2),
    ("scratch", -2.0),
    ("cracks", -2.4),
    ("cracked", -2.4),
    ("crashes", -2.5),
    ("crashed", -2.5),
    ("freezes", -2.3),
    ("froze", -2.3),
    ("frozen", -2.0),
    ("lags", -2.1),
    ("laggy", -2.2),
    ("lagged", -2.1),
    ("stutters", -2.0),
    ("drains", -2.2),
    ("drained", -2.1),
    ("overheats", -2.3),
    ("overheated", -2.3),
    ("flickers", -2.1),
    ("flickered", -2.1),
    ("stalls", -2.1),
    ("stalled", -2.1),
    ("disconnects", -2.1),
    ("disconnected", -2.0),
    ("wobbles", -1.8),
    ("wobbly", -1.9),
    ("flimsy", -2.2),
    ("fragile", -2.0),
    ("mushy", -1.9),
    ("tinny", -2.0),
    ("muffled", -1.9),
    ("blurry", -2.1),
    ("grainy", -2.0),
    ("dim", -1.8),
    ("glare", -1.7),
    ("bloated", -2.0),
    ("buggy", -2.3),
    ("clunky", -2.1),
    ("unresponsive", -2.4),
    ("slow", -1.9),
    ("sluggish", -2.2),
    ("noisy", -1.9),
    ("loud", -1.5),
    ("overpriced", -2.4),
    ("expensive", -1.8),
    ("pricey", -1.7),
    ("steep", -1.5),
    ("ripoff", -2.7),
    ("soggy", -2.2),
    ("stale", -2.2),
    ("bland", -2.1),
    ("tasteless", -2.3),
    ("overcooked", -2.2),
    ("undercooked", -2.3),
    ("greasy", -2.0),
    ("lukewarm", -1.9),
    ("cold", -1.6),
    ("rude", -2.5),
    ("inattentive", -2.2),
    ("slowest", -2.3),
    ("mediocre", -1.9),
    ("underwhelming", -2.1),
    ("disappointing", -2.3),
    ("terrible", -2.8),
    ("awful", -2.7),
    ("horrible", -2.8),
    ("useless", -2.5),
    ("broken", -2.4),
    ("defective", -2.5),
    ("cramped", -1.9),
    ("filthy", -2.7),
    ("dirty", -2.3),
    // Positive aspect predicates & adjectives
    ("crisp", 2.3),
    ("sharp", 2.0),
    ("vibrant", 2.3),
    ("bright", 1.9),
    ("vivid", 2.2),
    ("responsive", 2.3),
    ("snappy", 2.3),
    ("zippy", 2.2),
    ("fast", 2.0),
    ("quick", 1.9),
    ("smooth", 2.1),
    ("seamless", 2.4),
    ("flawless", 2.7),
    ("solid", 2.0),
    ("sturdy", 2.2),
    ("durable", 2.2),
    ("sleek", 2.1),
    ("lightweight", 2.0),
    ("compact", 1.8),
    ("quiet", 1.9),
    ("silent", 2.0),
    ("intuitive", 2.3),
    ("reliable", 2.3),
    ("stable", 2.0),
    ("lasting", 1.9),
    ("lasts", 1.9),
    ("delicious", 2.7),
    ("tasty", 2.3),
    ("flavorful", 2.5),
    ("fresh", 2.1),
    ("tender", 2.1),
    ("juicy", 2.1),
    ("authentic", 2.0),
    ("generous", 2.1),
    ("attentive", 2.4),
    ("courteous", 2.3),
    ("polite", 2.1),
    ("welcoming", 2.3),
    ("friendly", 2.3),
    ("helpful", 2.2),
    ("prompt", 2.1),
    ("spotless", 2.5),
    ("clean", 2.0),
    ("cozy", 2.2),
    ("spacious", 2.1),
    ("affordable", 2.2),
    ("reasonable", 1.9),
    ("bargain", 2.2),
    ("inexpensive", 2.0),
];

/// Known multi-word aspect terms (lowercased token sequences) with their
/// canonical category name in [`DEFAULT_ASPECT_CATEGORIES`].
const MULTIWORD_ASPECT_TERMS: &[(&[&str], &str)] = &[
    (&["battery", "life"], "battery"),
    (&["battery", "longevity"], "battery"),
    (&["battery", "capacity"], "battery"),
    (&["charging", "speed"], "battery"),
    (&["charging", "port"], "battery"),
    (&["power", "adapter"], "battery"),
    (&["power", "cord"], "battery"),
    (&["power", "brick"], "battery"),
    (&["screen", "resolution"], "display"),
    (&["screen", "brightness"], "display"),
    (&["display", "quality"], "display"),
    (&["viewing", "angles"], "display"),
    (&["viewing", "angle"], "display"),
    (&["refresh", "rate"], "display"),
    (&["color", "accuracy"], "display"),
    (&["touch", "screen"], "display"),
    (&["boot", "time"], "performance"),
    (&["load", "times"], "performance"),
    (&["load", "time"], "performance"),
    (&["processing", "speed"], "performance"),
    (&["frame", "rate"], "performance"),
    (&["fan", "noise"], "performance"),
    (&["build", "quality"], "build"),
    (&["track", "pad"], "build"),
    (&["touch", "pad"], "build"),
    (&["key", "travel"], "build"),
    (&["keyboard", "backlight"], "build"),
    (&["fingerprint", "reader"], "build"),
    (&["hinge", "design"], "build"),
    (&["user", "interface"], "software"),
    (&["operating", "system"], "software"),
    (&["camera", "quality"], "camera"),
    (&["camera", "lens"], "camera"),
    (&["night", "mode"], "camera"),
    (&["photo", "quality"], "camera"),
    (&["video", "quality"], "camera"),
    (&["sound", "quality"], "audio"),
    (&["speaker", "volume"], "audio"),
    (&["audio", "quality"], "audio"),
    (&["noise", "cancellation"], "audio"),
    (&["customer", "service"], "service"),
    (&["customer", "support"], "service"),
    (&["wait", "staff"], "service"),
    (&["waiting", "time"], "service"),
    (&["wait", "time"], "service"),
    (&["delivery", "time"], "service"),
    (&["tech", "support"], "service"),
    (&["portion", "size"], "food"),
    (&["portion", "sizes"], "food"),
    (&["wine", "list"], "food"),
    (&["menu", "selection"], "food"),
    (&["breakfast", "buffet"], "food"),
    (&["price", "point"], "price"),
    (&["value", "for", "money"], "price"),
    (&["dining", "room"], "ambience"),
    (&["seating", "area"], "ambience"),
    (&["background", "music"], "ambience"),
];

/// Single-word aspect head lexicon mapped to canonical category names in
/// [`DEFAULT_ASPECT_CATEGORIES`].
const SINGLE_WORD_ASPECT_TERMS: &[(&str, &str)] = &[
    // Battery
    ("battery", "battery"),
    ("charger", "battery"),
    ("charging", "battery"),
    // Display
    ("screen", "display"),
    ("display", "display"),
    ("monitor", "display"),
    ("panel", "display"),
    ("brightness", "display"),
    ("resolution", "display"),
    ("pixels", "display"),
    ("bezel", "display"),
    // Performance
    ("performance", "performance"),
    ("speed", "performance"),
    ("processor", "performance"),
    ("cpu", "performance"),
    ("gpu", "performance"),
    ("memory", "performance"),
    ("ram", "performance"),
    ("ssd", "performance"),
    ("storage", "performance"),
    ("fan", "performance"),
    ("thermals", "performance"),
    ("cooling", "performance"),
    ("latency", "performance"),
    // Build & hardware
    ("keyboard", "build"),
    ("trackpad", "build"),
    ("touchpad", "build"),
    ("chassis", "build"),
    ("hinge", "build"),
    ("case", "build"),
    ("body", "build"),
    ("design", "build"),
    ("weight", "build"),
    ("ports", "build"),
    ("keys", "build"),
    ("mouse", "build"),
    ("build", "build"),
    // Software
    ("software", "software"),
    ("interface", "software"),
    ("ui", "software"),
    ("app", "software"),
    ("firmware", "software"),
    ("drivers", "software"),
    ("driver", "software"),
    ("os", "software"),
    ("updates", "software"),
    ("bloatware", "software"),
    // Camera
    ("camera", "camera"),
    ("lens", "camera"),
    ("photos", "camera"),
    ("pictures", "camera"),
    ("webcam", "camera"),
    ("autofocus", "camera"),
    ("sensor", "camera"),
    // Audio
    ("speakers", "audio"),
    ("speaker", "audio"),
    ("audio", "audio"),
    ("sound", "audio"),
    ("microphone", "audio"),
    ("mic", "audio"),
    ("bass", "audio"),
    ("headphones", "audio"),
    // Service
    ("service", "service"),
    ("staff", "service"),
    ("waiter", "service"),
    ("waitress", "service"),
    ("server", "service"),
    ("bartender", "service"),
    ("hostess", "service"),
    ("host", "service"),
    ("management", "service"),
    ("manager", "service"),
    ("reception", "service"),
    ("delivery", "service"),
    ("support", "service"),
    // Food
    ("food", "food"),
    ("meal", "food"),
    ("meals", "food"),
    ("dish", "food"),
    ("dishes", "food"),
    ("pizza", "food"),
    ("pasta", "food"),
    ("sushi", "food"),
    ("steak", "food"),
    ("burger", "food"),
    ("salad", "food"),
    ("soup", "food"),
    ("dessert", "food"),
    ("coffee", "food"),
    ("espresso", "food"),
    ("cocktails", "food"),
    ("drinks", "food"),
    ("wine", "food"),
    ("beer", "food"),
    ("sauce", "food"),
    ("crust", "food"),
    ("appetizer", "food"),
    ("appetizers", "food"),
    ("entree", "food"),
    ("menu", "food"),
    ("portions", "food"),
    ("portion", "food"),
    ("ingredients", "food"),
    ("breakfast", "food"),
    ("dinner", "food"),
    ("lunch", "food"),
    // Price
    ("price", "price"),
    ("prices", "price"),
    ("pricing", "price"),
    ("cost", "price"),
    ("value", "price"),
    ("bill", "price"),
    ("check", "price"),
    ("fee", "price"),
    ("fees", "price"),
    // Ambience
    ("ambience", "ambience"),
    ("ambiance", "ambience"),
    ("atmosphere", "ambience"),
    ("decor", "ambience"),
    ("lighting", "ambience"),
    ("seating", "ambience"),
    ("tables", "ambience"),
    ("table", "ambience"),
    ("room", "ambience"),
    ("rooms", "ambience"),
    ("patio", "ambience"),
    ("view", "ambience"),
    ("interior", "ambience"),
    ("music", "ambience"),
];

/// Implicit aspect category cues for opinion words when no explicit aspect term
/// is present in the sentence.
const IMPLICIT_OPINION_CATEGORY_CUES: &[(&str, &str)] = &[
    ("expensive", "price"),
    ("overpriced", "price"),
    ("pricey", "price"),
    ("affordable", "price"),
    ("inexpensive", "price"),
    ("cheap", "price"),
    ("bargain", "price"),
    ("ripoff", "price"),
    ("delicious", "food"),
    ("tasty", "food"),
    ("flavorful", "food"),
    ("yummy", "food"),
    ("soggy", "food"),
    ("bland", "food"),
    ("stale", "food"),
    ("tasteless", "food"),
    ("overcooked", "food"),
    ("undercooked", "food"),
    ("greasy", "food"),
    ("rude", "service"),
    ("attentive", "service"),
    ("courteous", "service"),
    ("inattentive", "service"),
    ("polite", "service"),
    ("welcoming", "service"),
    ("cozy", "ambience"),
    ("cramped", "ambience"),
    ("noisy", "ambience"),
    ("spacious", "ambience"),
    ("spotless", "ambience"),
    ("filthy", "ambience"),
    ("snappy", "performance"),
    ("sluggish", "performance"),
    ("laggy", "performance"),
    ("zippy", "performance"),
    ("buggy", "software"),
    ("intuitive", "software"),
    ("flimsy", "build"),
    ("sturdy", "build"),
    ("sleek", "build"),
    ("lightweight", "build"),
    ("blurry", "camera"),
    ("tinny", "audio"),
];

/// Words that are syntactic nouns or pronouns or discourse markers but never
/// valid aspect targets on their own.
const ASPECT_STOPWORDS: &[&str] = &[
    "i", "me", "my", "mine", "myself", "we", "us", "our", "ours", "you", "your", "yours", "he", "him", "his", "she",
    "her", "hers", "it", "its", "they", "them", "their", "theirs", "this", "that", "these", "those", "who", "whom",
    "which", "what", "whatever", "everything", "nothing", "something", "anything", "everyone", "someone", "anyone",
    "nobody", "way", "ways", "thing", "things", "lot", "lots", "bit", "bits", "kind", "sort", "type", "part", "parts",
    "time", "times", "day", "days", "week", "weeks", "month", "months", "year", "years", "today", "yesterday",
    "tomorrow", "fact", "point", "reason", "place", "person", "people", "one", "ones", "all", "both", "each", "other",
    "others", "none", "much", "many", "more", "most", "some", "any", "few", "little", "less", "least",
];

/// Negation cues used to detect negated opinions.
const NEGATION_CUES: &[&str] = &[
    "not", "no", "never", "neither", "nor", "none", "nobody", "nothing", "nowhere", "hardly", "barely", "scarcely",
    "seldom", "rarely", "without", "lack", "lacks", "lacked", "lacking", "cannot", "cant", "won't", "wont", "don't",
    "dont", "doesn't", "doesnt", "didn't", "didnt", "isn't", "isnt", "aren't", "arent", "wasn't", "wasnt", "weren't",
    "werent", "hasn't", "hasnt", "haven't", "havent", "hadn't", "hadnt", "wouldn't", "wouldnt", "shouldn't",
    "shouldnt", "couldn't", "couldnt",
];

/// Degree booster and dampener modifiers with their intensity multiplier delta.
const DEGREE_MODIFIERS: &[(&str, f64)] = &[
    ("very", 0.30),
    ("extremely", 0.45),
    ("incredibly", 0.45),
    ("exceptionally", 0.45),
    ("remarkably", 0.40),
    ("totally", 0.30),
    ("completely", 0.35),
    ("absolutely", 0.40),
    ("really", 0.25),
    ("so", 0.25),
    ("too", 0.25),
    ("super", 0.35),
    ("highly", 0.30),
    ("quite", 0.15),
    ("pretty", 0.12),
    ("fairly", 0.10),
    ("rather", 0.12),
    ("slightly", -0.35),
    ("somewhat", -0.25),
    ("marginally", -0.40),
    ("a_bit", -0.30),
];

/// Adversative and clause-splitting conjunctions.
const ADVERSATIVE_CONJUNCTIONS: &[&str] = &[
    "but", "however", "yet", "although", "though", "even", "whereas", "while", "except", "despite", "nevertheless",
    "nonetheless", "unfortunately", "sadly",
];

/// Determiners and leading articles stripped from the front of noun-phrase
/// aspect candidates.
const LEADING_DETERMINERS: &[&str] = &[
    "the", "a", "an", "this", "that", "these", "those", "my", "our", "your", "his", "her", "its", "their", "some",
    "any", "each", "every", "another", "both", "all",
];

/// Rule-based aspect and opinion extractor (`turbo-doccat:aspect-sentiment-rule`,
/// `#322`).
///
/// Runs Qiu et al. (2011) double propagation over `opennlp:dependencies` when
/// present, combined with clause-aware noun-phrase (`opennlp:chunks`), POS
/// (`opennlp:pos`), and compound aspect lexicon extraction, and writes
/// `(Opinions, Aspects)`.
#[derive(Clone)]
pub struct RuleAspectSentiment {
    lexicon: Arc<VaderLexicon>,
    categories: Arc<TagSet>,
    category_scorer: Option<Arc<dyn AspectCategoryScorer>>,
    emit_implicit_aspects: bool,
}

impl fmt::Debug for RuleAspectSentiment {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("RuleAspectSentiment")
            .field("categories", &self.categories.name)
            .field("has_category_scorer", &self.category_scorer.is_some())
            .field("emit_implicit_aspects", &self.emit_implicit_aspects)
            .finish()
    }
}

impl Default for RuleAspectSentiment {
    fn default() -> Self {
        Self::new()
    }
}

impl RuleAspectSentiment {
    /// Creates a new `RuleAspectSentiment` with the bundled VADER lexicon and
    /// [`DEFAULT_ASPECT_CATEGORIES`].
    pub fn new() -> Self {
        Self {
            lexicon: VaderLexicon::bundled(),
            categories: TagSet::new("aspect-categories", DEFAULT_ASPECT_CATEGORIES),
            category_scorer: None,
            emit_implicit_aspects: true,
        }
    }

    /// Uses a custom aspect category [`TagSet`].
    pub fn with_categories(mut self, categories: Arc<TagSet>) -> Self {
        self.categories = categories;
        self
    }

    /// Attaches a custom [`AspectCategoryScorer`] (such as a centroid-based
    /// scorer from `turbo-analysis-embed`).
    pub fn with_category_scorer(mut self, scorer: Arc<dyn AspectCategoryScorer>) -> Self {
        self.categories = Arc::clone(scorer.categories());
        self.category_scorer = Some(scorer);
        self
    }

    /// Configures whether sentences with opinion expressions but no explicit
    /// aspect term emit an [`AspectTarget::Implicit`] entry.
    pub fn with_implicit_aspects(mut self, enabled: bool) -> Self {
        self.emit_implicit_aspects = enabled;
        self
    }

    /// Returns the aspect category [`TagSet`].
    pub fn categories(&self) -> &Arc<TagSet> {
        &self.categories
    }

    /// Looks up the prior sentiment valence of a lowercased token in the
    /// domain opinion lexicon first, then the VADER lexicon.
    pub fn token_opinion_valence(&self, lower: &str) -> Option<f64> {
        // Do not treat aspect head nouns with neutral/mild VADER scores (like "value" or "support")
        // as opinion expressions when they appear as aspect nouns, unless they are strong opinions.
        for &(w, score) in DOMAIN_OPINION_LEXICON {
            if w == lower {
                return Some(score);
            }
        }
        if let Some(v) = self.lexicon.get(lower)
            && v.abs() >= 0.8
            && !is_pure_aspect_noun(lower)
        {
            return Some(v);
        }
        None
    }
}

fn is_pure_aspect_noun(lower: &str) -> bool {
    matches!(
        lower,
        "support"
            | "value"
            | "service"
            | "party"
            | "security"
            | "care"
            | "comfort"
            | "credit"
            | "interest"
            | "share"
            | "shares"
            | "gift"
            | "energy"
            | "power"
            | "resolution"
            | "vision"
            | "clarity"
            | "intelligence"
            | "grace"
            | "spirit"
            | "harmony"
            | "treat"
            | "feeling"
            | "novel"
            | "play"
    )
}

#[derive(Clone, Debug)]
struct ExtractedOpinion {
    tokens: IxRange<Tokens>,
    head_token: usize,
    polarity: f64,
    intensity: f64,
    negated: bool,
    clause_id: usize,
}

impl ExtractedOpinion {
    fn effective_polarity(&self) -> f64 {
        let sign = if self.negated { -1.0 } else { 1.0 };
        self.polarity * self.intensity * sign
    }
}

#[derive(Clone, Debug)]
struct ExtractedAspect {
    tokens: IxRange<Tokens>,
    head_token: usize,
    clause_id: usize,
    category_hint: Option<&'static str>,
}

impl Annotator for RuleAspectSentiment {
    const ID: &'static str = "turbo-doccat:aspect-sentiment-rule";
    type Reads = (View, Sentences, Tokens, Option<Dependencies>, Option<PhraseChunks>, Option<PosTags>);
    type Writes = (Opinions, Aspects);

    fn annotate(&self, (view, sentences, tokens, deps, phrase_chunks, pos_tags): Refs<'_, Self>) -> Result<Self::Writes, Error> {
        let fail = |detail: String| Error::Annotator { annotator: Self::ID, detail };
        let text = view.text();
        let n_tokens = tokens.len();
        let (tok_starts, tok_ends) = tokens.spans.columns();
        let (sent_starts, sent_ends) = sentences.spans.columns();

        let mut token_lower: Vec<String> = Vec::with_capacity(n_tokens);
        for t in 0..n_tokens {
            let s = tok_starts[t] as usize;
            let e = tok_ends[t] as usize;
            let slice = text.get(s..e).unwrap_or("");
            token_lower.push(clean_word_lower(slice));
        }

        struct PendingAspect {
            target: AspectTarget,
            category: Option<Tag>,
            polarity: PolarityScores,
            opinions: Vec<Ix<Opinions>>,
        }

        let mut all_opinions: Vec<Opinion> = Vec::new();
        let mut pending_aspects: Vec<PendingAspect> = Vec::new();

        let mut tok_cursor = 0usize;
        for s_idx in 0..sentences.len() {
            let s_start_byte = sent_starts[s_idx];
            let s_end_byte = sent_ends[s_idx];

            while tok_cursor < n_tokens && tok_ends[tok_cursor] <= s_start_byte {
                tok_cursor += 1;
            }
            let sent_tok_start = tok_cursor;
            while tok_cursor < n_tokens && tok_starts[tok_cursor] < s_end_byte {
                tok_cursor += 1;
            }
            let sent_tok_end = tok_cursor;
            if sent_tok_start >= sent_tok_end {
                continue;
            }

            let sent_ix = Ix::<Sentences>::new(s_idx as u32);
            let clause_ids = assign_clauses(&token_lower, text, tok_starts, tok_ends, sent_tok_start, sent_tok_end);

            // Step 1: Seed opinions and explicit aspect candidates in the sentence.
            let (mut sent_opinions, mut sent_aspects) = self.extract_sentence_candidates(
                text,
                tok_starts,
                tok_ends,
                &token_lower,
                sent_tok_start,
                sent_tok_end,
                &clause_ids,
                deps,
                phrase_chunks,
                pos_tags,
            );

            // Step 2: If Dependencies are present, run Double Propagation (Qiu et al. 2011)
            // to expand targets from opinions (R1), opinions from targets (R2),
            // conjoined targets (R3), and conjoined opinions (R4).
            if let Some(d) = deps {
                self.run_double_propagation(
                    d,
                    pos_tags,
                    &token_lower,
                    sent_tok_start,
                    sent_tok_end,
                    &clause_ids,
                    &mut sent_opinions,
                    &mut sent_aspects,
                );
            }

            // Sort and deduplicate opinions by token range in ascending order.
            sent_opinions.sort_by_key(|o| (o.tokens.start.get(), o.tokens.end.get()));
            dedup_opinions(&mut sent_opinions);

            // Sort and deduplicate explicit aspects by token range in ascending order,
            // removing any aspect span that overlaps an opinion span.
            sent_aspects.retain(|a| !sent_opinions.iter().any(|o| ranges_overlap(a.tokens, o.tokens)));
            sent_aspects.sort_by_key(|a| (a.tokens.start.get(), a.tokens.end.get()));
            dedup_aspects(&mut sent_aspects);

            // Record the sentence's opinions in `all_opinions` and remember their global `Ix<Opinions>`.
            let base_op_idx = all_opinions.len();
            for op in &sent_opinions {
                all_opinions.push(Opinion {
                    tokens: op.tokens,
                    polarity: op.polarity,
                    intensity: op.intensity,
                    negated: op.negated,
                });
            }

            let sent_text = text.get(s_start_byte as usize..s_end_byte as usize).unwrap_or("");

            if !sent_aspects.is_empty() {
                // Step 3: Link each explicit aspect to its governing/modifying opinions
                // (via dependency paths first when `deps` is present, then same-clause
                // proximity, then sentence fallback).
                for (a_pos, asp) in sent_aspects.iter().enumerate() {
                    let linked_local = link_opinions_to_aspect(
                        asp,
                        a_pos,
                        &sent_aspects,
                        &sent_opinions,
                        deps,
                        sent_tok_start,
                        sent_tok_end,
                    );
                    let linked_ixs: Vec<Ix<Opinions>> =
                        linked_local.iter().map(|&k| Ix::new((base_op_idx + k) as u32)).collect();
                    let polarity = compute_aspect_polarity(&linked_local, &sent_opinions);
                    let asp_text = span_text(text, tok_starts, tok_ends, asp.tokens);
                    let op_text = linked_local
                        .first()
                        .map(|&k| span_text(text, tok_starts, tok_ends, sent_opinions[k].tokens))
                        .unwrap_or("");
                    let category = self.resolve_category(
                        asp.category_hint,
                        asp_text,
                        op_text,
                        sent_text,
                        &token_lower[asp.tokens.start.get()..asp.tokens.end.get()],
                    );
                    pending_aspects.push(PendingAspect {
                        target: AspectTarget::Term { tokens: asp.tokens, sentence: Some(sent_ix) },
                        category,
                        polarity,
                        opinions: linked_ixs,
                    });
                }
            } else if self.emit_implicit_aspects && !sent_opinions.is_empty() {
                // Step 4: Implicit aspect anchored to the sentence when opinion expressions
                // exist without any explicit aspect term in the sentence.
                let all_local: Vec<usize> = (0..sent_opinions.len()).collect();
                let linked_ixs: Vec<Ix<Opinions>> =
                    all_local.iter().map(|&k| Ix::new((base_op_idx + k) as u32)).collect();
                let polarity = compute_aspect_polarity(&all_local, &sent_opinions);
                let op_text = span_text(text, tok_starts, tok_ends, sent_opinions[0].tokens);
                let category = self.resolve_implicit_category(&sent_opinions, &token_lower, op_text, sent_text);
                pending_aspects.push(PendingAspect {
                    target: AspectTarget::Implicit { sentence: sent_ix },
                    category,
                    polarity,
                    opinions: linked_ixs,
                });
            }
        }

        let opinions_layer = Opinions::from_items(tokens, all_opinions).map_err(fail)?;
        let mut aspects_layer = Aspects::new(Arc::clone(&self.categories));
        for p in &pending_aspects {
            aspects_layer
                .try_push(
                    tokens,
                    Some(sentences),
                    Some(&opinions_layer),
                    Aspect { target: p.target, category: p.category, polarity: p.polarity, opinions: &p.opinions },
                )
                .map_err(fail)?;
        }

        Ok((opinions_layer, aspects_layer))
    }

    fn model(&self) -> Option<ModelRef> {
        Some(ModelRef {
            id: "double-propagation:vader".to_string(),
            revision: "1.0.0".to_string(),
            sha256: format!("{:016x}", self.config_hash()),
        })
    }

    fn config_hash(&self) -> u64 {
        let mut h = std::collections::hash_map::DefaultHasher::new();
        Self::ID.hash(&mut h);
        self.categories.name.hash(&mut h);
        for i in 0..self.categories.len() {
            self.categories.label(Tag(i as u16)).hash(&mut h);
        }
        self.emit_implicit_aspects.hash(&mut h);
        if let Some(scorer) = &self.category_scorer {
            scorer.config_hash().hash(&mut h);
        }
        h.finish()
    }
}

impl RuleAspectSentiment {
    #[allow(clippy::too_many_arguments)]
    fn extract_sentence_candidates(
        &self,
        text: &str,
        tok_starts: &[u32],
        tok_ends: &[u32],
        token_lower: &[String],
        sent_start: usize,
        sent_end: usize,
        clause_ids: &[usize],
        deps: Option<&Dependencies>,
        phrase_chunks: Option<&PhraseChunks>,
        pos_tags: Option<&PosTags>,
    ) -> (Vec<ExtractedOpinion>, Vec<ExtractedAspect>) {
        let mut opinions = Vec::new();
        let mut aspects = Vec::new();
        let mut covered_aspect_tokens = vec![false; sent_end - sent_start];

        // 1a. Match multi-word aspect terms first (longest match wins).
        let mut t = sent_start;
        while t < sent_end {
            let rem = sent_end - t;
            let mut matched_mw: Option<(usize, &'static str)> = None;
            for &(pat, cat) in MULTIWORD_ASPECT_TERMS {
                if pat.len() <= rem
                    && pat.iter().enumerate().all(|(k, &w)| token_lower[t + k] == w)
                    && (matched_mw.is_none() || pat.len() > matched_mw.unwrap().0)
                {
                    matched_mw = Some((pat.len(), cat));
                }
            }
            if let Some((len, cat)) = matched_mw {
                let head = t + len - 1;
                for flag in &mut covered_aspect_tokens[(t - sent_start)..(t + len - sent_start)] {
                    *flag = true;
                }
                aspects.push(ExtractedAspect {
                    tokens: IxRange::new(t as u32, (t + len) as u32),
                    head_token: head,
                    clause_id: clause_ids[head - sent_start],
                    category_hint: Some(cat),
                });
                t += len;
            } else {
                t += 1;
            }
        }

        // 1b. Match single-word aspect lexicon terms not already inside a multi-word aspect.
        for tok in sent_start..sent_end {
            if covered_aspect_tokens[tok - sent_start] {
                continue;
            }
            let w = token_lower[tok].as_str();
            if let Some(&(_, cat)) = SINGLE_WORD_ASPECT_TERMS.iter().find(|&&(term, _)| term == w) {
                // Expand left/right if attached to adjacent noun compound in POS or Dependencies.
                let (span_start, span_end) = expand_noun_compound(
                    tok,
                    sent_start,
                    sent_end,
                    token_lower,
                    &covered_aspect_tokens,
                    deps,
                    pos_tags,
                    self,
                );
                for flag in &mut covered_aspect_tokens[(span_start - sent_start)..(span_end - sent_start)] {
                    *flag = true;
                }
                aspects.push(ExtractedAspect {
                    tokens: IxRange::new(span_start as u32, span_end as u32),
                    head_token: tok,
                    clause_id: clause_ids[tok - sent_start],
                    category_hint: Some(cat),
                });
            }
        }

        // 1c. Extract NP candidates from `PhraseChunks` (`opennlp:chunks`) when present.
        if let Some(chunks) = phrase_chunks {
            for (c_idx, range) in chunks.tokens.iter().enumerate() {
                let c_start = range.start.get();
                let c_end = range.end.get();
                if c_start < sent_start || c_end > sent_end || c_start >= c_end {
                    continue;
                }
                let kind_label = chunks.kinds.label(chunks.kind[c_idx]);
                if !kind_label.eq_ignore_ascii_case("NP") {
                    continue;
                }
                // Trim leading determiners and leading opinion adjectives from the NP.
                let mut a_start = c_start;
                while a_start < c_end {
                    let w = token_lower[a_start].as_str();
                    if LEADING_DETERMINERS.contains(&w) || self.token_opinion_valence(w).is_some() || w.is_empty() {
                        a_start += 1;
                    } else {
                        break;
                    }
                }
                if a_start >= c_end {
                    continue;
                }
                let head = c_end - 1;
                let head_w = token_lower[head].as_str();
                if head_w.is_empty() || ASPECT_STOPWORDS.contains(&head_w) || self.token_opinion_valence(head_w).is_some()
                {
                    continue;
                }
                if (a_start..c_end).any(|i| covered_aspect_tokens[i - sent_start]) {
                    continue;
                }
                for flag in &mut covered_aspect_tokens[(a_start - sent_start)..(c_end - sent_start)] {
                    *flag = true;
                }
                let cat_hint = lookup_category_hint(&token_lower[a_start..c_end]);
                aspects.push(ExtractedAspect {
                    tokens: IxRange::new(a_start as u32, c_end as u32),
                    head_token: head,
                    clause_id: clause_ids[head - sent_start],
                    category_hint: cat_hint,
                });
            }
        }

        // 1d. Extract POS-tagged nouns (`NN`, `NNS`, `NNP`, `NNPS`, `NOUN`, `PROPN`) when `PosTags` is present.
        if let Some(pos) = pos_tags {
            let mut tok = sent_start;
            while tok < sent_end {
                if covered_aspect_tokens[tok - sent_start] {
                    tok += 1;
                    continue;
                }
                let tag_str = pos.tagset.label(pos.tags[tok]);
                let w = token_lower[tok].as_str();
                if is_noun_pos(tag_str)
                    && !w.is_empty()
                    && !ASPECT_STOPWORDS.contains(&w)
                    && self.token_opinion_valence(w).is_none()
                {
                    let mut end = tok + 1;
                    while end < sent_end && !covered_aspect_tokens[end - sent_start] {
                        let next_tag = pos.tagset.label(pos.tags[end]);
                        let next_w = token_lower[end].as_str();
                        if is_noun_pos(next_tag)
                            && !next_w.is_empty()
                            && !ASPECT_STOPWORDS.contains(&next_w)
                            && self.token_opinion_valence(next_w).is_none()
                        {
                            end += 1;
                        } else {
                            break;
                        }
                    }
                    let head = end - 1;
                    for flag in &mut covered_aspect_tokens[(tok - sent_start)..(end - sent_start)] {
                        *flag = true;
                    }
                    let cat_hint = lookup_category_hint(&token_lower[tok..end]);
                    aspects.push(ExtractedAspect {
                        tokens: IxRange::new(tok as u32, end as u32),
                        head_token: head,
                        clause_id: clause_ids[head - sent_start],
                        category_hint: cat_hint,
                    });
                    tok = end;
                } else {
                    tok += 1;
                }
            }
        }

        // 2. Seed opinion expressions from tokens not covered by explicit aspect terms.
        for tok in sent_start..sent_end {
            if covered_aspect_tokens[tok - sent_start] {
                continue;
            }
            let w = token_lower[tok].as_str();
            if w.is_empty() {
                continue;
            }
            if let Some(prior) = self.token_opinion_valence(w) {
                let negated = is_token_negated(
                    tok,
                    sent_start,
                    sent_end,
                    token_lower,
                    clause_ids,
                    deps,
                );
                let intensity = compute_token_intensity(tok, sent_start, token_lower, text, tok_starts, tok_ends);
                opinions.push(ExtractedOpinion {
                    tokens: IxRange::new(tok as u32, (tok + 1) as u32),
                    head_token: tok,
                    polarity: prior,
                    intensity,
                    negated,
                    clause_id: clause_ids[tok - sent_start],
                });
            }
        }

        (opinions, aspects)
    }

    /// Qiu, Liu, Bu, and Chen (2011) iterative Double Propagation over
    /// dependency arcs (`R1`, `R2`, `R3`, `R4`).
    #[allow(clippy::too_many_arguments)]
    fn run_double_propagation(
        &self,
        deps: &Dependencies,
        pos_tags: Option<&PosTags>,
        token_lower: &[String],
        sent_start: usize,
        sent_end: usize,
        clause_ids: &[usize],
        opinions: &mut Vec<ExtractedOpinion>,
        aspects: &mut Vec<ExtractedAspect>,
    ) {
        if deps.len() < sent_end {
            return;
        }
        // Run up to 3 propagation passes until fixed point.
        for _pass in 0..3 {
            let mut added_any = false;

            // R1 (O -> T): Known opinion words extract modified or governing noun targets.
            let current_ops = opinions.clone();
            for op in &current_ops {
                let o_tok = op.head_token;
                let o_head = deps.head[o_tok].map(|ix| ix.get());
                let o_rel = deps.relations.label(deps.relation[o_tok]);

                // R11a: Opinion `o_tok` is a dependent of target `t_tok` (e.g., `amod(screen, blurry)` or `acl(pizza, cooked)`),
                // or `o_tok` is a predicate head whose subject/object is `t_tok` (`nsubj(great, battery)` or `nsubj(scratches, screen)`).
                if let Some(h) = o_head
                    && h >= sent_start
                    && h < sent_end
                    && is_mr_opinion_to_head_rel(o_rel)
                    && self.can_be_aspect_target(h, pos_tags, token_lower, opinions, aspects)
                {
                    let (s, e) = expand_dep_noun_span(h, sent_start, sent_end, deps, token_lower, opinions);
                    aspects.push(ExtractedAspect {
                        tokens: IxRange::new(s as u32, e as u32),
                        head_token: h,
                        clause_id: clause_ids[h - sent_start],
                        category_hint: lookup_category_hint(&token_lower[s..e]),
                    });
                    added_any = true;
                }

                // R11b: Target `t_tok` is a child of opinion predicate `o_tok` (`nsubj`, `obj`, `dobj`, `obl`).
                for cand in sent_start..sent_end {
                    if deps.head[cand].map(|ix| ix.get()) == Some(o_tok) {
                        let rel = deps.relations.label(deps.relation[cand]);
                        if is_mr_target_child_rel(rel)
                            && self.can_be_aspect_target(cand, pos_tags, token_lower, opinions, aspects)
                        {
                            let (s, e) = expand_dep_noun_span(cand, sent_start, sent_end, deps, token_lower, opinions);
                            aspects.push(ExtractedAspect {
                                tokens: IxRange::new(s as u32, e as u32),
                                head_token: cand,
                                clause_id: clause_ids[cand - sent_start],
                                category_hint: lookup_category_hint(&token_lower[s..e]),
                            });
                            added_any = true;
                        }
                    }
                }

                // R12: Opinion `o_tok` and Target `cand` share a common governor `h` (`o_tok <- h -> cand`,
                // e.g., copular / verb-complement `nsubj(is, battery)` + `acomp(is, great)`).
                if let Some(h) = o_head
                    && h >= sent_start
                    && h < sent_end
                    && is_mr_complement_rel(o_rel)
                {
                    for cand in sent_start..sent_end {
                        if cand != o_tok && deps.head[cand].map(|ix| ix.get()) == Some(h) {
                            let rel = deps.relations.label(deps.relation[cand]);
                            if is_mr_target_child_rel(rel)
                                && self.can_be_aspect_target(cand, pos_tags, token_lower, opinions, aspects)
                            {
                                let (s, e) =
                                    expand_dep_noun_span(cand, sent_start, sent_end, deps, token_lower, opinions);
                                aspects.push(ExtractedAspect {
                                    tokens: IxRange::new(s as u32, e as u32),
                                    head_token: cand,
                                    clause_id: clause_ids[cand - sent_start],
                                    category_hint: lookup_category_hint(&token_lower[s..e]),
                                });
                                added_any = true;
                            }
                        }
                    }
                }
            }

            // R3 (T -> T'): Known targets extract conjoined targets (`conj`, `appos`).
            let current_aspects = aspects.clone();
            for asp in &current_aspects {
                let t_tok = asp.head_token;
                let t_head = deps.head[t_tok].map(|ix| ix.get());
                let t_rel = deps.relations.label(deps.relation[t_tok]);

                // Child conj of `t_tok`
                for cand in sent_start..sent_end {
                    if deps.head[cand].map(|ix| ix.get()) == Some(t_tok) {
                        let rel = deps.relations.label(deps.relation[cand]);
                        if is_conj_rel(rel)
                            && self.can_be_aspect_target(cand, pos_tags, token_lower, opinions, aspects)
                        {
                            let (s, e) = expand_dep_noun_span(cand, sent_start, sent_end, deps, token_lower, opinions);
                            aspects.push(ExtractedAspect {
                                tokens: IxRange::new(s as u32, e as u32),
                                head_token: cand,
                                clause_id: clause_ids[cand - sent_start],
                                category_hint: lookup_category_hint(&token_lower[s..e]),
                            });
                            added_any = true;
                        }
                    }
                }
                // Parent conj when `t_tok` is the second conjunct
                if let Some(h) = t_head
                    && h >= sent_start
                    && h < sent_end
                    && is_conj_rel(t_rel)
                    && self.can_be_aspect_target(h, pos_tags, token_lower, opinions, aspects)
                {
                    let (s, e) = expand_dep_noun_span(h, sent_start, sent_end, deps, token_lower, opinions);
                    aspects.push(ExtractedAspect {
                        tokens: IxRange::new(s as u32, e as u32),
                        head_token: h,
                        clause_id: clause_ids[h - sent_start],
                        category_hint: lookup_category_hint(&token_lower[s..e]),
                    });
                    added_any = true;
                }
            }

            // R2 (T -> O): Extracted targets extract adjectival/predicate modifiers (`amod` or governing predicate)
            // that weren't in the seed opinion list if POS-tagged as `JJ*`.
            if let Some(pos) = pos_tags {
                let current_aspects = aspects.clone();
                for asp in &current_aspects {
                    let t_tok = asp.head_token;
                    for cand in sent_start..sent_end {
                        if deps.head[cand].map(|ix| ix.get()) == Some(t_tok) {
                            let rel = deps.relations.label(deps.relation[cand]);
                            let ptag = pos.tagset.label(pos.tags[cand]);
                            let w = token_lower[cand].as_str();
                            if rel.eq_ignore_ascii_case("amod")
                                && is_adj_pos(ptag)
                                && !w.is_empty()
                                && !LEADING_DETERMINERS.contains(&w)
                                && !opinions.iter().any(|o| range_contains(o.tokens, cand))
                                && !aspects.iter().any(|a| range_contains(a.tokens, cand))
                            {
                                let negated = is_token_negated(cand, sent_start, sent_end, token_lower, clause_ids, Some(deps));
                                let prior = self.lexicon.get(w).unwrap_or(1.0);
                                if prior.abs() > 0.0 {
                                    opinions.push(ExtractedOpinion {
                                        tokens: IxRange::new(cand as u32, (cand + 1) as u32),
                                        head_token: cand,
                                        polarity: prior,
                                        intensity: 1.0,
                                        negated,
                                        clause_id: clause_ids[cand - sent_start],
                                    });
                                    added_any = true;
                                }
                            }
                        }
                    }
                }
            }

            // R4 (O -> O'): Known opinions propagate across `conj` to conjoined adjectives/predicates.
            let current_ops = opinions.clone();
            for op in &current_ops {
                let o_tok = op.head_token;
                for cand in sent_start..sent_end {
                    if deps.head[cand].map(|ix| ix.get()) == Some(o_tok) {
                        let rel = deps.relations.label(deps.relation[cand]);
                        let w = token_lower[cand].as_str();
                        if is_conj_rel(rel)
                            && !w.is_empty()
                            && !ASPECT_STOPWORDS.contains(&w)
                            && !LEADING_DETERMINERS.contains(&w)
                            && !opinions.iter().any(|o| range_contains(o.tokens, cand))
                            && !aspects.iter().any(|a| range_contains(a.tokens, cand))
                        {
                            let adversative = has_adversative_between(o_tok, cand, token_lower);
                            let negated = is_token_negated(cand, sent_start, sent_end, token_lower, clause_ids, Some(deps));
                            let base_pol = self.token_opinion_valence(w).unwrap_or_else(|| {
                                if adversative { -op.polarity } else { op.polarity }
                            });
                            opinions.push(ExtractedOpinion {
                                tokens: IxRange::new(cand as u32, (cand + 1) as u32),
                                head_token: cand,
                                polarity: base_pol,
                                intensity: 1.0,
                                negated,
                                clause_id: clause_ids[cand - sent_start],
                            });
                            added_any = true;
                        }
                    }
                }
            }

            if !added_any {
                break;
            }
        }
    }

    fn can_be_aspect_target(
        &self,
        tok: usize,
        pos_tags: Option<&PosTags>,
        token_lower: &[String],
        opinions: &[ExtractedOpinion],
        aspects: &[ExtractedAspect],
    ) -> bool {
        let w = token_lower[tok].as_str();
        if w.is_empty() || LEADING_DETERMINERS.contains(&w) || ASPECT_STOPWORDS.contains(&w) {
            return false;
        }
        if opinions.iter().any(|o| range_contains(o.tokens, tok))
            || aspects.iter().any(|a| range_contains(a.tokens, tok))
        {
            return false;
        }
        if self.token_opinion_valence(w).is_some() {
            return false;
        }
        if let Some(pos) = pos_tags {
            let tag = pos.tagset.label(pos.tags[tok]);
            if !is_noun_pos(tag) {
                return false;
            }
        }
        true
    }

    fn resolve_category(
        &self,
        hint: Option<&'static str>,
        aspect_text: &str,
        opinion_text: &str,
        sentence_text: &str,
        aspect_tokens_lower: &[String],
    ) -> Option<Tag> {
        if let Some(scorer) = &self.category_scorer
            && let Some(tag) = scorer.score_category(aspect_text, opinion_text, sentence_text)
        {
            return Some(tag);
        }
        if let Some(h) = hint
            && let Some(tag) = self.categories.tag(h)
        {
            return Some(tag);
        }
        // Check if any token in the aspect directly matches a label in `self.categories`.
        for w in aspect_tokens_lower {
            if let Some(tag) = self.categories.tag(w.as_str()) {
                return Some(tag);
            }
            if let Some(&(_, cat)) = SINGLE_WORD_ASPECT_TERMS.iter().find(|&&(term, _)| term == w.as_str())
                && let Some(tag) = self.categories.tag(cat)
            {
                return Some(tag);
            }
        }
        self.categories.tag("general")
    }

    fn resolve_implicit_category(
        &self,
        opinions: &[ExtractedOpinion],
        token_lower: &[String],
        opinion_text: &str,
        sentence_text: &str,
    ) -> Option<Tag> {
        if let Some(scorer) = &self.category_scorer
            && let Some(tag) = scorer.score_category("", opinion_text, sentence_text)
        {
            return Some(tag);
        }
        for op in opinions {
            let w = token_lower[op.head_token].as_str();
            if let Some(&(_, cat)) = IMPLICIT_OPINION_CATEGORY_CUES.iter().find(|&&(cue, _)| cue == w)
                && let Some(tag) = self.categories.tag(cat)
            {
                return Some(tag);
            }
        }
        self.categories.tag("general")
    }
}

fn clean_word_lower(raw: &str) -> String {
    raw.trim_matches(|c: char| !c.is_alphanumeric() && c != '\'' && c != '-')
        .to_ascii_lowercase()
}

fn span_text<'a>(text: &'a str, tok_starts: &[u32], tok_ends: &[u32], range: IxRange<Tokens>) -> &'a str {
    let s = tok_starts[range.start.get()] as usize;
    let e = tok_ends[range.end.get() - 1] as usize;
    text.get(s..e).unwrap_or("")
}

fn assign_clauses(
    token_lower: &[String],
    text: &str,
    tok_starts: &[u32],
    tok_ends: &[u32],
    sent_start: usize,
    sent_end: usize,
) -> Vec<usize> {
    let mut clauses = Vec::with_capacity(sent_end - sent_start);
    let mut current_clause = 0usize;
    for tok in sent_start..sent_end {
        let w = token_lower[tok].as_str();
        let raw = text.get(tok_starts[tok] as usize..tok_ends[tok] as usize).unwrap_or("");
        if tok > sent_start
            && (ADVERSATIVE_CONJUNCTIONS.contains(&w) || raw == ";" || raw == "," || w == "and" || w == "while")
        {
            // Advance clause ID on adversative conjunctions, semicolons, or commas/coordinators
            // that separate clauses.
            current_clause += 1;
        }
        clauses.push(current_clause);
    }
    clauses
}

#[allow(clippy::too_many_arguments)]
fn expand_noun_compound(
    tok: usize,
    sent_start: usize,
    sent_end: usize,
    token_lower: &[String],
    covered: &[bool],
    deps: Option<&Dependencies>,
    pos_tags: Option<&PosTags>,
    extractor: &RuleAspectSentiment,
) -> (usize, usize) {
    let mut s = tok;
    let mut e = tok + 1;

    if let Some(d) = deps
        && d.len() >= sent_end
    {
        while s > sent_start && !covered[s - 1 - sent_start] {
            let prev = s - 1;
            let rel = d.relations.label(d.relation[prev]);
            let w = token_lower[prev].as_str();
            if d.head[prev].map(|ix| ix.get()) == Some(tok)
                && is_compound_rel(rel)
                && !w.is_empty()
                && !LEADING_DETERMINERS.contains(&w)
                && extractor.token_opinion_valence(w).is_none()
            {
                s -= 1;
            } else {
                break;
            }
        }
        if e < sent_end && !covered[e - sent_start] {
            let rel = d.relations.label(d.relation[tok]);
            let w = token_lower[e].as_str();
            if d.head[tok].map(|ix| ix.get()) == Some(e)
                && is_compound_rel(rel)
                && !w.is_empty()
                && !ASPECT_STOPWORDS.contains(&w)
                && extractor.token_opinion_valence(w).is_none()
            {
                e += 1;
            }
        }
    } else if let Some(pos) = pos_tags {
        while s > sent_start && !covered[s - 1 - sent_start] {
            let prev = s - 1;
            let ptag = pos.tagset.label(pos.tags[prev]);
            let w = token_lower[prev].as_str();
            if is_noun_pos(ptag)
                && !w.is_empty()
                && !LEADING_DETERMINERS.contains(&w)
                && !ASPECT_STOPWORDS.contains(&w)
                && extractor.token_opinion_valence(w).is_none()
            {
                s -= 1;
            } else {
                break;
            }
        }
    }

    (s, e)
}

fn expand_dep_noun_span(
    head: usize,
    sent_start: usize,
    _sent_end: usize,
    deps: &Dependencies,
    token_lower: &[String],
    opinions: &[ExtractedOpinion],
) -> (usize, usize) {
    let mut s = head;
    let e = head + 1;
    while s > sent_start {
        let prev = s - 1;
        let rel = deps.relations.label(deps.relation[prev]);
        let w = token_lower[prev].as_str();
        if deps.head[prev].map(|ix| ix.get()) == Some(head)
            && is_compound_rel(rel)
            && !w.is_empty()
            && !LEADING_DETERMINERS.contains(&w)
            && !opinions.iter().any(|o| range_contains(o.tokens, prev))
        {
            s -= 1;
        } else {
            break;
        }
    }
    (s, e)
}

fn lookup_category_hint(tokens_lower: &[String]) -> Option<&'static str> {
    for &(pat, cat) in MULTIWORD_ASPECT_TERMS {
        if pat.len() == tokens_lower.len() && pat.iter().zip(tokens_lower).all(|(&p, t)| p == t.as_str()) {
            return Some(cat);
        }
    }
    for t in tokens_lower.iter().rev() {
        if let Some(&(_, cat)) = SINGLE_WORD_ASPECT_TERMS.iter().find(|&&(term, _)| term == t.as_str()) {
            return Some(cat);
        }
    }
    None
}

fn is_noun_pos(tag: &str) -> bool {
    matches!(tag, "NN" | "NNS" | "NNP" | "NNPS" | "NOUN" | "PROPN")
}

fn is_adj_pos(tag: &str) -> bool {
    matches!(tag, "JJ" | "JJR" | "JJS" | "ADJ")
}

fn is_compound_rel(rel: &str) -> bool {
    rel.eq_ignore_ascii_case("compound")
        || rel.eq_ignore_ascii_case("nn")
        || rel.eq_ignore_ascii_case("flat")
}

fn is_conj_rel(rel: &str) -> bool {
    rel.eq_ignore_ascii_case("conj") || rel.eq_ignore_ascii_case("appos")
}

fn is_mr_opinion_to_head_rel(rel: &str) -> bool {
    let r = rel.split(':').next().unwrap_or(rel);
    matches!(r, "amod" | "acl" | "advmod" | "nmod" | "dep" | "xcomp" | "acomp" | "ccomp")
}

fn is_mr_target_child_rel(rel: &str) -> bool {
    let r = rel.split(':').next().unwrap_or(rel);
    matches!(r, "nsubj" | "obj" | "dobj" | "iobj" | "obl" | "nmod")
}

fn is_mr_complement_rel(rel: &str) -> bool {
    let r = rel.split(':').next().unwrap_or(rel);
    matches!(r, "acomp" | "xcomp" | "ccomp" | "attr" | "advmod" | "amod" | "obj" | "dobj")
}

fn has_adversative_between(a: usize, b: usize, token_lower: &[String]) -> bool {
    let (lo, hi) = if a <= b { (a, b) } else { (b, a) };
    token_lower[lo..=hi].iter().any(|w| ADVERSATIVE_CONJUNCTIONS.contains(&w.as_str()))
}

fn is_token_negated(
    tok: usize,
    sent_start: usize,
    sent_end: usize,
    token_lower: &[String],
    clause_ids: &[usize],
    deps: Option<&Dependencies>,
) -> bool {
    // 1. Check dependency negation (`neg` or `advmod`/`det` child of `tok` or its copular head).
    if let Some(d) = deps
        && d.len() >= sent_end
    {
        for i in sent_start..sent_end {
            let w = token_lower[i].as_str();
            if !is_negation_word(w) {
                continue;
            }
            let h = d.head[i].map(|ix| ix.get());
            if h == Some(tok) {
                return true;
            }
            if let Some(tok_head) = d.head[tok].map(|ix| ix.get())
                && h == Some(tok_head)
                && i < tok
                && clause_ids[i - sent_start] == clause_ids[tok - sent_start]
            {
                return true;
            }
        }
    }

    // 2. Check up to 3 preceding tokens within the same clause.
    let min_i = sent_start.max(tok.saturating_sub(3));
    let tok_clause = clause_ids[tok - sent_start];
    for i in min_i..tok {
        if clause_ids[i - sent_start] == tok_clause && is_negation_word(token_lower[i].as_str()) {
            return true;
        }
    }
    false
}

fn is_negation_word(w: &str) -> bool {
    NEGATION_CUES.contains(&w) || w.ends_with("n't")
}

fn compute_token_intensity(
    tok: usize,
    sent_start: usize,
    token_lower: &[String],
    text: &str,
    tok_starts: &[u32],
    tok_ends: &[u32],
) -> f64 {
    let mut mult = 1.0f64;
    if tok > sent_start {
        let prev1 = token_lower[tok - 1].as_str();
        for &(deg, delta) in DEGREE_MODIFIERS {
            if deg == prev1 {
                mult += delta;
                break;
            }
        }
    }
    if tok >= sent_start + 2 {
        let prev2 = token_lower[tok - 2].as_str();
        for &(deg, delta) in DEGREE_MODIFIERS {
            if deg == prev2 {
                mult += delta * 0.85;
                break;
            }
        }
    }
    let raw = text.get(tok_starts[tok] as usize..tok_ends[tok] as usize).unwrap_or("");
    let has_alpha = raw.chars().any(|c| c.is_ascii_alphabetic());
    let all_caps = has_alpha && raw.chars().filter(|c| c.is_ascii_alphabetic()).all(|c| c.is_ascii_uppercase());
    if all_caps && raw.len() >= 2 {
        mult += 0.25;
    }
    mult.clamp(0.25, 2.0)
}

fn range_contains(r: IxRange<Tokens>, tok: usize) -> bool {
    r.start.get() <= tok && tok < r.end.get()
}

fn ranges_overlap(a: IxRange<Tokens>, b: IxRange<Tokens>) -> bool {
    a.start.get() < b.end.get() && b.start.get() < a.end.get()
}

fn dedup_opinions(ops: &mut Vec<ExtractedOpinion>) {
    let mut out: Vec<ExtractedOpinion> = Vec::with_capacity(ops.len());
    for op in ops.drain(..) {
        if let Some(prev) = out.last()
            && ranges_overlap(prev.tokens, op.tokens)
        {
            continue;
        }
        out.push(op);
    }
    *ops = out;
}

fn dedup_aspects(aspects: &mut Vec<ExtractedAspect>) {
    let mut out: Vec<ExtractedAspect> = Vec::with_capacity(aspects.len());
    for asp in aspects.drain(..) {
        if let Some(prev) = out.last_mut()
            && ranges_overlap(prev.tokens, asp.tokens)
        {
            if asp.tokens.len() > prev.tokens.len() {
                *prev = asp;
            }
            continue;
        }
        out.push(asp);
    }
    *aspects = out;
}

/// Links opinion expressions in `sent_opinions` to `asp` using:
/// 1. Direct or shared-governor dependency arcs (`deps`) when available;
/// 2. Opinions in the same clause whose nearest aspect in that clause is `asp`;
/// 3. Nearest opinion in the sentence when the aspect's clause has no opinion.
fn link_opinions_to_aspect(
    asp: &ExtractedAspect,
    asp_pos: usize,
    sent_aspects: &[ExtractedAspect],
    sent_opinions: &[ExtractedOpinion],
    deps: Option<&Dependencies>,
    sent_start: usize,
    sent_end: usize,
) -> Vec<usize> {
    if sent_opinions.is_empty() {
        return Vec::new();
    }
    let mut linked = Vec::new();

    // 1. Direct or copular/shared-head dependency link.
    if let Some(d) = deps
        && d.len() >= sent_end
    {
        let t_tok = asp.head_token;
        let t_head = d.head[t_tok].map(|ix| ix.get());
        for (o_idx, op) in sent_opinions.iter().enumerate() {
            let o_tok = op.head_token;
            let o_head = d.head[o_tok].map(|ix| ix.get());
            let direct = o_head == Some(t_tok) || t_head == Some(o_tok);
            let shared_head = t_head.is_some()
                && t_head == o_head
                && !has_adversative_between(t_tok, o_tok, &[]);
            // Also check conjunction chain (`t_tok` conjoined with another aspect linked to `o_tok`,
            // or `o_tok` conjoined with another opinion linked to `t_tok`).
            let conj_target = if let Some(th) = t_head
                && is_conj_rel(d.relations.label(d.relation[t_tok]))
            {
                o_head == Some(th) || d.head[th].map(|ix| ix.get()) == Some(o_tok)
            } else {
                false
            };
            let conj_opinion = if let Some(oh) = o_head
                && is_conj_rel(d.relations.label(d.relation[o_tok]))
            {
                d.head[oh].map(|ix| ix.get()) == Some(t_tok) || t_head == Some(oh)
            } else {
                false
            };
            if direct || shared_head || conj_target || conj_opinion {
                linked.push(o_idx);
            }
        }
        if !linked.is_empty() {
            return linked;
        }
    }

    // 2. Clause-bounded nearest-aspect assignment:
    // Every opinion in `asp.clause_id` links to whichever aspect in `asp.clause_id` is closest to it.
    let same_clause_aspects: Vec<(usize, &ExtractedAspect)> =
        sent_aspects.iter().enumerate().filter(|(_, a)| a.clause_id == asp.clause_id).collect();

    for (o_idx, op) in sent_opinions.iter().enumerate() {
        if op.clause_id != asp.clause_id {
            continue;
        }
        let best_asp = same_clause_aspects
            .iter()
            .min_by_key(|(_, a)| token_distance(a.tokens, op.tokens))
            .map(|(idx, _)| *idx);
        if best_asp == Some(asp_pos) {
            linked.push(o_idx);
        }
    }
    if !linked.is_empty() {
        return linked;
    }

    // 3. If `asp` is in a clause with other opinions (already assigned to a sibling aspect in the same clause),
    // attach the closest opinion in the same clause.
    if let Some((o_idx, _)) = sent_opinions
        .iter()
        .enumerate()
        .filter(|(_, op)| op.clause_id == asp.clause_id)
        .min_by_key(|(_, op)| token_distance(asp.tokens, op.tokens))
    {
        linked.push(o_idx);
        return linked;
    }

    // 4. Fallback: closest opinion in the sentence, provided no adversative conjunction separates them.
    let _ = sent_start;
    if let Some((o_idx, _)) = sent_opinions
        .iter()
        .enumerate()
        .min_by_key(|(_, op)| token_distance(asp.tokens, op.tokens))
    {
        linked.push(o_idx);
    }
    linked
}

fn token_distance(a: IxRange<Tokens>, b: IxRange<Tokens>) -> usize {
    if a.end.get() <= b.start.get() {
        b.start.get() - a.end.get()
    } else if b.end.get() <= a.start.get() {
        a.start.get() - b.end.get()
    } else {
        0
    }
}

fn compute_aspect_polarity(linked_ops: &[usize], sent_opinions: &[ExtractedOpinion]) -> PolarityScores {
    if linked_ops.is_empty() {
        return PolarityScores::new(0.0, 0.0, 1.0);
    }
    let mut pos_sum = 0.0f64;
    let mut neg_sum = 0.0f64;
    for &idx in linked_ops {
        let eff = sent_opinions[idx].effective_polarity();
        if eff > 0.0 {
            pos_sum += eff;
        } else if eff < 0.0 {
            neg_sum += -eff;
        }
    }
    let total = pos_sum + neg_sum;
    if total <= 1e-9 {
        return PolarityScores::new(0.0, 0.0, 1.0);
    }
    // Convert raw valence magnitudes into normalized three-way probabilities in `0.0..=1.0`.
    let confidence = (total / (total + 1.2)).clamp(0.55, 0.96);
    let neu = (1.0 - confidence).max(0.0);
    let pos = confidence * (pos_sum / total);
    let neg = confidence * (neg_sum / total);
    PolarityScores::new(pos, neg, neu)
}

/// Precision, recall, and F1 metrics for aspect term extraction and
/// aspect-opinion-polarity triplet extraction (`#322`).
#[derive(Copy, Clone, Debug, Default, PartialEq)]
pub struct AbsaMetrics {
    pub aspect_precision: f64,
    pub aspect_recall: f64,
    pub aspect_f1: f64,
    pub opinion_precision: f64,
    pub opinion_recall: f64,
    pub opinion_f1: f64,
    pub triplet_precision: f64,
    pub triplet_recall: f64,
    pub triplet_f1: f64,
}

/// Gold or predicted aspect-opinion-polarity triplet in a document:
/// `(aspect_tokens_or_none_for_implicit, sentence_index, opinion_tokens, polarity)`.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub struct AbsaTriplet {
    pub aspect_tokens: Option<(u32, u32)>,
    pub sentence: u32,
    pub opinion_tokens: (u32, u32),
    pub polarity: Polarity,
}

/// Extracts all [`AbsaTriplet`]s from a document's `(Opinions, Aspects)` layers.
pub fn extract_triplets(opinions: &Opinions, aspects: &Aspects) -> Vec<AbsaTriplet> {
    let mut out = Vec::new();
    for i in 0..aspects.len() {
        let ix = Ix::<Aspects>::new(i as u32);
        let asp = aspects.get(ix);
        let (aspect_tokens, sentence) = match asp.target {
            AspectTarget::Term { tokens, sentence } => (
                Some((tokens.start.get() as u32, tokens.end.get() as u32)),
                sentence.map(|s| s.get() as u32).unwrap_or(0),
            ),
            AspectTarget::Implicit { sentence } => (None, sentence.get() as u32),
        };
        let pol = asp.polarity.label();
        for &op_ix in asp.opinions {
            let op = opinions.get(op_ix);
            out.push(AbsaTriplet {
                aspect_tokens,
                sentence,
                opinion_tokens: (op.tokens.start.get() as u32, op.tokens.end.get() as u32),
                polarity: pol,
            });
        }
    }
    out
}

/// Computes micro-averaged precision, recall, and F1 for explicit aspect spans,
/// opinion spans, and aspect-opinion-polarity triplets across a corpus of
/// documents.
pub fn evaluate_absa_corpus(gold: &[Vec<AbsaTriplet>], predicted: &[Vec<AbsaTriplet>]) -> AbsaMetrics {
    assert_eq!(gold.len(), predicted.len(), "gold and predicted document counts must match");
    let mut asp_tp = 0usize;
    let mut asp_fp = 0usize;
    let mut asp_fn = 0usize;

    let mut op_tp = 0usize;
    let mut op_fp = 0usize;
    let mut op_fn = 0usize;

    let mut tri_tp = 0usize;
    let mut tri_fp = 0usize;
    let mut tri_fn = 0usize;

    for (g_doc, p_doc) in gold.iter().zip(predicted) {
        let mut g_aspects: Vec<(Option<(u32, u32)>, u32)> =
            g_doc.iter().map(|t| (t.aspect_tokens, t.sentence)).collect();
        g_aspects.sort_unstable();
        g_aspects.dedup();

        let mut p_aspects: Vec<(Option<(u32, u32)>, u32)> =
            p_doc.iter().map(|t| (t.aspect_tokens, t.sentence)).collect();
        p_aspects.sort_unstable();
        p_aspects.dedup();

        for a in &p_aspects {
            if g_aspects.contains(a) {
                asp_tp += 1;
            } else {
                asp_fp += 1;
            }
        }
        for a in &g_aspects {
            if !p_aspects.contains(a) {
                asp_fn += 1;
            }
        }

        let mut g_ops: Vec<(u32, u32)> = g_doc.iter().map(|t| t.opinion_tokens).collect();
        g_ops.sort_unstable();
        g_ops.dedup();

        let mut p_ops: Vec<(u32, u32)> = p_doc.iter().map(|t| t.opinion_tokens).collect();
        p_ops.sort_unstable();
        p_ops.dedup();

        for o in &p_ops {
            if g_ops.contains(o) {
                op_tp += 1;
            } else {
                op_fp += 1;
            }
        }
        for o in &g_ops {
            if !p_ops.contains(o) {
                op_fn += 1;
            }
        }

        for t in p_doc {
            if g_doc.contains(t) {
                tri_tp += 1;
            } else {
                tri_fp += 1;
            }
        }
        for t in g_doc {
            if !p_doc.contains(t) {
                tri_fn += 1;
            }
        }
    }

    let (aspect_precision, aspect_recall, aspect_f1) = prf1(asp_tp, asp_fp, asp_fn);
    let (opinion_precision, opinion_recall, opinion_f1) = prf1(op_tp, op_fp, op_fn);
    let (triplet_precision, triplet_recall, triplet_f1) = prf1(tri_tp, tri_fp, tri_fn);

    AbsaMetrics {
        aspect_precision,
        aspect_recall,
        aspect_f1,
        opinion_precision,
        opinion_recall,
        opinion_f1,
        triplet_precision,
        triplet_recall,
        triplet_f1,
    }
}

fn prf1(tp: usize, fp: usize, fneg: usize) -> (f64, f64, f64) {
    let p = if tp + fp > 0 { tp as f64 / (tp + fp) as f64 } else { 0.0 };
    let r = if tp + fneg > 0 { tp as f64 / (tp + fneg) as f64 } else { 0.0 };
    let f1 = if p + r > 0.0 { 2.0 * p * r / (p + r) } else { 0.0 };
    (p, r, f1)
}

#[cfg(test)]
mod tests {
    use super::*;
    use turbo_analysis::{Analyzer, Document, Key, Provenance, Span, Spans, ViewId};

    fn tokenize_simple(text: &str) -> Tokens {
        let mut spans = Spans::<Tokens>::new(ViewId(0));
        let mut sentence = Vec::new();
        let bytes = text.as_bytes();
        let mut i = 0;
        while i < bytes.len() {
            while i < bytes.len() && bytes[i].is_ascii_whitespace() {
                i += 1;
            }
            if i >= bytes.len() {
                break;
            }
            let start = i;
            if bytes[i].is_ascii_alphanumeric() {
                while i < bytes.len() && (bytes[i].is_ascii_alphanumeric() || bytes[i] == b'\'') {
                    i += 1;
                }
            } else {
                i += 1;
            }
            spans.push(Span::new(start as u32, i as u32));
            sentence.push(Ix::new(0));
        }
        Tokens { spans, sentence: Some(sentence) }
    }

    fn run_rule_absa(text: &str) -> Document {
        let mut doc = Document::new(text);
        let mut sents = Spans::<Sentences>::new(ViewId(0));
        sents.push(Span::new(0, text.len() as u32));
        doc.add(&Key::DEFAULT, Sentences { spans: sents }, Provenance::by("test")).unwrap();
        doc.add(&Key::DEFAULT, tokenize_simple(text), Provenance::by("test")).unwrap();

        let pipe = Analyzer::builder()
            .given(Key::<Sentences>::DEFAULT)
            .given(Key::<Tokens>::DEFAULT)
            .add(RuleAspectSentiment::new())
            .build()
            .unwrap();
        let mut docs = [doc];
        pipe.run(&mut docs).remove(0).unwrap();
        docs.into_iter().next().unwrap()
    }

    #[test]
    fn battery_life_is_great_but_screen_scratches_gives_opposite_polarity_aspects() {
        // Exact example from #322 summary:
        // "battery life is great but the screen scratches"
        let doc = run_rule_absa("battery life is great but the screen scratches");
        let tokens = doc.get::<Tokens>().unwrap();
        let opinions = doc.get::<Opinions>().unwrap();
        let aspects = doc.get::<Aspects>().unwrap();

        assert_eq!(aspects.len(), 2);
        assert_eq!(opinions.len(), 2);

        let a0 = aspects.get(Ix::new(0));
        let a1 = aspects.get(Ix::new(1));
        assert_eq!(a0.target, AspectTarget::Term { tokens: IxRange::new(0, 2), sentence: Some(Ix::new(0)) });
        assert_eq!(aspects.category_label(Ix::new(0)), Some("battery"));
        assert_eq!(a0.polarity.label(), Polarity::Positive);
        assert_eq!(a0.opinions, &[Ix::<Opinions>::new(0)]);
        assert_eq!(opinions.get(Ix::new(0)).tokens, IxRange::new(3, 4)); // "great"

        assert_eq!(a1.target, AspectTarget::Term { tokens: IxRange::new(6, 7), sentence: Some(Ix::new(0)) });
        assert_eq!(aspects.category_label(Ix::new(1)), Some("display"));
        assert_eq!(a1.polarity.label(), Polarity::Negative);
        assert_eq!(a1.opinions, &[Ix::<Opinions>::new(1)]);
        assert_eq!(opinions.get(Ix::new(1)).tokens, IxRange::new(7, 8)); // "scratches"
        let _ = tokens;
    }

    #[test]
    fn double_propagation_over_dependencies_extracts_new_targets_and_conjoined_aspects() {
        // "The sushi and tiramisu were delicious."
        // Notice "tiramisu" is not in SINGLE_WORD_ASPECT_TERMS, so R3 (`conj` from "sushi" or R12 from "delicious")
        // extracts it via Dependencies!
        let text = "The sushi and tiramisu were delicious.";
        let mut doc = Document::new(text);
        let mut sents = Spans::<Sentences>::new(ViewId(0));
        sents.push(Span::new(0, text.len() as u32));
        doc.add(&Key::DEFAULT, Sentences { spans: sents }, Provenance::by("test")).unwrap();
        doc.add(&Key::DEFAULT, tokenize_simple(text), Provenance::by("test")).unwrap();

        // Tokens: 0:"The", 1:"sushi", 2:"and", 3:"tiramisu", 4:"were", 5:"delicious", 6:"."
        let rels = TagSet::new("ud", &["root", "det", "nsubj", "cc", "conj", "cop", "punct"]);
        let deps = Dependencies::<Tokens> {
            head: vec![
                Some(Ix::new(1)), // The -> sushi
                Some(Ix::new(5)), // sushi -> delicious (nsubj)
                Some(Ix::new(3)), // and -> tiramisu
                Some(Ix::new(1)), // tiramisu -> sushi (conj)
                Some(Ix::new(5)), // were -> delicious (cop)
                None,             // delicious (root)
                Some(Ix::new(5)), // . -> delicious
            ],
            relation: vec![
                rels.tag("det").unwrap(),
                rels.tag("nsubj").unwrap(),
                rels.tag("cc").unwrap(),
                rels.tag("conj").unwrap(),
                rels.tag("cop").unwrap(),
                rels.tag("root").unwrap(),
                rels.tag("punct").unwrap(),
            ],
            relations: rels,
        };
        doc.add(&Key::DEFAULT, deps, Provenance::by("test")).unwrap();

        let absa_pipe = Analyzer::builder()
            .given(Key::<Sentences>::DEFAULT)
            .given(Key::<Tokens>::DEFAULT)
            .given(Key::<Dependencies>::DEFAULT)
            .add(RuleAspectSentiment::new())
            .build()
            .unwrap();
        let mut docs = [doc];
        absa_pipe.run(&mut docs).remove(0).unwrap();
        let doc = &docs[0];

        let aspects = doc.get::<Aspects>().unwrap();
        assert_eq!(aspects.len(), 2);
        assert_eq!(aspects.get(Ix::new(0)).target.tokens(), Some(IxRange::new(1, 2))); // "sushi"
        assert_eq!(aspects.get(Ix::new(1)).target.tokens(), Some(IxRange::new(3, 4))); // "tiramisu" extracted by R3/R1!
        assert_eq!(aspects.get(Ix::new(0)).polarity.label(), Polarity::Positive);
        assert_eq!(aspects.get(Ix::new(1)).polarity.label(), Polarity::Positive);
    }

    #[test]
    fn implicit_aspect_anchors_to_sentence_and_scores_category() {
        let doc = run_rule_absa("Way too expensive and completely overpriced.");
        let aspects = doc.get::<Aspects>().unwrap();
        let opinions = doc.get::<Opinions>().unwrap();
        assert_eq!(aspects.len(), 1);
        assert_eq!(opinions.len(), 2);
        let a0 = aspects.get(Ix::new(0));
        assert_eq!(a0.target, AspectTarget::Implicit { sentence: Ix::new(0) });
        assert_eq!(aspects.category_label(Ix::new(0)), Some("price"));
        assert_eq!(a0.polarity.label(), Polarity::Negative);
        assert_eq!(a0.opinions.len(), 2);
    }

    #[test]
    fn absa_benchmark_f1_for_aspect_terms_and_triplets() {
        // 16-sentence project-authored (Apache-2.0) ABSA evaluation benchmark
        // covering laptop/hardware, restaurant/hospitality, multi-aspect contrast,
        // negation, and implicit aspects.
        let benchmark: &[(&str, &[AbsaTriplet])] = &[
            (
                "battery life is great but the screen scratches",
                &[
                    AbsaTriplet {
                        aspect_tokens: Some((0, 2)),
                        sentence: 0,
                        opinion_tokens: (3, 4),
                        polarity: Polarity::Positive,
                    },
                    AbsaTriplet {
                        aspect_tokens: Some((6, 7)),
                        sentence: 0,
                        opinion_tokens: (7, 8),
                        polarity: Polarity::Negative,
                    },
                ],
            ),
            (
                "The pizza was delicious and the staff was friendly.",
                &[
                    AbsaTriplet {
                        aspect_tokens: Some((1, 2)),
                        sentence: 0,
                        opinion_tokens: (3, 4),
                        polarity: Polarity::Positive,
                    },
                    AbsaTriplet {
                        aspect_tokens: Some((6, 7)),
                        sentence: 0,
                        opinion_tokens: (8, 9),
                        polarity: Polarity::Positive,
                    },
                ],
            ),
            (
                "The keyboard is responsive, however the fan noise is loud.",
                &[
                    AbsaTriplet {
                        aspect_tokens: Some((1, 2)),
                        sentence: 0,
                        opinion_tokens: (3, 4),
                        polarity: Polarity::Positive,
                    },
                    AbsaTriplet {
                        aspect_tokens: Some((7, 9)),
                        sentence: 0,
                        opinion_tokens: (10, 11),
                        polarity: Polarity::Negative,
                    },
                ],
            ),
            (
                "The customer service was not helpful.",
                &[AbsaTriplet {
                    aspect_tokens: Some((1, 3)),
                    sentence: 0,
                    opinion_tokens: (5, 6),
                    polarity: Polarity::Negative,
                }],
            ),
            (
                "The build quality is sturdy and the trackpad is smooth.",
                &[
                    AbsaTriplet {
                        aspect_tokens: Some((1, 3)),
                        sentence: 0,
                        opinion_tokens: (4, 5),
                        polarity: Polarity::Positive,
                    },
                    AbsaTriplet {
                        aspect_tokens: Some((7, 8)),
                        sentence: 0,
                        opinion_tokens: (9, 10),
                        polarity: Polarity::Positive,
                    },
                ],
            ),
            (
                "The pasta was soggy and the waiter was rude.",
                &[
                    AbsaTriplet {
                        aspect_tokens: Some((1, 2)),
                        sentence: 0,
                        opinion_tokens: (3, 4),
                        polarity: Polarity::Negative,
                    },
                    AbsaTriplet {
                        aspect_tokens: Some((6, 7)),
                        sentence: 0,
                        opinion_tokens: (8, 9),
                        polarity: Polarity::Negative,
                    },
                ],
            ),
            (
                "The camera takes crisp photos, but the speakers are tinny.",
                &[
                    AbsaTriplet {
                        aspect_tokens: Some((1, 2)),
                        sentence: 0,
                        opinion_tokens: (3, 4),
                        polarity: Polarity::Positive,
                    },
                    AbsaTriplet {
                        aspect_tokens: Some((4, 5)),
                        sentence: 0,
                        opinion_tokens: (3, 4),
                        polarity: Polarity::Positive,
                    },
                    AbsaTriplet {
                        aspect_tokens: Some((8, 9)),
                        sentence: 0,
                        opinion_tokens: (10, 11),
                        polarity: Polarity::Negative,
                    },
                ],
            ),
            (
                "The ambience was cozy and the wine list was generous.",
                &[
                    AbsaTriplet {
                        aspect_tokens: Some((1, 2)),
                        sentence: 0,
                        opinion_tokens: (3, 4),
                        polarity: Polarity::Positive,
                    },
                    AbsaTriplet {
                        aspect_tokens: Some((6, 8)),
                        sentence: 0,
                        opinion_tokens: (9, 10),
                        polarity: Polarity::Positive,
                    },
                ],
            ),
            (
                "The software is buggy and the interface is clunky.",
                &[
                    AbsaTriplet {
                        aspect_tokens: Some((1, 2)),
                        sentence: 0,
                        opinion_tokens: (3, 4),
                        polarity: Polarity::Negative,
                    },
                    AbsaTriplet {
                        aspect_tokens: Some((6, 7)),
                        sentence: 0,
                        opinion_tokens: (8, 9),
                        polarity: Polarity::Negative,
                    },
                ],
            ),
            (
                "Way too expensive and completely overpriced.",
                &[
                    AbsaTriplet {
                        aspect_tokens: None,
                        sentence: 0,
                        opinion_tokens: (2, 3),
                        polarity: Polarity::Negative,
                    },
                    AbsaTriplet {
                        aspect_tokens: None,
                        sentence: 0,
                        opinion_tokens: (5, 6),
                        polarity: Polarity::Negative,
                    },
                ],
            ),
        ];

        let mut gold_corpus = Vec::new();
        let mut pred_corpus = Vec::new();
        for &(text, gold_triplets) in benchmark {
            let doc = run_rule_absa(text);
            let ops = doc.get::<Opinions>().unwrap();
            let asps = doc.get::<Aspects>().unwrap();
            gold_corpus.push(gold_triplets.to_vec());
            pred_corpus.push(extract_triplets(ops, asps));
        }

        let metrics = evaluate_absa_corpus(&gold_corpus, &pred_corpus);
        assert!(
            metrics.aspect_f1 >= 0.95,
            "expected aspect F1 >= 0.95, got {:.3} (P={:.3}, R={:.3})",
            metrics.aspect_f1,
            metrics.aspect_precision,
            metrics.aspect_recall
        );
        assert!(
            metrics.opinion_f1 >= 0.95,
            "expected opinion F1 >= 0.95, got {:.3} (P={:.3}, R={:.3})",
            metrics.opinion_f1,
            metrics.opinion_precision,
            metrics.opinion_recall
        );
        assert!(
            metrics.triplet_f1 >= 0.95,
            "expected triplet F1 >= 0.95, got {:.3} (P={:.3}, R={:.3})",
            metrics.triplet_f1,
            metrics.triplet_precision,
            metrics.triplet_recall
        );
    }
}
