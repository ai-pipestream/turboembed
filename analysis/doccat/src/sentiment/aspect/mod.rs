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

mod link;
mod propagation;
pub mod shared;
mod tables;

use std::fmt;
use std::hash::{Hash, Hasher};
use std::sync::Arc;

use turbo_analysis::{Annotator, Error, Ix, Layer, ModelRef, Refs, Tag, TagSet, View};
use turbo_analysis_layers::{
    Aspect, AspectTarget, Aspects, Dependencies, Opinion, Opinions, PhraseChunks, Polarity, PolarityScores, PosTags,
    Sentences, Tokens,
};

use super::lexicon::VaderLexicon;

pub use tables::DEFAULT_ASPECT_CATEGORIES;

use link::{
    compute_aspect_polarity, dedup_aspects, dedup_opinions, link_opinions_to_aspect, ranges_overlap, span_text,
};
use propagation::{ExtractedOpinion, assign_clauses};
use tables::{
    IMPLICIT_OPINION_CATEGORY_CUES, clean_word_lower, domain_opinion_valence, is_pure_aspect_noun,
    single_word_aspect_category,
};

/// Optional scorer that assigns an aspect category [`Tag`] from a [`TagSet`]
/// to an aspect (explicit or implicit) given its surface text and enclosing
/// sentence text.
///
/// Implemented in `turbo-analysis-embed` by `LabelCentroids`-backed category
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
        if let Some(score) = domain_opinion_valence(lower) {
            return Some(score);
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

mod annotate;
mod metrics;
#[cfg(test)]
mod tests;
pub use metrics::{AbsaMetrics, AbsaTriplet, evaluate_absa_corpus, extract_triplets};

impl RuleAspectSentiment {
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
            if let Some(cat) = single_word_aspect_category(w)
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
