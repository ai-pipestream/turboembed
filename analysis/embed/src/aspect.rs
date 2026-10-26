//! Learned aspect and opinion extraction over contextual subword vectors and
//! span-pooled embeddings (`#322`).
//!
//! - [`LearnedAspectSentiment`] (`turbo:aspect-sentiment-learned`) reads
//!   `(View, Sentences, Tokens, Subwords, Option<Vectors<Subwords>>, Option<Dependencies>, Option<PhraseChunks>)`
//!   and writes `(Opinions, Aspects)`, matching the output layer schema of
//!   `RuleAspectSentiment` (`turbo-doccat:aspect-sentiment-rule`) while
//!   recording the learned encoder, heads, and execution backend in
//!   provenance.
//! - **Subword BIO tagging**: scores each piece in [`Subwords`]
//!   (`opennlp:subwords`) over its contextual subword vector
//!   (`Vectors<Subwords>`) using a 5-class BIO [`ScoringHead`]
//!   ([`BIO_ASPECT_OPINION_LABELS`]: `["O", "B-ASP", "I-ASP", "B-OP", "I-OP"]`),
//!   projects subword tag distributions back onto [`Tokens`] (`opennlp:tokens`)
//!   via `Subwords` byte-span and token alignments, and decodes contiguous
//!   aspect term (`B-ASP`/`I-ASP`) and opinion expression (`B-OP`/`I-OP`)
//!   token ranges.
//! - **Span-pooled polarity (`#263`)**: pools the exact subword range of each
//!   extracted opinion and aspect (or linked aspect + opinion span) from retained raw window states via
//!   the sentence window policy and scores three-way [`PolarityScores`]
//!   with the polarity [`ScoringHead`].
//! - **Aspect categories**: scores each aspect's pooled embedding against
//!   [`LabelCentroids`] over an aspect category [`TagSet`] (for example,
//!   `"battery"`, `"display"`, `"service"`, `"food"`, `"price"`).
//! - **Implicit aspects**: when a sentence contains extracted opinion
//!   expressions (`B-OP`/`I-OP`) without any explicit aspect term, anchors an
//!   [`AspectTarget::Implicit`] aspect to the sentence, pools its span via
//!   `#263`, and scores polarity and category.

mod decode;
mod fit;
mod infer;
#[cfg(test)]
mod tests;

use std::sync::Arc;

use turbo_analysis::{Annotator, Error, Execution, Ix, IxRange, Layer, ModelRef, Refs, Span, Tag, TagSet, View};
use turbo_analysis_layers::{
    Aspect, AspectTarget, Aspects, Dependencies, Opinion, Opinions, PhraseChunks, PolarityScores, Sentences, Subwords,
    Tokens, Vectors,
};

use self::decode::{
    DecodedAspect, DecodedOpinion, TAG_O, assign_clauses, decode_bio_spans, is_token_negated, link_decoded_opinions,
    token_range_span, unit_l2,
};
use crate::centroids::{LabelCentroids, fnv1a};
use crate::heads::ScoringHead;
use crate::vectors::VectorAnnotator;
use crate::window::Which;

/// Standard 5-class BIO tag labels for joint aspect term and opinion
/// expression extraction (`O`, `B-ASP`, `I-ASP`, `B-OP`, `I-OP`).
pub const BIO_ASPECT_OPINION_LABELS: &[&str; 5] = &["O", "B-ASP", "I-ASP", "B-OP", "I-OP"];

/// Builds the canonical 5-label BIO [`TagSet`] (`"bio-aspect-opinion"`).
pub fn bio_aspect_opinion_tagset() -> Arc<TagSet> {
    TagSet::new("bio-aspect-opinion", BIO_ASPECT_OPINION_LABELS)
}

/// One labeled training document for fitting [`LearnedAspectSentiment`] heads:
/// its subwords, sentences, tokens, explicit aspect spans with category and
/// polarity label (`"positive"`, `"negative"`, `"neutral"`), and opinion
/// spans with polarity label.
pub struct LabeledAbsaDoc<'a> {
    pub subwords: &'a Subwords,
    pub sentences: &'a Sentences,
    pub tokens: &'a Tokens,
    /// Explicit aspect `(token_range, category_label, polarity_label)`.
    pub aspects: &'a [(IxRange<Tokens>, &'a str, &'a str)],
    /// Opinion `(token_range, polarity_label)`.
    pub opinions: &'a [(IxRange<Tokens>, &'a str)],
}

/// Learned aspect and opinion extractor (`turbo:aspect-sentiment-learned`,
/// `#322`).
#[derive(Clone)]
pub struct LearnedAspectSentiment {
    pub vectors: Arc<VectorAnnotator>,
    pub bio_head: ScoringHead,
    pub polarity_head: ScoringHead,
    pub category_centroids: Option<Arc<LabelCentroids>>,
    pub categories: Arc<TagSet>,
    pub emit_implicit_aspects: bool,
    bio_threshold: f64,
}

impl LearnedAspectSentiment {
    /// Creates a new `LearnedAspectSentiment` annotator.
    pub fn new(
        vectors: Arc<VectorAnnotator>,
        bio_head: impl Into<ScoringHead>,
        polarity_head: impl Into<ScoringHead>,
        category_centroids: Option<Arc<LabelCentroids>>,
    ) -> Self {
        let categories = category_centroids
            .as_ref()
            .map(|c| Arc::clone(c.tags()))
            .unwrap_or_else(|| TagSet::new("aspect-categories", &["general"]));
        Self {
            vectors,
            bio_head: bio_head.into(),
            polarity_head: polarity_head.into(),
            category_centroids,
            categories,
            emit_implicit_aspects: true,
            bio_threshold: 0.0,
        }
    }

    /// Requires a non-background BIO probability of at least `threshold`
    /// before decoding an aspect or opinion token. Zero keeps argmax decoding.
    pub fn with_bio_threshold(mut self, threshold: f64) -> Self {
        self.bio_threshold = if threshold.is_finite() { threshold.clamp(0.0, 1.0) } else { threshold };
        self
    }

    /// Configures whether sentences with extracted opinion expressions but no
    /// explicit aspect term emit an [`AspectTarget::Implicit`] entry.
    pub fn with_implicit_aspects(mut self, enabled: bool) -> Self {
        self.emit_implicit_aspects = enabled;
        self
    }

    /// Overrides the aspect category [`TagSet`] when `category_centroids` is
    /// `None`.
    pub fn with_categories(mut self, categories: Arc<TagSet>) -> Self {
        if self.category_centroids.is_none() {
            self.categories = categories;
        }
        self
    }
}

impl Annotator for LearnedAspectSentiment {
    const ID: &'static str = "turbo:aspect-sentiment-learned";
    const VERSION: &'static str = "3";
    type Reads =
        (View, Sentences, Tokens, Subwords, Option<Vectors<Subwords>>, Option<Dependencies>, Option<PhraseChunks>);
    type Writes = (Opinions, Aspects);

    fn annotate(&self, input: Refs<'_, Self>) -> Result<Self::Writes, Error> {
        self.annotate_batch(vec![input]).remove(0)
    }

    fn annotate_batch<'d>(&self, inputs: Vec<Refs<'d, Self>>) -> Vec<Result<Self::Writes, Error>> {
        let valid: Vec<_> = inputs.iter().map(|&i| Self::validate_views(i, self.bio_threshold)).collect();
        let windows: Vec<_> =
            inputs.iter().zip(&valid).filter(|(_, v)| v.is_ok()).map(|(i, _)| (i.3, &i.1.spans)).collect();
        let mut cached = self.vectors.cache_windows(Self::ID, &windows).into_iter();
        inputs
            .into_iter()
            .zip(valid)
            .map(|(input, valid)| {
                valid?;
                self.infer(input, &cached.next().expect("cache for valid document")?)
            })
            .collect()
    }

    fn model(&self) -> Option<ModelRef> {
        Some(self.vectors.model().clone())
    }

    fn execution(&self) -> Option<Execution> {
        Some(self.vectors.execution().clone())
    }

    fn config_hash(&self) -> u64 {
        let mut buf = Vec::with_capacity(41);
        buf.extend_from_slice(&self.vectors.config(Which::Sentences).to_le_bytes());
        buf.extend_from_slice(&self.vectors.token_config().to_le_bytes());
        buf.extend_from_slice(&self.bio_head.config_hash().to_le_bytes());
        buf.extend_from_slice(&self.polarity_head.config_hash().to_le_bytes());
        if let Some(c) = &self.category_centroids {
            buf.extend_from_slice(&c.config_hash().to_le_bytes());
        }
        buf.push(u8::from(self.emit_implicit_aspects));
        buf.extend_from_slice(&self.bio_threshold.to_bits().to_le_bytes());
        fnv1a(&buf)
    }
}

impl LearnedAspectSentiment {
    fn validate_views(
        (view, sentences, tokens, subwords, _, _, phrase_chunks): Refs<'_, Self>,
        bio_threshold: f64,
    ) -> Result<(), Error> {
        let fail = |detail: String| Error::Annotator { annotator: Self::ID, detail };
        if !bio_threshold.is_finite() {
            return Err(fail("BIO threshold must be finite".into()));
        }
        if sentences.spans.view() != view.id() {
            return Err(fail("sentences are not in the given view".into()));
        }
        if tokens.spans.view() != view.id() {
            return Err(fail("tokens are not in the given view".into()));
        }
        if subwords.spans.view() != view.id() {
            return Err(fail("subwords are not in the given view".into()));
        }
        if phrase_chunks.is_some_and(|pc| pc.spans.view() != view.id()) {
            return Err(fail("phrase chunks are not in the given view".into()));
        }

        Ok(())
    }
}
