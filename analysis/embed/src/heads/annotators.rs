//! Unified [`ScoringHead`] and vector/span-pooling head annotators.

use std::marker::PhantomData;
use std::sync::Arc;

use turbo_analysis::{Annotator, Error, Execution, Ix, Layer, ModelRef, Refs, Span, Tag, TagSet};
use turbo_analysis_layers::{
    Aspect, Aspects, Categories, CategoryScope, Chunks, Incivility, Opinions, PolarityScores, Pragmatics, Sentences,
    Subwords, Tokens, Vectors,
};

use super::ClassifierHead;
use crate::centroids::LabelCentroids;
use crate::vectors::VectorAnnotator;
use crate::window::Which;

/// Unified scoring head wrapping either a fixed-weight [`ClassifierHead`] or
/// a prototype [`LabelCentroids`] classifier.
#[derive(Clone, Debug)]
pub enum ScoringHead {
    Classifier(Arc<ClassifierHead>),
    Centroids(Arc<LabelCentroids>),
}

impl From<Arc<ClassifierHead>> for ScoringHead {
    fn from(h: Arc<ClassifierHead>) -> Self {
        ScoringHead::Classifier(h)
    }
}

impl From<ClassifierHead> for ScoringHead {
    fn from(h: ClassifierHead) -> Self {
        ScoringHead::Classifier(Arc::new(h))
    }
}

impl From<Arc<LabelCentroids>> for ScoringHead {
    fn from(c: Arc<LabelCentroids>) -> Self {
        ScoringHead::Centroids(c)
    }
}

impl From<LabelCentroids> for ScoringHead {
    fn from(c: LabelCentroids) -> Self {
        ScoringHead::Centroids(Arc::new(c))
    }
}

impl ScoringHead {
    pub fn model(&self) -> &ModelRef {
        match self {
            ScoringHead::Classifier(h) => h.model(),
            ScoringHead::Centroids(c) => c.model(),
        }
    }

    pub fn in_dim(&self) -> u32 {
        match self {
            ScoringHead::Classifier(h) => h.in_dim(),
            ScoringHead::Centroids(c) => c.dim(),
        }
    }

    pub fn tags(&self) -> &Arc<TagSet> {
        match self {
            ScoringHead::Classifier(h) => h.tags(),
            ScoringHead::Centroids(c) => c.tags(),
        }
    }

    pub fn config_hash(&self) -> u64 {
        match self {
            ScoringHead::Classifier(h) => h.config_hash(),
            ScoringHead::Centroids(c) => c.config_hash() ^ 0x9e37_79b9_7f4a_7c15,
        }
    }

    /// Checks that `vectors` come from this head's encoder, by `ModelRef`,
    /// and have its input dimension: a different encoder of the same width
    /// is a different feature space.
    pub fn check_vectors<L: Layer>(&self, annotator: &'static str, vectors: &Vectors<L>) -> Result<(), Error> {
        let fail = |detail: String| Error::Annotator { annotator, detail };
        let model = self.model();
        if vectors.model != *model {
            return Err(fail(format!(
                "vectors are from model {} (revision {}), head is for {} (revision {})",
                vectors.model.id, vectors.model.revision, model.id, model.revision
            )));
        }
        if vectors.dim != self.in_dim() {
            return Err(fail(format!("vectors have dim {}, head expects dim {}", vectors.dim, self.in_dim())));
        }
        Ok(())
    }

    /// Scores `vector` into `scores` (`f64` probabilities or multi-label
    /// scores) and returns the top `(Tag, score)`.
    pub fn score(&self, vector: &[f32], scores: &mut [f64]) -> Result<(Tag, f64), String> {
        match self {
            ScoringHead::Classifier(h) => h.score(vector, scores),
            ScoringHead::Centroids(c) => c.score_single_label(vector, scores),
        }
    }

    /// Scores every item of `vectors` (`opennlp:embeddings<L>`) into
    /// [`Categories<L>`] (`turbo:categories<L>`).
    pub fn categorize_vectors<L: CategoryScope>(
        &self,
        annotator: &'static str,
        vectors: &Vectors<L>,
    ) -> Result<Categories<L>, Error> {
        match self {
            ScoringHead::Classifier(h) => h.categorize_vectors(annotator, vectors),
            ScoringHead::Centroids(c) => c.categorize_vectors(annotator, vectors),
        }
    }

    /// Converts a probability distribution `probs` over this head's `TagSet`
    /// into three-way [`PolarityScores`] (`positive`, `negative`, `neutral`).
    pub fn polarity_scores_from_probs(&self, probs: &[f64]) -> PolarityScores {
        let mut pos = 0.0f64;
        let mut neg = 0.0f64;
        let mut neu = 0.0f64;
        let tags = self.tags();
        for (i, &p) in probs.iter().enumerate() {
            let label = tags.label(Tag(i as u16));
            if label.eq_ignore_ascii_case("positive") || label.eq_ignore_ascii_case("pos") {
                pos += p;
            } else if label.eq_ignore_ascii_case("negative") || label.eq_ignore_ascii_case("neg") {
                neg += p;
            } else if label.eq_ignore_ascii_case("neutral") || label.eq_ignore_ascii_case("neu") {
                neu += p;
            }
        }
        PolarityScores::new(pos, neg, neu)
    }
}

/// Annotator step that reads `Vectors<L>` (`opennlp:embeddings<L>`) and
/// writes `Categories<L>` (`turbo:categories<L>`) using a [`ScoringHead`].
#[derive(Clone)]
pub struct VectorHeadAnnotator<L: CategoryScope> {
    pub head: ScoringHead,
    _marker: PhantomData<fn() -> L>,
}

impl<L: CategoryScope> VectorHeadAnnotator<L> {
    pub fn new(head: impl Into<ScoringHead>) -> Self {
        VectorHeadAnnotator { head: head.into(), _marker: PhantomData }
    }
}

impl<L: CategoryScope> Annotator for VectorHeadAnnotator<L> {
    const ID: &'static str = "turbo:vector-head";
    type Reads = (L, Vectors<L>);
    type Writes = (Categories<L>,);

    fn annotate(&self, (scope, vectors): Refs<'_, Self>) -> Result<Self::Writes, Error> {
        if vectors.len() != scope.len() {
            return Err(Error::Annotator {
                annotator: Self::ID,
                detail: format!("vectors has {} items, scope has {}", vectors.len(), scope.len()),
            });
        }
        Ok((self.head.categorize_vectors(Self::ID, vectors)?,))
    }

    fn model(&self) -> Option<ModelRef> {
        Some(self.head.model().clone())
    }

    fn config_hash(&self) -> u64 {
        self.head.config_hash()
    }
}

/// Span-pooling head annotator over `Sentences`: pools each sentence's
/// subword range from `Subwords` via [`VectorAnnotator::embed_spans`] (`#263`)
/// and writes `Categories<Sentences>`.
#[derive(Clone)]
pub struct SentenceSpanHead {
    pub vectors: Arc<VectorAnnotator>,
    pub head: ScoringHead,
}

impl SentenceSpanHead {
    pub fn new(vectors: Arc<VectorAnnotator>, head: impl Into<ScoringHead>) -> Self {
        SentenceSpanHead { vectors, head: head.into() }
    }
}

impl Annotator for SentenceSpanHead {
    const ID: &'static str = "turbo:sentence-span-head";
    type Reads = (Subwords, Sentences);
    type Writes = (Categories<Sentences>,);

    fn annotate(&self, (subwords, sentences): Refs<'_, Self>) -> Result<Self::Writes, Error> {
        let spans: Vec<Span> = (0..sentences.len()).map(|i| sentences.spans.get(Ix::new(i as u32))).collect();
        let pooled: Vectors<Sentences> =
            self.vectors.embed_spans(Self::ID, subwords, Some(&sentences.spans), sentences.spans.view(), &spans)?;
        Ok((self.head.categorize_vectors(Self::ID, &pooled)?,))
    }

    fn model(&self) -> Option<ModelRef> {
        Some(self.vectors.model().clone())
    }

    fn execution(&self) -> Option<Execution> {
        Some(self.vectors.execution().clone())
    }

    fn config_hash(&self) -> u64 {
        self.vectors.config(Which::Sentences) ^ self.head.config_hash()
    }
}

/// Span-pooling head annotator over `Chunks`: pools each chunk's subword
/// range from `Subwords` via [`VectorAnnotator::embed_spans`] (`#263`) and
/// writes `Categories<Chunks>`.
#[derive(Clone)]
pub struct ChunkSpanHead {
    pub vectors: Arc<VectorAnnotator>,
    pub head: ScoringHead,
}

impl ChunkSpanHead {
    pub fn new(vectors: Arc<VectorAnnotator>, head: impl Into<ScoringHead>) -> Self {
        ChunkSpanHead { vectors, head: head.into() }
    }
}

impl Annotator for ChunkSpanHead {
    const ID: &'static str = "turbo:chunk-span-head";
    type Reads = (Subwords, Option<Sentences>, Chunks);
    type Writes = (Categories<Chunks>,);

    fn annotate(&self, (subwords, sentences, chunks): Refs<'_, Self>) -> Result<Self::Writes, Error> {
        let spans: Vec<Span> = (0..chunks.len()).map(|i| chunks.spans.get(Ix::new(i as u32))).collect();
        let pooled: Vectors<Chunks> =
            self.vectors.embed_spans(Self::ID, subwords, sentences.map(|s| &s.spans), chunks.spans.view(), &spans)?;
        Ok((self.head.categorize_vectors(Self::ID, &pooled)?,))
    }

    fn model(&self) -> Option<ModelRef> {
        Some(self.vectors.model().clone())
    }

    fn execution(&self) -> Option<Execution> {
        Some(self.vectors.execution().clone())
    }

    fn config_hash(&self) -> u64 {
        self.vectors.config(Which::Chunks) ^ self.head.config_hash()
    }
}

/// Span-pooling head annotator over `Opinions` (`turbo:opinions`): pools the
/// exact subword range of each opinion expression (`opinions.span(i, tokens)`),
/// never the whole sentence, from the sentence windows via `#263`
/// ([`VectorAnnotator::embed_spans`]), and writes `Categories<Opinions>`.
#[derive(Clone)]
pub struct OpinionSpanHead {
    pub vectors: Arc<VectorAnnotator>,
    pub head: ScoringHead,
}

impl OpinionSpanHead {
    pub fn new(vectors: Arc<VectorAnnotator>, head: impl Into<ScoringHead>) -> Self {
        OpinionSpanHead { vectors, head: head.into() }
    }

    /// Pools the exact subword range of every opinion span in `opinions`
    /// within the sentence windows (when `sentences` is `Some`).
    pub fn pool_opinions(
        &self,
        subwords: &Subwords,
        tokens: &Tokens,
        sentences: Option<&Sentences>,
        opinions: &Opinions,
    ) -> Result<Vectors<Opinions>, Error> {
        let spans: Vec<Span> = (0..opinions.len()).map(|i| opinions.span(Ix::new(i as u32), tokens)).collect();
        self.vectors.embed_spans(Self::ID, subwords, sentences.map(|s| &s.spans), tokens.spans.view(), &spans)
    }
}

impl Annotator for OpinionSpanHead {
    const ID: &'static str = "turbo:opinion-span-head";
    type Reads = (Subwords, Tokens, Option<Sentences>, Opinions);
    type Writes = (Categories<Opinions>,);

    fn annotate(&self, (subwords, tokens, sentences, opinions): Refs<'_, Self>) -> Result<Self::Writes, Error> {
        let pooled = self.pool_opinions(subwords, tokens, sentences, opinions)?;
        Ok((self.head.categorize_vectors(Self::ID, &pooled)?,))
    }

    fn model(&self) -> Option<ModelRef> {
        Some(self.vectors.model().clone())
    }

    fn execution(&self) -> Option<Execution> {
        Some(self.vectors.execution().clone())
    }

    fn config_hash(&self) -> u64 {
        self.vectors.config(Which::Sentences) ^ self.head.config_hash()
    }
}

/// Span-pooling head annotator over `Aspects` (`turbo:aspects`): pools the
/// exact subword range of each aspect term (`aspects.span(i, tokens, sentences)`,
/// which is the term's token range for explicit aspects or the sentence span
/// for implicit aspects) from the sentence windows via `#263`
/// ([`VectorAnnotator::embed_spans`]), and writes `Categories<Aspects>`.
#[derive(Clone)]
pub struct AspectSpanHead {
    pub vectors: Arc<VectorAnnotator>,
    pub head: ScoringHead,
}

impl AspectSpanHead {
    pub fn new(vectors: Arc<VectorAnnotator>, head: impl Into<ScoringHead>) -> Self {
        AspectSpanHead { vectors, head: head.into() }
    }

    /// Pools the exact subword range of every aspect in `aspects` within the
    /// sentence windows.
    pub fn pool_aspects(
        &self,
        subwords: &Subwords,
        tokens: &Tokens,
        sentences: Option<&Sentences>,
        aspects: &Aspects,
    ) -> Result<Vectors<Aspects>, Error> {
        let mut spans = Vec::with_capacity(aspects.len());
        for i in 0..aspects.len() {
            let ix = Ix::<Aspects>::new(i as u32);
            let span = aspects.span(ix, tokens, sentences).ok_or_else(|| Error::Annotator {
                annotator: Self::ID,
                detail: format!("implicit aspect {i} requires a Sentences layer to resolve its span"),
            })?;
            spans.push(span);
        }
        self.vectors.embed_spans(Self::ID, subwords, sentences.map(|s| &s.spans), tokens.spans.view(), &spans)
    }

    /// Pools each aspect's exact subword range and populates its three-way
    /// [`PolarityScores`] (`positive`, `negative`, `neutral`) from `self.head`.
    pub fn score_aspects_polarity(
        &self,
        subwords: &Subwords,
        tokens: &Tokens,
        sentences: Option<&Sentences>,
        opinions: Option<&Opinions>,
        aspects: &Aspects,
    ) -> Result<Aspects, Error> {
        let fail = |detail: String| Error::Annotator { annotator: Self::ID, detail };
        let pooled = self.pool_aspects(subwords, tokens, sentences, aspects)?;
        self.head.check_vectors(Self::ID, &pooled)?;
        let n_labels = self.head.tags().len();
        let mut probs = vec![0.0f64; n_labels];
        let mut out = Aspects::on(
            aspects.tokens_key,
            aspects.sentences_key,
            aspects.opinions_key,
            Arc::clone(&aspects.categories),
        );
        for i in 0..aspects.len() {
            let ix = Ix::<Aspects>::new(i as u32);
            let cur = aspects.get(ix);
            let polarity = if pooled.is_empty(ix) {
                PolarityScores::new(0.0, 0.0, 1.0)
            } else {
                self.head.score(pooled.row(ix), &mut probs).map_err(fail)?;
                self.head.polarity_scores_from_probs(&probs)
            };
            out.try_push(
                tokens,
                sentences,
                opinions,
                Aspect { target: cur.target, category: cur.category, polarity, opinions: cur.opinions },
            )
            .map_err(fail)?;
        }
        Ok(out)
    }
}

impl Annotator for AspectSpanHead {
    const ID: &'static str = "turbo:aspect-span-head";
    type Reads = (Subwords, Tokens, Option<Sentences>, Aspects);
    type Writes = (Categories<Aspects>,);

    fn annotate(&self, (subwords, tokens, sentences, aspects): Refs<'_, Self>) -> Result<Self::Writes, Error> {
        let pooled = self.pool_aspects(subwords, tokens, sentences, aspects)?;
        Ok((self.head.categorize_vectors(Self::ID, &pooled)?,))
    }

    fn model(&self) -> Option<ModelRef> {
        Some(self.vectors.model().clone())
    }

    fn execution(&self) -> Option<Execution> {
        Some(self.vectors.execution().clone())
    }

    fn config_hash(&self) -> u64 {
        self.vectors.config(Which::Sentences) ^ self.head.config_hash()
    }
}

/// Multi-label incivility annotator (`turbo:incivility-head`) reading
/// `(Tokens, Sentences, Vectors<Sentences>)` and writing [`Incivility`]
/// (`turbo:incivility`).
#[derive(Clone)]
pub struct IncivilityHeadAnnotator(pub Arc<ClassifierHead>);

impl Annotator for IncivilityHeadAnnotator {
    const ID: &'static str = "turbo:incivility-head";
    type Reads = (Tokens, Sentences, Vectors<Sentences>);
    type Writes = (Incivility,);

    fn annotate(&self, (tokens, sentences, vectors): Refs<'_, Self>) -> Result<Self::Writes, Error> {
        Ok((self.0.score_incivility_sentences(Self::ID, tokens, sentences, vectors)?,))
    }

    fn model(&self) -> Option<ModelRef> {
        Some(self.0.model().clone())
    }

    fn config_hash(&self) -> u64 {
        self.0.config_hash()
    }
}

/// Sentence-level sarcasm and irony head annotator (`turbo:pragmatics-head`)
/// reading `(Tokens, Sentences, Vectors<Sentences>)` and writing
/// [`Pragmatics`] (`turbo:pragmatics`) with `evidence: None` (`#324`).
#[derive(Clone)]
pub struct PragmaticsHeadAnnotator(pub Arc<ClassifierHead>);

impl Annotator for PragmaticsHeadAnnotator {
    const ID: &'static str = "turbo:pragmatics-head";
    type Reads = (Tokens, Sentences, Vectors<Sentences>);
    type Writes = (Pragmatics,);

    fn annotate(&self, (tokens, sentences, vectors): Refs<'_, Self>) -> Result<Self::Writes, Error> {
        Ok((self.0.score_pragmatics_sentences(Self::ID, sentences, tokens, vectors)?,))
    }

    fn model(&self) -> Option<ModelRef> {
        Some(self.0.model().clone())
    }

    fn config_hash(&self) -> u64 {
        self.0.config_hash()
    }
}
