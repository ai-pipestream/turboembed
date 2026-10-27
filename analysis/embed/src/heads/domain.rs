//! Domain-specific scoring methods on [`super::ClassifierHead`] for
//! [`Incivility`] (`turbo:incivility`) and [`Pragmatics`] (`turbo:pragmatics`).

use turbo_analysis::{Error, Ix, Layer, ModelRef, Tag};
use turbo_analysis_layers::{
    Chunks, Incivility, IncivilityItem, IncivilityLabel, IncivilityScores, IncivilityTarget, Pragmatics,
    PragmaticsItem, Sentences, Tokens, Vectors,
};

use super::{ClassifierHead, HeadMode};

impl ClassifierHead {
    /// Maps this head's labels to [`IncivilityLabel`]s when configured as a
    /// [`HeadMode::MultiLabel`] incivility head.
    fn incivility_label_map(&self, annotator: &'static str) -> Result<Vec<IncivilityLabel>, Error> {
        let fail = |detail: String| Error::Annotator { annotator, detail };
        if self.mode != HeadMode::MultiLabel {
            return Err(fail("incivility head must have HeadMode::MultiLabel".into()));
        }
        let mut map = Vec::with_capacity(self.tags.len());
        for i in 0..self.tags.len() {
            let label = self.tags.label(Tag(i as u16));
            let parsed = IncivilityLabel::from_name(label)
                .ok_or_else(|| fail(format!("head label {label:?} is not a valid IncivilityLabel")))?;
            map.push(parsed);
        }
        Ok(map)
    }

    /// Scores a single embedding vector `vector` (`&[f32]`) with a
    /// [`HeadMode::MultiLabel`] incivility head and returns its six
    /// [`IncivilityScores`] (`0.0..=1.0`).
    pub fn score_incivility_vector(&self, vector: &[f32]) -> Result<IncivilityScores, String> {
        if self.mode != HeadMode::MultiLabel {
            return Err("incivility head must have HeadMode::MultiLabel".into());
        }
        let mut buf = vec![0.0f64; self.tags.len()];
        self.score(vector, &mut buf)?;
        let mut out = IncivilityScores::default();
        for (i, &val) in buf.iter().enumerate() {
            let label = self.tags.label(Tag(i as u16));
            let parsed = IncivilityLabel::from_name(label)
                .ok_or_else(|| format!("head label {label:?} is not a valid IncivilityLabel"))?;
            out.set(parsed, val);
        }
        Ok(out)
    }

    /// Scores `vectors` (`Vectors<Sentences>`) with a multi-label incivility
    /// head and writes [`Incivility`] (`turbo:incivility`).
    pub fn score_incivility_sentences(
        &self,
        annotator: &'static str,
        tokens: &Tokens,
        sentences: &Sentences,
        vectors: &Vectors<Sentences>,
    ) -> Result<Incivility, Error> {
        let fail = |detail: String| Error::Annotator { annotator, detail };
        if tokens.spans.view() != sentences.spans.view() {
            return Err(fail(format!(
                "the tokens are in view {}, the sentences in view {}",
                tokens.spans.view().0,
                sentences.spans.view().0
            )));
        }
        if vectors.model != self.model {
            return Err(fail(format!(
                "vectors are from model {} (revision {}), head is for {} (revision {})",
                vectors.model.id, vectors.model.revision, self.model.id, self.model.revision
            )));
        }
        if vectors.dim != self.in_dim() {
            return Err(fail(format!("vectors have dim {}, head expects dim {}", vectors.dim, self.in_dim())));
        }
        if vectors.len() != sentences.len() {
            return Err(fail(format!(
                "sentence vectors has {} items, sentences has {}",
                vectors.len(),
                sentences.len()
            )));
        }
        let label_map = self.incivility_label_map(annotator)?;
        let mut out = Incivility::new();
        let mut buf = vec![0.0f64; self.tags.len()];
        for i in 0..sentences.len() {
            let ix = Ix::<Sentences>::new(i as u32);
            let mut item_scores = IncivilityScores::default();
            if !vectors.is_empty(ix) {
                self.score(vectors.row(ix), &mut buf).map_err(fail)?;
                for (&lbl, &val) in label_map.iter().zip(&buf) {
                    item_scores.set(lbl, val);
                }
            }
            out.try_push(
                tokens,
                Some(sentences),
                None,
                None,
                IncivilityItem { target: IncivilityTarget::Sentence(ix), scores: item_scores, evidence: &[] },
            )
            .map_err(fail)?;
        }
        Ok(out)
    }

    /// Scores `vectors` (`Vectors<Chunks>`) with a multi-label incivility head
    /// and writes [`Incivility`] (`turbo:incivility`).
    pub fn score_incivility_chunks(
        &self,
        annotator: &'static str,
        tokens: &Tokens,
        chunks: &Chunks,
        vectors: &Vectors<Chunks>,
    ) -> Result<Incivility, Error> {
        let fail = |detail: String| Error::Annotator { annotator, detail };
        if tokens.spans.view() != chunks.spans.view() {
            return Err(fail(format!(
                "the tokens are in view {}, the chunks in view {}",
                tokens.spans.view().0,
                chunks.spans.view().0
            )));
        }
        if vectors.model != self.model {
            return Err(fail(format!(
                "vectors are from model {} (revision {}), head is for {} (revision {})",
                vectors.model.id, vectors.model.revision, self.model.id, self.model.revision
            )));
        }
        if vectors.dim != self.in_dim() {
            return Err(fail(format!("vectors have dim {}, head expects dim {}", vectors.dim, self.in_dim())));
        }
        if vectors.len() != chunks.len() {
            return Err(fail(format!("chunk vectors has {} items, chunks has {}", vectors.len(), chunks.len())));
        }
        let label_map = self.incivility_label_map(annotator)?;
        let mut out = Incivility::new();
        let mut buf = vec![0.0f64; self.tags.len()];
        for i in 0..chunks.len() {
            let ix = Ix::<Chunks>::new(i as u32);
            let mut item_scores = IncivilityScores::default();
            if !vectors.is_empty(ix) {
                self.score(vectors.row(ix), &mut buf).map_err(fail)?;
                for (&lbl, &val) in label_map.iter().zip(&buf) {
                    item_scores.set(lbl, val);
                }
            }
            out.try_push(
                tokens,
                None,
                Some(chunks),
                None,
                IncivilityItem { target: IncivilityTarget::Chunk(ix), scores: item_scores, evidence: &[] },
            )
            .map_err(fail)?;
        }
        Ok(out)
    }

    /// Scores `vectors` (`Vectors<Sentences>`) with a sarcasm/irony head and
    /// writes [`Pragmatics`] (`turbo:pragmatics`) with `evidence: None`,
    /// recording that the score came from the embedding head alone (`#324`).
    pub fn score_pragmatics_sentences(
        &self,
        annotator: &'static str,
        sentences: &Sentences,
        tokens: &Tokens,
        vectors: &Vectors<Sentences>,
    ) -> Result<Pragmatics, Error> {
        let fail = |detail: String| Error::Annotator { annotator, detail };
        if tokens.spans.view() != sentences.spans.view() {
            return Err(fail(format!(
                "the tokens are in view {}, the sentences in view {}",
                tokens.spans.view().0,
                sentences.spans.view().0
            )));
        }
        if vectors.model != self.model {
            return Err(fail(format!(
                "vectors are from model {} (revision {}), head is for {} (revision {})",
                vectors.model.id, vectors.model.revision, self.model.id, self.model.revision
            )));
        }
        if vectors.dim != self.in_dim() {
            return Err(fail(format!("vectors have dim {}, head expects dim {}", vectors.dim, self.in_dim())));
        }
        if vectors.len() != sentences.len() {
            return Err(fail(format!(
                "sentence vectors has {} items, sentences has {}",
                vectors.len(),
                sentences.len()
            )));
        }
        let pos_idx = self
            .tags
            .tag("sarcastic")
            .or_else(|| self.tags.tag("ironic"))
            .or_else(|| self.tags.tag("sarcasm"))
            .ok_or_else(|| fail("pragmatics head has no \"sarcastic\", \"ironic\" or \"sarcasm\" label".into()))?
            .0 as usize;
        let mut out = Pragmatics::new();
        let mut buf = vec![0.0f64; self.tags.len()];
        for i in 0..sentences.len() {
            let ix = Ix::<Sentences>::new(i as u32);
            let score = if vectors.is_empty(ix) {
                0.0
            } else {
                self.score(vectors.row(ix), &mut buf).map_err(fail)?;
                buf[pos_idx].clamp(0.0, 1.0)
            };
            out.try_push(sentences, tokens, PragmaticsItem { score, evidence: None }).map_err(fail)?;
        }
        Ok(out)
    }

    /// Scores one sentence embedding as sarcasm or irony.
    pub fn score_pragmatics_vector(&self, vector: &[f32]) -> Result<f64, String> {
        if vector.len() != self.in_dim() as usize {
            return Err(format!("vector has dim {}, head expects dim {}", vector.len(), self.in_dim()));
        }
        let pos_idx = self
            .tags
            .tag("sarcastic")
            .or_else(|| self.tags.tag("ironic"))
            .or_else(|| self.tags.tag("sarcasm"))
            .ok_or_else(|| "pragmatics head has no sarcasm or irony label".to_owned())?
            .0 as usize;
        let mut buf = vec![0.0f64; self.tags.len()];
        self.score(vector, &mut buf)?;
        Ok(buf[pos_idx].clamp(0.0, 1.0))
    }
}

impl turbo_doccat::sentiment::pragmatics::PragmaticsVectorScorer for ClassifierHead {
    fn score_sentence_vector(&self, vec: &[f32]) -> Result<f64, String> {
        self.score_pragmatics_vector(vec)
    }

    fn config_hash(&self) -> u64 {
        ClassifierHead::config_hash(self)
    }

    fn model_ref(&self) -> Option<ModelRef> {
        Some(self.model().clone())
    }
}
