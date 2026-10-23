use turbo_analysis::{Annotator, Error, Ix, Layer, ModelRef, Refs, View};
use turbo_analysis_layers::{Chunks, EmotionItem, EmotionLayer, EmotionTarget, Sentences, Tokens, Vectors};

pub(super) use super::super::token_key::token_key as key;
use super::super::validation;
use super::{EmotionAnnotator, EmotionStage, EmotionUnit};

impl Annotator for EmotionAnnotator {
    const ID: &'static str = "turbo-doccat:emotion";
    const VERSION: &'static str = "1";
    type Reads = (View, Tokens, Option<Sentences>, Option<Chunks>, Option<Vectors<Sentences>>, Option<Vectors<Chunks>>);
    type Writes = (EmotionLayer,);

    fn annotate(
        &self,
        (view, tokens, sentences, chunks, sent_vecs, chunk_vecs): Refs<'_, Self>,
    ) -> Result<Self::Writes, Error> {
        let fail = |detail: String| Error::Annotator { annotator: Self::ID, detail };
        let text = view.text();
        if tokens.spans.view() != view.id()
            || sentences.is_some_and(|s| s.spans.view() != view.id())
            || chunks.is_some_and(|c| c.spans.view() != view.id())
        {
            return Err(fail("tokens, sentences or chunks are in another view".into()));
        }
        validation::tokens(Self::ID, tokens, sentences, text)?;
        if let Some(c) = chunks {
            validation::checked(Self::ID, c)?;
            validation::contiguous(Self::ID, &c.spans, text)?;
        }
        let reads_vectors = self.stage != EmotionStage::Lexicon;
        if reads_vectors && self.vector_scorer.is_none() {
            return Err(fail("the learned and hybrid stages need a vector scorer".into()));
        }
        let encoder = self.vector_scorer.as_ref().filter(|_| reads_vectors).and_then(|s| s.model());
        let check_encoder = |m: &ModelRef| match &encoder {
            Some(want) if want != m => Err(fail(format!(
                "vectors are from model {} (revision {}), the head is for {} (revision {})",
                m.id, m.revision, want.id, want.revision
            ))),
            _ => Ok(()),
        };
        let (starts, ends) = tokens.spans.columns();
        let keys: Vec<String> = (0..tokens.len()).map(|t| key(&text[starts[t] as usize..ends[t] as usize])).collect();
        // Each unit's byte span, target and vector.
        let units: Vec<(u32, u32, EmotionTarget, Option<&[f32]>)> = match self.unit {
            EmotionUnit::Sentences => {
                let s = sentences.ok_or_else(|| fail("EmotionUnit::Sentences needs a Sentences layer".into()))?;
                if let Some(v) = sent_vecs.filter(|_| reads_vectors) {
                    check_encoder(&v.model)?;
                    validation::vectors(Self::ID, v, s.len())?;
                }
                let (a, b) = s.spans.columns();
                (0..s.len())
                    .map(|i| {
                        let ix = Ix::<Sentences>::new(i as u32);
                        let vec = sent_vecs.filter(|v| !v.is_empty(ix)).map(|v| v.row(ix));
                        (a[i], b[i], EmotionTarget::Sentence(ix), vec)
                    })
                    .collect()
            }
            EmotionUnit::Chunks => {
                let c = chunks.ok_or_else(|| fail("EmotionUnit::Chunks needs a Chunks layer".into()))?;
                if let Some(v) = chunk_vecs.filter(|_| reads_vectors) {
                    check_encoder(&v.model)?;
                    validation::vectors(Self::ID, v, c.len())?;
                }
                let (a, b) = c.spans.columns();
                (0..c.len())
                    .map(|i| {
                        let ix = Ix::<Chunks>::new(i as u32);
                        let vec = chunk_vecs.filter(|v| !v.is_empty(ix)).map(|v| v.row(ix));
                        (a[i], b[i], EmotionTarget::Chunk(ix), vec)
                    })
                    .collect()
            }
        };
        let mut out = EmotionLayer::new();
        for (i, (from, to, target, vec)) in units.into_iter().enumerate() {
            let start = ends.partition_point(|&e| e <= from);
            let end = starts.partition_point(|&s| s < to).max(start);
            let (scores, evidence) =
                self.score_unit(&keys, start, end, vec).map_err(|e| fail(format!("unit {i}: {e}")))?;
            out.try_push(tokens, sentences, chunks, EmotionItem { target, scores, evidence: &evidence })
                .map_err(fail)?;
        }
        Ok((out,))
    }

    fn model(&self) -> Option<ModelRef> {
        Some(self.model_ref())
    }

    fn config_hash(&self) -> u64 {
        self.config()
    }
}
