//! Composable quality filtering, preserving chunk columns and ancestry.

use turbo_analysis::{Annotator, Error, Ix, Layer, Refs, Spans};
use turbo_analysis_layers::{Chunks, QualityFlags, Sentences, SpanQuality};

#[derive(Debug, Clone, Copy)]
pub struct QualityChunkFilter {
    min_score: f32,
    forbidden: QualityFlags,
}

impl QualityChunkFilter {
    pub fn new(min_score: f32, forbidden: QualityFlags) -> Result<Self, String> {
        if !min_score.is_finite() || !(0.0..=1.0).contains(&min_score) || forbidden.0 & !QualityFlags::ALL.0 != 0 {
            return Err("invalid chunk quality floor or forbidden flags".into());
        }
        Ok(Self { min_score, forbidden })
    }
}

impl Annotator for QualityChunkFilter {
    const ID: &'static str = "turbo:quality-chunk-filter";
    type Reads = (Chunks, SpanQuality, Sentences);
    type Writes = (Chunks,);

    fn annotate(&self, (chunks, quality, sentences): Refs<'_, Self>) -> Result<Self::Writes, Error> {
        let fail = |detail| Error::Annotator { annotator: Self::ID, detail };
        chunks.check().map_err(fail)?;
        if chunks.spans.view() != sentences.spans.view()
            || chunks.sentences.iter().any(|r| r.end.get() > sentences.len())
        {
            return Err(fail("chunk sentence ranges must resolve in the same view".into()));
        }
        let rejected = quality.rejected(&chunks.spans, self.min_score, self.forbidden).map_err(fail)?;
        let mut out = Chunks {
            spans: Spans::new(chunks.spans.view()),
            level: Vec::new(),
            parent: Vec::new(),
            sentences: Vec::new(),
            cut: Vec::new(),
            cost: Vec::new(),
            model_tokens: Vec::new(),
            over_budget: Vec::new(),
            prompt_overhead: chunks.prompt_overhead,
        };
        // Nearest retained ancestor, also for a rejected intermediate row.
        let mut nearest = Vec::with_capacity(chunks.len());
        for (i, &reject) in rejected.iter().enumerate() {
            let parent = chunks.parent[i].and_then(|p| nearest[p.get()]);
            if reject {
                nearest.push(parent);
                continue;
            }
            let old = Ix::new(i as u32);
            let new = if chunks.spans.is_discontinuous(old) {
                out.spans.push_fragments(&chunks.spans.fragments(old).collect::<Vec<_>>())
            } else {
                out.spans.push(chunks.spans.get(old))
            };
            out.level.push(chunks.level[i]);
            out.parent.push(parent);
            out.sentences.push(chunks.sentences[i]);
            out.cut.push(chunks.cut[i]);
            out.cost.push(chunks.cost[i]);
            out.model_tokens.push(chunks.model_tokens[i]);
            out.over_budget.push(chunks.over_budget[i]);
            nearest.push(Some(new));
        }
        out.check().map_err(fail)?;
        Ok((out,))
    }

    fn config_hash(&self) -> u64 {
        (u64::from(self.min_score.to_bits()) << 16) | u64::from(self.forbidden.0)
    }
}
