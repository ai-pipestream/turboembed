//! Native GLiNER markerV0 inference on the shared DeBERTa-v3 encoder.
//! No external inference engine. See docs/gliner.md for the supported profile.
mod codec;
mod config;
mod heads;
mod output;
pub mod package;
mod packing;
mod schema;
mod window;
pub use codec::{codecs, register_codecs};
pub use config::Config;
pub use output::{Decisions, GlinerAnnotator};
pub use packing::PackedInput;
pub use schema::{Decode, ExtractionSchema};
use turbo::{deberta::Encoder, sentencepiece::SentencePiece};
use turbo_analysis::ModelRef;

#[derive(Clone, Debug, PartialEq)]
pub struct Candidate {
    pub start: usize,
    pub end: usize,
    pub label: usize,
    pub logit: f32,
    pub score: f32,
}
/// Immutable checkpoint, safe to share between documents. Workspace is per call.
pub struct Model {
    encoder: Encoder,
    heads: heads::Heads,
    config: Config,
    tokenizer: SentencePiece,
    identity: ModelRef,
}
impl Model {
    pub fn new(
        encoder: Encoder,
        config: Config,
        heads: &[u8],
        sentencepiece: &[u8],
        identity: ModelRef,
    ) -> Result<Self, String> {
        if heads.len() > 256 << 20 || sentencepiece.len() > 16 << 20 {
            return Err("gliner package byte limit exceeded".into());
        }
        config.check(encoder.config())?;
        let heads = heads::Heads::load(&config, encoder.config(), heads)?;
        let tokenizer = SentencePiece::parse(sentencepiece).map_err(|e| e.message)?;
        if tokenizer.len() > encoder.config().vocabulary {
            return Err("gliner tokenizer exceeds encoder vocabulary".into());
        }
        Ok(Self { encoder, heads, config, tokenizer, identity })
    }
    pub fn config(&self) -> &Config {
        &self.config
    }
    pub fn identity(&self) -> &ModelRef {
        &self.identity
    }
    pub fn pack(&self, words: &[&str], schema: &ExtractionSchema) -> Result<PackedInput, String> {
        packing::pack(&self.tokenizer, &self.config, self.encoder.config(), words, schema)
    }
    pub fn predict(&self, words: &[&str], schema: &ExtractionSchema) -> Result<Vec<Candidate>, String> {
        let p = self.pack(words, schema)?;
        Ok(decode(self.score_packed(&p, schema)?, schema))
    }
    /// All valid span/label pairs in start, width, label order. Positions are
    /// first-subword positions; prompts use the ENT marker hidden state.
    pub fn score_packed(&self, p: &PackedInput, schema: &ExtractionSchema) -> Result<Vec<Candidate>, String> {
        p.check(&self.config, self.encoder.config(), schema.labels().len())?;
        let h = self.encoder.encode(&p.ids, &vec![true; p.ids.len()]).map_err(|e| e.message)?;
        self.heads.score(h.data(), h.width(), &p.prompts, &p.words, self.config.max_width)
    }
    pub(crate) fn score_packed_with_work(
        &self,
        p: &PackedInput,
        schema: &ExtractionSchema,
        work: &mut heads::Work,
    ) -> Result<Vec<Candidate>, String> {
        p.check(&self.config, self.encoder.config(), schema.labels().len())?;
        let h = self.encoder.encode(&p.ids, &vec![true; p.ids.len()]).map_err(|e| e.message)?;
        self.heads.score_with_work(h.data(), h.width(), &p.prompts, &p.words, self.config.max_width, work)
    }
}
/// Stable score-priority decoding. Nested mode allows containment but rejects
/// crossing intervals and multiple labels on exactly the same interval.
pub fn decode(mut candidates: Vec<Candidate>, schema: &ExtractionSchema) -> Vec<Candidate> {
    candidates.retain(|c| c.score.is_finite() && c.score > schema.threshold());
    candidates.sort_by(|a, b| b.score.total_cmp(&a.score));
    let mut out: Vec<Candidate> = Vec::new();
    for c in candidates {
        if out.iter().any(|b| {
            let overlaps = c.start < b.end && b.start < c.end;
            let contains = (c.start <= b.start && c.end >= b.end) || (b.start <= c.start && b.end >= c.end);
            overlaps && (schema.mode() == Decode::Flat || !contains || (c.start == b.start && c.end == b.end))
        }) {
            continue;
        }
        out.push(c);
    }
    out.sort_by_key(|c| c.start);
    out
}

#[cfg(test)]
mod tests;
