//! Native GLiNER markerV0 inference on the shared DeBERTa-v3 encoder.
//! No external inference engine. See docs/gliner.md for the supported profile.
mod codec;
mod config;
mod device;
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
use std::path::Path;
use turbo::{deberta::Encoder, sentencepiece::SentencePiece};
use turbo_analysis::ModelRef;
use turbo_analysis_embed::EncoderOptions;

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
    encoder: Backend,
    heads: heads::Heads,
    config: Config,
    tokenizer: SentencePiece,
    identity: ModelRef,
}
/// Where the encoder runs: the CPU encoder itself, or a core bundle through
/// the backend table on any device (device.rs).
enum Backend {
    Cpu(Encoder),
    Table(Box<device::Table>, turbo::deberta::Config),
}
impl Backend {
    fn config(&self) -> &turbo::deberta::Config {
        match self {
            Backend::Cpu(e) => e.config(),
            Backend::Table(_, c) => c,
        }
    }
}
impl Model {
    pub fn new(
        encoder: Encoder,
        config: Config,
        heads: &[u8],
        sentencepiece: &[u8],
        identity: ModelRef,
    ) -> Result<Self, String> {
        Self::with(Backend::Cpu(encoder), config, heads, sentencepiece, identity)
    }
    /// The same heads and tokenizer over the DeBERTa-v2 core bundle at
    /// `encoder`, on the device `options` names (cpu, cuda, levelzero),
    /// through the backend table. Its longest sequence is the session's.
    pub fn open_on(
        encoder: &Path,
        options: &EncoderOptions,
        config: Config,
        heads: &[u8],
        sentencepiece: &[u8],
        identity: ModelRef,
    ) -> Result<Self, String> {
        let (table, ec) = device::Table::open(encoder, options)?;
        Self::with(Backend::Table(Box::new(table), ec), config, heads, sentencepiece, identity)
    }
    fn with(
        encoder: Backend,
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
    /// Where the encoder runs: None for the CPU encoder itself, else the
    /// backend, device and precision of its session on the backend table.
    pub fn execution(&self) -> Option<&turbo_analysis::Execution> {
        match &self.encoder {
            Backend::Cpu(_) => None,
            Backend::Table(t, _) => Some(&t.execution),
        }
    }
    pub(crate) fn encoder_config(&self) -> &turbo::deberta::Config {
        self.encoder.config()
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
        self.score_packed_with_work(p, schema, &mut heads::Work::default())
    }
    pub(crate) fn score_packed_with_work(
        &self,
        p: &PackedInput,
        schema: &ExtractionSchema,
        work: &mut heads::Work,
    ) -> Result<Vec<Candidate>, String> {
        p.check(&self.config, self.encoder.config(), schema.labels().len())?;
        let width = self.encoder.config().hidden;
        match &self.encoder {
            Backend::Cpu(e) => {
                let h = e.encode(&p.ids, &vec![true; p.ids.len()]).map_err(|e| e.message)?;
                self.heads.score_with_work(h.data(), h.width(), &p.prompts, &p.words, self.config.max_width, work)
            }
            Backend::Table(t, _) => {
                // The prompts' and words' states, one row each, in order.
                let mut states = Vec::new();
                t.states(p, &mut states)?;
                let (np, nw) = (p.prompts.len(), p.words.len());
                let (prompts, words): (Vec<usize>, Vec<usize>) = ((0..np).collect(), (np..np + nw).collect());
                self.heads.score_with_work(&states, width, &prompts, &words, self.config.max_width, work)
            }
        }
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
