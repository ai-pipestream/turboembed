use super::*;

/// Pair construction and capacity for a CPU cross-encoder.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Rerank {
    pub max_seq: u32,
    pub max_batch: u32,
    pub query_tokens: u32,
    pub pair_template: PairTemplate,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum PairTemplate {
    #[serde(rename = "PAIR_BERT")]
    Bert,
    #[serde(rename = "PAIR_ROBERTA")]
    Roberta,
}

impl PairTemplate {
    pub fn specials(self) -> u32 {
        match self {
            Self::Bert => 3,
            Self::Roberta => 4,
        }
    }
}

impl Manifest {
    /// Encoder shape shared internally by embedding, reranking and QA tasks.
    pub fn encoder(&self) -> Embed {
        if let Some(r) = &self.rerank {
            Embed {
                dim: self.architecture.as_ref().map_or(0, |a| a.hidden),
                pooling: Pooling::Cls,
                normalize: Normalize::None,
                max_seq: r.max_seq,
                max_batch: r.max_batch,
                prefix_query: String::new(),
                prefix_document: String::new(),
                output_dims: Vec::new(),
            }
        } else if let Some(q) = &self.qa {
            Embed {
                dim: self.architecture.as_ref().map_or(0, |a| a.hidden),
                pooling: Pooling::Mean,
                normalize: Normalize::None,
                max_seq: q.max_seq,
                max_batch: q.max_batch,
                prefix_query: String::new(),
                prefix_document: String::new(),
                output_dims: Vec::new(),
            }
        } else {
            self.embed().clone()
        }
    }

    pub(super) fn validate_rerank(&self) -> Result<()> {
        if self.task != "TASK_RERANK" {
            if self.rerank.is_some() {
                return Err(invalid("manifest.json: rerank: only allowed for TASK_RERANK"));
            }
            return Ok(());
        }
        let r = self.rerank.as_ref().ok_or_else(|| invalid("manifest.json: rerank: required for TASK_RERANK"))?;
        if self.embed.is_some() {
            return Err(invalid("manifest.json: embed: not allowed for TASK_RERANK"));
        }
        let c = self
            .classifier
            .as_ref()
            .ok_or_else(|| invalid("manifest.json: classifier: TASK_RERANK requires a relevance head"))?;
        if c.kind != ClassifierKind::Sequence || c.labels.len() != 1 {
            return Err(invalid("manifest.json: classifier: TASK_RERANK requires one sequence logit"));
        }
        let a = self
            .architecture
            .as_ref()
            .ok_or_else(|| invalid("manifest.json: architecture: TASK_RERANK requires an encoder"))?;
        let expected = match a.family {
            Family::Bert => PairTemplate::Bert,
            Family::Roberta => PairTemplate::Roberta,
        };
        if r.pair_template != expected || (expected == PairTemplate::Bert && a.token_types < 2) {
            return Err(invalid("manifest.json: rerank.pair_template: incompatible architecture or token_types"));
        }
        positive("rerank.max_batch", r.max_batch)?;
        positive("rerank.query_tokens", r.query_tokens)?;
        if r.max_seq <= r.pair_template.specials() + 1 || r.query_tokens >= r.max_seq - r.pair_template.specials() {
            return Err(invalid("manifest.json: rerank.query_tokens: must leave a passage token within max_seq"));
        }
        if self.tokenizer.template.len() != 3 || self.tokenizer.template[1] != "$TEXT" {
            return Err(invalid("manifest.json: tokenizer.template: rerank requires start, $TEXT, end"));
        }
        if self.static_embedding.is_some() {
            return Err(invalid("manifest.json: static_embedding: incompatible with rerank"));
        }
        Ok(())
    }
}
