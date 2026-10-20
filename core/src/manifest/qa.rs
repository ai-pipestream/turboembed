use super::*;

/// Joint question/context input limits. `doc_stride` is the overlap in
/// context tokens between windows, following SQuAD preprocessing.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Qa {
    pub max_seq: u32,
    pub max_batch: u32,
    #[serde(default = "query_default")]
    pub max_query_tokens: u32,
    #[serde(default = "stride_default")]
    pub doc_stride: u32,
}
fn query_default() -> u32 {
    64
}
fn stride_default() -> u32 {
    128
}

impl Manifest {
    pub(super) fn validate_qa(&self) -> Result<()> {
        if self.task != "TASK_EXTRACTIVE_QA" {
            if self.qa.is_some() {
                return Err(invalid("qa: only allowed for TASK_EXTRACTIVE_QA"));
            }
            return Ok(());
        }
        let q = self.qa.as_ref().ok_or_else(|| invalid("qa: required for TASK_EXTRACTIVE_QA"))?;
        if self.embed.is_some() || self.static_embedding.is_some() || self.projection.is_some() {
            return Err(invalid("qa: embed, static_embedding and projection are not allowed"));
        }
        let a = self.architecture.as_ref().ok_or_else(|| invalid("qa: architecture is required"))?;
        if self.artifacts.len() != 1 || self.artifacts[0].format != Format::Safetensors {
            return Err(invalid("qa: this CPU implementation requires one raw safetensors artifact"));
        }
        let c = self.classifier.as_ref().ok_or_else(|| invalid("qa: two-output token classifier is required"))?;
        if c.kind != ClassifierKind::Token || c.labels != ["start", "end"] {
            return Err(invalid("qa: classifier must have CLASSIFIER_TOKEN labels [start, end]"));
        }
        let specials = if a.family == Family::Roberta { 4 } else { 3 };
        if a.family == Family::Bert && a.token_types < 2 {
            return Err(invalid("qa: BERT requires two token types"));
        }
        positive("qa.max_batch", q.max_batch)?;
        positive("qa.max_query_tokens", q.max_query_tokens)?;
        if q.max_seq <= specials + 1 || q.max_query_tokens >= q.max_seq - specials {
            return Err(invalid("qa: max_query_tokens must leave context space within max_seq"));
        }
        if q.doc_stride >= q.max_seq - specials - q.max_query_tokens {
            return Err(invalid("qa.doc_stride: overlap must be smaller than context capacity"));
        }
        if self.tokenizer.template.len() != 3 || self.tokenizer.template[1] != "$TEXT" {
            return Err(invalid("qa: tokenizer template must be start, $TEXT, end"));
        }
        for artifact in &self.artifacts {
            if artifact.tensor_names.contains_key(&TensorRole::PoolerWeight)
                || artifact.tensor_names.contains_key(&TensorRole::PoolerBias)
            {
                return Err(invalid("qa: token logits must not use a sequence pooler"));
            }
        }
        Ok(())
    }
}
