use super::*;
#[derive(Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Architecture {
    pub family: Family,
    pub layers: u32,
    pub hidden: u32,
    pub heads: u32,
    pub intermediate: u32,
    pub activation: Activation,
    pub layer_norm_eps: f64,
    pub position_embedding: PositionEmbedding,
    pub max_positions: u32,
    /// The row of the position table a row's first token reads: 0 for
    /// BERT; RoBERTa counts positions from its padding id plus one, so
    /// XLM-RoBERTa's is 2 and its table holds two rows more than it uses.
    #[serde(default, skip_serializing_if = "is_zero")]
    pub position_offset: u32,
    pub token_types: u32,
    pub vocab_size: u32,
}

/// A static model: one vector per vocabulary entry and no encoder
/// (docs/static.md). A row's vector is the mean of its tokens' rows,
/// each times its weight (turbo_backend.h, TURBO_FAMILY_STATIC). Its
/// vectors live in a space of their own, not that of any model it was
/// distilled from.
#[derive(Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct StaticEmbedding {
    pub vocab_size: u32,
    /// The table's rows when a token mapping picks them; without one, the
    /// table has vocab_size rows and this is 0.
    #[serde(default, skip_serializing_if = "is_zero")]
    pub rows: u32,
    /// The tokens TURBO_TRUNCATE_MODEL keeps. A text is first cut to max_length times the
    /// vocabulary's median entry length in characters, its encoding to
    /// max_length tokens, and then the unknown token is dropped.
    pub max_length: u32,
    /// The bundle the table was distilled from by `turbo-bundle distill`;
    /// absent for a model made elsewhere.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub distilled_from: Option<DistilledFrom>,
    /// What the distillation cost; with distilled_from.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub quality: Option<Quality>,
}

/// The bundle the static model was distilled from, as its manifest names
/// it.
#[derive(Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DistilledFrom {
    pub model_id: String,
    pub revision: String,
    pub manifest_sha256: String,
}

/// What the distillation cost, measured on the texts the bundle carries:
/// each text's nearest other text by cosine shares its group in
/// `base_top1` of them with the base model and `static_top1` with the
/// static one, and `similarity_spearman` is the rank correlation of the two
/// models' cosines over every pair of texts.
#[derive(Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Quality {
    pub texts: String,
    pub base_top1: f64,
    pub static_top1: f64,
    pub similarity_spearman: f64,
}

/// A classification head carried beside the encoder: the labels in id
/// order (`0..labels.len()`) and whether the head classifies each token
/// (`CLASSIFIER_TOKEN`) or the pooled sequence (`CLASSIFIER_SEQUENCE`).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Classifier {
    pub kind: ClassifierKind,
    pub labels: Vec<String>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ClassifierKind {
    #[serde(rename = "CLASSIFIER_TOKEN")]
    Token,
    #[serde(rename = "CLASSIFIER_SEQUENCE")]
    Sequence,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Family {
    #[serde(rename = "FAMILY_BERT")]
    Bert,
    /// BERT's encoder with the positions counted from `position_offset`.
    #[serde(rename = "FAMILY_ROBERTA")]
    Roberta,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Activation {
    #[serde(rename = "ACTIVATION_GELU_ERF")]
    GeluErf,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum PositionEmbedding {
    #[serde(rename = "POSITION_ABSOLUTE")]
    Absolute,
}
