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
    /// Limited lookahead: each layer's query attends to every key before
    /// it and to at most this many keys ahead of it, as SaT's `sat-1l` to
    /// `sat-12l` were trained (wtpsplit's `lookahead` over the layers).
    /// Absent, attention sees the whole row. Only the CPU backend runs it.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub lookahead: Option<u32>,
    pub token_types: u32,
    pub vocab_size: u32,
    /// FAMILY_DEBERTA_V2's log-bucketed relative positions, with
    /// POSITION_RELATIVE; absent for every other family.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub relative: Option<Relative>,
    /// POSITION_ROTARY's rotary bases; absent for every other position
    /// embedding.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub rope: Option<Rope>,
    /// Alternating local and global attention; absent, every layer
    /// attends to the whole row.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub attention: Option<Attention>,
    /// The feed-forward block: one projection, or a gated pair.
    #[serde(default, skip_serializing_if = "Mlp::is_dense")]
    pub mlp: Mlp,
    /// The normalization every LayerNorm of the encoder applies.
    #[serde(default, skip_serializing_if = "Norm::is_layer")]
    pub norm: Norm,
}

/// Rotary position embeddings over each head's halves (rotate_half, not
/// interleaved pairs): `global_theta` is the base on full-attention
/// layers, `local_theta` the base on sliding-window layers, present iff
/// `architecture.attention` is.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Rope {
    pub global_theta: f64,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub local_theta: Option<f64>,
}

/// Alternating attention: layer l attends to the whole row iff
/// `l % global_every == 0`; every other layer is local, its query i
/// reading the keys j with `abs(i - j) <= window / 2`. `window` is the
/// total width, as ModernBERT's `local_attention` gives it.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Attention {
    pub window: u32,
    pub global_every: u32,
}

impl Attention {
    /// Whether layer `l` attends to the whole row.
    pub fn is_global(&self, l: u32) -> bool {
        l.is_multiple_of(self.global_every)
    }
}

/// The feed-forward block. MLP_DENSE is BERT's `out(act(in x))`; MLP_GATED
/// is `out(act(gate x) * up x)`, GeGLU with ACTIVATION_GELU_ERF and
/// SwiGLU with ACTIVATION_SILU, its gated width `architecture.intermediate`.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
pub enum Mlp {
    #[default]
    #[serde(rename = "MLP_DENSE")]
    Dense,
    #[serde(rename = "MLP_GATED")]
    Gated,
}

impl Mlp {
    fn is_dense(&self) -> bool {
        *self == Mlp::Dense
    }
}

/// The normalization: LayerNorm with weight and bias, LayerNorm with a
/// weight alone, or RMSNorm (named, and refused until a backend runs it).
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
pub enum Norm {
    #[default]
    #[serde(rename = "NORM_LAYER")]
    Layer,
    #[serde(rename = "NORM_LAYER_NO_BIAS")]
    LayerNoBias,
    #[serde(rename = "NORM_RMS")]
    Rms,
}

impl Norm {
    fn is_layer(&self) -> bool {
        *self == Norm::Layer
    }
}

/// DeBERTa-v2's relative positions: `position_buckets` buckets each way
/// (the relative table has twice as many rows), log-spaced past half of
/// them out to `max_relative_positions`, as Hugging Face's
/// `make_log_bucket_position` gives.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Relative {
    pub position_buckets: u32,
    pub max_relative_positions: u32,
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

/// A linear projection head carried beside the encoder (as ColBERT and
/// late-interaction token-vector models carry): projects each token's
/// `architecture.hidden` state to `dim` before optional L2 normalization.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Projection {
    pub dim: u32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Family {
    #[serde(rename = "FAMILY_BERT")]
    Bert,
    /// BERT's encoder with the positions counted from `position_offset`.
    #[serde(rename = "FAMILY_ROBERTA")]
    Roberta,
    /// DeBERTa-v2 and v3: shared-key disentangled attention over relative
    /// positions (`architecture.relative`), with no absolute positions or
    /// token types.
    #[serde(rename = "FAMILY_DEBERTA_V2")]
    DebertaV2,
    /// ModernBERT: pre-norm layers with rotary positions, alternating
    /// local and global attention, a gated MLP and no biases; a final
    /// norm and no position or token type tables.
    #[serde(rename = "FAMILY_MODERNBERT")]
    Modernbert,
    /// nomic-bert: BERT's post-norm layers with rotary positions, a gated
    /// MLP and no projection biases; token types and no position table.
    #[serde(rename = "FAMILY_NOMIC_BERT")]
    NomicBert,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Activation {
    #[serde(rename = "ACTIVATION_GELU_ERF")]
    GeluErf,
    /// `x * sigmoid(x)`, SwiGLU's gate.
    #[serde(rename = "ACTIVATION_SILU")]
    Silu,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum PositionEmbedding {
    #[serde(rename = "POSITION_ABSOLUTE")]
    Absolute,
    #[serde(rename = "POSITION_RELATIVE")]
    Relative,
    /// Rotary positions on Q and K, with architecture.rope.
    #[serde(rename = "POSITION_ROTARY")]
    Rotary,
}
