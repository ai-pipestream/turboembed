use super::*;
#[derive(Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Artifact {
    pub name: String,
    pub format: Format,
    pub files: Vec<String>,
    pub backends: Vec<String>,
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub target: String,
    #[serde(default, skip_serializing_if = "is_zero")]
    pub fixed_seq: u32,
    #[serde(default, skip_serializing_if = "is_zero")]
    pub fixed_batch: u32,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub compute_dtype: Option<Dtype>,
    pub graph_input: GraphInput,
    pub graph_output: GraphOutput,
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub host_weights: String,
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub tensor_names: BTreeMap<TensorRole, String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub produced_by: Option<ProducedBy>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Format {
    #[serde(rename = "FORMAT_SAFETENSORS")]
    Safetensors,
    #[serde(rename = "FORMAT_OPENVINO_IR")]
    OpenvinoIr,
    #[serde(rename = "FORMAT_HEF")]
    Hef,
    #[serde(rename = "FORMAT_GGUF")]
    Gguf,
    #[serde(rename = "FORMAT_ONNX")]
    Onnx,
}

/// What a compiled artifact computes in: the header's TURBO_DTYPE_*.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Dtype {
    #[serde(rename = "DTYPE_I8")]
    I8,
    #[serde(rename = "DTYPE_I32")]
    I32,
    #[serde(rename = "DTYPE_F16")]
    F16,
    #[serde(rename = "DTYPE_BF16")]
    Bf16,
    #[serde(rename = "DTYPE_F32")]
    F32,
}

impl Format {
    /// TURBO_FORMAT_*.
    pub fn value(self) -> u32 {
        use crate::backend::*;
        match self {
            Format::Safetensors => TURBO_FORMAT_SAFETENSORS,
            Format::OpenvinoIr => TURBO_FORMAT_OPENVINO_IR,
            Format::Hef => TURBO_FORMAT_HEF,
            Format::Gguf => TURBO_FORMAT_GGUF,
            Format::Onnx => TURBO_FORMAT_ONNX,
        }
    }
}

impl Dtype {
    /// TURBO_DTYPE_*.
    pub fn value(self) -> u32 {
        match self {
            Dtype::I8 => crate::TURBO_DTYPE_I8,
            Dtype::I32 => crate::TURBO_DTYPE_I32,
            Dtype::F16 => crate::TURBO_DTYPE_F16,
            Dtype::Bf16 => crate::TURBO_DTYPE_BF16,
            Dtype::F32 => crate::TURBO_DTYPE_F32,
        }
    }
}

impl GraphInput {
    /// TURBO_INPUT_*.
    pub fn value(self) -> u32 {
        match self {
            GraphInput::TokenIds => crate::backend::TURBO_INPUT_TOKEN_IDS,
            GraphInput::Embeddings => crate::backend::TURBO_INPUT_EMBEDDINGS,
        }
    }
}

impl GraphOutput {
    /// TURBO_OUTPUT_*.
    pub fn value(self) -> u32 {
        match self {
            GraphOutput::HiddenStates => crate::backend::TURBO_OUTPUT_HIDDEN_STATES,
            GraphOutput::Embeddings => crate::backend::TURBO_OUTPUT_EMBEDDINGS,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum GraphInput {
    #[serde(rename = "INPUT_TOKEN_IDS")]
    TokenIds,
    #[serde(rename = "INPUT_EMBEDDINGS")]
    Embeddings,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum GraphOutput {
    #[serde(rename = "OUTPUT_HIDDEN_STATES")]
    HiddenStates,
    /// The pooled and normalized vectors: the graph carries the embed
    /// block's pooling and normalization, as a sentence-transformers
    /// export does.
    #[serde(rename = "OUTPUT_EMBEDDINGS")]
    Embeddings,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TensorRole {
    WordEmbeddings,
    PositionEmbeddings,
    TokenTypeEmbeddings,
    EmbeddingsLnWeight,
    EmbeddingsLnBias,
    QWeight,
    QBias,
    KWeight,
    KBias,
    VWeight,
    VBias,
    AttnOutWeight,
    AttnOutBias,
    AttnLnWeight,
    AttnLnBias,
    FfnInWeight,
    FfnInBias,
    FfnOutWeight,
    FfnOutBias,
    FfnLnWeight,
    FfnLnBias,
    /// A static model's table, `[rows, dim]`.
    StaticEmbeddings,
    /// A static model's weight per token, `[vocab_size]`; optional.
    StaticWeights,
    /// A static model's table row per token, `[vocab_size]`; optional.
    StaticMapping,
    /// A BERT pooler's linear weight, `[hidden, hidden]`; optional with
    /// `CLASSIFIER_SEQUENCE`.
    PoolerWeight,
    /// A BERT pooler's linear bias, `[hidden]`; optional with
    /// `CLASSIFIER_SEQUENCE`.
    PoolerBias,
    /// A classification head's linear weight, `[labels, hidden]`.
    ClassifierWeight,
    /// A classification head's linear bias, `[labels]`.
    ClassifierBias,
    /// An MLM head's transform weight, `[hidden, hidden]`; with
    /// `embed.sparse`.
    MlmTransformWeight,
    /// An MLM head's transform bias, `[hidden]`; with `embed.sparse`.
    MlmTransformBias,
    /// An MLM head's transform LayerNorm weight, `[hidden]`; with
    /// `embed.sparse`.
    MlmTransformLnWeight,
    /// An MLM head's transform LayerNorm bias, `[hidden]`; with
    /// `embed.sparse`.
    MlmTransformLnBias,
    /// An MLM head's decoder weight, `[vocab_size, hidden]`; optional,
    /// tied to `word_embeddings` when absent.
    MlmDecoderWeight,
    /// An MLM head's decoder bias, `[vocab_size]`; with `embed.sparse`.
    MlmDecoderBias,
}

impl TensorRole {
    /// Roles that name one tensor per encoder layer, through `{layer}`.
    pub fn per_layer(self) -> bool {
        !matches!(
            self,
            TensorRole::WordEmbeddings
                | TensorRole::PositionEmbeddings
                | TensorRole::TokenTypeEmbeddings
                | TensorRole::EmbeddingsLnWeight
                | TensorRole::EmbeddingsLnBias
                | TensorRole::StaticEmbeddings
                | TensorRole::StaticWeights
                | TensorRole::StaticMapping
                | TensorRole::PoolerWeight
                | TensorRole::PoolerBias
                | TensorRole::ClassifierWeight
                | TensorRole::ClassifierBias
                | TensorRole::MlmTransformWeight
                | TensorRole::MlmTransformBias
                | TensorRole::MlmTransformLnWeight
                | TensorRole::MlmTransformLnBias
                | TensorRole::MlmDecoderWeight
                | TensorRole::MlmDecoderBias
        )
    }

    /// Roles a static model's weights have, and an encoder's do not.
    pub fn is_static(self) -> bool {
        matches!(self, TensorRole::StaticEmbeddings | TensorRole::StaticWeights | TensorRole::StaticMapping)
    }

    /// Roles a classification head has.
    pub fn is_classifier(self) -> bool {
        matches!(
            self,
            TensorRole::PoolerWeight
                | TensorRole::PoolerBias
                | TensorRole::ClassifierWeight
                | TensorRole::ClassifierBias
        )
    }

    /// Roles a masked language model head has (the learned sparse output).
    pub fn is_mlm(self) -> bool {
        matches!(
            self,
            TensorRole::MlmTransformWeight
                | TensorRole::MlmTransformBias
                | TensorRole::MlmTransformLnWeight
                | TensorRole::MlmTransformLnBias
                | TensorRole::MlmDecoderWeight
                | TensorRole::MlmDecoderBias
        )
    }
}
