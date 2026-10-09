//! `manifest.json` in the one canonical JSON form docs/bundle.md
//! describes, parsed strictly and written by the bundle tool. An unknown field or enum value, a
//! missing required field, an over-long string or a bad path is
//! BUNDLE_INVALID naming the field.

use std::collections::{BTreeMap, HashSet};

use serde::{Deserialize, Serialize};

use crate::status::{Error, Result, UNSUPPORTED_TASK, invalid};

pub const BUNDLE_VERSION: u32 = 1;

fn is_zero(v: &u32) -> bool {
    *v == 0
}

#[derive(Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Manifest {
    pub bundle_version: u32,
    pub model: Model,
    pub task: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub embed: Option<Embed>,
    pub tokenizer: Tokenizer,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub architecture: Option<Architecture>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub static_embedding: Option<StaticEmbedding>,
    pub artifacts: Vec<Artifact>,
    pub reference: Reference,
    pub files: Vec<FileEntry>,
}

#[derive(Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Model {
    pub id: String,
    pub revision: String,
    pub source: Source,
    pub license: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub license_file: Option<String>,
}

#[derive(Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Source {
    pub repository: String,
    pub commit: String,
}

#[derive(Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Embed {
    pub dim: u32,
    pub pooling: Pooling,
    pub normalize: Normalize,
    pub max_seq: u32,
    pub max_batch: u32,
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub prefix_query: String,
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub prefix_document: String,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub output_dims: Vec<u32>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Pooling {
    #[serde(rename = "POOLING_MEAN")]
    Mean,
    #[serde(rename = "POOLING_CLS")]
    Cls,
    #[serde(rename = "POOLING_LAST")]
    Last,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Normalize {
    #[serde(rename = "NORMALIZE_NONE")]
    None,
    #[serde(rename = "NORMALIZE_L2")]
    L2,
}

#[derive(Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Tokenizer {
    pub file: String,
    /// BertNormalizer's switches, with `wordpiece`. A `unigram` tokenizer
    /// normalizes with the map its file carries, so it has none.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub normalizer: Option<Normalizer>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub wordpiece: Option<WordPiece>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub unigram: Option<Unigram>,
    pub special_tokens: Vec<SpecialToken>,
    pub template: Vec<String>,
    pub truncation: Truncation,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Normalizer {
    pub clean_text: bool,
    pub lowercase: bool,
    pub strip_accents: bool,
    pub split_cjk: bool,
    pub unicode_form: UnicodeForm,
}

/// Only UNICODE_NONE is read in this cut; the other forms arrive with a
/// tokenizer that needs them.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum UnicodeForm {
    #[serde(rename = "UNICODE_NONE")]
    None,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct WordPiece {
    pub continuing_prefix: String,
    pub max_chars_per_word: u32,
}

/// SentencePiece's Unigram as upstream `tokenizers` runs it: the text is
/// normalized by the precompiled character map the tokenizer file
/// carries (when `precompiled_charsmap`), runs of spaces are collapsed to
/// one (when `collapse_spaces`), every space becomes `metaspace` and each
/// piece of text between special tokens gets one in front (when
/// `add_prefix_space`); the text is then cut at each `metaspace` into
/// words, and each word into the vocabulary's pieces of highest total
/// score, a character no piece covers being the SPECIAL_UNK token.
///
/// After the collapse of spaces, as BGE-M3's static models normalize:
/// each ASCII punctuation character gets a space on each side (when
/// `space_punctuation`), each run of whitespace becomes one space (when
/// `collapse_whitespace`) and the text is trimmed (when `strip`). With
/// `whole_text` the text is not cut at the metaspace: it is segmented at
/// once, as a Metaspace pre-tokenizer with `split` false leaves it.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Unigram {
    pub precompiled_charsmap: bool,
    pub collapse_spaces: bool,
    pub metaspace: String,
    pub add_prefix_space: bool,
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    pub space_punctuation: bool,
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    pub collapse_whitespace: bool,
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    pub strip: bool,
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    pub whole_text: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SpecialToken {
    pub role: SpecialRole,
    pub content: String,
    pub id: u32,
    /// The token takes the whitespace before it in the text, as upstream's
    /// `lstrip` added tokens do.
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    pub lstrip: bool,
    /// The token is matched in the normalized text, its content
    /// normalized too, as upstream matches a `normalized` added token,
    /// and only as a whole word when `single_word`; it takes the
    /// whitespace after it when `rstrip`. A WordPiece tokenizer only.
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    pub normalized: bool,
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    pub single_word: bool,
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    pub rstrip: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum SpecialRole {
    #[serde(rename = "SPECIAL_PAD")]
    Pad,
    #[serde(rename = "SPECIAL_UNK")]
    Unk,
    #[serde(rename = "SPECIAL_BOS")]
    Bos,
    #[serde(rename = "SPECIAL_EOS")]
    Eos,
    #[serde(rename = "SPECIAL_MASK")]
    Mask,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Truncation {
    #[serde(rename = "TRUNCATE_NONE")]
    None,
    #[serde(rename = "TRUNCATE_RIGHT")]
    Right,
    #[serde(rename = "TRUNCATE_LEFT")]
    Left,
}

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

/// A static model: one vector per vocabulary entry and no encoder, as
/// Model2Vec makes them. A row's vector is the mean of its tokens' rows,
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
    /// The tokens TURBO_TRUNCATE_MODEL keeps: StaticModel's max_length.
    /// As StaticModel cuts, a text is first cut to max_length times the
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
        )
    }

    /// Roles a static model's weights have, and an encoder's do not.
    pub fn is_static(self) -> bool {
        matches!(self, TensorRole::StaticEmbeddings | TensorRole::StaticWeights | TensorRole::StaticMapping)
    }
}

#[derive(Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ProducedBy {
    pub tool: String,
    pub tool_version: String,
    pub container: String,
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub from: String,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub inputs: Vec<String>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub args: Vec<String>,
    pub reproducible: bool,
}

#[derive(Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Reference {
    pub file: String,
    pub cases: Vec<Case>,
    pub produced_by: ProducedBy,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Case {
    pub text: String,
    pub prompt_role: PromptRole,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum PromptRole {
    #[serde(rename = "PROMPT_NONE")]
    None,
    #[serde(rename = "PROMPT_QUERY")]
    Query,
    #[serde(rename = "PROMPT_DOCUMENT")]
    Document,
}

#[derive(Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FileEntry {
    pub path: String,
    pub size: u64,
    pub sha256: String,
}

impl Manifest {
    /// Parse and check everything that can be checked without opening
    /// another file.
    pub fn parse(bytes: &[u8]) -> Result<Manifest> {
        let de = &mut serde_json::Deserializer::from_slice(bytes);
        let task = peek_task(bytes);
        let m: Manifest = match serde_path_to_error::deserialize(de) {
            Ok(m) => m,
            Err(e) => {
                // A task this build does not have is reported as such, not
                // as a malformed manifest.
                if let Some(t) = task.filter(|t| t != "TASK_EMBED" && is_task_name(t)) {
                    return Err(unsupported_task(&t));
                }
                let path = e.path().to_string();
                return Err(invalid(format!("manifest.json: {path}: {}", e.inner())));
            }
        };
        m.validate()?;
        Ok(m)
    }

    pub fn embed(&self) -> &Embed {
        self.embed.as_ref().expect("validate() requires embed for TASK_EMBED")
    }

    pub fn file(&self, path: &str) -> &FileEntry {
        self.files.iter().find(|f| f.path == path).expect("validate() requires every path in files")
    }

    fn validate(&self) -> Result<()> {
        if self.bundle_version != BUNDLE_VERSION {
            return Err(invalid(format!(
                "manifest.json: bundle_version: {} is not {BUNDLE_VERSION}",
                self.bundle_version
            )));
        }
        if self.task != "TASK_EMBED" {
            if is_task_name(&self.task) {
                return Err(unsupported_task(&self.task));
            }
            return Err(invalid(format!("manifest.json: task: unknown value {:?}", self.task)));
        }

        // files[]: the list every other path must be in.
        let mut listed = HashSet::new();
        for (i, f) in self.files.iter().enumerate() {
            let field = format!("files[{i}].path");
            check_path(&field, &f.path)?;
            if f.path == "manifest.json" {
                return Err(invalid(format!("manifest.json: {field}: the manifest does not list itself")));
            }
            if !listed.insert(f.path.as_str()) {
                return Err(invalid(format!("manifest.json: {field}: {:?} is listed twice", f.path)));
            }
            if !is_hex64(&f.sha256) {
                return Err(invalid(format!("manifest.json: files[{i}].sha256: not 64 lowercase hex digits")));
            }
        }
        let listed_path = |field: &str, p: &str| -> Result<()> {
            check_path(field, p)?;
            if !listed.contains(p) {
                return Err(invalid(format!("manifest.json: {field}: {p:?} is not in files")));
            }
            Ok(())
        };

        // model
        required("model.id", &self.model.id)?;
        fits("model.id", &self.model.id, 128)?;
        required("model.revision", &self.model.revision)?;
        fits("model.revision", &self.model.revision, 64)?;
        required("model.source.repository", &self.model.source.repository)?;
        required("model.source.commit", &self.model.source.commit)?;
        required("model.license", &self.model.license)?;
        if let Some(p) = &self.model.license_file {
            listed_path("model.license_file", p)?;
        }

        // embed
        let Some(e) = &self.embed else {
            return Err(invalid("manifest.json: embed: required for TASK_EMBED"));
        };
        positive("embed.dim", e.dim)?;
        positive("embed.max_seq", e.max_seq)?;
        positive("embed.max_batch", e.max_batch)?;
        fits("embed.prefix_query", &e.prefix_query, 128)?;
        fits("embed.prefix_document", &e.prefix_document, 128)?;
        if e.output_dims.len() > crate::TURBO_OUTPUT_DIMS_MAX {
            return Err(invalid(format!(
                "manifest.json: embed.output_dims: {} widths, and turbo_model_info holds {}",
                e.output_dims.len(),
                crate::TURBO_OUTPUT_DIMS_MAX
            )));
        }
        let mut dims = HashSet::new();
        for (i, &d) in e.output_dims.iter().enumerate() {
            if d == 0 || d > e.dim {
                return Err(invalid(format!("manifest.json: embed.output_dims[{i}]: {d} is not in 1..={}", e.dim)));
            }
            if !dims.insert(d) {
                return Err(invalid(format!("manifest.json: embed.output_dims[{i}]: {d} is listed twice")));
            }
        }

        // tokenizer
        let t = &self.tokenizer;
        listed_path("tokenizer.file", &t.file)?;
        match (&t.wordpiece, &t.unigram) {
            (Some(w), None) => {
                required("tokenizer.wordpiece.continuing_prefix", &w.continuing_prefix)?;
                positive("tokenizer.wordpiece.max_chars_per_word", w.max_chars_per_word)?;
                if t.normalizer.is_none() {
                    return Err(invalid("manifest.json: tokenizer.normalizer: required with wordpiece"));
                }
            }
            (None, Some(u)) => {
                if u.metaspace.chars().count() != 1 {
                    return Err(invalid(format!(
                        "manifest.json: tokenizer.unigram.metaspace: {:?} is not one character",
                        u.metaspace
                    )));
                }
                if t.normalizer.is_some() {
                    return Err(invalid(
                        "manifest.json: tokenizer.normalizer: a unigram tokenizer normalizes with its file's map, not BertNormalizer",
                    ));
                }
            }
            _ => return Err(invalid("manifest.json: tokenizer: one of wordpiece and unigram")),
        }
        let mut roles = HashSet::new();
        let mut contents = HashSet::new();
        for (i, s) in t.special_tokens.iter().enumerate() {
            if !roles.insert(s.role) {
                return Err(invalid(format!("manifest.json: tokenizer.special_tokens[{i}].role: listed twice")));
            }
            required(&format!("tokenizer.special_tokens[{i}].content"), &s.content)?;
            if !contents.insert(s.content.as_str()) {
                return Err(invalid(format!("manifest.json: tokenizer.special_tokens[{i}].content: listed twice")));
            }
            if (s.single_word || s.rstrip) && !s.normalized {
                return Err(invalid(format!(
                    "manifest.json: tokenizer.special_tokens[{i}]: single_word and rstrip are read with normalized"
                )));
            }
            if s.normalized && t.wordpiece.is_none() {
                return Err(invalid(format!(
                    "manifest.json: tokenizer.special_tokens[{i}].normalized: a wordpiece tokenizer only"
                )));
            }
            if s.id > i32::MAX as u32 {
                return Err(invalid(format!("manifest.json: tokenizer.special_tokens[{i}].id: over INT32_MAX")));
            }
        }
        if !roles.contains(&SpecialRole::Unk) {
            return Err(invalid("manifest.json: tokenizer.special_tokens: the tokenizer needs a SPECIAL_UNK token"));
        }
        let texts = t.template.iter().filter(|s| *s == "$TEXT").count();
        if texts != 1 {
            return Err(invalid(format!("manifest.json: tokenizer.template: has {texts} \"$TEXT\" entries, not 1")));
        }
        for (i, s) in t.template.iter().enumerate() {
            if s != "$TEXT" && !contents.contains(s.as_str()) {
                return Err(invalid(format!("manifest.json: tokenizer.template[{i}]: {s:?} is not a special token")));
            }
        }
        // TRUNCATE_NONE is a caller's choice, not a model's: a bundle names
        // the side it cuts, and its reference carries a case that is cut.
        if t.truncation == Truncation::None {
            return Err(invalid(
                "manifest.json: tokenizer.truncation: TRUNCATE_NONE is a caller option; a bundle says TRUNCATE_RIGHT or TRUNCATE_LEFT",
            ));
        }
        let specials = t.template.len() as u32 - 1;
        if e.max_seq <= specials {
            return Err(invalid(format!(
                "manifest.json: embed.max_seq: {} leaves no room for text beside {specials} special tokens",
                e.max_seq
            )));
        }

        // architecture
        if let Some(a) = &self.architecture {
            for (field, v) in [
                ("layers", a.layers),
                ("hidden", a.hidden),
                ("heads", a.heads),
                ("intermediate", a.intermediate),
                ("max_positions", a.max_positions),
                ("token_types", a.token_types),
                ("vocab_size", a.vocab_size),
            ] {
                positive(&format!("architecture.{field}"), v)?;
            }
            if a.hidden % a.heads != 0 {
                return Err(invalid(format!(
                    "manifest.json: architecture.heads: {} does not divide hidden {}",
                    a.heads, a.hidden
                )));
            }
            if !(a.layer_norm_eps > 0.0 && a.layer_norm_eps.is_finite()) {
                return Err(invalid("manifest.json: architecture.layer_norm_eps: not a positive number"));
            }
            // The vectors are pooled hidden states, and no role projects them.
            if e.dim != a.hidden {
                return Err(invalid(format!(
                    "manifest.json: embed.dim: {} is not architecture.hidden {}, the width of the pooled hidden states",
                    e.dim, a.hidden
                )));
            }
            match a.family {
                Family::Bert if a.position_offset != 0 => {
                    return Err(invalid(
                        "manifest.json: architecture.position_offset: FAMILY_BERT counts positions from 0",
                    ));
                }
                Family::Roberta if a.position_offset == 0 => {
                    return Err(invalid(
                        "manifest.json: architecture.position_offset: required for FAMILY_ROBERTA (its padding id plus one)",
                    ));
                }
                _ => {}
            }
            if e.max_seq as u64 + a.position_offset as u64 > a.max_positions as u64 {
                return Err(invalid(format!(
                    "manifest.json: embed.max_seq: {} from position {} is over architecture.max_positions {}",
                    e.max_seq, a.position_offset, a.max_positions
                )));
            }
            for s in &t.special_tokens {
                if s.id >= a.vocab_size {
                    return Err(invalid(format!(
                        "manifest.json: tokenizer.special_tokens: id {} of {:?} is not under architecture.vocab_size {}",
                        s.id, s.content, a.vocab_size
                    )));
                }
            }
        }

        // static_embedding
        if let Some(st) = &self.static_embedding {
            if self.architecture.is_some() {
                return Err(invalid(
                    "manifest.json: static_embedding: a static model has no architecture; the bundle names one of them",
                ));
            }
            positive("static_embedding.vocab_size", st.vocab_size)?;
            positive("static_embedding.max_length", st.max_length)?;
            if st.max_length > e.max_seq {
                return Err(invalid(format!(
                    "manifest.json: static_embedding.max_length: {} is over embed.max_seq {}, the longest row a session takes",
                    st.max_length, e.max_seq
                )));
            }
            // StaticModel tokenizes the text alone and cuts it on the
            // right; the core does the same and nothing else.
            if t.template != ["$TEXT"] {
                return Err(invalid(
                    "manifest.json: tokenizer.template: a static model's rows are the text's tokens alone, [\"$TEXT\"]",
                ));
            }
            if t.truncation != Truncation::Right {
                return Err(invalid(
                    "manifest.json: tokenizer.truncation: a static model cuts on the right, TRUNCATE_RIGHT",
                ));
            }
            match (&st.distilled_from, &st.quality) {
                (Some(d), Some(q)) => {
                    required("static_embedding.distilled_from.model_id", &d.model_id)?;
                    required("static_embedding.distilled_from.revision", &d.revision)?;
                    if !is_hex64(&d.manifest_sha256) {
                        return Err(invalid(
                            "manifest.json: static_embedding.distilled_from.manifest_sha256: not 64 lowercase hex digits",
                        ));
                    }
                    listed_path("static_embedding.quality.texts", &q.texts)?;
                    for (field, v, lo) in [
                        ("base_top1", q.base_top1, 0.0),
                        ("static_top1", q.static_top1, 0.0),
                        ("similarity_spearman", q.similarity_spearman, -1.0),
                    ] {
                        if !(lo..=1.0).contains(&v) {
                            return Err(invalid(format!(
                                "manifest.json: static_embedding.quality.{field}: {v} is not in {lo}..=1"
                            )));
                        }
                    }
                }
                (None, None) => {}
                _ => {
                    return Err(invalid(
                        "manifest.json: static_embedding: distilled_from and quality come together, or neither",
                    ));
                }
            }
            for (i, a) in self.artifacts.iter().enumerate() {
                let names = |r: TensorRole| a.tensor_names.contains_key(&r);
                if !names(TensorRole::StaticEmbeddings) {
                    return Err(invalid(format!(
                        "manifest.json: artifacts[{i}].tensor_names: no static_embeddings, which a static model needs"
                    )));
                }
                if names(TensorRole::StaticMapping) != (st.rows != 0) {
                    return Err(invalid(format!(
                        "manifest.json: artifacts[{i}].tensor_names: static_mapping and static_embedding.rows come together, or neither"
                    )));
                }
            }
            for s in &t.special_tokens {
                if s.id >= st.vocab_size {
                    return Err(invalid(format!(
                        "manifest.json: tokenizer.special_tokens: id {} of {:?} is not under static_embedding.vocab_size {}",
                        s.id, s.content, st.vocab_size
                    )));
                }
            }
        }

        // artifacts
        if self.artifacts.is_empty() {
            return Err(invalid("manifest.json: artifacts: empty"));
        }
        let mut names = HashSet::new();
        for (i, a) in self.artifacts.iter().enumerate() {
            let at = |f: &str| format!("artifacts[{i}].{f}");
            required(&at("name"), &a.name)?;
            if !names.insert(a.name.as_str()) {
                return Err(invalid(format!("manifest.json: {}: {:?} is used twice", at("name"), a.name)));
            }
            if a.files.is_empty() {
                return Err(invalid(format!("manifest.json: {}: empty", at("files"))));
            }
            for (j, p) in a.files.iter().enumerate() {
                listed_path(&at(&format!("files[{j}]")), p)?;
            }
            for (j, b) in a.backends.iter().enumerate() {
                required(&at(&format!("backends[{j}]")), b)?;
                fits(&at(&format!("backends[{j}]")), b, 32)?;
            }
            fits(&at("target"), &a.target, 32)?;
            if a.format == Format::Hef {
                if a.target.is_empty() {
                    return Err(invalid(format!("manifest.json: {}: required for a compiled HEF", at("target"))));
                }
                // The core hands a backend a HEF as one block of bytes, over
                // the architecture raw weights would describe.
                if a.files.len() != 1 {
                    return Err(invalid(format!("manifest.json: {}: a HEF is one file", at("files"))));
                }
                if a.compute_dtype.is_none() {
                    return Err(invalid(format!(
                        "manifest.json: {}: required for a compiled HEF",
                        at("compute_dtype")
                    )));
                }
                if self.architecture.is_none() {
                    return Err(invalid(format!("manifest.json: architecture: required by the HEF in {}", at("name"))));
                }
                // A HEF's shape is compiled in; its backend has no dynamic one.
                for (field, v) in [("fixed_seq", a.fixed_seq), ("fixed_batch", a.fixed_batch)] {
                    if v == 0 {
                        return Err(invalid(format!("manifest.json: {}: required for a compiled HEF", at(field))));
                    }
                }
            }
            if a.format == Format::OpenvinoIr {
                // The core hands a backend the IR as two blocks of bytes:
                // the xml, then its weights.
                if a.files.len() != 2 {
                    return Err(invalid(format!(
                        "manifest.json: {}: an OpenVINO IR is two files, the xml then its weights",
                        at("files")
                    )));
                }
                if a.compute_dtype.is_none() {
                    return Err(invalid(format!(
                        "manifest.json: {}: required for a compiled OpenVINO IR",
                        at("compute_dtype")
                    )));
                }
                if self.architecture.is_none() {
                    return Err(invalid(format!(
                        "manifest.json: architecture: required by the OpenVINO IR in {}",
                        at("name")
                    )));
                }
                // The NPU compiles a static shape; the manifest names it,
                // the same way a HEF's compiled frame is named.
                for (field, v) in [("fixed_seq", a.fixed_seq), ("fixed_batch", a.fixed_batch)] {
                    if v == 0 {
                        return Err(invalid(format!(
                            "manifest.json: {}: required for a compiled OpenVINO IR",
                            at(field)
                        )));
                    }
                }
            }
            if a.graph_input == GraphInput::Embeddings && a.host_weights.is_empty() {
                return Err(invalid(format!(
                    "manifest.json: {}: required when graph_input is INPUT_EMBEDDINGS",
                    at("host_weights")
                )));
            }
            if self.static_embedding.is_some() && a.format != Format::Safetensors {
                return Err(invalid(format!(
                    "manifest.json: {}: a static model's artifacts are raw weights, FORMAT_SAFETENSORS",
                    at("format")
                )));
            }
            if a.format == Format::Safetensors {
                if a.tensor_names.is_empty() {
                    return Err(invalid(format!("manifest.json: {}: required for raw weights", at("tensor_names"))));
                }
                if self.architecture.is_none() && self.static_embedding.is_none() {
                    return Err(invalid(format!(
                        "manifest.json: architecture: required by raw weights in {}",
                        at("name")
                    )));
                }
                // Raw weights compute in what the session's precision says,
                // and start where the weights do: at token ids.
                if a.compute_dtype.is_some() {
                    return Err(invalid(format!(
                        "manifest.json: {}: fixed by a compilation; raw weights have none",
                        at("compute_dtype")
                    )));
                }
                if a.graph_input != GraphInput::TokenIds {
                    return Err(invalid(format!(
                        "manifest.json: {}: raw weights start at INPUT_TOKEN_IDS",
                        at("graph_input")
                    )));
                }
                if a.graph_output != GraphOutput::HiddenStates {
                    return Err(invalid(format!(
                        "manifest.json: {}: raw weights stop at OUTPUT_HIDDEN_STATES; the backend pools",
                        at("graph_output")
                    )));
                }
            }
            for (role, name) in &a.tensor_names {
                let field = at(&format!("tensor_names.{}", role_name(*role)));
                required(&field, name)?;
                if role.is_static() != self.static_embedding.is_some() {
                    let whose = if role.is_static() { "a static model's" } else { "an encoder's" };
                    return Err(invalid(format!("manifest.json: {field}: {whose} role, and the bundle is not one")));
                }
                if role.per_layer() != name.contains("{layer}") {
                    let want = if role.per_layer() { "must" } else { "must not" };
                    return Err(invalid(format!("manifest.json: {field}: {want} contain {{layer}}")));
                }
            }
            if let Some(p) = &a.produced_by {
                self.check_produced_by(&at("produced_by"), p, &listed_path)?;
                // Raw weights the tool wrote from the upstream checkpoint
                // come from the upstream model, as the reference does.
                if p.from.is_empty() && a.format != Format::Safetensors {
                    return Err(invalid(format!(
                        "manifest.json: {}: required for a converted artifact",
                        at("produced_by.from")
                    )));
                }
            }
        }
        for (i, a) in self.artifacts.iter().enumerate() {
            if !a.host_weights.is_empty() {
                let Some(h) = self.artifacts.iter().find(|h| h.name == a.host_weights) else {
                    return Err(invalid(format!(
                        "manifest.json: artifacts[{i}].host_weights: no artifact named {:?}",
                        a.host_weights
                    )));
                };
                // The lookup reads embedding tensors, which raw weights name.
                if h.format != Format::Safetensors {
                    return Err(invalid(format!(
                        "manifest.json: artifacts[{i}].host_weights: {:?} is not FORMAT_SAFETENSORS",
                        a.host_weights
                    )));
                }
            }
            if let Some(p) = &a.produced_by
                && !p.from.is_empty()
                && !names.contains(p.from.as_str())
            {
                return Err(invalid(format!(
                    "manifest.json: artifacts[{i}].produced_by.from: no artifact named {:?}",
                    p.from
                )));
            }
        }

        // reference
        listed_path("reference.file", &self.reference.file)?;
        if self.reference.cases.is_empty() {
            return Err(invalid("manifest.json: reference.cases: empty"));
        }
        self.check_produced_by("reference.produced_by", &self.reference.produced_by, &listed_path)?;
        if !self.reference.produced_by.from.is_empty() {
            return Err(invalid(
                "manifest.json: reference.produced_by.from: the reference is made from the upstream model, not an artifact",
            ));
        }
        Ok(())
    }

    fn check_produced_by(
        &self,
        field: &str,
        p: &ProducedBy,
        listed_path: &dyn Fn(&str, &str) -> Result<()>,
    ) -> Result<()> {
        required(&format!("{field}.tool"), &p.tool)?;
        required(&format!("{field}.tool_version"), &p.tool_version)?;
        required(&format!("{field}.container"), &p.container)?;
        for (j, input) in p.inputs.iter().enumerate() {
            listed_path(&format!("{field}.inputs[{j}]"), input)?;
        }
        Ok(())
    }
}

/// The manifest's snake_case name for a tensor role.
pub fn role_name(role: TensorRole) -> String {
    let mut out = String::new();
    for (i, c) in format!("{role:?}").chars().enumerate() {
        if c.is_ascii_uppercase() {
            if i > 0 {
                out.push('_');
            }
            out.push(c.to_ascii_lowercase());
        } else {
            out.push(c);
        }
    }
    out
}

/// Bundle paths: relative, `/`-separated, no `.` or `..` or empty
/// component, no backslash, no NUL.
fn check_path(field: &str, p: &str) -> Result<()> {
    let bad = |why: &str| Err(invalid(format!("manifest.json: {field}: {p:?} {why}")));
    if p.is_empty() {
        return bad("is empty");
    }
    if p.starts_with('/') {
        return bad("starts with /");
    }
    if p.contains('\\') || p.contains('\0') {
        return bad("contains a backslash or NUL");
    }
    for c in p.split('/') {
        match c {
            ".." => return bad("contains .."),
            "." | "" => return bad("has an empty or . component"),
            _ => {}
        }
    }
    Ok(())
}

fn is_hex64(s: &str) -> bool {
    s.len() == 64 && s.bytes().all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
}

fn required(field: &str, v: &str) -> Result<()> {
    if v.is_empty() {
        return Err(invalid(format!("manifest.json: {field}: required")));
    }
    Ok(())
}

fn positive(field: &str, v: u32) -> Result<()> {
    if v == 0 {
        return Err(invalid(format!("manifest.json: {field}: must be over 0")));
    }
    Ok(())
}

/// A string must fit its header buffer with the terminating NUL.
fn fits(field: &str, v: &str, buffer: usize) -> Result<()> {
    if v.len() >= buffer {
        return Err(invalid(format!(
            "manifest.json: {field}: {} bytes is over the header's {} for this field",
            v.len(),
            buffer - 1
        )));
    }
    Ok(())
}

fn is_task_name(s: &str) -> bool {
    s.len() > 5 && s.starts_with("TASK_") && s[5..].bytes().all(|b| b.is_ascii_uppercase() || b == b'_')
}

fn unsupported_task(t: &str) -> Error {
    Error::new(UNSUPPORTED_TASK, format!("manifest.json: task: this build has TASK_EMBED only, not {t}"))
}

/// The task, if the manifest is JSON with a string `task`: another task's
/// block is an unknown field to this build, and should read as
/// UNSUPPORTED_TASK rather than BUNDLE_INVALID.
fn peek_task(bytes: &[u8]) -> Option<String> {
    let v: serde_json::Value = serde_json::from_slice(bytes).ok()?;
    v.get("task")?.as_str().map(str::to_owned)
}
