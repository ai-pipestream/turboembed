//! The provider contract.
//!
//! A provider implements tasks at task granularity against its own runtime and
//! may execute each as one fused device pipeline. The core enforces handle
//! lifetimes, single-owner sessions, result leases, option validation against
//! the capability matrix, and error containment; the provider does the work.
//!
//! Every trait method that a provider does not offer has a default body that
//! returns `TURBO_E_UNSUPPORTED_TASK`. The core checks the capability cell
//! first, so those defaults are a second line of defense, not the contract.

use std::sync::Arc;

use crate::buffer::{BufferDesc, NativeHandle, ProviderBuffer};
use crate::bundle::Bundle;
use crate::error::{Error, Result};
use crate::types::{
    Aggregation, CapStatus, DType, DeviceKind, FinishReason, Modality, ModelKind, Normalize, OutputDType, Placement,
    Pooling, PromptRole, StagePlacements, StructuredKind, Task, Truncate,
};

/// Static description of one device offered by a provider.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DeviceInfo {
    /// Hardware class.
    pub kind: DeviceKind,
    /// Ordinal within the provider.
    pub ordinal: u32,
    /// PCI vendor id or 0.
    pub vendor_id: u32,
    /// `TURBO_CAP_*` bits.
    pub caps: u64,
    /// Total device memory or 0.
    pub memory_total: u64,
    /// Free device memory or 0.
    pub memory_free: u64,
    /// Human-readable name.
    pub name: String,
    /// Vendor name.
    pub vendor: String,
    /// Provider id.
    pub provider_id: String,
    /// Provider version.
    pub provider_version: String,
    /// Vendor runtime version.
    pub runtime_version: String,
    /// Driver version or empty.
    pub driver_version: String,
}

/// One cell of the capability matrix.
#[derive(Clone, Debug, PartialEq)]
pub struct Capability {
    /// Qualification status.
    pub status: CapStatus,
    /// Compute dtype, if applicable.
    pub dtype: Option<DType>,
    /// Reference dtype the precision figures were measured against.
    pub reference_dtype: Option<DType>,
    /// Measured cosine floor vs reference, 0 if unmeasured.
    pub cosine_floor: f32,
    /// Measured max absolute error vs reference, 0 if unmeasured.
    pub max_abs_error: f32,
    /// Bit-reproducible across runs.
    pub deterministic: bool,
    /// Qualification note.
    pub notes: String,
}

impl Capability {
    /// The unsupported cell.
    pub fn unsupported() -> Self {
        Self {
            status: CapStatus::Unsupported,
            dtype: None,
            reference_dtype: None,
            cosine_floor: 0.0,
            max_abs_error: 0.0,
            deterministic: false,
            notes: String::new(),
        }
    }

    /// True unless the status is `Unsupported`.
    pub fn is_offered(&self) -> bool {
        self.status != CapStatus::Unsupported
    }
}

/// Ordered key/value options passed to providers. Unknown keys are rejected
/// by the provider that receives them.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Options(pub Vec<(String, String)>);

impl Options {
    /// Value for a key.
    pub fn get(&self, key: &str) -> Option<&str> {
        self.0.iter().find(|(k, _)| k == key).map(|(_, v)| v.as_str())
    }

    /// Fail with `TURBO_E_INVALID_ARGUMENT` if any key is outside `known`.
    pub fn reject_unknown(&self, known: &[&str], what: &str) -> Result<()> {
        for (i, (k, _)) in self.0.iter().enumerate() {
            if !known.contains(&k.as_str()) {
                return Err(Error::invalid_argument(format!(
                    "{what} option `{k}` is not recognized; known options: {}",
                    if known.is_empty() { "(none)".to_string() } else { known.join(", ") }
                ))
                .with_field(i as u32 + 1));
            }
        }
        Ok(())
    }
}

/// Context creation parameters.
#[derive(Clone, Debug, Default)]
pub struct ContextDesc {
    /// Provider options.
    pub options: Options,
}

/// Model load parameters.
#[derive(Clone, Debug, Default)]
pub struct ModelDesc {
    /// Provider options.
    pub options: Options,
}

/// Named tensor description for RUN models. `-1` extents are dynamic.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TensorInfo {
    /// Name.
    pub name: String,
    /// Element type.
    pub dtype: DType,
    /// Extents, -1 for dynamic.
    pub shape: Vec<i64>,
}

/// What actually loaded.
#[derive(Clone, Debug, PartialEq)]
pub struct ModelInfo {
    /// Primary task.
    pub task: Task,
    /// Kind.
    pub kind: ModelKind,
    /// Modality.
    pub modality: Modality,
    /// Embedding dimension or 0.
    pub dim: u32,
    /// Labels in index order (classifiers).
    pub labels: Vec<String>,
    /// Contract pooling.
    pub pooling: Option<Pooling>,
    /// Contract normalization.
    pub normalize: Option<Normalize>,
    /// Contract aggregation (token classifiers).
    pub aggregation: Option<Aggregation>,
    /// Maximum sequence length.
    pub max_seq: u32,
    /// Maximum batch a session may declare.
    pub max_batch: u32,
    /// Compute dtype used.
    pub dtype_used: Option<DType>,
    /// Per-stage placement.
    pub stages: StagePlacements,
    /// Named inputs (RUN).
    pub inputs: Vec<TensorInfo>,
    /// Named outputs (RUN).
    pub outputs: Vec<TensorInfo>,
    /// Vocabulary size or 0.
    pub vocab_size: u32,
    /// Model identifier.
    pub model_id: String,
    /// Revision.
    pub revision: String,
    /// Tokenizer hash or empty.
    pub tokenizer_sha256: String,
    /// Provider id.
    pub provider_id: String,
    /// Query prefix.
    pub prefix_query: String,
    /// Document prefix.
    pub prefix_document: String,
}

/// Session creation parameters.
#[derive(Clone, Debug, Default)]
pub struct SessionDesc {
    /// Maximum batch; 0 = model default.
    pub max_batch: u32,
    /// Maximum sequence length; 0 = model default.
    pub max_seq: u32,
    /// Provider options.
    pub options: Options,
}

/// Per-call embedding options. Field indices (1-based, for error reporting)
/// follow the ABI struct: 2 truncate, 3 max_tokens, 4 prompt_role,
/// 5 normalize, 6 pooling, 7 output_dim, 8 output_dtype.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct EmbedOptions {
    /// Truncation.
    pub truncate: Truncate,
    /// Token budget, 0 = model.
    pub max_tokens: u32,
    /// Prompt prefix role.
    pub prompt_role: PromptRole,
    /// Normalization.
    pub normalize: Normalize,
    /// Pooling.
    pub pooling: Pooling,
    /// Output dimension, 0 = model.
    pub output_dim: u32,
    /// Output element type.
    pub output_dtype: OutputDType,
}

impl Default for EmbedOptions {
    fn default() -> Self {
        Self {
            truncate: Truncate::Model,
            max_tokens: 0,
            prompt_role: PromptRole::None,
            normalize: Normalize::Model,
            pooling: Pooling::Model,
            output_dim: 0,
            output_dtype: OutputDType::Model,
        }
    }
}

impl EmbedOptions {
    /// ABI field index of `truncate`.
    pub const FIELD_TRUNCATE: u32 = 2;
    /// ABI field index of `max_tokens`.
    pub const FIELD_MAX_TOKENS: u32 = 3;
    /// ABI field index of `prompt_role`.
    pub const FIELD_PROMPT_ROLE: u32 = 4;
    /// ABI field index of `normalize`.
    pub const FIELD_NORMALIZE: u32 = 5;
    /// ABI field index of `pooling`.
    pub const FIELD_POOLING: u32 = 6;
    /// ABI field index of `output_dim`.
    pub const FIELD_OUTPUT_DIM: u32 = 7;
    /// ABI field index of `output_dtype`.
    pub const FIELD_OUTPUT_DTYPE: u32 = 8;
}

/// Per-call rerank options. Field indices: 2 truncate, 3 max_tokens, 4 top_n,
/// 5 return_sorted, 6 raw_scores.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RerankOptions {
    /// Truncation.
    pub truncate: Truncate,
    /// Token budget per pair, 0 = model.
    pub max_tokens: u32,
    /// Keep only the best `top_n`, 0 = all.
    pub top_n: u32,
    /// Also produce the sorted index output.
    pub return_sorted: bool,
    /// Raw logits instead of activated scores.
    pub raw_scores: bool,
}

impl Default for RerankOptions {
    fn default() -> Self {
        Self { truncate: Truncate::Model, max_tokens: 0, top_n: 0, return_sorted: false, raw_scores: false }
    }
}

impl RerankOptions {
    /// ABI field index of `truncate`.
    pub const FIELD_TRUNCATE: u32 = 2;
    /// ABI field index of `max_tokens`.
    pub const FIELD_MAX_TOKENS: u32 = 3;
    /// ABI field index of `top_n`.
    pub const FIELD_TOP_N: u32 = 4;
    /// ABI field index of `return_sorted`.
    pub const FIELD_RETURN_SORTED: u32 = 5;
    /// ABI field index of `raw_scores`.
    pub const FIELD_RAW_SCORES: u32 = 6;
}

/// Per-call classification options. Field indices: 2 truncate, 3 max_tokens,
/// 4 aggregation, 5 raw_scores.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ClassifyOptions {
    /// Truncation.
    pub truncate: Truncate,
    /// Token budget, 0 = model.
    pub max_tokens: u32,
    /// Span aggregation (token classification).
    pub aggregation: Aggregation,
    /// Raw logits instead of activated scores.
    pub raw_scores: bool,
}

impl Default for ClassifyOptions {
    fn default() -> Self {
        Self { truncate: Truncate::Model, max_tokens: 0, aggregation: Aggregation::Model, raw_scores: false }
    }
}

impl ClassifyOptions {
    /// ABI field index of `truncate`.
    pub const FIELD_TRUNCATE: u32 = 2;
    /// ABI field index of `max_tokens`.
    pub const FIELD_MAX_TOKENS: u32 = 3;
    /// ABI field index of `aggregation`.
    pub const FIELD_AGGREGATION: u32 = 4;
    /// ABI field index of `raw_scores`.
    pub const FIELD_RAW_SCORES: u32 = 5;
}

/// Per-call RUN options: request parameters.
#[derive(Clone, Debug, Default)]
pub struct RunOptions {
    /// Parameters.
    pub params: Options,
}

/// Borrowed caller-prepared token batch.
#[derive(Clone, Copy, Debug)]
pub struct TokenBatch<'a> {
    /// Rows.
    pub batch: u32,
    /// Columns.
    pub seq: u32,
    /// Elements between row starts.
    pub row_stride: u32,
    /// Ids, at least `(batch-1)*row_stride + seq` long.
    pub ids: &'a [i32],
    /// Mask, same length rule.
    pub mask: &'a [i32],
    /// Types, or None for all zero.
    pub types: Option<&'a [i32]>,
}

impl<'a> TokenBatch<'a> {
    /// Minimum slice length for these dimensions, or an error on overflow.
    pub fn required_len(batch: u32, seq: u32, row_stride: u32) -> Result<usize> {
        if batch == 0 || seq == 0 {
            return Err(Error::invalid_shape("token batch must have batch >= 1 and seq >= 1"));
        }
        if row_stride < seq {
            return Err(Error::invalid_shape(format!("row_stride {row_stride} is smaller than seq {seq}")));
        }
        (batch as usize - 1)
            .checked_mul(row_stride as usize)
            .and_then(|n| n.checked_add(seq as usize))
            .ok_or_else(|| Error::invalid_shape("token batch size overflows"))
    }

    /// Validate lengths and mask/type values against the vocabulary.
    pub fn validate(&self, vocab_size: u32) -> Result<()> {
        let need = Self::required_len(self.batch, self.seq, self.row_stride)?;
        if self.ids.len() < need {
            return Err(Error::invalid_shape(format!("ids has {} elements but {need} are needed", self.ids.len())));
        }
        if self.mask.len() < need {
            return Err(Error::invalid_shape(format!("mask has {} elements but {need} are needed", self.mask.len())));
        }
        if let Some(t) = self.types {
            if t.len() < need {
                return Err(Error::invalid_shape(format!("types has {} elements but {need} are needed", t.len())));
            }
        }
        for row in 0..self.batch as usize {
            let start = row * self.row_stride as usize;
            for col in 0..self.seq as usize {
                let i = start + col;
                let id = self.ids[i];
                if id < 0 || (vocab_size != 0 && id as u32 >= vocab_size) {
                    return Err(Error::invalid_argument(format!(
                        "token id {id} at row {row} column {col} is outside 0..{vocab_size}"
                    )));
                }
                if !matches!(self.mask[i], 0 | 1) {
                    return Err(Error::invalid_argument(format!(
                        "mask value {} at row {row} column {col} is not 0 or 1",
                        self.mask[i]
                    )));
                }
                if let Some(t) = self.types {
                    if !matches!(t[i], 0 | 1) {
                        return Err(Error::invalid_argument(format!(
                            "token type {} at row {row} column {col} is not 0 or 1",
                            t[i]
                        )));
                    }
                }
            }
        }
        Ok(())
    }

    /// Row `r` of `ids`, `seq` long.
    pub fn ids_row(&self, r: usize) -> &'a [i32] {
        let s = r * self.row_stride as usize;
        &self.ids[s..s + self.seq as usize]
    }

    /// Row `r` of `mask`, `seq` long.
    pub fn mask_row(&self, r: usize) -> &'a [i32] {
        let s = r * self.row_stride as usize;
        &self.mask[s..s + self.seq as usize]
    }
}

/// Session counters. Adapter-level only.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct SessionStats {
    /// Completed runs.
    pub runs: u64,
    /// Host heap allocations the adapter made on the run path since
    /// warmup, or `None` when the adapter does not count them (`UINT64_MAX`
    /// on the ABI). A provider that counts reports the number; one that
    /// does not must not report zero.
    pub host_allocs: Option<u64>,
    /// Explicit host-to-device bytes.
    pub h2d_bytes: u64,
    /// Explicit device-to-host bytes.
    pub d2h_bytes: u64,
    /// Bytes of bound input storage.
    pub input_bytes: u64,
    /// Bytes of bound output storage.
    pub output_bytes: u64,
    /// Provider-internal allocations, or `None` if unknown.
    pub provider_allocs: Option<u64>,
}

/// One output tensor of a run.
#[derive(Clone)]
pub struct Output {
    /// Name (`embeddings`, `scores`, `sorted`, `labels`, or the graph's own name).
    /// Shared so cloning a result does not allocate.
    pub name: Arc<str>,
    /// Storage. Reused by the provider across runs; the core's lease keeps it
    /// stable while a result is outstanding.
    pub buffer: Arc<dyn ProviderBuffer>,
    /// Logical shape (may be smaller than the buffer's capacity shape).
    pub shape: Vec<u64>,
}

impl Output {
    /// Element type.
    pub fn dtype(&self) -> DType {
        self.buffer.desc().dtype
    }

    /// Placement.
    pub fn placement(&self) -> Placement {
        self.buffer.desc().placement
    }

    /// Bytes covered by the logical shape (packed).
    pub fn logical_bytes(&self) -> Result<u64> {
        BufferDesc::packed(self.placement(), self.dtype(), &self.shape).map(|d| d.bytes)
    }
}

/// One aggregated span from token classification.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Span {
    /// Input row.
    pub row: u32,
    /// Byte start in that row's input.
    pub byte_start: u64,
    /// Byte end (exclusive).
    pub byte_end: u64,
    /// Label index.
    pub label: u32,
    /// Score.
    pub score: f32,
}

/// Result of one run.
#[derive(Clone, Default)]
pub struct ProviderResult {
    /// Output tensors; output 0 is the primary.
    pub outputs: Vec<Output>,
    /// Aggregated spans (token classification).
    pub spans: Vec<Span>,
}

/// One chat message.
#[derive(Clone, Copy, Debug)]
pub struct Message<'a> {
    /// Role.
    pub role: &'a str,
    /// Content.
    pub content: &'a str,
}

/// Generation parameters. Field indices follow the ABI struct.
#[derive(Clone, Debug)]
pub struct GenerateDesc {
    /// Maximum new tokens, 0 = model default.
    pub max_new_tokens: u32,
    /// Minimum new tokens.
    pub min_new_tokens: u32,
    /// Sequences, 0 or 1 = one.
    pub n_sequences: u32,
    /// Temperature, 0 = greedy.
    pub temperature: f32,
    /// Top-k, 0 = off.
    pub top_k: u32,
    /// Top-p, 0 or 1 = off.
    pub top_p: f32,
    /// Min-p, 0 = off.
    pub min_p: f32,
    /// Repeat penalty, 0 or 1 = off.
    pub repeat_penalty: f32,
    /// Presence penalty.
    pub presence_penalty: f32,
    /// Frequency penalty.
    pub frequency_penalty: f32,
    /// Seed, if any.
    pub seed: Option<u64>,
    /// Stop strings.
    pub stop: Vec<String>,
    /// Stop token ids.
    pub stop_tokens: Vec<i32>,
    /// Logit bias.
    pub logit_bias: Vec<(i32, f32)>,
    /// Top logprobs per token, 0 = none.
    pub logprobs: u32,
    /// Structured output kind.
    pub structured_kind: StructuredKind,
    /// Schema or grammar.
    pub structured: String,
    /// Include the prompt in output.
    pub echo: bool,
    /// Tool definitions (JSON).
    pub tools: Vec<String>,
    /// Provider options.
    pub options: Options,
}

impl Default for GenerateDesc {
    fn default() -> Self {
        Self {
            max_new_tokens: 0,
            min_new_tokens: 0,
            n_sequences: 1,
            temperature: 0.0,
            top_k: 0,
            top_p: 0.0,
            min_p: 0.0,
            repeat_penalty: 0.0,
            presence_penalty: 0.0,
            frequency_penalty: 0.0,
            seed: None,
            stop: Vec::new(),
            stop_tokens: Vec::new(),
            logit_bias: Vec::new(),
            logprobs: 0,
            structured_kind: StructuredKind::None,
            structured: String::new(),
            echo: false,
            tools: Vec::new(),
            options: Options::default(),
        }
    }
}

impl GenerateDesc {
    /// ABI field index of `n_sequences`.
    pub const FIELD_N_SEQUENCES: u32 = 4;
    /// ABI field index of `repeat_penalty`.
    pub const FIELD_REPEAT_PENALTY: u32 = 9;
    /// ABI field index of `presence_penalty`.
    pub const FIELD_PRESENCE_PENALTY: u32 = 10;
    /// ABI field index of `frequency_penalty`.
    pub const FIELD_FREQUENCY_PENALTY: u32 = 11;
    /// ABI field index of `has_seed`.
    pub const FIELD_HAS_SEED: u32 = 12;
    /// ABI field index of `n_stop`.
    pub const FIELD_N_STOP: u32 = 14;
    /// ABI field index of `n_logit_bias`.
    pub const FIELD_N_LOGIT_BIAS: u32 = 18;
    /// ABI field index of `logprobs`.
    pub const FIELD_LOGPROBS: u32 = 19;
    /// ABI field index of `structured_kind`.
    pub const FIELD_STRUCTURED_KIND: u32 = 21;
    /// ABI field index of `n_tools`.
    pub const FIELD_N_TOOLS: u32 = 24;
}

/// One generation step. Buffers are reused across steps by the caller.
#[derive(Clone, Debug, Default)]
pub struct Chunk {
    /// Sequence index.
    pub sequence: u32,
    /// New token ids.
    pub tokens: Vec<i32>,
    /// Decoded text for the new tokens.
    pub text: String,
    /// Logprobs, `tokens.len() * logprobs` entries.
    pub logprobs: Vec<f32>,
    /// Finished.
    pub done: bool,
    /// Why, when done.
    pub finish_reason: FinishReason,
    /// Prompt tokens consumed.
    pub prompt_tokens: u32,
    /// Generated tokens so far.
    pub generated_tokens: u32,
}

impl Chunk {
    /// Clear for reuse.
    pub fn clear(&mut self) {
        self.sequence = 0;
        self.tokens.clear();
        self.text.clear();
        self.logprobs.clear();
        self.done = false;
        self.finish_reason = FinishReason::None;
        self.prompt_tokens = 0;
        self.generated_tokens = 0;
    }
}

/// A provider: a set of devices and the ability to run tasks on them.
pub trait Provider: Send + Sync {
    /// Stable id (`mock`, `static`, `cuda`, `openvino`, `metal`, `hailo`, `ggml`).
    fn id(&self) -> &str;

    /// Provider library version.
    fn version(&self) -> &str;

    /// Enumerate devices. A probe failure is an error the runtime records
    /// and logs; the provider then contributes no devices.
    fn devices(&self) -> Result<Vec<DeviceInfo>>;

    /// Capability cell for (device ordinal, task, modality).
    fn capability(&self, ordinal: u32, task: Task, modality: Modality) -> Capability;

    /// Per-request feasibility check without allocation.
    fn can_run(&self, ordinal: u32, bundle: &Bundle, task: Task, modality: Modality) -> Result<()>;

    /// Create a context on a device.
    fn create_context(&self, ordinal: u32, desc: &ContextDesc) -> Result<Arc<dyn ProviderContext>>;
}

/// A device plus memory domain.
pub trait ProviderContext: Send + Sync {
    /// Device ordinal within the provider.
    fn ordinal(&self) -> u32;

    /// Allocate a buffer.
    fn alloc(&self, desc: &BufferDesc) -> Result<Arc<dyn ProviderBuffer>>;

    /// Wrap caller memory without copying.
    fn import(&self, desc: &BufferDesc, handle: &NativeHandle) -> Result<Arc<dyn ProviderBuffer>>;

    /// Load a verified bundle.
    fn load_model(&self, bundle: Arc<Bundle>, desc: &ModelDesc) -> Result<Arc<dyn ProviderModel>>;
}

/// A loaded, immutable model.
pub trait ProviderModel: Send + Sync {
    /// What loaded.
    fn info(&self) -> &ModelInfo;

    /// Create a session with a fixed maximum shape.
    fn create_session(&self, desc: &SessionDesc) -> Result<Box<dyn ProviderSession>>;

    /// Create a generation. Default: unsupported.
    fn create_generation(&self, _desc: &GenerateDesc) -> Result<Box<dyn ProviderGeneration>> {
        Err(Error::unsupported_task(format!(
            "model `{}` on provider `{}` does not generate",
            self.info().model_id,
            self.info().provider_id
        )))
    }
}

/// An execution workspace. Single owner; the core serializes calls.
pub trait ProviderSession: Send {
    /// Write texts for embedding.
    fn write_text(&mut self, _texts: &[&str], _opts: &EmbedOptions) -> Result<()> {
        Err(Error::unsupported_task("this session does not embed text"))
    }

    /// Write caller-prepared tokens.
    fn write_tokens(&mut self, _batch: &TokenBatch<'_>) -> Result<()> {
        Err(Error::unsupported_task("this session does not accept prepared tokens"))
    }

    /// Write a query and documents for reranking.
    fn write_pairs(&mut self, _query: &str, _docs: &[&str], _opts: &RerankOptions) -> Result<()> {
        Err(Error::unsupported_task("this session does not rerank"))
    }

    /// Write texts for classification or token classification.
    fn write_text_classify(&mut self, _texts: &[&str], _opts: &ClassifyOptions) -> Result<()> {
        Err(Error::unsupported_task("this session does not classify"))
    }

    /// Bind a named input or output (RUN).
    fn bind(&mut self, _name: &str, _buffer: Arc<dyn ProviderBuffer>) -> Result<()> {
        Err(Error::unsupported_task("this session does not take named bindings"))
    }

    /// Execute. Must not allocate on the host after warmup where the
    /// provider claims `host_allocs == 0`.
    fn run(&mut self, opts: &RunOptions) -> Result<ProviderResult>;

    /// Counters.
    fn stats(&self) -> SessionStats;
}

/// Streaming generation state. Single owner.
pub trait ProviderGeneration: Send {
    /// Apply the chat template and tokenize.
    fn prompt(&mut self, messages: &[Message<'_>]) -> Result<()>;

    /// Use caller-supplied prompt tokens.
    fn prompt_tokens(&mut self, ids: &[i32]) -> Result<()>;

    /// Produce the next chunk into `out` (cleared first).
    fn step(&mut self, out: &mut Chunk) -> Result<()>;

    /// Cancel; subsequent steps report `Cancelled`.
    fn cancel(&mut self);
}

#[cfg(test)]
mod tests {
    use super::*;
    use turbo_abi as abi;

    #[test]
    fn token_batch_validation() {
        let ids = [1, 2, 3, 4];
        let mask = [1, 1, 1, 0];
        let b = TokenBatch { batch: 2, seq: 2, row_stride: 2, ids: &ids, mask: &mask, types: None };
        b.validate(10).unwrap();
        assert_eq!(b.ids_row(1), &[3, 4]);
        let bad = TokenBatch { ids: &ids[..3], ..b };
        assert_eq!(bad.validate(10).unwrap_err().code(), abi::TURBO_E_INVALID_SHAPE);
        let oob = TokenBatch { ids: &[1, 2, 3, 99], ..b };
        assert_eq!(oob.validate(10).unwrap_err().code(), abi::TURBO_E_INVALID_ARGUMENT);
        let badmask = TokenBatch { mask: &[1, 2, 1, 0], ..b };
        assert_eq!(badmask.validate(10).unwrap_err().code(), abi::TURBO_E_INVALID_ARGUMENT);
    }

    #[test]
    fn options_reject_unknown_with_field_index() {
        let o = Options(vec![("a".into(), "1".into()), ("zzz".into(), "2".into())]);
        let err = o.reject_unknown(&["a"], "context").unwrap_err();
        assert_eq!(err.code(), abi::TURBO_E_INVALID_ARGUMENT);
        assert_eq!(err.field(), 2);
    }
}
