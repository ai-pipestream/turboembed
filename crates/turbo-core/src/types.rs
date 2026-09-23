//! Typed views of the `u32` constants in `turbo-abi`.
//!
//! Every enumeration here round-trips through its ABI constant and rejects
//! unknown values with `TURBO_E_INVALID_ENUM` rather than mapping them to a
//! default.

use turbo_abi as abi;

use crate::error::{Error, Result};

macro_rules! abi_enum {
    (
        $(#[$meta:meta])*
        $name:ident, $what:literal, {
            $( $(#[$vmeta:meta])* $variant:ident = $konst:path ),+ $(,)?
        }
    ) => {
        $(#[$meta])*
        #[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
        #[repr(u32)]
        pub enum $name {
            $( $(#[$vmeta])* #[doc = stringify!($konst)] $variant = $konst ),+
        }

        impl $name {
            /// Parse the ABI constant; unknown values are `TURBO_E_INVALID_ENUM`.
            pub fn from_abi(value: u32) -> Result<Self> {
                match value {
                    $( x if x == $konst => Ok(Self::$variant), )+
                    other => Err(Error::invalid_enum($what, other)),
                }
            }

            /// The ABI constant.
            pub const fn as_abi(self) -> u32 {
                self as u32
            }

            /// All variants, in constant order.
            pub const ALL: &'static [Self] = &[ $( Self::$variant ),+ ];
        }
    };
}

abi_enum!(
    /// Hardware class of a device.
    DeviceKind, "device kind", {
        Cpu = abi::TURBO_DEVICE_CPU,
        Gpu = abi::TURBO_DEVICE_GPU,
        IGpu = abi::TURBO_DEVICE_IGPU,
        Npu = abi::TURBO_DEVICE_NPU,
        Accel = abi::TURBO_DEVICE_ACCEL,
    }
);

abi_enum!(
    /// Task a model or session performs.
    Task, "task", {
        Embed = abi::TURBO_TASK_EMBED,
        Rerank = abi::TURBO_TASK_RERANK,
        Classify = abi::TURBO_TASK_CLASSIFY,
        TokenClassify = abi::TURBO_TASK_TOKEN_CLASSIFY,
        Generate = abi::TURBO_TASK_GENERATE,
        Tokenize = abi::TURBO_TASK_TOKENIZE,
        Run = abi::TURBO_TASK_RUN,
        Chunk = abi::TURBO_TASK_CHUNK,
    }
);

abi_enum!(
    /// Input modality.
    Modality, "modality", {
        Text = abi::TURBO_MODALITY_TEXT,
        Audio = abi::TURBO_MODALITY_AUDIO,
        Image = abi::TURBO_MODALITY_IMAGE,
        Video = abi::TURBO_MODALITY_VIDEO,
    }
);

abi_enum!(
    /// Qualification status of a capability cell.
    CapStatus, "capability status", {
        Unsupported = abi::TURBO_CAP_UNSUPPORTED,
        Planned = abi::TURBO_CAP_PLANNED,
        Experimental = abi::TURBO_CAP_EXPERIMENTAL,
        Supported = abi::TURBO_CAP_SUPPORTED,
    }
);

abi_enum!(
    /// What kind of model a bundle holds.
    ModelKind, "model kind", {
        Embedding = abi::TURBO_MODEL_EMBEDDING,
        Reranker = abi::TURBO_MODEL_RERANKER,
        Classifier = abi::TURBO_MODEL_CLASSIFIER,
        TokenClassifier = abi::TURBO_MODEL_TOKEN_CLASSIFIER,
        Generative = abi::TURBO_MODEL_GENERATIVE,
        Generic = abi::TURBO_MODEL_GENERIC,
    }
);

abi_enum!(
    /// Element type.
    DType, "dtype", {
        Bool = abi::TURBO_DTYPE_BOOL,
        U8 = abi::TURBO_DTYPE_U8,
        U16 = abi::TURBO_DTYPE_U16,
        U32 = abi::TURBO_DTYPE_U32,
        U64 = abi::TURBO_DTYPE_U64,
        I8 = abi::TURBO_DTYPE_I8,
        I16 = abi::TURBO_DTYPE_I16,
        I32 = abi::TURBO_DTYPE_I32,
        I64 = abi::TURBO_DTYPE_I64,
        F16 = abi::TURBO_DTYPE_F16,
        BF16 = abi::TURBO_DTYPE_BF16,
        F32 = abi::TURBO_DTYPE_F32,
        F64 = abi::TURBO_DTYPE_F64,
        Bytes = abi::TURBO_DTYPE_BYTES,
    }
);

impl DType {
    /// Element size in bytes, or `None` for variable-length `Bytes`.
    pub const fn element_size(self) -> Option<usize> {
        match self {
            DType::Bool | DType::U8 | DType::I8 => Some(1),
            DType::U16 | DType::I16 | DType::F16 | DType::BF16 => Some(2),
            DType::U32 | DType::I32 | DType::F32 => Some(4),
            DType::U64 | DType::I64 | DType::F64 => Some(8),
            DType::Bytes => None,
        }
    }

    /// Lower-case name used in bundle manifests and messages.
    pub const fn name(self) -> &'static str {
        match self {
            DType::Bool => "bool",
            DType::U8 => "u8",
            DType::U16 => "u16",
            DType::U32 => "u32",
            DType::U64 => "u64",
            DType::I8 => "i8",
            DType::I16 => "i16",
            DType::I32 => "i32",
            DType::I64 => "i64",
            DType::F16 => "f16",
            DType::BF16 => "bf16",
            DType::F32 => "f32",
            DType::F64 => "f64",
            DType::Bytes => "bytes",
        }
    }

    /// Parse a manifest name.
    pub fn from_name(name: &str) -> Result<Self> {
        Self::ALL
            .iter()
            .copied()
            .find(|d| d.name() == name)
            .ok_or_else(|| Error::bundle_invalid(format!("unknown dtype name `{name}`")))
    }
}

abi_enum!(
    /// Memory placement.
    Placement, "placement", {
        Host = abi::TURBO_PLACE_HOST,
        Pinned = abi::TURBO_PLACE_PINNED,
        Device = abi::TURBO_PLACE_DEVICE,
        Shared = abi::TURBO_PLACE_SHARED,
    }
);

impl Placement {
    /// Whether the host can map this placement for direct access.
    pub const fn host_visible(self) -> bool {
        matches!(self, Placement::Host | Placement::Pinned | Placement::Shared)
    }
}

abi_enum!(
    /// Truncation policy.
    Truncate, "truncate", {
        Model = abi::TURBO_TRUNCATE_MODEL,
        None = abi::TURBO_TRUNCATE_NONE,
        Right = abi::TURBO_TRUNCATE_RIGHT,
        Left = abi::TURBO_TRUNCATE_LEFT,
    }
);

abi_enum!(
    /// Prompt prefix role.
    PromptRole, "prompt_role", {
        None = abi::TURBO_PROMPT_NONE,
        Query = abi::TURBO_PROMPT_QUERY,
        Document = abi::TURBO_PROMPT_DOCUMENT,
    }
);

abi_enum!(
    /// Normalization policy.
    Normalize, "normalize", {
        Model = abi::TURBO_NORMALIZE_MODEL,
        None = abi::TURBO_NORMALIZE_NONE,
        L2 = abi::TURBO_NORMALIZE_L2,
    }
);

abi_enum!(
    /// Pooling policy.
    Pooling, "pooling", {
        Model = abi::TURBO_POOLING_MODEL,
        Mean = abi::TURBO_POOLING_MEAN,
        Cls = abi::TURBO_POOLING_CLS,
        Last = abi::TURBO_POOLING_LAST,
    }
);

impl Pooling {
    /// Parse a bundle contract name.
    pub fn from_name(name: &str) -> Result<Self> {
        match name {
            "mean" => Ok(Pooling::Mean),
            "cls" => Ok(Pooling::Cls),
            "last" | "lasttoken" | "last_token" => Ok(Pooling::Last),
            other => Err(Error::bundle_invalid(format!("unknown pooling `{other}`"))),
        }
    }
}

abi_enum!(
    /// Output element type request.
    OutputDType, "output_dtype", {
        Model = abi::TURBO_OUTPUT_MODEL,
        F32 = abi::TURBO_OUTPUT_F32,
        F16 = abi::TURBO_OUTPUT_F16,
        I8 = abi::TURBO_OUTPUT_I8,
    }
);

abi_enum!(
    /// Token-classification span aggregation.
    Aggregation, "aggregation", {
        Model = abi::TURBO_AGGREGATE_MODEL,
        None = abi::TURBO_AGGREGATE_NONE,
        Simple = abi::TURBO_AGGREGATE_SIMPLE,
        First = abi::TURBO_AGGREGATE_FIRST,
        Max = abi::TURBO_AGGREGATE_MAX,
    }
);

impl Aggregation {
    /// Parse a bundle contract name.
    pub fn from_name(name: &str) -> Result<Self> {
        match name {
            "none" => Ok(Aggregation::None),
            "simple" => Ok(Aggregation::Simple),
            "first" => Ok(Aggregation::First),
            "max" => Ok(Aggregation::Max),
            other => Err(Error::bundle_invalid(format!("unknown aggregation `{other}`"))),
        }
    }
}

abi_enum!(
    /// Where a pipeline stage executes.
    StagePlacement, "stage placement", {
        Unused = abi::TURBO_STAGE_UNUSED,
        Host = abi::TURBO_STAGE_HOST,
        Device = abi::TURBO_STAGE_DEVICE,
        Fused = abi::TURBO_STAGE_FUSED,
    }
);

abi_enum!(
    /// Why a generated sequence ended.
    #[derive(Default)]
    FinishReason, "finish_reason", {
        #[default]
        None = abi::TURBO_FINISH_NONE,
        Eos = abi::TURBO_FINISH_EOS,
        Stop = abi::TURBO_FINISH_STOP,
        Length = abi::TURBO_FINISH_LENGTH,
        Cancelled = abi::TURBO_FINISH_CANCELLED,
    }
);

abi_enum!(
    /// Structured-output constraint kind.
    StructuredKind, "structured_kind", {
        None = abi::TURBO_STRUCTURED_NONE,
        JsonSchema = abi::TURBO_STRUCTURED_JSON_SCHEMA,
        Grammar = abi::TURBO_STRUCTURED_GRAMMAR,
    }
);

abi_enum!(
    /// Native memory handle kind.
    HandleKind, "native handle kind", {
        HostPtr = abi::TURBO_HANDLE_HOST_PTR,
        CudaPtr = abi::TURBO_HANDLE_CUDA_PTR,
        ClMem = abi::TURBO_HANDLE_CL_MEM,
        ZeUsm = abi::TURBO_HANDLE_ZE_USM,
        MtlBuffer = abi::TURBO_HANDLE_MTL_BUFFER,
        DmabufFd = abi::TURBO_HANDLE_DMABUF_FD,
    }
);

abi_enum!(
    /// Device selection policy.
    #[derive(Default)]
    SelectPolicy, "select policy", {
        #[default]
        Auto = abi::TURBO_SELECT_AUTO,
        Explicit = abi::TURBO_SELECT_EXPLICIT,
    }
);

/// Pipeline stage indices, matching `TURBO_STAGE_*`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u32)]
pub enum Stage {
    /// Tokenization.
    Tokenize = abi::TURBO_STAGE_TOKENIZE,
    /// Encoder / forward.
    Encode = abi::TURBO_STAGE_ENCODE,
    /// Pooling.
    Pool = abi::TURBO_STAGE_POOL,
    /// Normalization.
    Normalize = abi::TURBO_STAGE_NORMALIZE,
    /// Post-processing.
    Postprocess = abi::TURBO_STAGE_POSTPROCESS,
}

/// Per-stage placement report, one entry per [`Stage`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct StagePlacements(pub [StagePlacement; abi::TURBO_STAGE_COUNT]);

impl StagePlacements {
    /// All stages unused.
    pub const NONE: Self = Self([StagePlacement::Unused; abi::TURBO_STAGE_COUNT]);

    /// Set one stage.
    pub fn with(mut self, stage: Stage, placement: StagePlacement) -> Self {
        self.0[stage as usize] = placement;
        self
    }

    /// True when every used stage is on the device (`Device` or `Fused`).
    pub fn fully_accelerated(&self) -> bool {
        self.0.iter().all(|p| matches!(p, StagePlacement::Unused | StagePlacement::Device | StagePlacement::Fused))
            && self.0.iter().any(|p| !matches!(p, StagePlacement::Unused))
    }

    /// ABI array.
    pub fn as_abi(&self) -> [u32; abi::TURBO_STAGE_COUNT] {
        let mut out = [0u32; abi::TURBO_STAGE_COUNT];
        for (o, p) in out.iter_mut().zip(self.0.iter()) {
            *o = p.as_abi();
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn unknown_enum_values_are_rejected() {
        let err = Task::from_abi(99).unwrap_err();
        assert_eq!(err.code(), abi::TURBO_E_INVALID_ENUM);
        assert!(err.message().contains("task"));
    }

    #[test]
    fn dtype_sizes() {
        assert_eq!(DType::F32.element_size(), Some(4));
        assert_eq!(DType::Bytes.element_size(), None);
        assert_eq!(DType::from_name("bf16").unwrap(), DType::BF16);
        assert!(DType::from_name("float").is_err());
    }

    #[test]
    fn full_acceleration_requires_a_used_stage() {
        assert!(!StagePlacements::NONE.fully_accelerated());
        let p =
            StagePlacements::NONE.with(Stage::Encode, StagePlacement::Device).with(Stage::Pool, StagePlacement::Fused);
        assert!(p.fully_accelerated());
        let q = p.with(Stage::Tokenize, StagePlacement::Host);
        assert!(!q.fully_accelerated());
    }
}
