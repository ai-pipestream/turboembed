"""Low-level ctypes bindings for libturbo (`include/turbo/turbo.h`)."""

from __future__ import annotations

import ctypes
from ctypes import (
    CFUNCTYPE,
    POINTER,
    Structure,
    c_char,
    c_char_p,
    c_double,
    c_float,
    c_int,
    c_int32,
    c_int64,
    c_uint8,
    c_uint32,
    c_uint64,
    c_void_p,
)
import os
from pathlib import Path
import sys

TURBO_ERROR_MESSAGE_LEN = 496
TURBO_STAGE_MAX = 16
TURBO_OUTPUT_DIMS_MAX = 16
TURBO_CHOICES_LEN = 1024

TURBO_OK = 0
TURBO_E_INVALID_ARGUMENT = 1
TURBO_E_INVALID_STRUCT_SIZE = 2
TURBO_E_INVALID_UTF8 = 3
TURBO_E_INVALID_HANDLE = 4
TURBO_E_INVALID_SHAPE = 5
TURBO_E_INVALID_STATE = 6
TURBO_E_INVALID_ENUM = 7
TURBO_E_UNSUPPORTED = 8
TURBO_E_UNSUPPORTED_OPTION = 9
TURBO_E_UNSUPPORTED_TASK = 10
TURBO_E_OUT_OF_MEMORY = 11
TURBO_E_BUSY = 12
TURBO_E_CAPACITY = 13
TURBO_E_DEVICE_NOT_FOUND = 14
TURBO_E_DEVICE_UNAVAILABLE = 15
TURBO_E_RUNTIME = 16
TURBO_E_BUNDLE_NOT_FOUND = 17
TURBO_E_BUNDLE_INVALID = 18
TURBO_E_BUNDLE_INTEGRITY = 19
TURBO_E_BUNDLE_NO_ARTIFACT = 20
TURBO_E_INTERNAL = 21
TURBO_E_PANIC = 22

TURBO_TASK_EMBED = 1
TURBO_TASK_RERANK = 2
TURBO_TASK_EXTRACTIVE_QA = 3

TURBO_DEVICE_CPU = 1
TURBO_DEVICE_GPU = 2
TURBO_DEVICE_IGPU = 3
TURBO_DEVICE_NPU = 4

TURBO_DTYPE_I8 = 6
TURBO_DTYPE_U8 = 7
TURBO_DTYPE_I32 = 8
TURBO_DTYPE_I64 = 9
TURBO_DTYPE_F16 = 10
TURBO_DTYPE_BF16 = 11
TURBO_DTYPE_F32 = 12
TURBO_DTYPE_F64 = 13

TURBO_PLACE_HOST = 1
TURBO_PLACE_PINNED = 2
TURBO_PLACE_DEVICE = 3
TURBO_PLACE_SHARED = 4

TURBO_HANDLE_HOST_PTR = 1

TURBO_PRECISION_MODEL = 0
TURBO_PRECISION_FASTEST = 1
TURBO_PRECISION_EXACT = 2

TURBO_AUTOTUNE_RUNTIME = 0
TURBO_AUTOTUNE_OFF = 1
TURBO_AUTOTUNE_ON = 2
TURBO_AUTOTUNE_RETUNE = 3

TURBO_TRUNCATE_MODEL = 0
TURBO_TRUNCATE_NONE = 1
TURBO_TRUNCATE_RIGHT = 2
TURBO_TRUNCATE_LEFT = 3

TURBO_PROMPT_NONE = 0
TURBO_PROMPT_QUERY = 1
TURBO_PROMPT_DOCUMENT = 2

TURBO_NORMALIZE_MODEL = 0
TURBO_NORMALIZE_NONE = 1
TURBO_NORMALIZE_L2 = 2

TURBO_POOLING_MODEL = 0
TURBO_POOLING_MEAN = 1
TURBO_POOLING_CLS = 2
TURBO_POOLING_LAST = 3

TURBO_QUANTIZE_MODEL = 0
TURBO_QUANTIZE_NONE = 1
TURBO_QUANTIZE_INT8 = 2
TURBO_QUANTIZE_UBINARY = 3

TURBO_ANALYSIS_LAYER_SENTENCES = 1
TURBO_ANALYSIS_LAYER_TOKENS = 2
TURBO_ANALYSIS_LAYER_POS = 3
TURBO_ANALYSIS_LAYER_LEMMAS = 4
TURBO_ANALYSIS_LAYER_ENTITIES = 5
TURBO_ANALYSIS_LAYER_PII = 6
TURBO_ANALYSIS_LAYER_CHUNKS = 7

TURBO_SPAN_VIEW_RANGE = 1 << 0
TURBO_SPAN_SOURCE_RANGE = 1 << 1
TURBO_SPAN_TOKEN_RANGE = 1 << 2
TURBO_SPAN_LABEL = 1 << 3
TURBO_SPAN_VALUE = 1 << 4
TURBO_SPAN_PROBABILITY = 1 << 5

TURBO_PII_CONFIDENCE_FORMAT = 1
TURBO_PII_CONFIDENCE_CHECKSUM = 2
TURBO_PII_CONFIDENCE_LABELLED = 3

ARROW_FLAG_DICTIONARY_ORDERED = 1
ARROW_FLAG_NULLABLE = 2
ARROW_FLAG_MAP_KEYS_SORTED = 4


class TurboText(Structure):
    _fields_ = [
        ("ptr", c_char_p),
        ("len", c_uint64),
    ]


class TurboError(Structure):
    _fields_ = [
        ("struct_size", c_uint32),
        ("code", c_int32),
        ("field", c_uint32),
        ("message", c_char * TURBO_ERROR_MESSAGE_LEN),
    ]


class TurboRuntimeDesc(Structure):
    _fields_ = [
        ("struct_size", c_uint32),
        ("reserved", c_uint32),
        ("log", c_void_p),
        ("log_user_data", c_void_p),
    ]


class TurboDeviceInfo(Structure):
    _fields_ = [
        ("struct_size", c_uint32),
        ("kind", c_uint32),
        ("ordinal", c_uint32),
        ("unified_memory", c_uint32),
        ("memory_total", c_uint64),
        ("memory_free", c_uint64),
        ("arch", c_char * 32),
        ("name", c_char * 128),
        ("vendor", c_char * 64),
        ("backend", c_char * 32),
        ("runtime_version", c_char * 64),
        ("driver_version", c_char * 64),
    ]


class TurboNativeHandle(Structure):
    _fields_ = [
        ("struct_size", c_uint32),
        ("kind", c_uint32),
        ("handle", c_uint64),
        ("aux", c_uint64),
        ("offset", c_uint64),
    ]


class TurboBufferDesc(Structure):
    _fields_ = [
        ("struct_size", c_uint32),
        ("placement", c_uint32),
        ("dtype", c_uint32),
        ("ndim", c_uint32),
        ("shape", c_uint64 * 2),
        ("bytes", c_uint64),
    ]


class TurboModelInfo(Structure):
    _fields_ = [
        ("struct_size", c_uint32),
        ("task", c_uint32),
        ("dim", c_uint32),
        ("pooling", c_uint32),
        ("normalize", c_uint32),
        ("max_seq", c_uint32),
        ("max_batch", c_uint32),
        ("dtype", c_uint32),
        ("model_id", c_char * 128),
        ("revision", c_char * 64),
        ("manifest_sha256", c_char * 72),
        ("artifact_sha256", c_char * 72),
        ("tokenizer_sha256", c_char * 72),
        ("prefix_query", c_char * 128),
        ("prefix_document", c_char * 128),
        ("output_dims_count", c_uint32),
        ("output_dims", c_uint32 * TURBO_OUTPUT_DIMS_MAX),
        ("rerank_query_tokens", c_uint32),
        ("rerank_pair_template", c_uint32),
    ]


class TurboTokenizerInfo(Structure):
    _fields_ = [
        ("struct_size", c_uint32),
        ("vocab_size", c_uint32),
        ("max_seq", c_uint32),
        ("specials_per_sequence", c_uint32),
        ("pad_id", c_int32),
        ("bos_id", c_int32),
        ("eos_id", c_int32),
        ("unk_id", c_int32),
        ("kind", c_char * 32),
        ("sha256", c_char * 72),
        ("manifest_sha256", c_char * 72),
    ]


class TurboEncodeOptions(Structure):
    _fields_ = [
        ("struct_size", c_uint32),
        ("omit_special_tokens", c_uint32),
        ("truncate", c_uint32),
        ("max_tokens", c_uint32),
        ("prompt_role", c_uint32),
    ]


class TurboSessionDesc(Structure):
    _fields_ = [
        ("struct_size", c_uint32),
        ("max_batch", c_uint32),
        ("max_seq", c_uint32),
        ("precision", c_uint32),
        ("tuning", c_uint32),
        ("tuning_budget_ms", c_uint32),
        ("cpu_threads", c_uint32),
        ("max_ranges", c_uint32),
    ]


class TurboEmbedOptions(Structure):
    _fields_ = [
        ("struct_size", c_uint32),
        ("truncate", c_uint32),
        ("max_tokens", c_uint32),
        ("prompt_role", c_uint32),
        ("normalize", c_uint32),
        ("pooling", c_uint32),
        ("output_dim", c_uint32),
        ("quantization", c_uint32),
        ("int8_ranges", POINTER(c_float)),
    ]


class TurboTokenBatch(Structure):
    _fields_ = [
        ("struct_size", c_uint32),
        ("batch", c_uint32),
        ("seq", c_uint32),
        ("row_stride", c_uint32),
        ("ids", POINTER(c_int32)),
        ("mask", POINTER(c_int32)),
        ("types", POINTER(c_int32)),
    ]


class TurboTokenRange(Structure):
    _fields_ = [
        ("row", c_uint32),
        ("start", c_uint32),
        ("end", c_uint32),
    ]


class TurboRerankOptions(Structure):
    _fields_ = [
        ("struct_size", c_uint32),
        ("query_tokens", c_uint32),
        ("max_tokens", c_uint32),
        ("stride", c_uint32),
    ]


class TurboTextPair(Structure):
    _fields_ = [
        ("query", TurboText),
        ("passage", TurboText),
    ]


class TurboResultInfo(Structure):
    _fields_ = [
        ("struct_size", c_uint32),
        ("task", c_uint32),
        ("batch", c_uint32),
        ("dim", c_uint32),
        ("dtype", c_uint32),
        ("compute_dtype", c_uint32),
        ("placement", c_uint32),
        ("device", c_uint32),
        ("bytes", c_uint64),
        ("h2d_bytes", c_uint64),
        ("d2h_bytes", c_uint64),
        ("host_allocs", c_uint64),
        ("device_allocs", c_uint64),
        ("stage_count", c_uint32),
        ("reserved", c_uint32),
        ("stage", c_uint32 * TURBO_STAGE_MAX),
        ("backend", c_char * 32),
        ("arch", c_char * 32),
        ("runtime_version", c_char * 64),
        ("manifest_sha256", c_char * 72),
        ("artifact_sha256", c_char * 72),
        ("tokenizer_sha256", c_char * 72),
    ]


class TurboAnalyzerDesc(Structure):
    _fields_ = [
        ("struct_size", c_uint32),
        ("flags", c_uint32),
        ("language", TurboText),
        ("sent_model_path", TurboText),
        ("tok_model_path", TurboText),
        ("pos_model_path", TurboText),
        ("lemma_model_path", TurboText),
        ("ner_model_path", TurboText),
    ]


class TurboSpan(Structure):
    _fields_ = [
        ("layer_kind", c_uint32),
        ("flags", c_uint32),
        ("item_index", c_uint32),
        ("fragment_index", c_uint32),
        ("view_id", c_uint32),
        ("classification", c_uint32),
        ("view_byte_start", c_uint32),
        ("view_byte_end", c_uint32),
        ("view_utf16_start", c_uint32),
        ("view_utf16_end", c_uint32),
        ("view_codepoint_start", c_uint32),
        ("view_codepoint_end", c_uint32),
        ("source_byte_start", c_uint32),
        ("source_byte_end", c_uint32),
        ("source_utf16_start", c_uint32),
        ("source_utf16_end", c_uint32),
        ("source_codepoint_start", c_uint32),
        ("source_codepoint_end", c_uint32),
        ("token_start", c_uint32),
        ("token_end", c_uint32),
        ("layer_offset", c_uint32),
        ("layer_len", c_uint32),
        ("instance_offset", c_uint32),
        ("instance_len", c_uint32),
        ("label_offset", c_uint32),
        ("label_len", c_uint32),
        ("value_offset", c_uint32),
        ("value_len", c_uint32),
        ("prob", c_double),
    ]


class ArrowSchema(Structure):
    pass


ArrowSchemaReleaseFn = CFUNCTYPE(None, POINTER(ArrowSchema))

ArrowSchema._fields_ = [
    ("format", c_char_p),
    ("name", c_char_p),
    ("metadata", c_char_p),
    ("flags", c_int64),
    ("n_children", c_int64),
    ("children", POINTER(POINTER(ArrowSchema))),
    ("dictionary", POINTER(ArrowSchema)),
    ("release", ArrowSchemaReleaseFn),
    ("private_data", c_void_p),
]


class ArrowArray(Structure):
    pass


ArrowArrayReleaseFn = CFUNCTYPE(None, POINTER(ArrowArray))

ArrowArray._fields_ = [
    ("length", c_int64),
    ("null_count", c_int64),
    ("offset", c_int64),
    ("n_buffers", c_int64),
    ("n_children", c_int64),
    ("buffers", POINTER(c_void_p)),
    ("children", POINTER(POINTER(ArrowArray))),
    ("dictionary", POINTER(ArrowArray)),
    ("release", ArrowArrayReleaseFn),
    ("private_data", c_void_p),
]


class ArrowArrayStream(Structure):
    pass


ArrowStreamGetSchemaFn = CFUNCTYPE(c_int, POINTER(ArrowArrayStream), POINTER(ArrowSchema))
ArrowStreamGetNextFn = CFUNCTYPE(c_int, POINTER(ArrowArrayStream), POINTER(ArrowArray))
ArrowStreamGetLastErrorFn = CFUNCTYPE(c_char_p, POINTER(ArrowArrayStream))
ArrowStreamReleaseFn = CFUNCTYPE(None, POINTER(ArrowArrayStream))

ArrowArrayStream._fields_ = [
    ("get_schema", ArrowStreamGetSchemaFn),
    ("get_next", ArrowStreamGetNextFn),
    ("get_last_error", ArrowStreamGetLastErrorFn),
    ("release", ArrowStreamReleaseFn),
    ("private_data", c_void_p),
]


class TurboEmbedError(RuntimeError):
    """Exception raised when a libturbo C ABI call returns a non-zero status."""

    def __init__(self, code: int, status_name: str, field: int, message: str) -> None:
        self.code = code
        self.status_name = status_name
        self.field = field
        self.detail = message
        detail_suffix = f": {message}" if message else ""
        field_suffix = f" (field {field})" if field else ""
        super().__init__(f"{status_name} ({code}){field_suffix}{detail_suffix}")


def make_error() -> TurboError:
    err = TurboError()
    err.struct_size = ctypes.sizeof(TurboError)
    err.code = 0
    err.field = 0
    return err


def make_text(text_bytes: bytes) -> TurboText:
    return TurboText(ptr=ctypes.c_char_p(text_bytes), len=len(text_bytes))


def _platform_lib_names() -> list[str]:
    if sys.platform == "darwin":
        return ["libturbo.dylib"]
    if sys.platform == "win32":
        return ["turbo.dll", "libturbo.dll"]
    return ["libturbo.so"]


def _discover_lib_path() -> Path:
    for env_key in ("TURBOEMBED_LIBRARY_PATH", "TURBO_LIBRARY_PATH"):
        val = os.environ.get(env_key)
        if val:
            candidate = Path(val).expanduser().resolve()
            if candidate.is_file():
                return candidate
            raise FileNotFoundError(f"{env_key} points to missing file: {candidate}")

    pkg_dir = Path(__file__).resolve().parent
    names = _platform_lib_names()
    search_dirs = [
        pkg_dir,
        pkg_dir / "_lib",
    ]
    # Search upward for Cargo target directories when running in a repo checkout.
    for parent in pkg_dir.parents:
        if (parent / "Cargo.toml").is_file() and (parent / "core" / "Cargo.toml").is_file():
            target_dir = Path(os.environ.get("CARGO_TARGET_DIR", str(parent / "target")))
            search_dirs.extend([
                target_dir / "release",
                target_dir / "debug",
            ])
            break

    for directory in search_dirs:
        for name in names:
            candidate = directory / name
            if candidate.is_file():
                return candidate

    raise FileNotFoundError(
        "Could not locate libturbo shared library. Build `cargo build -p turbo` or set TURBOEMBED_LIBRARY_PATH."
    )


_LIB: ctypes.CDLL | None = None


def get_lib() -> ctypes.CDLL:
    global _LIB
    if _LIB is not None:
        return _LIB
    path = _discover_lib_path()
    # ctypes.CDLL releases the Python GIL around every C function call.
    lib = ctypes.CDLL(str(path))
    _bind_signatures(lib)
    _LIB = lib
    return lib


def check_status(code: int, err: TurboError) -> None:
    if code == TURBO_OK:
        return
    lib = get_lib()
    raw_name = lib.turbo_status_name(code)
    status_name = raw_name.decode("utf-8", errors="replace") if raw_name else f"TURBO_E_{code}"
    msg = err.message.decode("utf-8", errors="replace")
    raise TurboEmbedError(code, status_name, int(err.field), msg)


def _bind_signatures(lib: ctypes.CDLL) -> None:
    lib.turbo_status_name.argtypes = [c_int32]
    lib.turbo_status_name.restype = c_char_p

    lib.turbo_version.argtypes = []
    lib.turbo_version.restype = c_char_p

    lib.turbo_last_error_message.argtypes = []
    lib.turbo_last_error_message.restype = c_char_p

    lib.turbo_runtime_create.argtypes = [POINTER(TurboRuntimeDesc), POINTER(c_void_p), POINTER(TurboError)]
    lib.turbo_runtime_create.restype = c_int32
    lib.turbo_runtime_release.argtypes = [c_void_p]
    lib.turbo_runtime_release.restype = None

    lib.turbo_runtime_device_count.argtypes = [c_void_p, POINTER(c_uint32), POINTER(TurboError)]
    lib.turbo_runtime_device_count.restype = c_int32
    lib.turbo_runtime_device_info.argtypes = [c_void_p, c_uint32, POINTER(TurboDeviceInfo), POINTER(TurboError)]
    lib.turbo_runtime_device_info.restype = c_int32

    lib.turbo_context_create.argtypes = [c_void_p, c_uint32, POINTER(c_void_p), POINTER(TurboError)]
    lib.turbo_context_create.restype = c_int32
    lib.turbo_context_release.argtypes = [c_void_p]
    lib.turbo_context_release.restype = None

    lib.turbo_buffer_get_desc.argtypes = [c_void_p, POINTER(TurboBufferDesc), POINTER(TurboError)]
    lib.turbo_buffer_get_desc.restype = c_int32
    lib.turbo_buffer_host_ptr.argtypes = [c_void_p, POINTER(c_void_p), POINTER(TurboError)]
    lib.turbo_buffer_host_ptr.restype = c_int32
    lib.turbo_buffer_export.argtypes = [c_void_p, c_uint32, POINTER(TurboNativeHandle), POINTER(TurboError)]
    lib.turbo_buffer_export.restype = c_int32
    lib.turbo_buffer_release.argtypes = [c_void_p]
    lib.turbo_buffer_release.restype = None

    lib.turbo_model_load.argtypes = [c_void_p, TurboText, POINTER(c_void_p), POINTER(TurboError)]
    lib.turbo_model_load.restype = c_int32
    lib.turbo_model_get_info.argtypes = [c_void_p, POINTER(TurboModelInfo), POINTER(TurboError)]
    lib.turbo_model_get_info.restype = c_int32
    lib.turbo_model_release.argtypes = [c_void_p]
    lib.turbo_model_release.restype = None

    lib.turbo_tokenizer_create.argtypes = [c_void_p, TurboText, POINTER(c_void_p), POINTER(TurboError)]
    lib.turbo_tokenizer_create.restype = c_int32
    lib.turbo_tokenizer_get_info.argtypes = [c_void_p, POINTER(TurboTokenizerInfo), POINTER(TurboError)]
    lib.turbo_tokenizer_get_info.restype = c_int32
    lib.turbo_tokenizer_count.argtypes = [
        c_void_p,
        TurboText,
        c_uint32,
        POINTER(c_uint32),
        POINTER(TurboError),
    ]
    lib.turbo_tokenizer_count.restype = c_int32
    lib.turbo_tokenizer_encode.argtypes = [
        c_void_p,
        POINTER(TurboText),
        c_uint32,
        POINTER(TurboEncodeOptions),
        POINTER(c_int32),
        POINTER(c_int32),
        POINTER(c_int32),
        c_uint32,
        POINTER(c_uint32),
        POINTER(TurboError),
    ]
    lib.turbo_tokenizer_encode.restype = c_int32
    lib.turbo_tokenizer_release.argtypes = [c_void_p]
    lib.turbo_tokenizer_release.restype = None

    lib.turbo_session_create.argtypes = [
        c_void_p,
        POINTER(TurboSessionDesc),
        POINTER(c_void_p),
        POINTER(TurboError),
    ]
    lib.turbo_session_create.restype = c_int32
    lib.turbo_session_release.argtypes = [c_void_p]
    lib.turbo_session_release.restype = None

    lib.turbo_embed_write_text.argtypes = [
        c_void_p,
        POINTER(TurboText),
        c_uint32,
        POINTER(TurboEmbedOptions),
        POINTER(TurboError),
    ]
    lib.turbo_embed_write_text.restype = c_int32
    lib.turbo_embed_write_tokens.argtypes = [
        c_void_p,
        POINTER(TurboTokenBatch),
        POINTER(TurboEmbedOptions),
        POINTER(TurboError),
    ]
    lib.turbo_embed_write_tokens.restype = c_int32
    lib.turbo_embed_write_ranges.argtypes = [
        c_void_p,
        POINTER(TurboTokenBatch),
        POINTER(TurboTokenRange),
        c_uint32,
        POINTER(TurboEmbedOptions),
        POINTER(TurboError),
    ]
    lib.turbo_embed_write_ranges.restype = c_int32
    lib.turbo_rerank_write_pairs.argtypes = [
        c_void_p,
        POINTER(TurboTextPair),
        c_uint32,
        POINTER(TurboRerankOptions),
        POINTER(TurboError),
    ]
    lib.turbo_rerank_write_pairs.restype = c_int32

    lib.turbo_session_run.argtypes = [c_void_p, POINTER(c_void_p), POINTER(TurboError)]
    lib.turbo_session_run.restype = c_int32

    lib.turbo_result_get_info.argtypes = [c_void_p, POINTER(TurboResultInfo), POINTER(TurboError)]
    lib.turbo_result_get_info.restype = c_int32
    lib.turbo_result_read.argtypes = [c_void_p, c_void_p, c_uint64, POINTER(c_uint64), POINTER(TurboError)]
    lib.turbo_result_read.restype = c_int32
    lib.turbo_result_buffer.argtypes = [c_void_p, POINTER(c_void_p), POINTER(TurboError)]
    lib.turbo_result_buffer.restype = c_int32
    lib.turbo_result_release.argtypes = [c_void_p]
    lib.turbo_result_release.restype = None

    # Analysis & Arrow C ABI
    lib.turbo_analyzer_create.argtypes = [
        POINTER(TurboAnalyzerDesc),
        POINTER(c_void_p),
        POINTER(TurboError),
    ]
    lib.turbo_analyzer_create.restype = c_int32
    lib.turbo_analyzer_retain.argtypes = [c_void_p]
    lib.turbo_analyzer_retain.restype = c_void_p
    lib.turbo_analyzer_release.argtypes = [c_void_p]
    lib.turbo_analyzer_release.restype = None

    for fn_name in (
        "turbo_analyzer_detect_sentences",
        "turbo_analyzer_tokenize",
        "turbo_analyzer_find_text",
        "turbo_analyzer_detect_pii",
        "turbo_analyzer_analyze_graph",
    ):
        fn = getattr(lib, fn_name)
        fn.argtypes = [c_void_p, TurboText, POINTER(c_void_p), POINTER(TurboError)]
        fn.restype = c_int32

    for fn_name in ("turbo_analyzer_pos_tag", "turbo_analyzer_find_tokens"):
        fn = getattr(lib, fn_name)
        fn.argtypes = [c_void_p, POINTER(TurboText), c_uint32, POINTER(c_void_p), POINTER(TurboError)]
        fn.restype = c_int32

    lib.turbo_analyzer_lemmatize.argtypes = [
        c_void_p,
        POINTER(TurboText),
        POINTER(TurboText),
        c_uint32,
        POINTER(c_void_p),
        POINTER(TurboError),
    ]
    lib.turbo_analyzer_lemmatize.restype = c_int32

    lib.turbo_analyzer_chunk_text.argtypes = [
        c_void_p,
        TurboText,
        c_uint32,
        c_uint32,
        POINTER(c_void_p),
        POINTER(TurboError),
    ]
    lib.turbo_analyzer_chunk_text.restype = c_int32

    lib.turbo_analyzer_inspect_graph.argtypes = [
        c_void_p,
        POINTER(c_uint8),
        c_uint64,
        POINTER(c_void_p),
        POINTER(TurboError),
    ]
    lib.turbo_analyzer_inspect_graph.restype = c_int32

    lib.turbo_analysis_result_spans.argtypes = [
        c_void_p,
        POINTER(POINTER(TurboSpan)),
        POINTER(c_uint32),
        POINTER(TurboError),
    ]
    lib.turbo_analysis_result_spans.restype = c_int32

    lib.turbo_analysis_result_strings.argtypes = [
        c_void_p,
        POINTER(c_void_p),
        POINTER(c_uint64),
        POINTER(TurboError),
    ]
    lib.turbo_analysis_result_strings.restype = c_int32

    lib.turbo_analysis_result_graph.argtypes = [
        c_void_p,
        POINTER(POINTER(c_uint8)),
        POINTER(c_uint64),
        POINTER(TurboError),
    ]
    lib.turbo_analysis_result_graph.restype = c_int32

    lib.turbo_analysis_result_retain.argtypes = [c_void_p]
    lib.turbo_analysis_result_retain.restype = c_void_p
    lib.turbo_analysis_result_release.argtypes = [c_void_p]
    lib.turbo_analysis_result_release.restype = None

    lib.turbo_analysis_result_to_arrow.argtypes = [
        c_void_p,
        c_uint32,
        POINTER(ArrowSchema),
        POINTER(ArrowArray),
        POINTER(TurboError),
    ]
    lib.turbo_analysis_result_to_arrow.restype = c_int32

    lib.turbo_analysis_result_to_arrow_stream.argtypes = [
        c_void_p,
        c_uint32,
        POINTER(ArrowArrayStream),
        POINTER(TurboError),
    ]
    lib.turbo_analysis_result_to_arrow_stream.restype = c_int32
