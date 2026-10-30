"""In-process Runtime, Bundle, Tokenizer, Session, and zero-copy EmbeddingBatch."""

from __future__ import annotations

from collections.abc import Sequence
import ctypes
from ctypes import (
    POINTER,
    byref,
    c_char,
    c_float,
    c_int8,
    c_int32,
    c_uint8,
    c_uint32,
    c_uint64,
    c_void_p,
)
from dataclasses import dataclass
from pathlib import Path
import threading

from ._ffi import (
    TURBO_AUTOTUNE_OFF,
    TURBO_DTYPE_F32,
    TURBO_DTYPE_I8,
    TURBO_DTYPE_U8,
    TURBO_NORMALIZE_MODEL,
    TURBO_POOLING_MODEL,
    TURBO_PRECISION_MODEL,
    TURBO_PROMPT_NONE,
    TURBO_QUANTIZE_MODEL,
    TURBO_TRUNCATE_MODEL,
    TurboDeviceInfo,
    TurboEmbedOptions,
    TurboEncodeOptions,
    TurboModelInfo,
    TurboRerankOptions,
    TurboResultInfo,
    TurboSessionDesc,
    TurboText,
    TurboTextPair,
    TurboTokenBatch,
    TurboTokenizerInfo,
    TurboTokenRange,
    check_status,
    get_lib,
    make_error,
    make_text,
)


@dataclass(frozen=True)
class DeviceInfo:
    ordinal: int
    kind: int
    unified_memory: bool
    memory_total: int
    memory_free: int
    arch: str
    name: str
    vendor: str
    backend: str
    runtime_version: str
    driver_version: str


@dataclass(frozen=True)
class ModelInfo:
    task: int
    dim: int
    pooling: int
    normalize: int
    max_seq: int
    max_batch: int
    dtype: int
    model_id: str
    revision: str
    manifest_sha256: str
    artifact_sha256: str
    tokenizer_sha256: str
    output_dims: tuple[int, ...]


@dataclass(frozen=True)
class TokenizedBatch:
    batch: int
    seq: int
    row_stride: int
    ids: list[list[int]]
    mask: list[list[int]]
    types: list[list[int]]
    lengths: list[int]


class EmbeddingBatch:
    """Zero-copy view over a libturbo result buffer (`f32`, `i8`, or packed `u8`).

    Exposes both PEP 3118 / PEP 688 buffer protocol (`memoryview(batch)` /
    `batch.memoryview()`) and NumPy's `__array_interface__` (`np.asarray(batch)`)
    directly backed by the native C buffer without copying.
    """

    def __init__(
        self,
        result_handle: c_void_p,
        buffer_handle: c_void_p | None,
        host_ptr: int,
        fallback_buf: ctypes.Array[ctypes.c_uint8] | None,
        info: TurboResultInfo,
        on_close: object = None,
    ) -> None:
        self._result = result_handle
        self._buffer = buffer_handle
        self._host_ptr = host_ptr
        self._fallback_buf = fallback_buf
        self._on_close = on_close
        self._closed = False
        self._views: list[memoryview] = []

        self.batch = int(info.batch)
        self.dim = int(info.dim)
        self.dtype = int(info.dtype)
        self.bytes_len = int(info.bytes)
        self.backend = info.backend.decode("utf-8", errors="replace")
        self.arch = info.arch.decode("utf-8", errors="replace")

        if self.dtype == TURBO_DTYPE_F32:
            self._ctype = c_float
            self.typestr = "<f4"
            self.itemsize = 4
            self.cols = self.dim
        elif self.dtype == TURBO_DTYPE_I8:
            self._ctype = c_int8
            self.typestr = "|i1"
            self.itemsize = 1
            self.cols = self.dim
        elif self.dtype == TURBO_DTYPE_U8:
            self._ctype = c_uint8
            self.typestr = "|u1"
            self.itemsize = 1
            self.cols = (self.dim + 7) // 8
        else:
            raise ValueError(f"Unsupported embedding result dtype: {self.dtype}")

    def _check_open(self) -> None:
        if self._closed:
            raise RuntimeError("EmbeddingBatch has been closed")

    @property
    def shape(self) -> tuple[int, int]:
        self._check_open()
        return (self.batch, self.cols)

    @property
    def data_ptr(self) -> int:
        self._check_open()
        return self._host_ptr

    @property
    def __array_interface__(self) -> dict[str, object]:
        self._check_open()
        return {
            "version": 3,
            "shape": (self.batch, self.cols),
            "typestr": self.typestr,
            "data": (self._host_ptr, True),
            "strides": (self.cols * self.itemsize, self.itemsize),
        }

    def memoryview(self) -> memoryview:
        """Return a zero-copy 2D memoryview over the native result buffer."""
        self._check_open()
        total_bytes = self.batch * self.cols * self.itemsize
        byte_arr_type = c_uint8 * total_bytes
        c_arr = byte_arr_type.from_address(self._host_ptr) if total_bytes > 0 else byte_arr_type()
        raw_mv = memoryview(c_arr)
        byte_mv = raw_mv.cast("B")
        if self.dtype == TURBO_DTYPE_F32:
            fmt = "f"
        elif self.dtype == TURBO_DTYPE_I8:
            fmt = "b"
        else:
            fmt = "B"
        shaped = byte_mv.cast(fmt, shape=(self.batch, self.cols))
        self._views.append(shaped)
        self._views.append(byte_mv)
        self._views.append(raw_mv)
        return shaped

    def __buffer__(self, flags: int) -> memoryview:
        return self.memoryview()

    def row(self, index: int) -> memoryview:
        self._check_open()
        if index < 0 or index >= self.batch:
            raise IndexError(f"row index {index} out of range for batch {self.batch}")
        total_bytes = self.batch * self.cols * self.itemsize
        byte_arr_type = c_uint8 * total_bytes
        c_arr = byte_arr_type.from_address(self._host_ptr) if total_bytes > 0 else byte_arr_type()
        raw_mv = memoryview(c_arr)
        byte_mv = raw_mv.cast("B")
        fmt = "f" if self.dtype == TURBO_DTYPE_F32 else ("b" if self.dtype == TURBO_DTYPE_I8 else "B")
        flat_mv = byte_mv.cast(fmt)
        sub = flat_mv[index * self.cols : (index + 1) * self.cols]
        self._views.extend([sub, flat_mv, byte_mv, raw_mv])
        return sub

    def to_list(self) -> list[list[float | int]]:
        self._check_open()
        return self.memoryview().tolist()  # type: ignore[return-value]

    def close(self) -> None:
        if self._closed:
            return
        for v in self._views:
            try:
                v.release()
            except BufferError:
                pass
        self._views.clear()
        lib = get_lib()
        if self._buffer is not None and bool(self._buffer):
            lib.turbo_buffer_release(self._buffer)
            self._buffer = None
        if self._result is not None and bool(self._result):
            lib.turbo_result_release(self._result)
            self._result = None
        self._fallback_buf = None
        self._host_ptr = 0
        self._closed = True
        if callable(self._on_close):
            cb = self._on_close
            self._on_close = None
            cb()

    def __enter__(self) -> EmbeddingBatch:
        self._check_open()
        return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        self.close()

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass


class Runtime:
    """In-process libturbo runtime managing devices, bundles, and tokenizers."""

    def __init__(self) -> None:
        lib = get_lib()
        handle = c_void_p()
        err = make_error()
        code = lib.turbo_runtime_create(None, byref(handle), byref(err))
        check_status(code, err)
        self._handle: c_void_p | None = handle

    def _require_handle(self) -> c_void_p:
        if self._handle is None or not bool(self._handle):
            raise RuntimeError("Runtime has been closed")
        return self._handle

    def version(self) -> str:
        raw = get_lib().turbo_version()
        return raw.decode("utf-8", errors="replace") if raw else ""

    def devices(self) -> list[DeviceInfo]:
        rt = self._require_handle()
        lib = get_lib()
        count = c_uint32(0)
        err = make_error()
        check_status(lib.turbo_runtime_device_count(rt, byref(count), byref(err)), err)
        out: list[DeviceInfo] = []
        for idx in range(count.value):
            info = TurboDeviceInfo()
            info.struct_size = ctypes.sizeof(TurboDeviceInfo)
            check_status(lib.turbo_runtime_device_info(rt, idx, byref(info), byref(err)), err)
            out.append(
                DeviceInfo(
                    ordinal=int(info.ordinal),
                    kind=int(info.kind),
                    unified_memory=bool(info.unified_memory),
                    memory_total=int(info.memory_total),
                    memory_free=int(info.memory_free),
                    arch=info.arch.decode("utf-8", errors="replace"),
                    name=info.name.decode("utf-8", errors="replace"),
                    vendor=info.vendor.decode("utf-8", errors="replace"),
                    backend=info.backend.decode("utf-8", errors="replace"),
                    runtime_version=info.runtime_version.decode("utf-8", errors="replace"),
                    driver_version=info.driver_version.decode("utf-8", errors="replace"),
                )
            )
        return out

    def open_bundle(self, path: str | Path, *, device: int = 0) -> Bundle:
        self._require_handle()
        return Bundle(self, Path(path).expanduser().resolve(), device=device)

    def open_tokenizer(self, path: str | Path) -> Tokenizer:
        rt = self._require_handle()
        lib = get_lib()
        path_bytes = str(Path(path).expanduser().resolve()).encode("utf-8")
        handle = c_void_p()
        err = make_error()
        code = lib.turbo_tokenizer_create(rt, make_text(path_bytes), byref(handle), byref(err))
        check_status(code, err)
        return Tokenizer(handle)

    def close(self) -> None:
        if self._handle is not None and bool(self._handle):
            get_lib().turbo_runtime_release(self._handle)
            self._handle = None

    def __enter__(self) -> Runtime:
        self._require_handle()
        return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        self.close()

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass


class Tokenizer:
    """In-process libturbo tokenizer."""

    def __init__(self, handle: c_void_p) -> None:
        self._handle: c_void_p | None = handle
        lib = get_lib()
        info = TurboTokenizerInfo()
        info.struct_size = ctypes.sizeof(TurboTokenizerInfo)
        err = make_error()
        check_status(lib.turbo_tokenizer_get_info(handle, byref(info), byref(err)), err)
        self.vocab_size = int(info.vocab_size)
        self.max_seq = int(info.max_seq)
        self.kind = info.kind.decode("utf-8", errors="replace")

    def _require_handle(self) -> c_void_p:
        if self._handle is None or not bool(self._handle):
            raise RuntimeError("Tokenizer has been closed")
        return self._handle

    def encode(
        self,
        texts: Sequence[str],
        *,
        omit_special_tokens: bool = False,
        truncate: int = TURBO_TRUNCATE_MODEL,
        max_tokens: int = 0,
        prompt_role: int = TURBO_PROMPT_NONE,
    ) -> TokenizedBatch:
        tok = self._require_handle()
        n = len(texts)
        if n == 0:
            return TokenizedBatch(batch=0, seq=0, row_stride=0, ids=[], mask=[], types=[], lengths=[])
        lib = get_lib()
        encoded_bytes = [t.encode("utf-8") for t in texts]
        c_texts = (TurboText * n)(*[make_text(b) for b in encoded_bytes])

        opts = TurboEncodeOptions()
        opts.struct_size = ctypes.sizeof(TurboEncodeOptions)
        opts.omit_special_tokens = 1 if omit_special_tokens else 0
        opts.truncate = truncate
        opts.max_tokens = max_tokens
        opts.prompt_role = prompt_role

        stride = max_tokens if max_tokens > 0 else self.max_seq
        stride = max(stride, 1)
        total = n * stride
        ids_buf = (c_int32 * total)()
        mask_buf = (c_int32 * total)()
        types_buf = (c_int32 * total)()
        lengths_buf = (c_uint32 * n)()

        err = make_error()
        check_status(
            lib.turbo_tokenizer_encode(
                tok,
                c_texts,
                n,
                byref(opts),
                ids_buf,
                mask_buf,
                types_buf,
                stride,
                lengths_buf,
                byref(err),
            ),
            err,
        )
        lengths = [int(lengths_buf[i]) for i in range(n)]
        seq = max(lengths, default=0)
        ids = [[int(ids_buf[i * stride + j]) for j in range(lengths[i])] for i in range(n)]
        mask = [[int(mask_buf[i * stride + j]) for j in range(lengths[i])] for i in range(n)]
        types = [[int(types_buf[i * stride + j]) for j in range(lengths[i])] for i in range(n)]
        return TokenizedBatch(
            batch=n,
            seq=seq,
            row_stride=stride,
            ids=ids,
            mask=mask,
            types=types,
            lengths=lengths,
        )

    def close(self) -> None:
        if self._handle is not None and bool(self._handle):
            get_lib().turbo_tokenizer_release(self._handle)
            self._handle = None

    def __enter__(self) -> Tokenizer:
        self._require_handle()
        return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        self.close()

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass


class Bundle:
    """Opened model bundle on a runtime device."""

    def __init__(self, runtime: Runtime, path: Path, *, device: int = 0) -> None:
        self._runtime = runtime
        self.path = path
        self._device = device
        rt = runtime._require_handle()
        lib = get_lib()
        err = make_error()

        ctx_handle = c_void_p()
        check_status(lib.turbo_context_create(rt, device, byref(ctx_handle), byref(err)), err)
        model_handle = c_void_p()
        try:
            path_b = str(path).encode("utf-8")
            check_status(lib.turbo_model_load(ctx_handle, make_text(path_b), byref(model_handle), byref(err)), err)
            raw = TurboModelInfo()
            raw.struct_size = ctypes.sizeof(TurboModelInfo)
            check_status(lib.turbo_model_get_info(model_handle, byref(raw), byref(err)), err)
            dims = tuple(int(raw.output_dims[i]) for i in range(int(raw.output_dims_count)))
            self.info = ModelInfo(
                task=int(raw.task),
                dim=int(raw.dim),
                pooling=int(raw.pooling),
                normalize=int(raw.normalize),
                max_seq=int(raw.max_seq),
                max_batch=int(raw.max_batch),
                dtype=int(raw.dtype),
                model_id=raw.model_id.decode("utf-8", errors="replace"),
                revision=raw.revision.decode("utf-8", errors="replace"),
                manifest_sha256=raw.manifest_sha256.decode("utf-8", errors="replace"),
                artifact_sha256=raw.artifact_sha256.decode("utf-8", errors="replace"),
                tokenizer_sha256=raw.tokenizer_sha256.decode("utf-8", errors="replace"),
                output_dims=dims,
            )
            self._context: c_void_p | None = ctx_handle
            self._model: c_void_p | None = model_handle
        except Exception:
            if bool(model_handle):
                lib.turbo_model_release(model_handle)
            if bool(ctx_handle):
                lib.turbo_context_release(ctx_handle)
            raise

    def _require_model(self) -> c_void_p:
        if self._model is None or not bool(self._model):
            raise RuntimeError("Bundle has been closed")
        return self._model

    def tokenizer(self) -> Tokenizer:
        self._require_model()
        return self._runtime.open_tokenizer(self.path)

    def create_session(
        self,
        *,
        max_batch: int = 0,
        max_seq: int = 0,
        precision: int = TURBO_PRECISION_MODEL,
        tuning: int = TURBO_AUTOTUNE_OFF,
        tuning_budget_ms: int = 0,
        cpu_threads: int = 0,
        max_ranges: int = 0,
    ) -> Session:
        model_handle = self._require_model()
        lib = get_lib()
        err = make_error()
        desc = TurboSessionDesc()
        desc.struct_size = ctypes.sizeof(TurboSessionDesc)
        desc.max_batch = max_batch
        desc.max_seq = max_seq
        desc.precision = precision
        desc.tuning = tuning
        desc.tuning_budget_ms = tuning_budget_ms
        desc.cpu_threads = cpu_threads
        desc.max_ranges = max_ranges

        sess_handle = c_void_p()
        check_status(lib.turbo_session_create(model_handle, byref(desc), byref(sess_handle), byref(err)), err)
        tok = self.tokenizer()
        return Session(sess_handle, tok, self.info)

    def close(self) -> None:
        lib = get_lib()
        if self._model is not None and bool(self._model):
            lib.turbo_model_release(self._model)
            self._model = None
        if self._context is not None and bool(self._context):
            lib.turbo_context_release(self._context)
            self._context = None

    def __enter__(self) -> Bundle:
        self._require_model()
        return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        self.close()

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass


class Session:
    """In-process inference session (`embed`, `embed_ranges`, `rerank`, `extractive_qa`, `tokenize`)."""

    def __init__(
        self,
        session_handle: c_void_p,
        tokenizer: Tokenizer,
        model_info: ModelInfo,
    ) -> None:
        self._session: c_void_p | None = session_handle
        self._tokenizer = tokenizer
        self.info = model_info
        self._lock = threading.Lock()

    def _require_handle(self) -> c_void_p:
        if self._session is None or not bool(self._session):
            raise RuntimeError("Session has been closed")
        return self._session

    def tokenize(self, texts: Sequence[str], **kwargs: object) -> TokenizedBatch:
        return self._tokenizer.encode(texts, **kwargs)  # type: ignore[arg-type]

    def _encode_to_c_batch(
        self,
        texts: Sequence[str],
        *,
        truncate: int,
        max_tokens: int,
        prompt_role: int,
    ) -> tuple[
        TurboTokenBatch,
        ctypes.Array[c_int32],
        ctypes.Array[c_int32],
        ctypes.Array[c_int32],
    ]:
        tok = self._tokenizer._require_handle()
        lib = get_lib()
        n = len(texts)
        encoded_bytes = [t.encode("utf-8") for t in texts]
        c_texts = (TurboText * n)(*[make_text(b) for b in encoded_bytes])

        enc_opts = TurboEncodeOptions()
        enc_opts.struct_size = ctypes.sizeof(TurboEncodeOptions)
        enc_opts.omit_special_tokens = 0
        enc_opts.truncate = truncate
        enc_opts.max_tokens = max_tokens
        enc_opts.prompt_role = prompt_role

        stride = max_tokens if max_tokens > 0 else self._tokenizer.max_seq
        stride = max(stride, 1)
        total = max(n * stride, 1)
        ids_buf = (c_int32 * total)()
        mask_buf = (c_int32 * total)()
        types_buf = (c_int32 * total)()
        lengths_buf = (c_uint32 * max(n, 1))()

        err = make_error()
        check_status(
            lib.turbo_tokenizer_encode(
                tok,
                c_texts,
                n,
                byref(enc_opts),
                ids_buf,
                mask_buf,
                types_buf,
                stride,
                lengths_buf,
                byref(err),
            ),
            err,
        )
        seq = max((int(lengths_buf[i]) for i in range(n)), default=0)

        batch = TurboTokenBatch()
        batch.struct_size = ctypes.sizeof(TurboTokenBatch)
        batch.batch = n
        batch.seq = seq
        batch.row_stride = stride
        batch.ids = ids_buf
        batch.mask = mask_buf
        batch.types = types_buf
        return batch, ids_buf, mask_buf, types_buf

    def _run_and_wrap(self, sess: c_void_p) -> EmbeddingBatch:
        lib = get_lib()
        err = make_error()
        res_handle = c_void_p()
        check_status(lib.turbo_session_run(sess, byref(res_handle), byref(err)), err)
        try:
            info = TurboResultInfo()
            info.struct_size = ctypes.sizeof(TurboResultInfo)
            check_status(lib.turbo_result_get_info(res_handle, byref(info), byref(err)), err)

            buf_handle = c_void_p()
            check_status(lib.turbo_result_buffer(res_handle, byref(buf_handle), byref(err)), err)
            host_ptr_out = c_void_p()
            code = lib.turbo_buffer_host_ptr(buf_handle, byref(host_ptr_out), byref(err))
            if code == 0 and bool(host_ptr_out):
                return EmbeddingBatch(
                    res_handle,
                    buf_handle,
                    int(host_ptr_out.value or 0),
                    None,
                    info,
                    on_close=self._lock.release,
                )

            # Fallback if result buffer lives in device-only memory without direct host pointer:
            nbytes = int(info.bytes)
            fallback = (c_uint8 * max(nbytes, 1))()
            written = c_uint64(0)
            check_status(
                lib.turbo_result_read(res_handle, ctypes.cast(fallback, c_void_p), nbytes, byref(written), byref(err)),
                err,
            )
            return EmbeddingBatch(
                res_handle,
                buf_handle,
                ctypes.addressof(fallback),
                fallback,
                info,
                on_close=self._lock.release,
            )
        except Exception:
            if bool(res_handle):
                lib.turbo_result_release(res_handle)
            raise

    def embed(
        self,
        texts: Sequence[str],
        *,
        truncate: int = TURBO_TRUNCATE_MODEL,
        max_tokens: int = 0,
        prompt_role: int = TURBO_PROMPT_NONE,
        normalize: int = TURBO_NORMALIZE_MODEL,
        pooling: int = TURBO_POOLING_MODEL,
        output_dim: int = 0,
        quantization: int = TURBO_QUANTIZE_MODEL,
        int8_ranges: Sequence[float] | None = None,
    ) -> EmbeddingBatch:
        sess = self._require_handle()
        n = len(texts)
        encoded_bytes = [t.encode("utf-8") for t in texts]
        c_texts = (TurboText * max(n, 1))(*[make_text(b) for b in encoded_bytes])

        opts = TurboEmbedOptions()
        opts.struct_size = ctypes.sizeof(TurboEmbedOptions)
        opts.truncate = truncate
        opts.max_tokens = max_tokens
        opts.prompt_role = prompt_role
        opts.normalize = normalize
        opts.pooling = pooling
        opts.output_dim = output_dim
        opts.quantization = quantization
        c_ranges: ctypes.Array[c_float] | None = None
        if int8_ranges is not None:
            c_ranges = (c_float * len(int8_ranges))(*[float(x) for x in int8_ranges])
            opts.int8_ranges = c_ranges

        lib = get_lib()
        err = make_error()
        self._lock.acquire()
        try:
            check_status(
                lib.turbo_embed_write_text(
                    sess,
                    c_texts if n > 0 else POINTER(TurboText)(),
                    n,
                    byref(opts),
                    byref(err),
                ),
                err,
            )
            return self._run_and_wrap(sess)
        except Exception:
            self._lock.release()
            raise

    def embed_ranges(
        self,
        texts: Sequence[str],
        ranges: Sequence[tuple[int, int, int]],
        *,
        truncate: int = TURBO_TRUNCATE_MODEL,
        max_tokens: int = 0,
        prompt_role: int = TURBO_PROMPT_NONE,
        normalize: int = TURBO_NORMALIZE_MODEL,
        pooling: int = TURBO_POOLING_MODEL,
        output_dim: int = 0,
        quantization: int = TURBO_QUANTIZE_MODEL,
        int8_ranges: Sequence[float] | None = None,
    ) -> EmbeddingBatch:
        sess = self._require_handle()
        batch, _ids, _mask, _types = self._encode_to_c_batch(
            texts,
            truncate=truncate,
            max_tokens=max_tokens,
            prompt_role=prompt_role,
        )
        n_ranges = len(ranges)
        c_range_arr = (TurboTokenRange * max(n_ranges, 1))(
            *[TurboTokenRange(row=r[0], start=r[1], end=r[2]) for r in ranges]
        )
        opts = TurboEmbedOptions()
        opts.struct_size = ctypes.sizeof(TurboEmbedOptions)
        opts.truncate = 0
        opts.max_tokens = max_tokens
        opts.prompt_role = 0
        opts.normalize = normalize
        opts.pooling = pooling
        opts.output_dim = output_dim
        opts.quantization = quantization
        c_i8: ctypes.Array[c_float] | None = None
        if int8_ranges is not None:
            c_i8 = (c_float * len(int8_ranges))(*[float(x) for x in int8_ranges])
            opts.int8_ranges = c_i8

        lib = get_lib()
        err = make_error()
        self._lock.acquire()
        try:
            check_status(
                lib.turbo_embed_write_ranges(
                    sess,
                    byref(batch),
                    c_range_arr if n_ranges > 0 else POINTER(TurboTokenRange)(),
                    n_ranges,
                    byref(opts),
                    byref(err),
                ),
                err,
            )
            return self._run_and_wrap(sess)
        except Exception:
            self._lock.release()
            raise

    def rerank(
        self,
        query: str,
        documents: Sequence[str],
        *,
        query_tokens: int = 0,
        max_tokens: int = 0,
        stride: int = 0,
    ) -> list[float]:
        sess = self._require_handle()
        lib = get_lib()
        q_bytes = query.encode("utf-8")
        doc_bytes = [d.encode("utf-8") for d in documents]
        n = len(documents)
        pairs = (TurboTextPair * max(n, 1))(
            *[TurboTextPair(query=make_text(q_bytes), passage=make_text(b)) for b in doc_bytes]
        )
        opts = TurboRerankOptions()
        opts.struct_size = ctypes.sizeof(TurboRerankOptions)
        opts.query_tokens = query_tokens
        opts.max_tokens = max_tokens
        opts.stride = stride
        err = make_error()
        self._lock.acquire()
        try:
            check_status(
                lib.turbo_rerank_write_pairs(
                    sess,
                    pairs if n > 0 else POINTER(TurboTextPair)(),
                    n,
                    byref(opts),
                    byref(err),
                ),
                err,
            )
            batch = self._run_and_wrap(sess)
        except Exception:
            self._lock.release()
            raise
        with batch:
            return [float(row[0]) for row in batch.to_list()]

    def extractive_qa(
        self,
        texts: Sequence[str],
        ranges: Sequence[tuple[int, int, int]],
        **kwargs: object,
    ) -> EmbeddingBatch:
        return self.embed_ranges(texts, ranges, **kwargs)  # type: ignore[arg-type]

    def close(self) -> None:
        lib = get_lib()
        if self._session is not None and bool(self._session):
            lib.turbo_session_release(self._session)
            self._session = None
        self._tokenizer.close()

    def __enter__(self) -> Session:
        self._require_handle()
        return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        self.close()

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass
