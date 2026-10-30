"""Apache Arrow C Data Interface and PyCapsule (`__arrow_c_array__` / `__arrow_c_stream__`) export."""

from __future__ import annotations

import ctypes
from ctypes import (
    CFUNCTYPE,
    POINTER,
    byref,
    c_char_p,
    c_double,
    c_int,
    c_int32,
    c_uint8,
    c_uint32,
    c_void_p,
    py_object,
)
import threading

from ._ffi import (
    ArrowArray,
    ArrowArrayStream,
    ArrowSchema,
    check_status,
    get_lib,
    make_error,
)

_CAPSULE_SCHEMA_NAME = b"arrow_schema"
_CAPSULE_ARRAY_NAME = b"arrow_array"
_CAPSULE_STREAM_NAME = b"arrow_array_stream"

_PyCapsuleDestructor = CFUNCTYPE(None, c_void_p)

_PyCapsule_New = ctypes.pythonapi.PyCapsule_New
_PyCapsule_New.argtypes = [c_void_p, c_char_p, _PyCapsuleDestructor]
_PyCapsule_New.restype = py_object

_PyCapsule_GetPointer = ctypes.pythonapi.PyCapsule_GetPointer
_PyCapsule_GetPointer.argtypes = [c_void_p, c_char_p]
_PyCapsule_GetPointer.restype = c_void_p

_PyCapsule_IsValid = ctypes.pythonapi.PyCapsule_IsValid
_PyCapsule_IsValid.argtypes = [py_object, c_char_p]
_PyCapsule_IsValid.restype = c_int

_PyErr_Clear = ctypes.pythonapi.PyErr_Clear
_PyErr_Clear.argtypes = []
_PyErr_Clear.restype = None

_LIVE_LOCK = threading.Lock()
_LIVE_STRUCTS: dict[int, object] = {}


def _release_c_struct(c_struct: ArrowSchema | ArrowArray | ArrowArrayStream) -> None:
    if bool(c_struct.release):
        c_struct.release(byref(c_struct))


@_PyCapsuleDestructor
def _schema_capsule_destructor(capsule_ptr: int) -> None:
    addr = _PyCapsule_GetPointer(capsule_ptr, _CAPSULE_SCHEMA_NAME)
    if not addr:
        _PyErr_Clear()
        return
    with _LIVE_LOCK:
        schema = _LIVE_STRUCTS.pop(int(addr), None)
    if isinstance(schema, ArrowSchema):
        _release_c_struct(schema)


@_PyCapsuleDestructor
def _array_capsule_destructor(capsule_ptr: int) -> None:
    addr = _PyCapsule_GetPointer(capsule_ptr, _CAPSULE_ARRAY_NAME)
    if not addr:
        _PyErr_Clear()
        return
    with _LIVE_LOCK:
        array = _LIVE_STRUCTS.pop(int(addr), None)
    if isinstance(array, ArrowArray):
        _release_c_struct(array)


@_PyCapsuleDestructor
def _stream_capsule_destructor(capsule_ptr: int) -> None:
    addr = _PyCapsule_GetPointer(capsule_ptr, _CAPSULE_STREAM_NAME)
    if not addr:
        _PyErr_Clear()
        return
    with _LIVE_LOCK:
        stream = _LIVE_STRUCTS.pop(int(addr), None)
    if isinstance(stream, ArrowArrayStream):
        _release_c_struct(stream)


def is_valid_capsule(capsule: object, expected_name: bytes) -> bool:
    return bool(_PyCapsule_IsValid(capsule, expected_name))


def capsule_pointer(capsule: object, expected_name: bytes) -> int:
    ptr = _PyCapsule_GetPointer(ctypes.cast(id(capsule), c_void_p), expected_name)
    if not ptr:
        _PyErr_Clear()
        raise ValueError(f"Invalid PyCapsule for {expected_name!r}")
    return int(ptr)


class ArrowTableExport:
    """Zero-copy Arrow C Data Interface wrapper for a `Document` layer or full span table.

    Implements both `__arrow_c_array__` and `__arrow_c_stream__` per the
    Apache Arrow PyCapsule Interface so `polars.from_arrow(...)`,
    `pyarrow.table(...)`, `pyarrow.record_batch(...)`, and DuckDB can consume
    TurboEmbed analysis layers directly without copying.
    """

    def __init__(self, result_handle: c_void_p, layer_kind: int = 0) -> None:
        lib = get_lib()
        retained = lib.turbo_analysis_result_retain(result_handle)
        if not retained:
            raise RuntimeError("Invalid turbo_analysis_result handle for Arrow export")
        self._result: c_void_p | None = c_void_p(retained)
        self.layer_kind = int(layer_kind)

    def _require_handle(self) -> c_void_p:
        if self._result is None or not bool(self._result):
            raise RuntimeError("ArrowTableExport has been closed")
        return self._result

    def export_c_structs(self) -> tuple[ArrowSchema, ArrowArray]:
        """Export fresh `ArrowSchema` and `ArrowArray` C structures."""
        res = self._require_handle()
        lib = get_lib()
        schema = ArrowSchema()
        array = ArrowArray()
        err = make_error()
        code = lib.turbo_analysis_result_to_arrow(
            res,
            self.layer_kind,
            byref(schema),
            byref(array),
            byref(err),
        )
        check_status(code, err)
        return schema, array

    def export_c_stream(self) -> ArrowArrayStream:
        """Export a fresh `ArrowArrayStream` C structure."""
        res = self._require_handle()
        lib = get_lib()
        stream = ArrowArrayStream()
        err = make_error()
        code = lib.turbo_analysis_result_to_arrow_stream(
            res,
            self.layer_kind,
            byref(stream),
            byref(err),
        )
        check_status(code, err)
        return stream

    def __arrow_c_array__(self, requested_schema: object = None) -> tuple[object, object]:
        schema, array = self.export_c_structs()
        schema_addr = ctypes.addressof(schema)
        array_addr = ctypes.addressof(array)
        with _LIVE_LOCK:
            _LIVE_STRUCTS[schema_addr] = schema
            _LIVE_STRUCTS[array_addr] = array
        try:
            schema_cap = _PyCapsule_New(c_void_p(schema_addr), _CAPSULE_SCHEMA_NAME, _schema_capsule_destructor)
            array_cap = _PyCapsule_New(c_void_p(array_addr), _CAPSULE_ARRAY_NAME, _array_capsule_destructor)
            return (schema_cap, array_cap)
        except Exception:
            with _LIVE_LOCK:
                _LIVE_STRUCTS.pop(schema_addr, None)
                _LIVE_STRUCTS.pop(array_addr, None)
            _release_c_struct(schema)
            _release_c_struct(array)
            raise

    def __arrow_c_stream__(self, requested_schema: object = None) -> object:
        stream = self.export_c_stream()
        stream_addr = ctypes.addressof(stream)
        with _LIVE_LOCK:
            _LIVE_STRUCTS[stream_addr] = stream
        try:
            return _PyCapsule_New(c_void_p(stream_addr), _CAPSULE_STREAM_NAME, _stream_capsule_destructor)
        except Exception:
            with _LIVE_LOCK:
                _LIVE_STRUCTS.pop(stream_addr, None)
            _release_c_struct(stream)
            raise

    def to_pydict(self) -> dict[str, list[int | float | str]]:
        """Decode the exported Arrow C buffers directly into a column dictionary."""
        schema, array = self.export_c_structs()
        try:
            n_rows = int(array.length)
            n_cols = int(schema.n_children)
            out: dict[str, list[int | float | str]] = {}
            for col_idx in range(n_cols):
                child_schema = schema.children[col_idx].contents
                child_array = array.children[col_idx].contents
                name = child_schema.name.decode("utf-8") if child_schema.name else f"col_{col_idx}"
                fmt = child_schema.format.decode("utf-8") if child_schema.format else ""
                if fmt == "I":
                    val_ptr = child_array.buffers[1]
                    if n_rows == 0 or not val_ptr:
                        out[name] = []
                    else:
                        vals = ctypes.cast(val_ptr, POINTER(c_uint32))
                        out[name] = [int(vals[i]) for i in range(n_rows)]
                elif fmt == "g":
                    val_ptr = child_array.buffers[1]
                    if n_rows == 0 or not val_ptr:
                        out[name] = []
                    else:
                        vals = ctypes.cast(val_ptr, POINTER(c_double))
                        out[name] = [float(vals[i]) for i in range(n_rows)]
                elif fmt == "u":
                    off_ptr = child_array.buffers[1]
                    data_ptr = child_array.buffers[2]
                    if n_rows == 0 or not off_ptr:
                        out[name] = []
                    else:
                        offs = ctypes.cast(off_ptr, POINTER(c_int32))
                        total_bytes = int(offs[n_rows])
                        raw_bytes = (
                            bytes(ctypes.cast(data_ptr, POINTER(c_uint8 * total_bytes)).contents)
                            if (total_bytes > 0 and data_ptr)
                            else b""
                        )
                        col_strs: list[int | float | str] = []
                        for i in range(n_rows):
                            s_off = int(offs[i])
                            e_off = int(offs[i + 1])
                            col_strs.append(raw_bytes[s_off:e_off].decode("utf-8", errors="replace"))
                        out[name] = col_strs
                else:
                    raise ValueError(f"Unsupported Arrow format {fmt!r} on column {name!r}")
            return out
        finally:
            _release_c_struct(schema)
            _release_c_struct(array)

    def to_pylist(self) -> list[dict[str, int | float | str]]:
        cols = self.to_pydict()
        if not cols:
            return []
        keys = list(cols.keys())
        n_rows = len(cols[keys[0]])
        return [{k: cols[k][i] for k in keys} for i in range(n_rows)]

    def close(self) -> None:
        if self._result is not None and bool(self._result):
            get_lib().turbo_analysis_result_release(self._result)
            self._result = None

    def __enter__(self) -> ArrowTableExport:
        self._require_handle()
        return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        self.close()

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass
