#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Turbo Python demo over the C ABI with ctypes (no extension module).

    python3 demo/python/turbo_demo.py [--lib <libturbo.so>] [--provider-lib <so>]
        [--provider <id> --ordinal <n>] --bundle <dir> text...

Embeds the texts on the selected device (AUTO never picks a CPU) and prints
the cosine similarity matrix. Every C call is checked; a failure raises
TurboError with the status name and the library's message.
"""

import argparse
import ctypes as C
import math
import os
import sys

TURBO_OK = 0
TURBO_SELECT_EXPLICIT = 1
TURBO_MODEL_EMBEDDING = 1
TURBO_DTYPE_F32 = 12
ERROR_MESSAGE_LEN = 496


class Text(C.Structure):
    _fields_ = [("ptr", C.c_char_p), ("len", C.c_uint64)]


class Error(C.Structure):
    _fields_ = [("struct_size", C.c_uint32), ("code", C.c_int32), ("field", C.c_uint32), ("message", C.c_char * ERROR_MESSAGE_LEN)]


class RuntimeDesc(C.Structure):
    _fields_ = [("struct_size", C.c_uint32), ("flags", C.c_uint32), ("n_provider_paths", C.c_uint32), ("reserved", C.c_uint32), ("provider_paths", C.POINTER(Text))]


class DeviceSelector(C.Structure):
    _fields_ = [("struct_size", C.c_uint32), ("policy", C.c_uint32), ("kind_mask", C.c_uint32), ("ordinal", C.c_uint32), ("provider_id", Text), ("vendor", Text)]


class DeviceInfo(C.Structure):
    _fields_ = [("struct_size", C.c_uint32), ("kind", C.c_uint32), ("ordinal", C.c_uint32), ("vendor_id", C.c_uint32),
                ("caps", C.c_uint64), ("memory_total", C.c_uint64), ("memory_free", C.c_uint64),
                ("name", C.c_char * 128), ("vendor", C.c_char * 64), ("provider_id", C.c_char * 32),
                ("provider_version", C.c_char * 32), ("runtime_version", C.c_char * 64), ("driver_version", C.c_char * 64)]


class SessionDesc(C.Structure):
    _fields_ = [("struct_size", C.c_uint32), ("max_batch", C.c_uint32), ("max_seq", C.c_uint32), ("n_options", C.c_uint32), ("options", C.c_void_p), ("next", C.c_void_p)]


class ResultInfo(C.Structure):
    _fields_ = [("struct_size", C.c_uint32), ("n_outputs", C.c_uint32), ("batch", C.c_uint32), ("dim", C.c_uint32), ("dtype", C.c_uint32), ("placement", C.c_uint32), ("bytes", C.c_uint64)]


class TurboError(RuntimeError):
    pass


def text(s: str) -> Text:
    b = s.encode("utf-8")
    t = Text()
    t.ptr = b  # ctypes keeps the bytes alive through the structure
    t.len = len(b)
    t._keep = b
    return t


# Every function the demo calls, with its C signature. ctypes defaults an
# undeclared argument to a C int, which silently truncates the 64-bit
# capacity of turbo_result_read and shifts every argument after it on an ABI
# that passes a 64-bit value in a register pair. Declaring all of them is the
# only way the demo is honest about the ABI it documents.
VOID = C.c_void_p
SIGNATURES = {
    "turbo_status_name": (C.c_char_p, [C.c_int32]),
    "turbo_abi_version": (C.c_uint32, []),
    "turbo_runtime_create": (C.c_int32, [C.POINTER(RuntimeDesc), C.POINTER(VOID), C.POINTER(Error)]),
    "turbo_runtime_select_device": (C.c_int32, [VOID, C.POINTER(DeviceSelector), C.POINTER(C.c_uint32), C.POINTER(Error)]),
    "turbo_runtime_device_info": (C.c_int32, [VOID, C.c_uint32, C.POINTER(DeviceInfo), C.POINTER(Error)]),
    "turbo_context_create": (C.c_int32, [VOID, C.c_uint32, VOID, C.POINTER(VOID), C.POINTER(Error)]),
    "turbo_model_load": (C.c_int32, [VOID, Text, VOID, C.POINTER(VOID), C.POINTER(Error)]),
    "turbo_session_create": (C.c_int32, [VOID, C.POINTER(SessionDesc), C.POINTER(VOID), C.POINTER(Error)]),
    "turbo_session_write_text": (C.c_int32, [VOID, C.POINTER(Text), C.c_uint32, VOID, C.POINTER(Error)]),
    "turbo_session_run": (C.c_int32, [VOID, VOID, C.POINTER(VOID), C.POINTER(Error)]),
    "turbo_result_get_info": (C.c_int32, [VOID, C.POINTER(ResultInfo), C.POINTER(Error)]),
    "turbo_result_read": (C.c_int32, [VOID, C.c_uint32, VOID, C.c_uint64, C.POINTER(C.c_uint64), C.POINTER(Error)]),
    "turbo_result_release": (None, [VOID]),
    "turbo_session_release": (None, [VOID]),
    "turbo_model_release": (None, [VOID]),
    "turbo_context_release": (None, [VOID]),
    "turbo_runtime_release": (None, [VOID]),
}


class Turbo:
    """A thin, checked wrapper over the handful of calls the demo needs."""

    def __init__(self, lib_path: str):
        self.lib = C.CDLL(lib_path)
        for name, (restype, argtypes) in SIGNATURES.items():
            fn = getattr(self.lib, name)
            fn.restype = restype
            fn.argtypes = argtypes
        self.err = Error()
        self.err.struct_size = C.sizeof(Error)

    def check(self, rc: int, what: str) -> None:
        if rc != TURBO_OK:
            name = self.lib.turbo_status_name(rc).decode()
            msg = self.err.message.decode(errors="replace")
            field = f" (field {self.err.field})" if self.err.field else ""
            raise TurboError(f"{what}: {name}{field}: {msg}")

    def sized(self, cls):
        s = cls()
        s.struct_size = C.sizeof(cls)
        return s


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    lib_name = "libturbo.dylib" if sys.platform == "darwin" else "libturbo.so"
    ap.add_argument("--lib", default=os.path.join(root, "target", "debug", lib_name))
    ap.add_argument("--provider-lib")
    ap.add_argument("--provider")
    ap.add_argument("--ordinal", type=int)
    ap.add_argument("--bundle", required=True)
    ap.add_argument("texts", nargs="+")
    a = ap.parse_args()
    if (a.provider is None) != (a.ordinal is None):
        ap.error("--provider and --ordinal go together; omit both for AUTO")

    t = Turbo(a.lib)
    lib, err = t.lib, t.err
    print(f"libturbo ABI {lib.turbo_abi_version()} from {a.lib}")

    rd = t.sized(RuntimeDesc)
    paths = None
    if a.provider_lib:
        paths = (Text * 1)(text(a.provider_lib))
        rd.n_provider_paths = 1
        rd.provider_paths = paths
    rt = C.c_void_p()
    t.check(lib.turbo_runtime_create(C.byref(rd), C.byref(rt), C.byref(err)), "turbo_runtime_create")

    dev = C.c_uint32()
    if a.provider:
        sel = t.sized(DeviceSelector)
        sel.policy = TURBO_SELECT_EXPLICIT
        sel.provider_id = text(a.provider)
        sel.ordinal = a.ordinal
        t.check(lib.turbo_runtime_select_device(rt, C.byref(sel), C.byref(dev), C.byref(err)), "turbo_runtime_select_device")
    else:
        t.check(lib.turbo_runtime_select_device(rt, None, C.byref(dev), C.byref(err)), "turbo_runtime_select_device")
    di = t.sized(DeviceInfo)
    t.check(lib.turbo_runtime_device_info(rt, dev.value, C.byref(di), C.byref(err)), "turbo_runtime_device_info")
    print(f"device: {di.name.decode()} ({di.provider_id.decode()}:{di.ordinal}, kind {di.kind}, runtime {di.runtime_version.decode()})")

    ctx = C.c_void_p()
    t.check(lib.turbo_context_create(rt, dev.value, None, C.byref(ctx), C.byref(err)), "turbo_context_create")
    model = C.c_void_p()
    t.check(lib.turbo_model_load(ctx, text(a.bundle), None, C.byref(model), C.byref(err)), "turbo_model_load")

    sd = t.sized(SessionDesc)
    sd.max_batch = len(a.texts)
    sd.max_seq = 0  # the model's own limit
    session = C.c_void_p()
    t.check(lib.turbo_session_create(model, C.byref(sd), C.byref(session), C.byref(err)), "turbo_session_create")

    views = (Text * len(a.texts))(*[text(s) for s in a.texts])
    t.check(lib.turbo_session_write_text(session, views, len(a.texts), None, C.byref(err)), "turbo_session_write_text")
    result = C.c_void_p()
    t.check(lib.turbo_session_run(session, None, C.byref(result), C.byref(err)), "turbo_session_run")
    ri = t.sized(ResultInfo)
    t.check(lib.turbo_result_get_info(result, C.byref(ri), C.byref(err)), "turbo_result_get_info")
    if ri.dtype != TURBO_DTYPE_F32:
        raise TurboError(f"result dtype {ri.dtype} is not f32")
    expected = ri.batch * ri.dim * 4
    if ri.bytes != expected:
        raise TurboError(f"result claims {ri.bytes} bytes for {ri.batch} x {ri.dim} f32 ({expected} expected)")
    buf = (C.c_float * (ri.bytes // 4))()
    written = C.c_uint64()
    t.check(lib.turbo_result_read(result, 0, C.cast(buf, VOID), ri.bytes, C.byref(written), C.byref(err)), "turbo_result_read")
    # The library reports what it wrote; the untouched tail of buf is zeros,
    # so a short read would print similarities of 0.000 or NaN as if they
    # were the model's answer.
    if written.value != ri.bytes:
        raise TurboError(f"turbo_result_read wrote {written.value} of {ri.bytes} bytes")
    dim = ri.dim
    rows = [list(buf[r * dim:(r + 1) * dim]) for r in range(ri.batch)]
    print(f"embeddings: {ri.batch} x {dim} (placement {ri.placement})")
    print("cosine similarity:")
    for i, row in enumerate(rows):
        line = ""
        for other in rows:
            dot = sum(x * y for x, y in zip(row, other))
            line += f" {dot / (math.sqrt(sum(x * x for x in row)) * math.sqrt(sum(y * y for y in other))):6.3f}"
        print(f"{line}  {a.texts[i]}")

    lib.turbo_result_release(result)
    lib.turbo_session_release(session)
    lib.turbo_model_release(model)
    lib.turbo_context_release(ctx)
    lib.turbo_runtime_release(rt)
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except TurboError as e:
        print(f"error: {e}", file=sys.stderr)
        sys.exit(1)
