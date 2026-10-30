"""In-process Analyzer and Document bindings over libturbo C ABI."""

from __future__ import annotations

from collections.abc import Sequence
import ctypes
from ctypes import (
    POINTER,
    byref,
    c_uint8,
    c_uint32,
    c_uint64,
    c_void_p,
)
from dataclasses import dataclass
import math
from pathlib import Path

from ._arrow import ArrowTableExport
from ._ffi import (
    TURBO_ANALYSIS_LAYER_CHUNKS,
    TURBO_ANALYSIS_LAYER_ENTITIES,
    TURBO_ANALYSIS_LAYER_LEMMAS,
    TURBO_ANALYSIS_LAYER_PII,
    TURBO_ANALYSIS_LAYER_POS,
    TURBO_ANALYSIS_LAYER_SENTENCES,
    TURBO_ANALYSIS_LAYER_TOKENS,
    TURBO_SPAN_LABEL,
    TURBO_SPAN_PROBABILITY,
    TURBO_SPAN_SOURCE_RANGE,
    TURBO_SPAN_TOKEN_RANGE,
    TURBO_SPAN_VALUE,
    TURBO_SPAN_VIEW_RANGE,
    TurboAnalyzerDesc,
    TurboSpan,
    TurboText,
    check_status,
    get_lib,
    make_error,
    make_text,
)


@dataclass(frozen=True)
class SpanRow:
    layer_kind: int
    layer: str
    instance: str
    item_index: int
    fragment_index: int
    view_id: int
    classification: int
    view_byte_span: tuple[int, int] | None
    view_utf16_span: tuple[int, int] | None
    view_codepoint_span: tuple[int, int] | None
    source_byte_span: tuple[int, int] | None
    source_utf16_span: tuple[int, int] | None
    source_codepoint_span: tuple[int, int] | None
    token_span: tuple[int, int] | None
    label: str | None
    value: str | None
    prob: float | None
    text: str | None


class Document:
    """In-process analyzed document holding typed annotation layers, binary
    `DocumentGraph` (`TGRF`) wire bytes, and zero-copy Apache Arrow exports.
    """

    def __init__(self, handle: c_void_p, source_text: str | None = None) -> None:
        self._handle: c_void_p | None = handle
        self.source_text = source_text
        self._source_bytes = source_text.encode("utf-8") if source_text is not None else None

    def _require_handle(self) -> c_void_p:
        if self._handle is None or not bool(self._handle):
            raise RuntimeError("Document has been closed")
        return self._handle

    def spans(self, layer_kind: int | None = None) -> list[SpanRow]:
        res = self._require_handle()
        lib = get_lib()
        err = make_error()

        spans_ptr = POINTER(TurboSpan)()
        count = c_uint32(0)
        check_status(lib.turbo_analysis_result_spans(res, byref(spans_ptr), byref(count), byref(err)), err)

        pool_ptr = c_void_p()
        pool_len = c_uint64(0)
        check_status(lib.turbo_analysis_result_strings(res, byref(pool_ptr), byref(pool_len), byref(err)), err)

        n = int(count.value)
        if n == 0:
            return []

        p_len = int(pool_len.value)
        pool_bytes = (
            bytes(ctypes.cast(pool_ptr, POINTER(c_uint8 * p_len)).contents)
            if (p_len > 0 and bool(pool_ptr))
            else b""
        )

        def _slice_pool(offset: int, length: int) -> str:
            if length == 0:
                return ""
            return pool_bytes[offset : offset + length].decode("utf-8", errors="replace")

        out: list[SpanRow] = []
        for idx in range(n):
            sp = spans_ptr[idx]
            kind = int(sp.layer_kind)
            if layer_kind is not None and kind != layer_kind:
                continue
            flags = int(sp.flags)
            has_view = bool(flags & TURBO_SPAN_VIEW_RANGE)
            has_source = bool(flags & TURBO_SPAN_SOURCE_RANGE)
            has_tokens = bool(flags & TURBO_SPAN_TOKEN_RANGE)
            has_label = bool(flags & TURBO_SPAN_LABEL)
            has_value = bool(flags & TURBO_SPAN_VALUE)
            has_prob = bool(flags & TURBO_SPAN_PROBABILITY)

            src_bytes = (int(sp.source_byte_start), int(sp.source_byte_end)) if has_source else None
            covered_text: str | None = None
            if src_bytes is not None and self._source_bytes is not None:
                s_b, e_b = src_bytes
                if 0 <= s_b <= e_b <= len(self._source_bytes):
                    covered_text = self._source_bytes[s_b:e_b].decode("utf-8", errors="replace")

            prob_val = float(sp.prob)
            out.append(
                SpanRow(
                    layer_kind=kind,
                    layer=_slice_pool(int(sp.layer_offset), int(sp.layer_len)),
                    instance=_slice_pool(int(sp.instance_offset), int(sp.instance_len)),
                    item_index=int(sp.item_index),
                    fragment_index=int(sp.fragment_index),
                    view_id=int(sp.view_id),
                    classification=int(sp.classification),
                    view_byte_span=(int(sp.view_byte_start), int(sp.view_byte_end)) if has_view else None,
                    view_utf16_span=(int(sp.view_utf16_start), int(sp.view_utf16_end)) if has_view else None,
                    view_codepoint_span=(int(sp.view_codepoint_start), int(sp.view_codepoint_end))
                    if has_view
                    else None,
                    source_byte_span=src_bytes,
                    source_utf16_span=(int(sp.source_utf16_start), int(sp.source_utf16_end)) if has_source else None,
                    source_codepoint_span=(int(sp.source_codepoint_start), int(sp.source_codepoint_end))
                    if has_source
                    else None,
                    token_span=(int(sp.token_start), int(sp.token_end)) if has_tokens else None,
                    label=_slice_pool(int(sp.label_offset), int(sp.label_len)) if has_label else None,
                    value=_slice_pool(int(sp.value_offset), int(sp.value_len)) if has_value else None,
                    prob=prob_val if (has_prob and not math.isnan(prob_val)) else None,
                    text=covered_text,
                )
            )
        return out

    def sentences(self) -> list[SpanRow]:
        return self.spans(TURBO_ANALYSIS_LAYER_SENTENCES)

    def tokens(self) -> list[SpanRow]:
        return self.spans(TURBO_ANALYSIS_LAYER_TOKENS)

    def pos_tags(self) -> list[SpanRow]:
        return self.spans(TURBO_ANALYSIS_LAYER_POS)

    def lemmas(self) -> list[SpanRow]:
        return self.spans(TURBO_ANALYSIS_LAYER_LEMMAS)

    def entities(self) -> list[SpanRow]:
        return self.spans(TURBO_ANALYSIS_LAYER_ENTITIES)

    def pii(self) -> list[SpanRow]:
        return self.spans(TURBO_ANALYSIS_LAYER_PII)

    def chunks(self) -> list[SpanRow]:
        return self.spans(TURBO_ANALYSIS_LAYER_CHUNKS)

    def to_graph_bytes(self) -> bytes:
        """Return the serialized `DocumentGraph` (`TGRF` binary wire format)."""
        res = self._require_handle()
        lib = get_lib()
        data_ptr = POINTER(c_uint8)()
        data_len = c_uint64(0)
        err = make_error()
        check_status(lib.turbo_analysis_result_graph(res, byref(data_ptr), byref(data_len), byref(err)), err)
        n = int(data_len.value)
        if n == 0 or not bool(data_ptr):
            return b""
        return bytes(ctypes.cast(data_ptr, POINTER(c_uint8 * n)).contents)

    def spans_arrow(self, layer_kind: int = 0) -> ArrowTableExport:
        return ArrowTableExport(self._require_handle(), layer_kind=layer_kind)

    def sentences_arrow(self) -> ArrowTableExport:
        return ArrowTableExport(self._require_handle(), layer_kind=TURBO_ANALYSIS_LAYER_SENTENCES)

    def tokens_arrow(self) -> ArrowTableExport:
        return ArrowTableExport(self._require_handle(), layer_kind=TURBO_ANALYSIS_LAYER_TOKENS)

    def pos_arrow(self) -> ArrowTableExport:
        return ArrowTableExport(self._require_handle(), layer_kind=TURBO_ANALYSIS_LAYER_POS)

    def lemmas_arrow(self) -> ArrowTableExport:
        return ArrowTableExport(self._require_handle(), layer_kind=TURBO_ANALYSIS_LAYER_LEMMAS)

    def entities_arrow(self) -> ArrowTableExport:
        return ArrowTableExport(self._require_handle(), layer_kind=TURBO_ANALYSIS_LAYER_ENTITIES)

    def pii_arrow(self) -> ArrowTableExport:
        return ArrowTableExport(self._require_handle(), layer_kind=TURBO_ANALYSIS_LAYER_PII)

    def chunks_arrow(self) -> ArrowTableExport:
        return ArrowTableExport(self._require_handle(), layer_kind=TURBO_ANALYSIS_LAYER_CHUNKS)

    def __arrow_c_array__(self, requested_schema: object = None) -> tuple[object, object]:
        with self.spans_arrow(0) as exp:
            return exp.__arrow_c_array__(requested_schema=requested_schema)

    def __arrow_c_stream__(self, requested_schema: object = None) -> object:
        with self.spans_arrow(0) as exp:
            return exp.__arrow_c_stream__(requested_schema=requested_schema)

    def close(self) -> None:
        if self._handle is not None and bool(self._handle):
            get_lib().turbo_analysis_result_release(self._handle)
            self._handle = None

    def __enter__(self) -> Document:
        self._require_handle()
        return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        self.close()

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass


class Analyzer:
    """In-process text analysis pipeline backed by `turbo_analyzer` (`libturbo`).

    All native C calls release the Python GIL and are safe to invoke concurrently
    from multiple Python threads.
    """

    def __init__(
        self,
        language: str = "en",
        *,
        sent_model_path: str | Path = "",
        tok_model_path: str | Path = "",
        pos_model_path: str | Path = "",
        lemma_model_path: str | Path = "",
        ner_model_path: str | Path = "",
    ) -> None:
        lib = get_lib()
        lang_b = language.encode("utf-8")
        sent_b = str(sent_model_path).encode("utf-8") if sent_model_path else b""
        tok_b = str(tok_model_path).encode("utf-8") if tok_model_path else b""
        pos_b = str(pos_model_path).encode("utf-8") if pos_model_path else b""
        lemma_b = str(lemma_model_path).encode("utf-8") if lemma_model_path else b""
        ner_b = str(ner_model_path).encode("utf-8") if ner_model_path else b""

        desc = TurboAnalyzerDesc()
        desc.struct_size = ctypes.sizeof(TurboAnalyzerDesc)
        desc.flags = 0
        desc.language = make_text(lang_b)
        desc.sent_model_path = make_text(sent_b)
        desc.tok_model_path = make_text(tok_b)
        desc.pos_model_path = make_text(pos_b)
        desc.lemma_model_path = make_text(lemma_b)
        desc.ner_model_path = make_text(ner_b)

        handle = c_void_p()
        err = make_error()
        code = lib.turbo_analyzer_create(byref(desc), byref(handle), byref(err))
        check_status(code, err)
        self._handle: c_void_p | None = handle

    def _require_handle(self) -> c_void_p:
        if self._handle is None or not bool(self._handle):
            raise RuntimeError("Analyzer has been closed")
        return self._handle

    def analyze(self, text: str) -> Document:
        """Run the configured analysis pipeline, build the binary `DocumentGraph`
        (`TGRF`), and return a `Document` with all layers and chunks.
        """
        a = self._require_handle()
        lib = get_lib()
        text_b = text.encode("utf-8")
        out = c_void_p()
        err = make_error()
        code = lib.turbo_analyzer_analyze_graph(a, make_text(text_b), byref(out), byref(err))
        check_status(code, err)
        return Document(out, source_text=text)

    def from_graph_bytes(self, graph_bytes: bytes, *, source_text: str | None = None) -> Document:
        """Validate and import a binary `DocumentGraph` (`TGRF`) wire buffer."""
        a = self._require_handle()
        lib = get_lib()
        n = len(graph_bytes)
        buf = (c_uint8 * max(n, 1)).from_buffer_copy(graph_bytes if n > 0 else b"\x00")
        out = c_void_p()
        err = make_error()
        code = lib.turbo_analyzer_inspect_graph(
            a,
            buf if n > 0 else POINTER(c_uint8)(),
            n,
            byref(out),
            byref(err),
        )
        check_status(code, err)
        return Document(out, source_text=source_text)

    def detect_sentences(self, text: str) -> Document:
        a = self._require_handle()
        lib = get_lib()
        text_b = text.encode("utf-8")
        out = c_void_p()
        err = make_error()
        code = lib.turbo_analyzer_detect_sentences(a, make_text(text_b), byref(out), byref(err))
        check_status(code, err)
        return Document(out, source_text=text)

    def tokenize(self, text: str) -> Document:
        a = self._require_handle()
        lib = get_lib()
        text_b = text.encode("utf-8")
        out = c_void_p()
        err = make_error()
        code = lib.turbo_analyzer_tokenize(a, make_text(text_b), byref(out), byref(err))
        check_status(code, err)
        return Document(out, source_text=text)

    def pos_tag(self, tokens: Sequence[str]) -> Document:
        a = self._require_handle()
        lib = get_lib()
        n = len(tokens)
        tok_bytes = [t.encode("utf-8") for t in tokens]
        c_toks = (TurboText * max(n, 1))(*[make_text(b) for b in tok_bytes])
        out = c_void_p()
        err = make_error()
        code = lib.turbo_analyzer_pos_tag(
            a,
            c_toks if n > 0 else POINTER(TurboText)(),
            n,
            byref(out),
            byref(err),
        )
        check_status(code, err)
        return Document(out)

    def lemmatize(self, tokens: Sequence[str], tags: Sequence[str]) -> Document:
        if len(tokens) != len(tags):
            raise ValueError("tokens and tags must have the same length")
        a = self._require_handle()
        lib = get_lib()
        n = len(tokens)
        tok_bytes = [t.encode("utf-8") for t in tokens]
        tag_bytes = [t.encode("utf-8") for t in tags]
        c_toks = (TurboText * max(n, 1))(*[make_text(b) for b in tok_bytes])
        c_tags = (TurboText * max(n, 1))(*[make_text(b) for b in tag_bytes])
        out = c_void_p()
        err = make_error()
        code = lib.turbo_analyzer_lemmatize(
            a,
            c_toks if n > 0 else POINTER(TurboText)(),
            c_tags if n > 0 else POINTER(TurboText)(),
            n,
            byref(out),
            byref(err),
        )
        check_status(code, err)
        return Document(out)

    def find_tokens(self, tokens: Sequence[str]) -> Document:
        a = self._require_handle()
        lib = get_lib()
        n = len(tokens)
        tok_bytes = [t.encode("utf-8") for t in tokens]
        c_toks = (TurboText * max(n, 1))(*[make_text(b) for b in tok_bytes])
        out = c_void_p()
        err = make_error()
        code = lib.turbo_analyzer_find_tokens(
            a,
            c_toks if n > 0 else POINTER(TurboText)(),
            n,
            byref(out),
            byref(err),
        )
        check_status(code, err)
        return Document(out)

    def find_text(self, text: str) -> Document:
        a = self._require_handle()
        lib = get_lib()
        text_b = text.encode("utf-8")
        out = c_void_p()
        err = make_error()
        code = lib.turbo_analyzer_find_text(a, make_text(text_b), byref(out), byref(err))
        check_status(code, err)
        return Document(out, source_text=text)

    def detect_pii(self, text: str) -> Document:
        a = self._require_handle()
        lib = get_lib()
        text_b = text.encode("utf-8")
        out = c_void_p()
        err = make_error()
        code = lib.turbo_analyzer_detect_pii(a, make_text(text_b), byref(out), byref(err))
        check_status(code, err)
        return Document(out, source_text=text)

    def chunk(self, text: str, *, target_tokens: int = 64, max_tokens: int = 128) -> Document:
        a = self._require_handle()
        lib = get_lib()
        text_b = text.encode("utf-8")
        out = c_void_p()
        err = make_error()
        code = lib.turbo_analyzer_chunk_text(
            a,
            make_text(text_b),
            int(target_tokens),
            int(max_tokens),
            byref(out),
            byref(err),
        )
        check_status(code, err)
        return Document(out, source_text=text)

    def close(self) -> None:
        if self._handle is not None and bool(self._handle):
            get_lib().turbo_analyzer_release(self._handle)
            self._handle = None

    def __enter__(self) -> Analyzer:
        self._require_handle()
        return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        self.close()

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass
