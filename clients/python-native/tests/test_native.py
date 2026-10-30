#!/usr/bin/env python3
"""Integration tests for turboembed-native (`clients/python-native`, issue #1469)."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import ctypes
from ctypes import byref
import gc
import math
from pathlib import Path
import sys
import tempfile
import unittest
import zipfile

REPO_ROOT = Path(__file__).resolve().parents[3]
CLIENT_ROOT = Path(__file__).resolve().parents[1]
if str(CLIENT_ROOT) not in sys.path:
    sys.path.insert(0, str(CLIENT_ROOT))

import build_wheel as wheel_builder  # noqa: E402
import turboembed_native as te  # noqa: E402
from turboembed_native._ffi import (  # noqa: E402
    ArrowArray,
    ArrowArrayStream,
    ArrowSchema,
)

TINY_BERT_BUNDLE = REPO_ROOT / "testdata" / "tiny-bert-bundle"


class TestTurboEmbedNative(unittest.TestCase):
    def test_runtime_bundle_session_and_zero_copy_embeddings(self) -> None:
        with te.Runtime() as rt:
            self.assertTrue(rt.version().startswith("0.1.0"))
            devices = rt.devices()
            self.assertGreaterEqual(len(devices), 1)
            self.assertEqual(devices[0].backend, "cpu")

            with rt.open_bundle(TINY_BERT_BUNDLE) as bundle:
                self.assertEqual(bundle.info.dim, 32)
                self.assertEqual(bundle.info.task, te.TURBO_TASK_EMBED)

                with bundle.create_session(max_batch=4, max_ranges=8) as session:
                    texts = [
                        "The quick brown fox jumps over the lazy dog.",
                        "reset a password",
                    ]
                    tok_batch = session.tokenize(texts)
                    self.assertEqual(tok_batch.batch, 2)
                    self.assertEqual(len(tok_batch.lengths), 2)
                    self.assertGreater(tok_batch.lengths[0], 0)

                    # 1. Float32 zero-copy embedding view
                    with session.embed(texts, prompt_role=te.TURBO_PROMPT_QUERY) as batch:
                        self.assertEqual(batch.shape, (2, 32))
                        self.assertEqual(batch.dtype, te.TURBO_DTYPE_F32)
                        self.assertEqual(batch.typestr, "<f4")
                        self.assertGreater(batch.data_ptr, 0)

                        iface = batch.__array_interface__
                        self.assertEqual(iface["version"], 3)
                        self.assertEqual(iface["shape"], (2, 32))
                        self.assertEqual(iface["typestr"], "<f4")
                        self.assertEqual(iface["data"], (batch.data_ptr, True))

                        mv = batch.memoryview()
                        self.assertEqual(mv.shape, (2, 32))
                        self.assertEqual(mv.format, "f")
                        rows = batch.to_list()
                        self.assertEqual(len(rows), 2)
                        norm0 = sum(float(x) * float(x) for x in rows[0])
                        self.assertAlmostEqual(norm0, 1.0, places=5)

                        try:
                            import numpy as np  # type: ignore[import-not-found]

                            arr = np.asarray(batch)
                            self.assertEqual(arr.shape, (2, 32))
                            self.assertEqual(arr.dtype, np.float32)
                            self.assertEqual(arr.__array_interface__["data"][0], batch.data_ptr)
                        except ImportError:
                            pass

                    # Lifetime safety after close()
                    with self.assertRaises(RuntimeError):
                        _ = batch.memoryview()
                    with self.assertRaises(RuntimeError):
                        _ = batch.__array_interface__
                    with self.assertRaises(RuntimeError):
                        _ = batch.to_list()

                    # 2. Int8 quantized zero-copy embedding view
                    int8_ranges = [-1.0] * 32 + [1.0] * 32
                    with session.embed(
                        texts,
                        prompt_role=te.TURBO_PROMPT_QUERY,
                        quantization=te.TURBO_QUANTIZE_INT8,
                        int8_ranges=int8_ranges,
                    ) as q_i8:
                        self.assertEqual(q_i8.shape, (2, 32))
                        self.assertEqual(q_i8.dtype, te.TURBO_DTYPE_I8)
                        self.assertEqual(q_i8.typestr, "|i1")
                        mv_i8 = q_i8.memoryview()
                        self.assertEqual(mv_i8.shape, (2, 32))
                        self.assertEqual(mv_i8.format, "b")

                    # 3. UBinary packed uint8 zero-copy embedding view
                    with session.embed(
                        texts,
                        prompt_role=te.TURBO_PROMPT_QUERY,
                        quantization=te.TURBO_QUANTIZE_UBINARY,
                    ) as q_u8:
                        self.assertEqual(q_u8.shape, (2, 4))
                        self.assertEqual(q_u8.dtype, te.TURBO_DTYPE_U8)
                        self.assertEqual(q_u8.typestr, "|u1")
                        mv_u8 = q_u8.memoryview()
                        self.assertEqual(mv_u8.shape, (2, 4))
                        self.assertEqual(mv_u8.format, "B")

                    # 4. Late pooling token ranges (`embed_ranges`)
                    with session.embed_ranges(
                        texts,
                        ranges=[(0, 1, 4), (1, 1, 3)],
                    ) as r_batch:
                        self.assertEqual(r_batch.shape, (2, 32))
                        r_rows = r_batch.to_list()
                        self.assertEqual(len(r_rows), 2)
                        for row in r_rows:
                            self.assertAlmostEqual(sum(float(x) * float(x) for x in row), 1.0, places=5)

    def test_analyzer_pipeline_utf16_offsets_and_graph_round_trip(self) -> None:
        text = "Hello 🚀 Café 東京! Dr. John Smith visited New York and emailed alice@example.com."
        utf8_bytes = text.encode("utf-8")
        utf16_bytes = text.encode("utf-16-le")

        with te.Analyzer("en") as analyzer:
            # 1. Sentence detection
            with analyzer.detect_sentences(text) as sent_doc:
                sents = sent_doc.sentences()
                self.assertEqual(len(sents), 2)
                for sp in sents:
                    self.assertIsNotNone(sp.source_byte_span)
                    self.assertIsNotNone(sp.source_utf16_span)
                    self.assertIsNotNone(sp.source_codepoint_span)
                    b0, b1 = sp.source_byte_span  # type: ignore[misc]
                    u0, u1 = sp.source_utf16_span  # type: ignore[misc]
                    c0, c1 = sp.source_codepoint_span  # type: ignore[misc]
                    self.assertEqual(utf8_bytes[b0:b1].decode("utf-8"), sp.text)
                    self.assertEqual(utf16_bytes[u0 * 2 : u1 * 2].decode("utf-16-le"), sp.text)
                    self.assertEqual(text[c0:c1], sp.text)

            # 2. Tokenization
            with analyzer.tokenize(text) as tok_doc:
                toks = tok_doc.tokens()
                self.assertGreater(len(toks), 10)
                texts = [t.text for t in toks]
                self.assertIn("Café", texts)
                self.assertIn("John", texts)
                for idx, tok in enumerate(toks):
                    self.assertEqual(tok.token_span, (idx, idx + 1))
                    u0, u1 = tok.source_utf16_span  # type: ignore[misc]
                    self.assertEqual(utf16_bytes[u0 * 2 : u1 * 2].decode("utf-16-le"), tok.text)

            # 3. Structured PII detection
            with analyzer.detect_pii(text) as pii_doc:
                pii_rows = pii_doc.pii()
                self.assertTrue(any(r.label == "email" and r.value == "alice@example.com" for r in pii_rows))

            # 4. DP chunking
            with analyzer.chunk(text, target_tokens=6, max_tokens=10) as chunk_doc:
                chunks = chunk_doc.chunks()
                self.assertGreaterEqual(len(chunks), 2)
                self.assertEqual(chunks[0].label, "document_start")
                for ch in chunks:
                    self.assertLessEqual(ch.classification, 10)
                    self.assertIsNotNone(ch.token_span)
                    t0, t1 = ch.token_span  # type: ignore[misc]
                    self.assertEqual(ch.classification, t1 - t0)

            # 5. Full DocumentGraph ("TGRF") binary wire round-trip
            with analyzer.analyze(text) as doc:
                wire = doc.to_graph_bytes()
                self.assertTrue(wire.startswith(b"TGRF"))
                self.assertGreater(len(wire), 16)
                self.assertGreater(len(doc.sentences()), 0)
                self.assertGreater(len(doc.tokens()), 0)
                self.assertGreater(len(doc.pii()), 0)
                self.assertGreater(len(doc.chunks()), 0)

                with analyzer.from_graph_bytes(wire, source_text=text) as imported:
                    self.assertEqual(len(imported.sentences()), len(doc.sentences()))
                    self.assertEqual(len(imported.tokens()), len(doc.tokens()))
                    self.assertEqual(len(imported.pii()), len(doc.pii()))
                    self.assertEqual(len(imported.chunks()), len(doc.chunks()))
                    self.assertEqual([t.text for t in imported.tokens()], [t.text for t in doc.tokens()])

            # 6. Unconfigured model tasks raise TurboEmbedError (UNSUPPORTED_TASK)
            with self.assertRaises(te.TurboEmbedError) as ctx:
                analyzer.pos_tag(["Hello", "world"])
            self.assertEqual(ctx.exception.status_name, "TURBO_E_UNSUPPORTED_TASK")

        # 7. Configured dictionary lemmatizer
        with tempfile.TemporaryDirectory() as tmp:
            dict_path = Path(tmp) / "lemma.tsv"
            dict_path.write_text("went\tVERB\tgo\nchildren\tNOUN\tchild\n", encoding="utf-8")
            with te.Analyzer("en", lemma_model_path=dict_path) as lemma_analyzer:
                with lemma_analyzer.lemmatize(["went", "children", "unknown"], ["VERB", "NOUN", "NOUN"]) as ldoc:
                    lemmas = ldoc.lemmas()
                    self.assertEqual([r.label for r in lemmas], ["go", "child", None])

    def test_arrow_c_data_interface_and_pycapsule_protocol(self) -> None:
        text = "First sentence has five words. Second sentence emails bob@example.org today."
        with te.Analyzer("en") as analyzer:
            with analyzer.analyze(text) as doc:
                # 1. Verify tokens_arrow() and chunks_arrow() column dictionaries
                with doc.tokens_arrow() as tok_arrow:
                    tok_dict = tok_arrow.to_pydict()
                    self.assertEqual(len(tok_dict["item_index"]), len(doc.tokens()))
                    self.assertTrue(all(k == te.TURBO_ANALYSIS_LAYER_TOKENS for k in tok_dict["layer_kind"]))
                    self.assertTrue(all(layer == "opennlp:tokens" for layer in tok_dict["layer"]))

                    # Verify __arrow_c_array__ PyCapsules
                    schema_cap, array_cap = tok_arrow.__arrow_c_array__()
                    self.assertTrue(te.is_valid_capsule(schema_cap, b"arrow_schema"))
                    self.assertTrue(te.is_valid_capsule(array_cap, b"arrow_array"))

                    schema_ptr = te.capsule_pointer(schema_cap, b"arrow_schema")
                    array_ptr = te.capsule_pointer(array_cap, b"arrow_array")
                    c_schema = ArrowSchema.from_address(schema_ptr)
                    c_array = ArrowArray.from_address(array_ptr)
                    self.assertEqual(c_schema.format, b"+s")
                    self.assertEqual(c_schema.n_children, 24)
                    self.assertEqual(c_array.length, len(doc.tokens()))
                    self.assertEqual(c_array.n_children, 24)
                    self.assertTrue(bool(c_schema.release))
                    self.assertTrue(bool(c_array.release))

                    # Drop capsules and run GC to verify capsule destructors release cleanly
                    del schema_cap
                    del array_cap
                    gc.collect()

                # 2. Verify chunks_arrow() and __arrow_c_stream__ PyCapsule
                with doc.chunks_arrow() as chunk_arrow:
                    chunk_rows = chunk_arrow.to_pylist()
                    self.assertEqual(len(chunk_rows), len(doc.chunks()))
                    self.assertEqual(chunk_rows[0]["label"], "document_start")
                    self.assertEqual(chunk_rows[0]["layer"], "turbo:chunks")

                    stream_cap = chunk_arrow.__arrow_c_stream__()
                    self.assertTrue(te.is_valid_capsule(stream_cap, b"arrow_array_stream"))
                    stream_ptr = te.capsule_pointer(stream_cap, b"arrow_array_stream")
                    c_stream = ArrowArrayStream.from_address(stream_ptr)

                    out_schema = ArrowSchema()
                    self.assertEqual(c_stream.get_schema(byref(c_stream), byref(out_schema)), 0)
                    self.assertEqual(out_schema.format, b"+s")
                    out_schema.release(byref(out_schema))

                    batch1 = ArrowArray()
                    self.assertEqual(c_stream.get_next(byref(c_stream), byref(batch1)), 0)
                    self.assertTrue(bool(batch1.release))
                    self.assertEqual(batch1.length, len(doc.chunks()))
                    batch1.release(byref(batch1))

                    batch2 = ArrowArray()
                    self.assertEqual(c_stream.get_next(byref(c_stream), byref(batch2)), 0)
                    self.assertFalse(bool(batch2.release))

                    del stream_cap
                    gc.collect()

                # 3. Verify optional pyarrow / polars zero-copy import if available
                try:
                    import pyarrow as pa  # type: ignore[import-not-found]

                    pa_table = pa.table(doc.tokens_arrow())
                    self.assertEqual(pa_table.num_rows, len(doc.tokens()))
                except ImportError:
                    pass

                try:
                    import polars as pl  # type: ignore[import-not-found]

                    df = pl.from_arrow(doc.chunks_arrow())
                    self.assertEqual(len(df), len(doc.chunks()))
                except ImportError:
                    pass

    def test_multithreaded_gil_release_concurrency(self) -> None:
        with te.Runtime() as rt, rt.open_bundle(TINY_BERT_BUNDLE) as bundle, bundle.create_session(max_batch=4) as session:
            with te.Analyzer("en") as analyzer:

                def worker(idx: int) -> tuple[int, int, float]:
                    text = f"Worker {idx} processes text with email user{idx}@example.com. Second sentence here."
                    with analyzer.analyze(text) as doc:
                        n_tok = len(doc.tokens())
                        n_chunks = len(doc.chunks())
                    with session.embed([text], prompt_role=te.TURBO_PROMPT_QUERY) as emb:
                        norm = sum(float(x) * float(x) for x in emb.to_list()[0])
                    return (n_tok, n_chunks, norm)

                with ThreadPoolExecutor(max_workers=8) as pool:
                    results = list(pool.map(worker, range(24)))

                self.assertEqual(len(results), 24)
                for n_tok, n_chunks, norm in results:
                    self.assertGreater(n_tok, 8)
                    self.assertGreaterEqual(n_chunks, 1)
                    self.assertTrue(math.isfinite(norm))
                    self.assertAlmostEqual(norm, 1.0, places=5)

    def test_build_cp310_abi3_wheel(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            out_dir = Path(tmp)
            whl_path = wheel_builder.build_wheel(out_dir=out_dir, version="0.1.0")
            self.assertTrue(whl_path.is_file())
            self.assertIn("-cp310-abi3-", whl_path.name)

            with zipfile.ZipFile(whl_path, "r") as zf:
                names = set(zf.namelist())
                self.assertIn("turboembed_native/__init__.py", names)
                self.assertIn("turboembed_native/_ffi.py", names)
                self.assertIn("turboembed_native/_runtime.py", names)
                self.assertIn("turboembed_native/_analysis.py", names)
                self.assertIn("turboembed_native/_arrow.py", names)
                self.assertIn("turboembed_native-0.1.0.dist-info/METADATA", names)
                self.assertIn("turboembed_native-0.1.0.dist-info/WHEEL", names)
                self.assertIn("turboembed_native-0.1.0.dist-info/RECORD", names)
                self.assertIn("turboembed_native-0.1.0.dist-info/licenses/LICENSE", names)
                self.assertIn("turboembed_native-0.1.0.dist-info/licenses/NOTICE", names)
                self.assertTrue(
                    any(n.startswith("turboembed_native/") and n.endswith((".so", ".dylib", ".dll")) for n in names)
                )
                wheel_meta = zf.read("turboembed_native-0.1.0.dist-info/WHEEL").decode("utf-8")
                self.assertIn("Root-Is-Purelib: false", wheel_meta)
                self.assertIn("Tag: cp310-abi3-", wheel_meta)


if __name__ == "__main__":
    unittest.main(verbosity=2)
