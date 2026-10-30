# `turboembed-native` (`clients/python-native`)

In-process native Python library for `libturbo` (`Runtime`, `Bundle`, `Tokenizer`, `Session`) and `turbo-analysis` (`Analyzer`, `Document`, `ArrowTableExport`) with:

- **Automatic GIL Release**: Every inference and text-analysis call into `libturbo` releases the Python GIL so multi-threaded Python callers execute in parallel.
- **Zero-Copy Embedding Views (`f32`, `i8`, `u8`)**: `EmbeddingBatch` exposes PEP 3118 / PEP 688 `memoryview` and NumPy's `__array_interface__` (`np.asarray(batch)`) directly over the native `turbo_buffer` host pointer without copying, with deterministic `close()` / `with` context management.
- **Zero-Copy Apache Arrow & Polars Export (`__arrow_c_array__` / `__arrow_c_stream__`)**: `Document` and `ArrowTableExport` (`doc.tokens_arrow()`, `doc.sentences_arrow()`, `doc.entities_arrow()`, `doc.pii_arrow()`, `doc.chunks_arrow()`) implement the Apache Arrow PyCapsule Interface over `ArrowSchema`, `ArrowArray`, and `ArrowArrayStream` so `polars.from_arrow(doc.chunks_arrow())` and `pyarrow.table(doc.tokens_arrow())` import in zero copy.

## Quick Start

```python
import numpy as np
import turboembed_native as te

# 1. In-process text analysis, DocumentGraph wire round-trip, and Arrow export
with te.Analyzer("en") as analyzer:
    doc = analyzer.analyze("Hello 🚀 Café! Contact alice@example.com for details.")
    tokens_table = doc.tokens_arrow().to_pydict()
    chunks_table = doc.chunks_arrow().to_pydict()
    wire_bytes = doc.to_graph_bytes()

# 2. In-process embedding session with zero-copy buffer protocol / NumPy view
with te.Runtime() as rt:
    with rt.open_bundle("path/to/bundle") as bundle:
        with bundle.create_session() as session:
            with session.embed(["first text", "second text"]) as batch:
                arr = np.asarray(batch)  # zero-copy view over native f32/i8/u8 buffer
```

## Building the `cp310-abi3` Wheel

```bash
cargo build --release -p turbo
python3 clients/python-native/build_wheel.py --out-dir target/wheels
```
