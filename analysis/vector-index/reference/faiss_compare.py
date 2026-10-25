"""Pinned FAISS SQ4/SQ8 on quant_bench's actual rows and query split.
Usage: python faiss_compare.py DUMP_DIR DIM K OUTPUT_JSON
Run in Dockerfile.faiss with one CPU thread and the dump mounted read-only.
"""
import hashlib
import json
import statistics
import sys
import time
from pathlib import Path

import faiss
import numpy as np

root, dim, k, output = sys.argv[1:]
dim, k = int(dim), int(k)
faiss.omp_set_num_threads(1)
rows = np.fromfile(Path(root) / "rows.f32", dtype="<f4").reshape(-1, dim)
queries = np.fromfile(Path(root) / "queries.f32", dtype="<f4").reshape(-1, dim)
truth = np.fromfile(Path(root) / "truth.u32", dtype="<u4").reshape(-1, k)
assert len(truth) == len(queries)
records = []
for name, kind in [("FlatIP", None), ("SQ4", faiss.ScalarQuantizer.QT_4bit),
                   ("SQ8", faiss.ScalarQuantizer.QT_8bit)]:
    index = faiss.IndexFlatIP(dim) if kind is None else faiss.IndexScalarQuantizer(dim, kind, faiss.METRIC_INNER_PRODUCT)
    start = time.perf_counter()
    index.train(rows)
    index.add(rows)
    encode_s = time.perf_counter() - start
    _, ids = index.search(queries, k)
    recall = sum(len(set(g) & set(t)) for g, t in zip(ids, truth)) / truth.size
    for _ in range(3):
        for q in queries:
            index.search(q[None, :], k)
        index.search(queries, k)
    single = []
    for _ in range(3):
        for q in queries:
            start = time.perf_counter()
            index.search(q[None, :], k)
            single.append((time.perf_counter() - start) * 1000)
    batch = []
    for _ in range(5):
        start = time.perf_counter()
        index.search(queries, k)
        batch.append((time.perf_counter() - start) * 1000 / len(queries))
    records.append(dict(index=name, recall_at_k=recall,
                        bytes_per_row=len(faiss.serialize_index(index)) / len(rows),
                        encode_rows_per_s=len(rows) / encode_s,
                        query_ms=statistics.median(single), batch_ms_per_query=statistics.median(batch)))
report = dict(faiss=faiss.__version__, numpy=np.__version__, threads=1, dim=dim, k=k,
              rows=len(rows), queries=len(queries), records=records,
              sha256={p: hashlib.sha256((Path(root) / p).read_bytes()).hexdigest()
                      for p in ["rows.f32", "queries.f32", "truth.u32"]})
Path(output).write_text(json.dumps(report, indent=2) + "\n")
print(json.dumps(report, indent=2))
