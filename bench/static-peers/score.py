"""One tool's vectors for the golden texts against the goldens.

Arguments: the golden safetensors file (golden-512), the vectors (raw F32,
one row per golden text, NaN rows for texts the tool refused), a label,
and optionally texts.json, to list up to ten rows under cosine 0.999 with their
norms and the start of their text. Prints the rows that are the golden's to the bit, the worst
cosine, the largest absolute difference, and the rows under cosine
0.99999 and 0.999.
"""

import json
import struct
import sys

import numpy as np


def golden(path):
    with open(path, "rb") as f:
        b = f.read()
    n = struct.unpack("<Q", b[:8])[0]
    h = json.loads(b[8 : 8 + n])["embeddings"]
    lo, hi = h["data_offsets"]
    return np.frombuffer(b[8 + n + lo : 8 + n + hi], "<f4").reshape(h["shape"])


def main(golden_path, vectors, label, texts=None):
    g = golden(golden_path)
    v = np.fromfile(vectors, "<f4")
    if v.size != g.size:
        print(f"{label}: {v.size // g.shape[1]} rows for {g.shape[0]} texts")
        return
    v = v.reshape(g.shape)
    refused = np.isnan(v).any(axis=1)
    ok = ~refused
    exact = int((v[ok].view("<u4") == g[ok].view("<u4")).all(axis=1).sum())
    gn = np.linalg.norm(g[ok].astype(np.float64), axis=1)
    vn = np.linalg.norm(v[ok].astype(np.float64), axis=1)
    dot = (g[ok].astype(np.float64) * v[ok]).sum(axis=1)
    both_zero = (gn == 0) & (vn == 0)
    cos = np.where(both_zero, 1.0, dot / np.maximum(gn * vn, 1e-300))
    diff = np.abs(g[ok].astype(np.float64) - v[ok]).max() if ok.any() else float("nan")
    print(
        f"{label}: {g.shape[0]} texts, {exact} to the bit, {int(refused.sum())} refused, "
        f"worst cosine {cos.min():.6f}, max abs diff {diff:.3e}, "
        f"{int((cos < 0.99999).sum())} under 0.99999, {int((cos < 0.999).sum())} under 0.999"
    )
    if texts:
        with open(texts, encoding="utf-8") as f:
            t = [x for x, k in zip(json.load(f), ok) if k]
        for i in np.argsort(cos)[: min(10, int((cos < 0.999).sum()))]:
            print(f"  cosine {cos[i]:.6f}, golden norm {gn[i]:.4f}, tool norm {vn[i]:.4f}, {len(t[i])} chars: {t[i][:60]!r}")


if __name__ == "__main__":
    main(*sys.argv[1:])
