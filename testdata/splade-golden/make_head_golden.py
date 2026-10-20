#!/usr/bin/env python3
"""The reference for turbo's MLM head (core/src/mlm.rs), in numpy.

No SPLADE checkpoint is fetchable here (see make_golden.py), so this script
is the independent implementation the head is held to: the BERT
`cls.predictions` forward (transform, GELU erf, LayerNorm, decoder) over a
hand-built tiny head. head_golden.json feeds core/tests/mlm.rs.

Run: python3 make_head_golden.py   (rewrites head_golden.json deterministically)
"""

import json
import math

import numpy as np

HIDDEN = 4
VOCAB = 8
EPS = 1e-12


def gelu(x: np.ndarray) -> np.ndarray:
    """0.5 x (1 + erf(x / sqrt 2)), in float64 like upstream's torch."""
    return (0.5 * x * (1.0 + np.vectorize(math.erf)(x / math.sqrt(2.0)))).astype(np.float32)


def layer_norm(y: np.ndarray, w: np.ndarray, b: np.ndarray) -> np.ndarray:
    mean = np.mean(y, dtype=np.float64)
    var = np.mean((y.astype(np.float64) - mean) ** 2)
    inv = 1.0 / math.sqrt(var + EPS)
    return ((y.astype(np.float64) - mean) * inv).astype(np.float32) * w + b


def main() -> None:
    rng = np.random.default_rng(708)
    transform_w = rng.uniform(-0.5, 0.5, size=(HIDDEN, HIDDEN)).astype(np.float32)
    transform_b = rng.uniform(-0.5, 0.5, size=HIDDEN).astype(np.float32)
    ln_w = rng.uniform(0.5, 1.5, size=HIDDEN).astype(np.float32)
    ln_b = rng.uniform(-0.5, 0.5, size=HIDDEN).astype(np.float32)
    decoder_w = rng.uniform(-0.5, 0.5, size=(VOCAB, HIDDEN)).astype(np.float32)
    decoder_b = rng.uniform(-0.5, 0.5, size=VOCAB).astype(np.float32)
    states = rng.uniform(-2.0, 2.0, size=(3, HIDDEN)).astype(np.float32)

    # y = x W^T + b, then GELU, LayerNorm, then the decoder.
    t = states @ transform_w.T + transform_b
    t = gelu(t)
    n = np.stack([layer_norm(row, ln_w, ln_b) for row in t])
    logits = n @ decoder_w.T + decoder_b

    out = {
        "hidden": HIDDEN,
        "vocab": VOCAB,
        "eps": EPS,
        "transform_weight": transform_w.tolist(),
        "transform_bias": transform_b.tolist(),
        "ln_weight": ln_w.tolist(),
        "ln_bias": ln_b.tolist(),
        "decoder_weight": decoder_w.tolist(),
        "decoder_bias": decoder_b.tolist(),
        "states": states.tolist(),
        "logits": logits.tolist(),
    }
    with open("head_golden.json", "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1)
        f.write("\n")
    print(f"head_golden.json: hidden {HIDDEN}, vocab {VOCAB}, {states.shape[0]} states")


if __name__ == "__main__":
    main()
