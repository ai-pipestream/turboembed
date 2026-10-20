#!/usr/bin/env python3
"""The reference for turbo's SPLADE pooling (core/src/sparse.rs), in numpy.

`naver/splade-v3` and the other SPLADE checkpoints are licence-gated and
not fetchable in the CI environment, so the acceptance golden for #708 is
synthetic: a small hand-seeded logits tensor stands in for an MLM head's
output, and this script pools it exactly as SPLADE pools
(arXiv 2403.06789): weight_v = max over tokens of log(1 + relu(logit_tv)),
cut by top-k or a weight floor. golden.json feeds core/tests/sparse_golden.rs,
which must match this output within float tolerance.

Run: python3 make_golden.py   (rewrites golden.json deterministically)
"""

import json

import numpy as np

VOCAB = 64
TOKENS = 7


def make_logits() -> np.ndarray:
    """A deterministic logits tensor [TOKENS, VOCAB] in float32 with all
    the shapes the selection must handle: heavy and light positives, exact
    ties at a top-k boundary, all-negative columns, and zeros."""
    rng = np.random.default_rng(20261020)
    logits = rng.uniform(-3.0, 3.0, size=(TOKENS, VOCAB)).astype(np.float32)
    # Term 3: two tokens tie at exactly 2.5, one more at 2.5 below the cut.
    logits[1, 3] = np.float32(2.5)
    logits[4, 3] = np.float32(2.5)
    logits[5, 3] = np.float32(-1.0)
    # Term 9: three-way tie with term 3's weight, to cut across ties.
    logits[2, 9] = np.float32(2.5)
    # Term 17: negative everywhere: pools to 0 and always drops.
    logits[:, 17] = np.float32(-0.25)
    # Term 23: zeros only: relu gives 0, weight log(1) = 0, drops at floor 0.
    logits[:, 23] = np.float32(0.0)
    # Term 40: one strong token, the rest negative.
    logits[6, 40] = np.float32(4.25)
    # Term 41: exactly the mirror, to pair with 40 at cuts.
    logits[0, 41] = np.float32(4.25)
    return logits


def pool(logits: np.ndarray) -> np.ndarray:
    """SPLADE max pooling in float32, as sparse.rs computes it."""
    return np.max(np.log1p(np.maximum(logits, np.float32(0.0))), axis=0)


def select(weights: np.ndarray, top_k: int, floor: float) -> list[list[float]]:
    """(id, weight) pairs over the floor, at most top_k, the heaviest,
    ties by id, sorted by id: select_terms in sparse.rs."""
    keep = [(int(v), float(w)) for v, w in enumerate(weights) if w > floor]
    if top_k > 0 and len(keep) > top_k:
        keep.sort(key=lambda t: (-t[1], t[0]))
        keep = keep[:top_k]
    keep.sort(key=lambda t: t[0])
    return [[v, w] for v, w in keep]


def main() -> None:
    logits = make_logits()
    weights = pool(logits)
    cases = [
        {"options": {"top_k": 0, "floor": 0.0}, "terms": select(weights, 0, 0.0)},
        {"options": {"top_k": 3, "floor": 0.0}, "terms": select(weights, 3, 0.0)},
        {"options": {"top_k": 0, "floor": 0.5}, "terms": select(weights, 0, 0.5)},
        {"options": {"top_k": 5, "floor": 0.75}, "terms": select(weights, 5, 0.75)},
        {"options": {"top_k": 1, "floor": 0.0}, "terms": select(weights, 1, 0.0)},
    ]
    out = {
        "vocab": VOCAB,
        "tokens": TOKENS,
        "logits": [[float(x) for x in row] for row in logits.tolist()],
        "cases": cases,
    }
    with open("golden.json", "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1)
        f.write("\n")
    print(f"golden.json: {VOCAB} terms, {TOKENS} tokens, {len(cases)} cases")


if __name__ == "__main__":
    main()
