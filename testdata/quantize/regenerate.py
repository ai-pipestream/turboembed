# Regenerates sentence-transformers-145ef47d.txt: int8 and ubinary codes for
# synthetic f32 rows, computed with the expressions of
# sentence_transformers/util/quantization.py (quantize_embeddings) at
# commit 145ef47d5a4bcb9c88ab5e0684e14a4222ef408f, Apache-2.0, with
# explicit ranges. numpy only; run with: python3 -I regenerate.py > sentence-transformers-145ef47d.txt
import numpy as np


def int8(embeddings, ranges):
    ranges = ranges.astype(np.float32, copy=False)
    starts = ranges[0, :]
    steps = (ranges[1, :] - starts) / 255
    steps = np.where(steps == 0, 1, steps)
    q_vals = np.clip(np.floor((embeddings - starts) / steps), 0, 255)
    return (q_vals - 128).astype(np.int8)


def ubinary(embeddings):
    return np.packbits(embeddings > 0, axis=-1)


def hexrow(a):
    return a.view(np.uint8).tobytes().hex()


rng = np.random.default_rng(1462)
print(f"# numpy {np.__version__}; rows: dim, then hex of f32 values, ranges [2, dim], int8 codes, ubinary codes")
for dim in (1, 7, 8, 9, 64):
    x = rng.standard_normal((6, dim)).astype(np.float32)
    calib = rng.standard_normal((200, dim)).astype(np.float32)
    ranges = np.vstack((calib.min(axis=0), calib.max(axis=0))).astype(np.float32)
    # Edge rows: values below min and above max, exact bucket edges, zeros.
    x[0] = ranges[0] - 1
    x[1] = ranges[1] + 1
    x[2] = ranges[0] + (ranges[1] - ranges[0]) / 255 * 3
    x[3] = np.where(np.arange(dim) % 2 == 0, np.float32(0.0), np.float32(-0.0))
    if dim > 1:
        ranges[:, 1] = np.float32(0.25)  # a step of 0 becomes 1
    for r in range(x.shape[0]):
        row = x[r : r + 1]
        print(dim, hexrow(row), hexrow(ranges), hexrow(int8(row, ranges)), hexrow(ubinary(row)))
