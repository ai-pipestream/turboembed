"""MLX's BERT encoder on the token rows turbo-bench measured, for the metal
backend's MLX reference (docs/benchmarks.md, Reference programs).

The encoder is bench/metal-peers/mlx_bench.py's: every weight cast to the
dtype it runs in, `mx.fast` LayerNorm and attention, `mx.compile`d, mean
pooling over the mask and L2 normalization in F32. It runs on the rows as
turbo-bench wrote them (ids, mask and types, batch x seq), a padded batch,
so it computes every position. Untimed runs, then timed ones, each waited
for with `mx.eval`; the vectors of the last run are written back for the
tool to hold against the bundle's reference.

Arguments: --model <dir with config.json and model.safetensors> --dtype
float16|float32 --rows <rows.json> --warmup N --iterations N --version V
--out <receipt.json>. Refuses an MLX whose version is not V. Writes one
JSON object: the MLX version, the SHA-256 of the library that runs the
kernels (lib/libmlx.dylib), of the kernels themselves (lib/mlx.metallib)
and of the Python module (mlx.core), the host's load averages before the
runs, the dtype, batch and seq, each timed run in ms (host wall time of
the encoder and pooling, with the rows already on the device and the
vectors not read back) and the vectors.
"""

import argparse
import hashlib
import json
import math
import os
import time

import mlx.core as mx


def load(model_dir, dtype):
    with open(f"{model_dir}/config.json", encoding="utf-8") as f:
        cfg = json.load(f)
    w = {k.removeprefix("bert."): v.astype(dtype) for k, v in mx.load(f"{model_dir}/model.safetensors").items()}
    # The linear layers' weights transposed once, [in, out].
    for k in list(w):
        if k.endswith(".weight") and w[k].ndim == 2 and "embeddings" not in k:
            w[k] = mx.array(w[k].T)
    mx.eval(w)
    return cfg, w


def encoder(cfg, w):
    layers, heads, hidden = cfg["num_hidden_layers"], cfg["num_attention_heads"], cfg["hidden_size"]
    eps, hd = cfg["layer_norm_eps"], hidden // heads

    def ln(x, p):
        return mx.fast.layer_norm(x, w[p + ".weight"], w[p + ".bias"], eps)

    def linear(x, p):
        return x @ w[p + ".weight"] + w[p + ".bias"]

    def gelu(x):
        return 0.5 * x * (1.0 + mx.erf(x * (1.0 / math.sqrt(2.0))))

    def forward(ids, mask, types):
        b, s = ids.shape
        x = w["embeddings.word_embeddings.weight"][ids]
        x = x + w["embeddings.position_embeddings.weight"][:s] + w["embeddings.token_type_embeddings.weight"][types]
        x = ln(x, "embeddings.LayerNorm")
        add = ((1.0 - mask.astype(x.dtype)) * -1e4)[:, None, None, :]
        for i in range(layers):
            p = f"encoder.layer.{i}."

            def split(t):
                return t.reshape(b, s, heads, hd).transpose(0, 2, 1, 3)

            q = split(linear(x, p + "attention.self.query"))
            k = split(linear(x, p + "attention.self.key"))
            v = split(linear(x, p + "attention.self.value"))
            a = mx.fast.scaled_dot_product_attention(q, k, v, scale=1.0 / math.sqrt(hd), mask=add)
            a = a.transpose(0, 2, 1, 3).reshape(b, s, hidden)
            x = ln(x + linear(a, p + "attention.output.dense"), p + "attention.output.LayerNorm")
            f = gelu(linear(x, p + "intermediate.dense"))
            x = ln(x + linear(f, p + "output.dense"), p + "output.LayerNorm")
        m = mask.astype(mx.float32)[:, :, None]
        pooled = (x.astype(mx.float32) * m).sum(axis=1) / m.sum(axis=1)
        return pooled / mx.maximum(mx.linalg.norm(pooled, axis=1, keepdims=True), 1e-12)

    return mx.compile(forward)


def main():
    a = argparse.ArgumentParser()
    a.add_argument("--model", required=True)
    a.add_argument("--dtype", required=True, choices=["float16", "float32"])
    a.add_argument("--rows", required=True)
    a.add_argument("--warmup", type=int, required=True)
    a.add_argument("--iterations", type=int, required=True)
    a.add_argument("--version", required=True)
    a.add_argument("--out", required=True)
    o = a.parse_args()
    if mx.__version__ != o.version:
        raise SystemExit(f"MLX {mx.__version__} is not the version asked for, {o.version}")
    load = list(os.getloadavg())
    with open(o.rows, encoding="utf-8") as f:
        rows = json.load(f)
    b, s = rows["batch"], rows["seq"]
    shape = lambda v: mx.array(v, dtype=mx.int32).reshape(b, s)
    ids, mask, types = shape(rows["ids"]), shape(rows["mask"]), shape(rows["types"])
    cfg, w = load(o.model, {"float32": mx.float32, "float16": mx.float16}[o.dtype])
    run = encoder(cfg, w)
    for _ in range(o.warmup):
        mx.eval(run(ids, mask, types))
    times = []
    out = None
    for _ in range(o.iterations):
        t0 = time.perf_counter()
        out = run(ids, mask, types)
        mx.eval(out)
        times.append((time.perf_counter() - t0) * 1e3)
    def sha256(path):
        with open(path, "rb") as f:
            return hashlib.sha256(f.read()).hexdigest()

    # mlx.core is the Python module; the kernels run from the library and
    # the metallib the wheel ships beside it, in mlx/lib.
    lib = os.path.join(os.path.dirname(mx.__file__), "lib")
    receipt = {
        "version": mx.__version__,
        "libmlx_sha256": sha256(os.path.join(lib, "libmlx.dylib")),
        "metallib_sha256": sha256(os.path.join(lib, "mlx.metallib")),
        "core_sha256": sha256(mx.__file__),
        "loadavg": load,
        "dtype": o.dtype,
        "batch": b,
        "seq": s,
        "times_ms": times,
        "vectors": out.astype(mx.float32).tolist(),
    }
    with open(o.out, "w", encoding="utf-8") as f:
        json.dump(receipt, f)


if __name__ == "__main__":
    main()
