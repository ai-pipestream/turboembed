"""A BERT sentence encoder written on MLX, timed the way turbo-bench times
the Metal backend: rows of token ids through the encoder, mean pooling and
L2 normalization, 5 untimed runs, then 50 timed, each waited for.

Arguments: the model directory (config.json and model.safetensors), the
dtype (float32 or float16), the batch and the sequence length. The rows
are dense, every token live, as turbo-bench's --rows dense: batch rows of
seq tokens. Prints one JSON line: p50, p99 and the least time in ms, and
rows per second at the p50.
"""

import json
import math
import sys
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

    def forward(ids, mask):
        b, s = ids.shape
        x = w["embeddings.word_embeddings.weight"][ids]
        x = x + w["embeddings.position_embeddings.weight"][:s] + w["embeddings.token_type_embeddings.weight"][0]
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


def main(model_dir, dtype, batch, seq):
    batch, seq = int(batch), int(seq)
    cfg, w = load(model_dir, {"float32": mx.float32, "float16": mx.float16}[dtype])
    run = encoder(cfg, w)
    vocab = cfg["vocab_size"]
    ids = [[1000 + (r * 131 + p * 17) % (vocab - 1000) for p in range(seq)] for r in range(batch)]
    ids, mask = mx.array(ids, dtype=mx.int32), mx.ones((batch, seq), dtype=mx.int32)
    for _ in range(5):
        mx.eval(run(ids, mask))
    times = []
    for _ in range(50):
        t0 = time.perf_counter()
        mx.eval(run(ids, mask))
        times.append((time.perf_counter() - t0) * 1e3)
    times.sort()
    p50 = times[len(times) // 2]
    print(
        json.dumps(
            {
                "tool": "mlx",
                "version": mx.__version__,
                "dtype": dtype,
                "rows": "dense",
                "batch": batch,
                "seq": seq,
                "live_tokens": batch * seq,
                "p50_ms": round(p50, 3),
                "p99_ms": round(times[min(len(times) - 1, int(len(times) * 0.99))], 3),
                "min_ms": round(times[0], 3),
                "rows_per_second": round(batch * 1e3 / p50, 1),
            }
        )
    )


if __name__ == "__main__":
    main(*sys.argv[1:])
