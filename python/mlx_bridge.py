"""Persistent stdio bridge between inferstream-backend-apple and MLX.

Spawned ONCE by the Rust server and kept alive: loaded models are cached in
this process, so after the first request a model is hot in unified memory
and per-call latency is just the Metal compute.

Protocol: newline-delimited JSON. One request object per line on stdin;
replies on stdout, one JSON object per line.

Unary ops (exactly one reply line):

    {"op": "ping"}
        -> {"status": "ok", "result": {"mlx_version": ..., "matmul_ok": true,
                                       "device": ...}}

    {"op": "embed", "model": <hf repo id or dir>, "texts": [...],
     "normalize": bool}
        -> {"status": "ok", "result": {"model": ..., "dimensions": N,
                                       "vectors": [[...], ...]}}

    {"op": "describe", "model": <hf repo id or dir>}
        -> loads (and warms) the embedding model, returns its dimension:
           {"status": "ok", "result": {"model": ..., "dimensions": N}}

Streaming ops (N reply lines):

    {"op": "generate", "model": <hf repo id or dir>, "prompt": str,
     "max_tokens": int}
        -> {"status": "chunk", "token": <decoded text piece>}   (per token)
           ...
           {"status": "done", "result": {"model": ...,
                                         "tokens_generated": N}}

Errors come back as {"status": "error", "message": "..."} and the loop keeps
serving — nonzero exits mean the bridge itself crashed. Diagnostics (model
download progress, warnings) go to stderr, never stdout.

Run inside the venv created by scripts/setup-mlx.sh. Owns all MLX/Metal
specifics so the Rust side stays runtime-agnostic.
"""

from __future__ import annotations

import json
import sys

# model id -> (model, tokenizer/processor); separate caches because
# mlx-embeddings and mlx-lm load different module types.
_EMBED_CACHE: dict[str, tuple] = {}
_LM_CACHE: dict[str, tuple] = {}


def _reply(obj: dict) -> None:
    sys.stdout.write(json.dumps(obj) + "\n")
    sys.stdout.flush()


def _error(message: str) -> None:
    _reply({"status": "error", "message": message})


def _load_embed_model(model_id: str) -> tuple:
    if model_id not in _EMBED_CACHE:
        try:
            from mlx_embeddings import load  # type: ignore
        except ImportError as exc:
            raise RuntimeError(
                "mlx-embeddings is not installed in this venv; "
                "run scripts/setup-mlx.sh (uv pip install mlx-embeddings)"
            ) from exc
        print(f"[mlx_bridge] loading embedding model {model_id}", file=sys.stderr)
        _EMBED_CACHE[model_id] = load(model_id)
    return _EMBED_CACHE[model_id]


def _load_lm(model_id: str) -> tuple:
    if model_id not in _LM_CACHE:
        try:
            from mlx_lm import load  # type: ignore
        except ImportError as exc:
            raise RuntimeError(
                "mlx-lm is not installed in this venv; "
                "run scripts/setup-mlx.sh (uv pip install mlx-lm)"
            ) from exc
        print(f"[mlx_bridge] loading LM {model_id}", file=sys.stderr)
        _LM_CACHE[model_id] = load(model_id)
    return _LM_CACHE[model_id]


def op_ping() -> dict:
    import mlx.core as mx

    a = mx.random.normal((64, 64))
    b = mx.random.normal((64, 64))
    c = (a @ b).sum()
    mx.eval(c)  # forces the Metal kernel to actually run
    metal = getattr(mx, "metal", None)
    metal_available = bool(metal is not None and metal.is_available())
    result = {
        "mlx_version": mx.__version__,
        "matmul_ok": bool(c.item() == c.item()),  # not NaN
        "device": str(mx.default_device()),
        "metal_available": metal_available,
    }
    if metal_available:
        result["active_memory"] = int(metal.get_active_memory())
        result["peak_memory"] = int(metal.get_peak_memory())
    return result


def _embed_vectors(model_id: str, texts: list[str], normalize: bool) -> list[list[float]]:
    import mlx.core as mx
    from mlx_embeddings import generate  # type: ignore

    model, tokenizer = _load_embed_model(model_id)
    output = generate(model, tokenizer, texts=texts)
    vectors = output.text_embeds  # (batch, dim), already pooled
    if normalize:
        norms = mx.maximum(mx.linalg.norm(vectors, axis=1, keepdims=True), 1e-12)
        vectors = vectors / norms
    return [[float(x) for x in row] for row in vectors.tolist()]


def op_embed(req: dict) -> dict:
    model_id = req["model"]
    texts = req["texts"]
    normalize = bool(req.get("normalize", True))
    if not texts:
        raise ValueError("embed called with no texts")
    vectors = _embed_vectors(model_id, texts, normalize)
    return {"model": model_id, "dimensions": len(vectors[0]), "vectors": vectors}


def op_describe(req: dict) -> dict:
    model_id = req["model"]
    vectors = _embed_vectors(model_id, ["warmup"], False)
    return {"model": model_id, "dimensions": len(vectors[0])}


def op_generate(req: dict) -> None:
    """Streaming op: emits chunk lines itself, then the done line."""
    model_id = req["model"]
    prompt = req["prompt"]
    max_tokens = int(req.get("max_tokens", 256))

    try:
        from mlx_lm import stream_generate  # type: ignore
    except ImportError as exc:
        raise RuntimeError(
            "mlx-lm is not installed in this venv; "
            "run scripts/setup-mlx.sh (uv pip install mlx-lm)"
        ) from exc

    model, tokenizer = _load_lm(model_id)
    # Chat models decode garbage on bare prompts; apply the model's chat
    # template when it has one.
    if getattr(tokenizer, "chat_template", None):
        prompt = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
        )

    count = 0
    for response in stream_generate(model, tokenizer, prompt=prompt, max_tokens=max_tokens):
        _reply({"status": "chunk", "token": response.text})
        count += 1
    _reply({"status": "done", "result": {"model": model_id, "tokens_generated": count}})


def main() -> None:
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            req = json.loads(line)
        except json.JSONDecodeError as exc:
            _error(f"malformed request JSON: {exc}")
            continue

        op = req.get("op")
        try:
            if op == "ping":
                _reply({"status": "ok", "result": op_ping()})
            elif op == "embed":
                _reply({"status": "ok", "result": op_embed(req)})
            elif op == "describe":
                _reply({"status": "ok", "result": op_describe(req)})
            elif op == "generate":
                op_generate(req)  # streams its own chunk/done lines
            else:
                _error(f"unknown op: {op!r}")
        except Exception as exc:  # protocol-level error, deliberate catch-all
            _error(f"{type(exc).__name__}: {exc}")


if __name__ == "__main__":
    main()
